"""Export item embeddings from a SPA or CCA adapter checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .checkpoint import load_checkpoint_metadata, load_encoder
from .config import resolve_path
from .io import (
    collect_image_paths,
    load_item2id,
    read_item_text_records,
    save_embeddings,
    sanitize_name,
    validate_output_paths,
)
from .tokenization import ItemTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", default="Office", help="Comma-separated dataset names")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--model-path", default=None, help="Relocated copy of the same base model")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"))
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float32"), default="auto")
    parser.add_argument(
        "--input-mode", choices=("text_image", "text", "image"), default="text_image"
    )
    parser.add_argument("--item-col", default="parent_asin")
    parser.add_argument("--title-col", default="title")
    parser.add_argument("--duplicate", choices=("keep_first", "keep_last"), default="keep_first")
    parser.add_argument("--feature-name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-text", action="store_true")
    parser.add_argument("--allow-missing-image", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    for name in ("data-dir", "item-map", "text-file", "image-dir", "output-dir"):
        parser.add_argument(f"--{name}", default=None, help="Single-dataset path override")
    return parser


def prepare_dataset(args: argparse.Namespace, metadata: dict[str, Any]) -> dict[str, Any]:
    data_dir = (
        resolve_path(args.data_dir)
        if args.data_dir
        else resolve_path(args.processed_root) / args.dataset
    )
    item_map = resolve_path(args.item_map) if args.item_map else data_dir / "item2id.json"
    text_file = resolve_path(args.text_file) if args.text_file else data_dir / "text.tsv"
    image_dir = resolve_path(args.image_dir) if args.image_dir else data_dir / "images"
    output_dir = resolve_path(args.output_dir) if args.output_dir else data_dir / "features"
    feature_name = args.feature_name or f"{metadata['stage']}_embeddings"
    if args.feature_name is None:
        if args.input_mode != "text_image":
            feature_name += f"_{args.input_mode}"
        if args.limit is not None:
            feature_name += f"_limit{args.limit}"
    feature_name = sanitize_name(feature_name)
    validate_output_paths(output_dir, feature_name, args.allow_overwrite)
    item2id = load_item2id(item_map)
    records: list[dict[str, str] | None] = [None] * len(item2id)
    image_paths: list[Path | None] = [None] * len(item2id)
    skipped_text = duplicate_text = skipped_images = 0
    if args.input_mode != "image":
        records, skipped_text, duplicate_text = read_item_text_records(
            text_file,
            item2id,
            args.item_col,
            args.title_col,
            args.duplicate,
        )
    if args.input_mode != "text":
        image_paths, skipped_images, _ = collect_image_paths(image_dir, item2id)
    missing_text = [
        i for i, value in enumerate(records) if args.input_mode != "image" and value is None
    ]
    missing_images = [
        i for i, value in enumerate(image_paths) if args.input_mode != "text" and value is None
    ]
    if missing_text and not args.allow_missing_text:
        raise ValueError(f"[{args.dataset}] {len(missing_text)} items have no usable title")
    if missing_images and not args.allow_missing_image:
        raise ValueError(f"[{args.dataset}] {len(missing_images)} items have no image")
    missing_set = set(missing_text) | set(missing_images)
    item_ids = [i for i in range(len(item2id)) if i not in missing_set]
    if args.limit is not None:
        item_ids = item_ids[: args.limit]
    if not item_ids:
        raise ValueError(f"[{args.dataset}] No items are eligible for export")
    missing_ids = sorted(set(range(len(item2id))) - set(item_ids))
    return {
        "item2id": item2id,
        "records": records,
        "image_paths": image_paths,
        "item_ids": item_ids,
        "missing_ids": missing_ids,
        "output_dir": output_dir,
        "feature_name": feature_name,
        "text_file": text_file,
        "image_dir": image_dir,
        "missing_text_items": len(missing_text),
        "missing_image_items": len(missing_images),
        "skipped_text_rows": skipped_text,
        "duplicate_text_rows": duplicate_text,
        "skipped_image_files": skipped_images,
    }


def export_dataset(
    args: argparse.Namespace,
    metadata: dict[str, Any],
    encoder: Any = None,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prepared = prepare_dataset(args, metadata) if prepared is None else prepared
    item_ids, item2id = prepared["item_ids"], prepared["item2id"]
    summary = {
        "dataset": args.dataset,
        "total_items": len(item2id),
        "covered_items": len(item_ids),
        "output": str(prepared["output_dir"] / f"{prepared['feature_name']}.npy"),
    }
    if args.dry_run:
        return summary
    if encoder is None:
        encoder, metadata = load_encoder(
            args.checkpoint,
            model_path=args.model_path,
            device=args.device,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
        )
    device = next(encoder.parameters()).device
    tokenizer = ItemTokenizer(
        encoder.processor,
        metadata["item_prompt"],
        max_length=metadata["max_length"],
        max_char_length=metadata["max_char_length"],
        item_text_mode=metadata["item_text_mode"],
        input_mode=args.input_mode,
    )
    normalization = metadata.get(
        "title_normalization",
        "collapse_whitespace_then_strip" if metadata["stage"] == "cca" else "strip",
    )
    started_utc, started = datetime.now(timezone.utc), time.monotonic()
    matrix = None
    encoder.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(item_ids), args.batch_size), desc=args.dataset):
            batch_ids = item_ids[start : start + args.batch_size]
            items = []
            for index in batch_ids:
                record = prepared["records"][index]
                title = record["title"] if record is not None else ""
                if normalization == "collapse_whitespace_then_strip":
                    title = re.sub(r"\s+", " ", title).strip()
                items.append({"title": title, "image_path": prepared["image_paths"][index]})
            encoded = {key: value.to(device) for key, value in tokenizer(items).items()}
            embeddings = encoder.encode(encoded).float()
            if embeddings.ndim != 2 or embeddings.size(0) != len(batch_ids):
                raise ValueError("Encoder returned an unexpected embedding shape")
            if not torch.isfinite(embeddings).all() or torch.any(embeddings.norm(dim=-1) == 0):
                raise ValueError("Encoder returned nonfinite or zero embeddings")
            vectors = F.normalize(embeddings, p=2, dim=-1).cpu().numpy().astype(np.float32)
            if matrix is None:
                matrix = np.zeros((len(item2id), vectors.shape[1]), dtype=np.float32)
            matrix[batch_ids] = vectors
    elapsed = time.monotonic() - started
    output_metadata = {
        "stage": metadata["stage"],
        "dataset": args.dataset,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "checkpoint_metadata": metadata,
        "input_mode": args.input_mode,
        "item_prompt": metadata["item_prompt"],
        "pooling": metadata["pooling"],
        "title_normalization": normalization,
        "l2_normalized": True,
        "batch_size": args.batch_size,
        "device": str(device),
        "missing_text_items": prepared["missing_text_items"],
        "missing_image_items": prepared["missing_image_items"],
        "skipped_text_rows": prepared["skipped_text_rows"],
        "duplicate_text_rows": prepared["duplicate_text_rows"],
        "export_runtime": {
            "scope": "tokenization_and_encoding",
            "started_at_utc": started_utc.isoformat(timespec="seconds"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 6),
            "items_per_second": round(len(item_ids) / max(elapsed, 1e-9), 3),
        },
    }
    path = save_embeddings(
        matrix,
        prepared["output_dir"],
        prepared["feature_name"],
        item2id,
        output_metadata,
        prepared["missing_ids"],
        args.allow_overwrite,
    )
    return dict(summary, output=str(path), elapsed_seconds=round(elapsed, 3))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    datasets = args.datasets.split(",")
    if len(set(datasets)) != len(datasets) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in datasets
    ):
        raise ValueError("datasets must contain distinct comma-separated names")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("batch-size and limit must be positive")
    if len(datasets) > 1 and any(
        getattr(args, name)
        for name in ("data_dir", "item_map", "text_file", "image_dir", "output_dir")
    ):
        raise ValueError(
            "Path overrides apply to one dataset; use processed-root for multiple datasets"
        )
    args.checkpoint = str(resolve_path(args.checkpoint))
    metadata = load_checkpoint_metadata(args.checkpoint)
    jobs = []
    for dataset in datasets:
        dataset_args = argparse.Namespace(**vars(args), dataset=dataset)
        jobs.append((dataset_args, prepare_dataset(dataset_args, metadata)))
    encoder = None
    if not args.dry_run:
        encoder, metadata = load_encoder(
            args.checkpoint,
            model_path=args.model_path,
            device=args.device,
            attn_implementation=args.attn_implementation,
            dtype=args.dtype,
        )
    for dataset_args, prepared in jobs:
        result = export_dataset(dataset_args, metadata, encoder, prepared)
        print(json.dumps(dict(result, dry_run=args.dry_run), ensure_ascii=False))


if __name__ == "__main__":
    main()
