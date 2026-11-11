"""Processed item catalog IO and embedding matrix persistence."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sanitize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not name:
        raise ValueError("Feature name must contain a letter or number")
    return name


def load_item2id(path: str | Path) -> dict[str, int]:
    with Path(path).open(encoding="utf-8") as stream:
        item2id = json.load(stream)
    if not isinstance(item2id, dict) or not item2id:
        raise ValueError("item2id.json must be a nonempty object")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in item2id.values()):
        raise ValueError("Mapped item IDs must be integers")
    if sorted(item2id.values()) != list(range(len(item2id))):
        raise ValueError("Mapped item IDs must be unique and contiguous from zero")
    return item2id


def read_item_text_records(
    text_path: str | Path,
    item2id: dict[str, int],
    item_col: str = "parent_asin",
    title_col: str = "title",
    duplicate: str = "keep_first",
) -> tuple[list[dict[str, str] | None], int, int]:
    if duplicate not in {"keep_first", "keep_last"}:
        raise ValueError("duplicate must be keep_first or keep_last")
    records: list[dict[str, str] | None] = [None] * len(item2id)
    skipped = duplicates = 0
    with Path(text_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if not {item_col, title_col}.issubset(reader.fieldnames or []):
            raise ValueError(f"TSV must contain {item_col!r} and {title_col!r} columns")
        for row in reader:
            raw_id, title = row[item_col], (row[title_col] or "").strip()
            if raw_id not in item2id or not title or title.lower() == "nan":
                skipped += 1
                continue
            index = item2id[raw_id]
            if records[index] is not None:
                duplicates += 1
                if duplicate == "keep_first":
                    continue
            records[index] = {"raw_item_id": raw_id, "title": title}
    return records, skipped, duplicates


def collect_image_paths(
    image_dir: str | Path,
    item2id: dict[str, int],
) -> tuple[list[Path | None], int, int]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths: list[Path | None] = [None] * len(item2id)
    for raw_id, index in item2id.items():
        if Path(raw_id).name != raw_id or raw_id in {".", ".."}:
            raise ValueError(f"Item ID cannot be used as an image filename: {raw_id!r}")
        primary = image_dir / f"{raw_id}.jpg"
        if primary.is_file():
            paths[index] = primary
    fallback: dict[str, list[Path]] = {}
    skipped = 0
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem not in item2id:
            skipped += 1
        elif paths[item2id[path.stem]] is None:
            fallback.setdefault(path.stem, []).append(path)
    for raw_id, candidates in fallback.items():
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous fallback images for {raw_id}: {candidates}")
        paths[item2id[raw_id]] = candidates[0]
    return paths, skipped, 0


def validate_output_paths(
    output_dir: str | Path,
    feature_name: str,
    allow_overwrite: bool = False,
) -> None:
    output_dir, feature_name = Path(output_dir), sanitize_name(feature_name)
    existing = [
        output_dir / f"{feature_name}{suffix}"
        for suffix in (".npy", ".meta.json", ".missing_items.tsv")
        if (output_dir / f"{feature_name}{suffix}").exists()
    ]
    if existing and not allow_overwrite:
        raise FileExistsError(f"Feature files already exist: {existing}")


def save_embeddings(
    matrix: np.ndarray,
    output_dir: str | Path,
    feature_name: str,
    item2id: dict[str, int],
    metadata: dict[str, Any],
    missing_ids: list[int],
    allow_overwrite: bool = False,
) -> Path:
    if matrix.ndim != 2 or matrix.shape[0] != len(item2id):
        raise ValueError("Embedding rows must match item2id.json")
    output_dir, feature_name = Path(output_dir), sanitize_name(feature_name)
    validate_output_paths(output_dir, feature_name, allow_overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / f"{feature_name}.npy"
    np.save(matrix_path, matrix.astype(np.float32, copy=False))
    coverage = {
        "feature_name": feature_name,
        "total_items": len(item2id),
        "covered_items": len(item2id) - len(missing_ids),
        "missing_items": len(missing_ids),
    }
    saved_metadata = dict(
        metadata,
        feature_name=feature_name,
        matrix_file=matrix_path.name,
        shape=list(matrix.shape),
        dtype="float32",
        coverage=coverage,
        item_id_semantics="row i corresponds to mapped item i in item2id.json",
    )
    with matrix_path.with_suffix(".meta.json").open("w", encoding="utf-8") as stream:
        json.dump(saved_metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    id2item = {index: raw_id for raw_id, index in item2id.items()}
    with matrix_path.with_suffix(".missing_items.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["item_id", "raw_item_id"])
        writer.writerows((index, id2item[index]) for index in missing_ids)
    coverage_path = output_dir / "feature_coverage.tsv"
    rows = []
    if coverage_path.exists():
        with coverage_path.open(encoding="utf-8", newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream, delimiter="\t")
                if row["feature_name"] != feature_name
            ]
    rows.append(coverage)
    with coverage_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["feature_name"]))
    return matrix_path
