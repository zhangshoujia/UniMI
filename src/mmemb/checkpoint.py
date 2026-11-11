"""Adapter metadata, legacy import, and content-based resume checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .config import resolve_path


CONFIG_NAME = "embedding_config.json"
SCHEMA_VERSION = 1
REPRESENTATION_FIELDS = (
    "item_text_mode",
    "pooling",
    "max_length",
    "max_char_length",
    "image_min_pixels",
    "image_max_pixels",
)


def read_json(path: str | Path) -> dict:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_weight_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = checkpoint / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Adapter weights not found: {checkpoint}")


def validate_adapter(checkpoint: str | Path) -> dict:
    checkpoint = Path(checkpoint)
    config = read_json(checkpoint / "adapter_config.json")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(f"Expected a LoRA checkpoint: {checkpoint}")
    for key in ("target_modules", "modules_to_save"):
        modules = config.get(key) or []
        modules = [modules] if isinstance(modules, str) else modules
        if any("lm_head" in str(name).lower() for name in modules):
            raise ValueError("The item encoder does not train lm_head adapters")
    adapter_weight_path(checkpoint)
    return config


def _model_reference(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or str(value).startswith(".") or path.exists():
        return str(path.resolve())
    return value.rstrip("/")


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict:
    """Normalize native or released SPA/CCA metadata to one representation."""
    checkpoint = resolve_path(checkpoint_path)
    adapter = validate_adapter(checkpoint)
    native = checkpoint / CONFIG_NAME
    if native.is_file():
        metadata = read_json(native)
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema: {native}")
        expected_adapter_base = metadata.get("adapter_base_model")
    else:
        candidates = (
            ("semantic_preference_config.json", "spa", "semantic_preference_alignment"),
            ("collaborative_contrastive_config.json", "cca", "collaborative_contrastive_alignment"),
        )
        found = [
            (checkpoint / name, stage, old_stage)
            for name, stage, old_stage in candidates
            if (checkpoint / name).is_file()
        ]
        if len(found) != 1:
            raise ValueError(f"Expected one SPA or CCA metadata file in {checkpoint}")
        source, stage, old_stage = found[0]
        metadata = read_json(source)
        if metadata.get("stage") != old_stage:
            raise ValueError(f"Unexpected stage in {source}")
        if metadata.get("direct_pooling_method") != "last":
            raise ValueError("This release supports checkpoints trained with terminal EOS pooling")
        if stage == "spa" and metadata.get("preference_loss") != "pairwise":
            raise ValueError("This release supports pairwise SPA checkpoints")
        if stage == "cca" and (
            metadata.get("num_hard_negatives", 0) != 0 or metadata.get("in_batch_negatives", False)
        ):
            raise ValueError(
                "This release supports CCA checkpoints with one explicit negative pool"
            )
        metadata = dict(metadata)
        expected_adapter_base = (
            metadata.get("stage1_model_path", metadata.get("model_path"))
            if stage == "cca"
            else metadata.get("model_path")
        )
        metadata.update(
            stage=stage,
            pooling="eos",
            model_path=metadata.get("runtime_model_path", metadata.get("model_path")),
            title_normalization="strip" if stage == "spa" else "collapse_whitespace_then_strip",
            legacy_metadata_file=source.name,
        )
        if stage == "cca":
            metadata["num_negatives"] = metadata["num_normal_negatives"]
    if metadata.get("stage") not in {"spa", "cca"}:
        raise ValueError(f"Invalid checkpoint stage: {checkpoint}")
    if metadata.get("pooling") != "eos" or metadata.get("item_text_mode") != "title":
        raise ValueError("This release requires title-only text and terminal EOS pooling")
    expected_normalization = (
        "strip" if metadata["stage"] == "spa" else "collapse_whitespace_then_strip"
    )
    if metadata.get("title_normalization") != expected_normalization:
        raise ValueError(f"Invalid title normalization for {metadata['stage']}")
    for key in ("item_prompt", "model_path", "attn_implementation"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Missing or invalid checkpoint field: {key}")
    for key in ("max_length", "max_char_length", "image_min_pixels", "image_max_pixels"):
        if type(metadata.get(key)) is not int or metadata[key] < 1:
            raise ValueError(f"Missing or invalid checkpoint field: {key}")
    if type(metadata.get("bf16")) is not bool:
        raise ValueError("Missing or invalid checkpoint field: bf16")
    if metadata["max_length"] < 2 or metadata["image_min_pixels"] > metadata["image_max_pixels"]:
        raise ValueError("Checkpoint sequence length or image pixel bounds are invalid")
    if metadata["attn_implementation"] not in {"sdpa", "flash_attention_2", "eager"}:
        raise ValueError("Unsupported checkpoint attention backend")
    objective_key = "beta" if metadata["stage"] == "spa" else "temperature"
    objective_value = metadata.get(objective_key)
    if (
        isinstance(objective_value, bool)
        or not isinstance(objective_value, (float, int))
        or not math.isfinite(objective_value)
        or objective_value <= 0
    ):
        raise ValueError(f"Missing or invalid checkpoint field: {objective_key}")
    if metadata["stage"] == "cca" and (
        type(metadata.get("num_negatives")) is not int or metadata["num_negatives"] < 1
    ):
        raise ValueError("Missing or invalid checkpoint field: num_negatives")
    adapter_base = adapter.get("base_model_name_or_path")
    if not isinstance(adapter_base, str) or not adapter_base.strip():
        raise ValueError("The adapter does not identify its base model")
    if not isinstance(expected_adapter_base, str) or not expected_adapter_base.strip():
        raise ValueError("Checkpoint metadata does not identify the adapter's original base model")
    if _model_reference(adapter_base) != _model_reference(expected_adapter_base):
        raise ValueError("Checkpoint metadata and adapter disagree on the original base model")
    metadata["adapter_base_model"] = adapter_base
    return metadata


def resolve_model_path(metadata: dict, override: str | Path | None = None) -> Path:
    path = resolve_path(override or metadata["model_path"])
    if not path.is_dir():
        raise FileNotFoundError(
            f"Base-model directory not found: {path}; pass --model-path if the weights moved"
        )
    config_file = path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Base-model config not found: {config_file}")
    recorded_hash = metadata.get("base_model_config_sha256")
    old_config = Path(metadata["model_path"]).expanduser() / "config.json"
    if recorded_hash is None and old_config.is_file():
        recorded_hash = sha256(old_config)
    if recorded_hash is not None and sha256(config_file) != recorded_hash:
        raise ValueError("Relocated base-model configuration differs from the checkpoint")
    return path


def choose_processor_source(
    checkpoint_path: str | Path, metadata: dict, base_path: str | Path
) -> Path:
    """Use the processor saved with native adapters; legacy exports used base files."""
    if metadata.get("schema_version") != SCHEMA_VERSION:
        return Path(base_path)
    checkpoint = resolve_path(checkpoint_path)
    for name in ("preprocessor_config.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(
                f"Native checkpoint processor file not found: {checkpoint / name}"
            )
    return checkpoint


def load_encoder(
    checkpoint_path: str | Path,
    model_path: str | Path | None = None,
    device: str = "auto",
    attn_implementation: str | None = None,
    dtype: str = "auto",
):
    """Load the base model and one cumulative adapter for inference."""
    import torch
    from peft import PeftModel

    from .encoder import Qwen2VLItemEncoder

    metadata = load_checkpoint_metadata(checkpoint_path)
    runtime_path = resolve_model_path(metadata, model_path)
    if dtype not in {"auto", "bfloat16", "float32"}:
        raise ValueError("dtype must be auto, bfloat16, or float32")
    selected_dtype = "bfloat16" if metadata["bf16"] else "float32"
    selected_dtype = selected_dtype if dtype == "auto" else dtype
    selected_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    encoder = Qwen2VLItemEncoder(
        model_path=str(runtime_path),
        processor_path=str(choose_processor_source(checkpoint_path, metadata, runtime_path)),
        image_min_pixels=metadata["image_min_pixels"],
        image_max_pixels=metadata["image_max_pixels"],
        attn_implementation=attn_implementation or metadata["attn_implementation"],
        torch_dtype=getattr(torch, selected_dtype),
        pooling=metadata["pooling"],
    )
    encoder.model = PeftModel.from_pretrained(
        encoder.model, str(resolve_path(checkpoint_path)), is_trainable=False
    )
    encoder.model.gradient_checkpointing_disable()
    encoder.model.config.use_cache = False
    encoder.requires_grad_(False).to(selected_device).eval()
    metadata = dict(metadata)
    metadata.update(
        runtime_model_path=str(runtime_path),
        export_dtype=selected_dtype,
        export_attn_implementation=attn_implementation or metadata["attn_implementation"],
    )
    return encoder, metadata


def resolve_resume_checkpoint(value: str | None, output_dir: str | Path) -> Path | None:
    if value is None:
        return None
    if value == "latest":
        candidates = [
            p
            for p in Path(output_dir).glob("checkpoint-*")
            if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* found under {output_dir}")
        checkpoint = max(candidates, key=lambda p: int(p.name.removeprefix("checkpoint-")))
    else:
        checkpoint = resolve_path(value)
    required = (
        CONFIG_NAME,
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        raise FileNotFoundError(
            f"Incomplete native resume checkpoint {checkpoint}: {', '.join(missing)}"
        )
    load_checkpoint_metadata(checkpoint)
    return checkpoint.resolve()


def validate_resume(checkpoint: Path | None, current: dict) -> None:
    if checkpoint is None:
        return
    previous = load_checkpoint_metadata(checkpoint)
    # Locations and logging intervals may change; optimization, data content,
    # adapter initialization, and the representation must stay identical.
    keys = (
        "stage",
        "item_prompt",
        "title_normalization",
        *REPRESENTATION_FIELDS,
        "beta",
        "temperature",
        "num_negatives",
        "seed",
        "validation_ratio",
        "max_train_samples",
        "batch_size",
        "gradient_accumulation_steps",
        "world_size",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "lr_scheduler_type",
        "optim",
        "max_grad_norm",
        "max_steps",
        "num_epochs",
        "bf16",
        "tf32",
        "attn_implementation",
        "dataloader_num_workers",
        "train_file_sha256",
        "base_model_config_sha256",
        "lora_config_sha256",
        "stage1_adapter_sha256",
    )
    mismatches = [key for key in keys if previous.get(key) != current.get(key)]
    if mismatches:
        raise ValueError(f"Resume settings or input content differ: {', '.join(mismatches)}")
