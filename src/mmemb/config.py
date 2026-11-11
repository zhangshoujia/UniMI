"""Project paths and the configuration shared by Python and shell entry points."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("MMEMB_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"MMEMB_PROJECT_ROOT has no pyproject.toml: {root}")
        return root
    for root in Path(__file__).resolve().parents:
        manifest = root / "pyproject.toml"
        if manifest.is_file() and 'name = "unimi"' in manifest.read_text():
            return root
    raise RuntimeError(
        "Cannot locate the UniMI checkout. Install it with pip install -e . "
        "or set MMEMB_PROJECT_ROOT to the checkout directory."
    )


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root() / path).resolve()


_PATH_FIELDS = {
    "model_path",
    "train_file",
    "image_root",
    "output_dir",
    "lora_config",
    "stage1_checkpoint",
}
_NULL_INT_FIELDS = {
    "max_train_samples",
    "max_length",
    "max_char_length",
    "image_min_pixels",
    "image_max_pixels",
}


def load_config(stage: str, argv: list[str] | None = None) -> dict:
    """Load the stage defaults, an optional JSON overlay, and CLI overrides."""
    if stage not in {"spa", "cca"}:
        raise ValueError(f"Unknown training stage: {stage}")
    default_path = project_root() / "configs" / f"{stage}.json"
    defaults = json.loads(default_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=f"Train {stage.upper()} item embeddings.")
    parser.add_argument("--config", default=str(default_path), help="JSON configuration overlay.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without loading data or weights.",
    )
    for name, value in defaults.items():
        if name == "stage":
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
            )
        else:
            value_type = (
                int if name in _NULL_INT_FIELDS else (type(value) if value is not None else str)
            )
            parser.add_argument(flag, type=value_type, default=argparse.SUPPRESS)
    arguments = vars(parser.parse_args(argv))
    config_path = resolve_path(arguments.pop("config"))
    dry_run = arguments.pop("dry_run")
    overlay = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(overlay, dict):
        raise ValueError(f"Configuration must be a JSON object: {config_path}")
    unknown = set(overlay) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    config = defaults | overlay | arguments
    if config["stage"] != stage:
        raise ValueError(f"Expected stage={stage!r} in {config_path}")
    for name in _PATH_FIELDS:
        if config.get(name) is not None:
            config[name] = str(resolve_path(config[name]))
    if config.get("resume_from_checkpoint") not in {None, "latest"}:
        config["resume_from_checkpoint"] = str(resolve_path(config["resume_from_checkpoint"]))
    config["configuration_file"] = str(config_path)
    config["dry_run"] = dry_run
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    positive_ints = {
        "batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "save_steps",
        "eval_steps",
        "save_total_limit",
        "reference_num_processes",
        "max_length",
        "max_char_length",
        "image_min_pixels",
        "image_max_pixels",
        "num_negatives",
        "max_train_samples",
    }
    for name in positive_ints:
        value = config.get(name)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    for name in ("seed", "dataloader_num_workers"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("beta", "temperature", "learning_rate", "num_epochs"):
        if name in config and (
            not isinstance(config[name], (int, float))
            or not math.isfinite(config[name])
            or config[name] <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "max_grad_norm"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if type(config["max_steps"]) is not int or config["max_steps"] == 0 or config["max_steps"] < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    for name, limit in (("validation_ratio", 1.0), ("warmup_ratio", 1.000000000001)):
        if not math.isfinite(config[name]) or not 0 <= config[name] < limit:
            raise ValueError(f"Invalid {name}: {config[name]}")
    for name in ("bf16", "tf32", "overwrite_output_dir"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config["attn_implementation"] not in {"sdpa", "flash_attention_2", "eager"}:
        raise ValueError("attn_implementation must be sdpa, flash_attention_2, or eager")
    if config.get("pooling") not in {None, "eos"} or config.get("item_text_mode") not in {
        None,
        "title",
    }:
        raise ValueError("This release uses EOS pooling and title-only text")
    if config["stage"] == "spa" and any(
        config.get(name) is None
        for name in (
            "model_path",
            "lora_config",
            "pooling",
            "item_text_mode",
            "max_length",
            "max_char_length",
            "image_min_pixels",
            "image_max_pixels",
        )
    ):
        raise ValueError("SPA model, LoRA, and representation settings must be specified")
    if config.get("max_length") is not None and config["max_length"] < 2:
        raise ValueError("max_length must leave room for a terminal EOS")
    expected_normalization = (
        "strip" if config["stage"] == "spa" else "collapse_whitespace_then_strip"
    )
    if config["title_normalization"] != expected_normalization:
        raise ValueError(
            f"title_normalization must be {expected_normalization!r} for {config['stage']}"
        )
    if not isinstance(config["item_prompt"], str) or not config["item_prompt"].strip():
        raise ValueError("item_prompt must be nonempty")
    if config.get("image_min_pixels") is not None and config.get("image_max_pixels") is not None:
        if config["image_min_pixels"] > config["image_max_pixels"]:
            raise ValueError("image_min_pixels cannot exceed image_max_pixels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a configured launcher setting.")
    parser.add_argument("stage", choices=("spa", "cca"))
    parser.add_argument(
        "--print-field",
        choices=("reference_num_processes", "mixed_precision"),
        default="reference_num_processes",
    )
    args, remaining = parser.parse_known_args()
    config = load_config(args.stage, remaining)
    print(
        ("bf16" if config["bf16"] else "no")
        if args.print_field == "mixed_precision"
        else config["reference_num_processes"]
    )


if __name__ == "__main__":
    main()
