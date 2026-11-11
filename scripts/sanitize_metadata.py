#!/usr/bin/env python3
"""Remove machine paths and credentials from a JSON publication copy."""

import argparse
import hashlib
import json
import ntpath
from pathlib import Path


PRIVATE_FIELDS = {
    "api_key", "access_token", "hub_token", "push_to_hub_token", "password",
    "secret", "token", "hostname", "host_name", "username",
}
MODEL_FIELDS = {
    "model_path", "runtime_model_path", "stage1_model_path",
    "base_model_name_or_path", "adapter_base_model", "model_name_or_path",
}
PUBLIC_PATHS = {
    "stage1_checkpoint": "/path/to/UniMI_data/checkpoints/unimi_spa",
    "image_root": "/path/to/UniMI_data/datasets/processed",
    "processed_root": "/path/to/UniMI_data/datasets/processed",
    "lora_config": "/path/to/UniMI/configs/lora.json",
    "lora_template_path": "/path/to/UniMI/configs/lora.json",
}


def sanitize_metadata(value, key=""):
    if isinstance(value, dict):
        return {
            name: sanitize_metadata(item, name)
            for name, item in value.items() if name.lower() not in PRIVATE_FIELDS
        }
    if isinstance(value, list):
        return [sanitize_metadata(item, key) for item in value]
    if isinstance(value, str) and (ntpath.isabs(value) or value.startswith("file://")):
        if key in MODEL_FIELDS:
            return "/path/to/Qwen2-VL-7B"
        if key in PUBLIC_PATHS:
            return PUBLIC_PATHS[key]
        if value.startswith("/path/to/"):
            return value
        if key in {"train_file", "train_data"}:
            return "/path/to/train.jsonl"
        if key in {"checkpoint", "output_dir", "logging_dir", "resume_from_checkpoint"}:
            return "/path/to/checkpoint"
        return "/path/to/resource"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Choose a new output path; keep the original runtime metadata.")
    original = args.source.read_bytes()
    cleaned = (json.dumps(sanitize_metadata(json.loads(original)), indent=2,
                          ensure_ascii=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(cleaned)
    print("Source SHA-256:", hashlib.sha256(original).hexdigest())
    print("Release SHA-256:", hashlib.sha256(cleaned).hexdigest())


if __name__ == "__main__":
    main()
