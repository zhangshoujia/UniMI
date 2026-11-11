"""Shared JSONL, item metadata, image lookup, and dataset splitting."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Subset


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def require_string(value: object, label: str, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string at {location}")
    return value.strip()


def require_name(value: object, label: str, location: str) -> str:
    name = require_string(value, label, location)
    if name in {".", ".."} or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe {label} {name!r} at {location}")
    return name


def require_number(value: object, label: str, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number at {location}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite at {location}")
    return number


def require_integer(value: object, label: str, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum} at {location}")
    return value


def read_jsonl(
    path: str | Path, max_samples: int | None = None
) -> Iterator[tuple[dict[str, Any], str]]:
    path = Path(path).expanduser().resolve()
    if max_samples is not None:
        require_integer(max_samples, "max_samples", str(path), minimum=1)
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if max_samples is not None and count >= max_samples:
                break
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {location}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {location}")
            count += 1
            yield record, location


class ItemRegistry:
    """Share repeated items and resolve JPG-first images within a dataset."""

    def __init__(
        self, image_root: str | Path, max_char_length: int, *, collapse_whitespace: bool
    ) -> None:
        self.image_root = Path(image_root).expanduser().resolve()
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root not found: {self.image_root}")
        require_integer(max_char_length, "max_char_length", "dataset", minimum=1)
        self.max_char_length = max_char_length
        self.collapse_whitespace = collapse_whitespace
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self._mapped_ids: dict[tuple[str, int], str] = {}
        self._fallbacks: dict[str, dict[str, list[Path]]] = {}

    def _image_path(self, dataset: str, item_id: str) -> Path:
        directory = self.image_root / dataset / "images"
        primary = directory / f"{item_id}.jpg"
        if primary.is_file():
            return primary
        if dataset not in self._fallbacks:
            if not directory.is_dir():
                raise FileNotFoundError(f"Image directory not found: {directory}")
            index: dict[str, list[Path]] = {}
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    index.setdefault(path.stem, []).append(path)
            self._fallbacks[dataset] = index
        candidates = self._fallbacks[dataset].get(item_id, [])
        if len(candidates) > 1:
            raise ValueError(f"Multiple fallback images for {dataset}/{item_id}")
        if not candidates:
            raise FileNotFoundError(f"Image not found: {primary}")
        return candidates[0]

    def intern(
        self, value: object, dataset: str, location: str, *, mapped_id: bool = False
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"Item must be an object at {location}")
        item_id = require_name(value.get("item_id"), "item_id", location)
        title = require_string(value.get("title"), "title", location)
        if self.collapse_whitespace:
            title = re.sub(r"\s+", " ", title).strip()
        item: dict[str, Any] = {
            "dataset": dataset,
            "item_id": item_id,
            "title": title[: self.max_char_length],
        }
        if mapped_id:
            item["mapped_item_id"] = require_integer(
                value.get("mapped_item_id"), "mapped_item_id", location
            )
            mapped_key = (dataset, item["mapped_item_id"])
            previous_id = self._mapped_ids.get(mapped_key)
            if previous_id is not None and previous_id != item_id:
                raise ValueError(
                    f"mapped_item_id {mapped_key} identifies both {previous_id!r} "
                    f"and {item_id!r} at {location}"
                )
            self._mapped_ids[mapped_key] = item_id
        if "interaction_count" in value:
            item["interaction_count"] = require_integer(
                value["interaction_count"], "interaction_count", location
            )
        if "description" in value:
            if not isinstance(value["description"], str):
                raise ValueError(f"description must be a string at {location}")
            item["description"] = value["description"].strip()[: self.max_char_length]
        key = (dataset, item_id)
        previous = self.items.get(key)
        if previous is not None:
            for field in item.keys() & previous.keys():
                if previous[field] != item[field]:
                    raise ValueError(f"Conflicting {field} for {key} at {location}")
            for field, field_value in item.items():
                previous.setdefault(field, field_value)
            return previous
        item["image_path"] = str(self._image_path(dataset, item_id))
        self.items[key] = item
        return item


class BatchItems:
    """Map item occurrences to one encoder input per dataset/item identity."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.indices: dict[tuple[str, str], int] = {}

    def add(self, item: dict[str, Any]) -> int:
        key = (item["dataset"], item["item_id"])
        if key not in self.indices:
            self.indices[key] = len(self.items)
            self.items.append(item)
        return self.indices[key]


def split_dataset(
    dataset: Dataset, validation_ratio: float, seed: int
) -> tuple[Dataset, Dataset | None]:
    validation_ratio = require_number(validation_ratio, "validation_ratio", "split")
    require_integer(seed, "seed", "split")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0 or len(dataset) < 2:
        return dataset, None
    validation_size = min(max(1, round(len(dataset) * validation_ratio)), len(dataset) - 1)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    return Subset(dataset, order[validation_size:]), Subset(dataset, order[:validation_size])
