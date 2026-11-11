"""Collaborative positive/negative records with fixed per-anchor sampling."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .common import (
    BatchItems,
    ItemRegistry,
    read_jsonl,
    require_integer,
    require_name,
    require_number,
)


def stable_sha256_seed(global_seed: int, *parts: object) -> int:
    require_integer(global_seed, "global_seed", "negative sampling")
    payload = "\0".join([str(global_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fixed_sample_without_replacement(
    pool: Sequence[dict[str, Any]],
    requested: int,
    *,
    global_seed: int,
    dataset: str,
    anchor_mapped_item_id: int,
) -> tuple[list[dict[str, Any]], int]:
    require_integer(requested, "requested negatives", "negative sampling", minimum=1)
    require_integer(anchor_mapped_item_id, "anchor_mapped_item_id", "negative sampling")
    if len(pool) < requested:
        raise ValueError(
            f"Insufficient negatives for {dataset}/{anchor_mapped_item_id}: "
            f"requested={requested}, available={len(pool)}"
        )
    shuffled = sorted(pool, key=lambda item: (int(item["mapped_item_id"]), str(item["item_id"])))
    # Keep the published run's seed namespace even though the public schema
    # now has a single negative pool.
    seed = stable_sha256_seed(
        global_seed, "stage2-fixed-negative-sampling", dataset, anchor_mapped_item_id, "normal"
    )
    random.Random(seed).shuffle(shuffled)
    return shuffled[:requested], seed


def _negative_pool(values: object, location: str) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"negatives must be a nonempty list at {location}")
    if any(not isinstance(item, dict) for item in values):
        raise ValueError(f"Every negative must be an object at {location}")
    if not any("negative_type" in item for item in values):
        return values
    pool = []
    for item in values:
        category = item.get("negative_type")
        if category not in {"normal", "hard"}:
            raise ValueError(f"Legacy negative_type must be normal or hard at {location}")
        if category == "normal":
            pool.append(item)
    return pool


def _validate_relation(value: dict[str, Any], *, positive: bool, location: str) -> None:
    if "cooccurrence" in value:
        count = require_integer(
            value["cooccurrence"], "cooccurrence", location, minimum=int(positive)
        )
        if not positive and count != 0:
            raise ValueError(f"Negative cooccurrence must equal zero at {location}")
    if "collaborative_score" in value:
        score = require_number(value["collaborative_score"], "collaborative_score", location)
        if not 0 <= score <= 1 or (not positive and score != 0):
            raise ValueError(f"Invalid collaborative_score at {location}")
    if "semantic_similarity" in value:
        require_number(value["semantic_similarity"], "semantic_similarity", location)
    if "semantic_rank" in value:
        if positive:
            require_integer(value["semantic_rank"], "semantic_rank", location, minimum=1)
        elif value["semantic_rank"] is not None:
            raise ValueError(f"Negative semantic_rank must be null at {location}")


class CCADataset(Dataset):
    def __init__(
        self,
        train_file: str | Path,
        image_root: str | Path,
        num_negatives: int = 10,
        seed: int = 42,
        max_char_length: int = 20480,
        max_train_samples: int | None = None,
    ) -> None:
        require_integer(num_negatives, "num_negatives", "CCA dataset", minimum=1)
        require_integer(seed, "seed", "CCA dataset")
        registry = ItemRegistry(image_root, max_char_length, collapse_whitespace=True)
        self.samples: list[dict[str, Any]] = []
        seen_anchors: set[tuple[str, str]] = set()
        seen_mapped_anchors: set[tuple[str, int]] = set()
        for record, location in read_jsonl(train_file, max_train_samples):
            dataset = require_name(record.get("dataset"), "dataset", location)
            anchor = registry.intern(
                record.get("anchor"), dataset, f"{location}.anchor", mapped_id=True
            )
            positive_raw = record.get("positive")
            if not isinstance(positive_raw, dict):
                raise ValueError(f"positive must be an object at {location}")
            _validate_relation(positive_raw, positive=True, location=f"{location}.positive")
            positive = registry.intern(
                positive_raw, dataset, f"{location}.positive", mapped_id=True
            )
            anchor_key = (dataset, anchor["item_id"])
            mapped_key = (dataset, anchor["mapped_item_id"])
            if anchor_key in seen_anchors or mapped_key in seen_mapped_anchors:
                raise ValueError(f"Duplicate anchor at {location}")
            seen_anchors.add(anchor_key)
            seen_mapped_anchors.add(mapped_key)
            item_ids = {anchor["item_id"]}
            mapped_ids = {anchor["mapped_item_id"]}

            def add_identity(item: dict[str, Any], item_location: str) -> None:
                if item["item_id"] in item_ids or item["mapped_item_id"] in mapped_ids:
                    raise ValueError(
                        f"Repeated anchor/positive/negative identity at {item_location}"
                    )
                item_ids.add(item["item_id"])
                mapped_ids.add(item["mapped_item_id"])

            add_identity(positive, f"{location}.positive")
            pool = []
            for index, raw in enumerate(_negative_pool(record.get("negatives"), location)):
                item_location = f"{location}.negatives[{index}]"
                _validate_relation(raw, positive=False, location=item_location)
                item = registry.intern(raw, dataset, item_location, mapped_id=True)
                add_identity(item, item_location)
                pool.append(item)
            negatives, sampling_seed = fixed_sample_without_replacement(
                pool,
                num_negatives,
                global_seed=seed,
                dataset=dataset,
                anchor_mapped_item_id=anchor["mapped_item_id"],
            )
            self.samples.append(
                {
                    "dataset": dataset,
                    "anchor": anchor,
                    "positive": positive,
                    "negatives": negatives,
                    "sampling_seed": sampling_seed,
                }
            )
        if not self.samples:
            raise ValueError(f"No training records found in {train_file}")
        self.unique_item_count = len(registry.items)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class CCACollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        items = BatchItems()
        anchors, positives, negatives = [], [], []
        count = len(features[0]["negatives"])
        for feature in features:
            if len(feature["negatives"]) != count:
                raise ValueError("Negative count must be constant within a batch")
            anchors.append(items.add(feature["anchor"]))
            positives.append(items.add(feature["positive"]))
            negatives.append([items.add(item) for item in feature["negatives"]])
        # Trainer infers the batch size from the first tensor, which must
        # count anchors rather than deduplicated encoder inputs.
        return {
            "anchor_indices": torch.tensor(anchors, dtype=torch.long),
            "item_inputs": self.tokenizer(items.items),
            "positive_indices": torch.tensor(positives, dtype=torch.long),
            "negative_indices": torch.tensor(negatives, dtype=torch.long),
        }
