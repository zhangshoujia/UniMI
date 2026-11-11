"""Three-candidate semantic preference records and collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .common import BatchItems, ItemRegistry, read_jsonl, require_name, require_number


def rank_candidates_by_reward(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: (-item["semantic_alignment_score"], item["item_id"]))


class SPADataset(Dataset):
    def __init__(
        self,
        train_file: str | Path,
        image_root: str | Path,
        max_char_length: int = 20480,
        max_train_samples: int | None = None,
    ) -> None:
        registry = ItemRegistry(image_root, max_char_length, collapse_whitespace=False)
        self.samples: list[dict[str, Any]] = []
        seen_anchors: set[tuple[str, str]] = set()
        for record, location in read_jsonl(train_file, max_train_samples):
            dataset = require_name(record.get("dataset"), "dataset", location)
            anchor = registry.intern(record.get("anchor"), dataset, f"{location}.anchor")
            anchor_key = (dataset, anchor["item_id"])
            if anchor_key in seen_anchors:
                raise ValueError(f"Duplicate anchor {anchor_key} at {location}")
            seen_anchors.add(anchor_key)
            candidates = record.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 3:
                raise ValueError(f"Expected exactly three candidates at {location}")
            candidate_ids = {anchor["item_id"]}
            validated = []
            for index, candidate in enumerate(candidates):
                item_location = f"{location}.candidates[{index}]"
                item = registry.intern(candidate, dataset, item_location)
                if item["item_id"] in candidate_ids:
                    raise ValueError(f"Repeated anchor/candidate item at {item_location}")
                candidate_ids.add(item["item_id"])
                reward = require_number(
                    candidate.get("semantic_alignment_score"),
                    "semantic_alignment_score",
                    item_location,
                )
                if not -1e-7 <= reward <= 1 + 1e-7:
                    raise ValueError(
                        f"semantic_alignment_score must be in [0, 1] at {item_location}"
                    )
                if "cosine_similarity" in candidate:
                    require_number(
                        candidate["cosine_similarity"], "cosine_similarity", item_location
                    )
                validated.append(
                    {"item": item, "semantic_alignment_score": reward, "item_id": item["item_id"]}
                )
            ranked = rank_candidates_by_reward(validated)
            self.samples.append(
                {
                    "dataset": dataset,
                    "anchor": anchor,
                    "candidates": [entry["item"] for entry in ranked],
                    "reward_scores": [entry["semantic_alignment_score"] for entry in ranked],
                }
            )
        if not self.samples:
            raise ValueError(f"No training records found in {train_file}")
        self.unique_item_count = len(registry.items)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class SPACollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        items = BatchItems()
        anchors, candidates, rewards = [], [], []
        for feature in features:
            anchors.append(items.add(feature["anchor"]))
            candidates.append([items.add(item) for item in feature["candidates"]])
            rewards.append(feature["reward_scores"])
        # Trainer infers the batch size from the first tensor, which must
        # count anchors rather than deduplicated encoder inputs.
        return {
            "anchor_indices": torch.tensor(anchors, dtype=torch.long),
            "item_inputs": self.tokenizer(items.items),
            "candidate_indices": torch.tensor(candidates, dtype=torch.long),
            "reward_scores": torch.tensor(rewards, dtype=torch.float32),
        }
