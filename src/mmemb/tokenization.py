"""Product prompts and terminal-EOS tokenization shared by training and export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


def last_active_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2 or torch.any(attention_mask.bool().sum(dim=1) == 0):
        raise ValueError("Every input must contain at least one active token")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    return (
        positions.expand_as(attention_mask)
        .masked_fill(~attention_mask.bool(), -1)
        .max(dim=1)
        .values
    )


def validate_terminal_eos(encoded: Any, eos_token_id: int) -> torch.Tensor:
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    positions = last_active_positions(attention_mask)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    if torch.any(input_ids[rows, positions] != eos_token_id):
        raise ValueError("The last active token must be EOS")
    return positions


@dataclass
class ItemTokenizer:
    processor: Any
    item_prompt: str
    max_length: int = 512
    max_char_length: int = 20480
    item_text_mode: str = "title"
    input_mode: str = "text_image"

    def __post_init__(self) -> None:
        if self.max_length < 2 or self.max_char_length < 1:
            raise ValueError("max_length must be >= 2 and max_char_length must be positive")
        if self.item_text_mode != "title":
            raise ValueError("This release uses title-only item text")
        if self.input_mode not in {"text_image", "text", "image"}:
            raise ValueError(f"Unsupported input mode: {self.input_mode}")
        if not self.item_prompt.strip():
            raise ValueError("item_prompt must not be empty")
        if self.processor.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id")

    def format_text(self, item: dict[str, Any]) -> str:
        prefix = f"Instruct: {self.item_prompt}\n\n"
        if self.input_mode != "text":
            image_token = f"<|vision_start|>{self.processor.image_token}<|vision_end|>"
            prefix = f"{image_token}\n{prefix}"
        if self.input_mode == "image":
            return f"{prefix}Item image:"
        return f"{prefix}Item title: {item['title'][:self.max_char_length]}"

    def __call__(self, items: list[dict[str, Any]]) -> Any:
        if not items:
            raise ValueError("Cannot tokenize an empty item batch")
        kwargs = {
            "text": [self.format_text(item) for item in items],
            "padding": True,
            "truncation": True,
            "max_length": self.max_length - 1,
            "return_tensors": "pt",
        }
        if self.input_mode != "text":
            images = []
            for item in items:
                with Image.open(item["image_path"]) as image:
                    images.append(image.convert("RGB"))
            kwargs["images"] = images
        encoded = self.processor(**kwargs)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        # Reserve one token before truncation so the pooled EOS is always present.
        eos = torch.full(
            (ids.size(0), 1),
            self.processor.tokenizer.eos_token_id,
            dtype=ids.dtype,
            device=ids.device,
        )
        encoded["input_ids"] = torch.cat((ids, eos), dim=1)
        encoded["attention_mask"] = torch.cat((mask, mask.new_ones((mask.size(0), 1))), dim=1)
        return encoded
