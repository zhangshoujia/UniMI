"""Qwen2-VL backbone with terminal-EOS pooling for product embeddings."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .tokenization import validate_terminal_eos


class Qwen2VLItemEncoder(torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        image_min_pixels: int = 3136,
        image_max_pixels: int = 262144,
        attn_implementation: str = "sdpa",
        torch_dtype: torch.dtype = torch.bfloat16,
        pooling: str = "eos",
        processor_path: str | None = None,
    ) -> None:
        super().__init__()
        if pooling != "eos":
            raise ValueError("Only terminal-EOS pooling is supported")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_path or model_path,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
            local_files_only=True,
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.pooling = pooling

    def backbone(self) -> torch.nn.Module:
        host = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return host.model

    def encode(self, item_inputs: Any) -> torch.Tensor:
        input_ids = item_inputs["input_ids"]
        attention_mask = item_inputs["attention_mask"]
        readout = validate_terminal_eos(item_inputs, self.tokenizer.eos_token_id)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        backbone = self.backbone()
        image_grid = item_inputs.get("image_grid_thw")
        pixel_values = item_inputs.get("pixel_values")
        if pixel_values is not None:
            image_embeds = backbone.get_image_features(pixel_values, image_grid)
            if isinstance(image_embeds, (list, tuple)):
                image_embeds = torch.cat(image_embeds, dim=0)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask, _ = backbone.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            if image_mask[..., 0].sum().item() != image_embeds.size(0):
                raise ValueError("Image placeholder count does not match encoded image tokens")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        position_ids, rope_deltas = backbone.get_rope_index(
            input_ids,
            image_grid,
            None,
            attention_mask,
        )
        backbone.rope_deltas = rope_deltas
        hidden = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[rows, readout]

    def forward(self, item_inputs: Any) -> torch.Tensor:
        return self.encode(item_inputs)
