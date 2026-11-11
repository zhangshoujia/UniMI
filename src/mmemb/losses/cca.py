"""Explicit-negative contrastive cross-entropy for collaborative alignment."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def contrastive_loss(
    positive_similarities: torch.Tensor,
    negative_similarities: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """The positive is column zero of [positive | negatives] / temperature."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if (
        positive_similarities.ndim != 1
        or negative_similarities.ndim != 2
        or positive_similarities.size(0) != negative_similarities.size(0)
        or positive_similarities.numel() == 0
        or negative_similarities.size(1) == 0
    ):
        raise ValueError("CCA requires positive [batch] and negative [batch, negatives] scores")
    if (
        not torch.isfinite(positive_similarities).all()
        or not torch.isfinite(negative_similarities).all()
    ):
        raise ValueError("Similarities must be finite")
    logits = torch.cat((positive_similarities[:, None], negative_similarities), dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits / temperature, labels)
