"""Shared item encoder with stage-specific alignment losses and diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .losses.cca import contrastive_loss
from .losses.spa import pairwise_preference_loss, preference_metrics


class SPAModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, beta: float = 14.2857) -> None:
        super().__init__()
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.encoder = encoder
        self.beta = beta

    def forward(self, item_inputs, anchor_indices, candidate_indices, reward_scores):
        items = F.normalize(self.encoder(item_inputs).float(), p=2, dim=-1)
        if anchor_indices.ndim != 1 or candidate_indices.ndim != 2:
            raise ValueError("SPA indices must have shape [batch] and [batch, candidates]")
        similarities = torch.einsum("bd,bkd->bk", items[anchor_indices], items[candidate_indices])
        rewards = reward_scores.float()
        loss = pairwise_preference_loss(self.beta * similarities, rewards)
        diagnostics = preference_metrics(similarities, rewards)
        anchor_count = anchor_indices.numel()
        pair_count = diagnostics.pop("preference_pair_count")
        metrics = {
            "spa_loss": (loss.detach(), anchor_count),
            "top1_accuracy": (diagnostics.pop("top1_accuracy"), anchor_count),
            "mean_policy_similarity": (
                diagnostics.pop("mean_policy_similarity"),
                similarities.numel(),
            ),
        }
        metrics.update({name: (value, pair_count) for name, value in diagnostics.items()})
        return {"loss": loss, "metrics": metrics}


class CCAModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, temperature: float = 0.1) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.encoder = encoder
        self.temperature = temperature

    def forward(self, item_inputs, anchor_indices, positive_indices, negative_indices):
        items = F.normalize(self.encoder(item_inputs).float(), p=2, dim=-1)
        if anchor_indices.ndim != 1 or positive_indices.shape != anchor_indices.shape:
            raise ValueError("CCA anchor and positive indices must have matching shape [batch]")
        if negative_indices.ndim != 2 or negative_indices.size(0) != anchor_indices.size(0):
            raise ValueError("CCA negative indices must have shape [batch, negatives]")
        anchors = items[anchor_indices]
        positives = torch.einsum("bd,bd->b", anchors, items[positive_indices])
        negatives = torch.einsum("bd,bnd->bn", anchors, items[negative_indices])
        loss = contrastive_loss(positives, negatives, self.temperature)
        with torch.no_grad():
            margins = positives[:, None] - negatives
            pair_count = negatives.numel()
            metrics = {
                "cca_loss": (loss.detach(), anchor_indices.numel()),
                "negative_preference_accuracy": ((margins > 0).float().mean(), pair_count),
                "positive_negative_margin": (margins.mean(), pair_count),
            }
        return {"loss": loss, "metrics": metrics}
