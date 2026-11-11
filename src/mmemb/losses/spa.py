"""Reward-margin-weighted pairwise semantic preference loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _pairwise_margins(
    policy_similarities: torch.Tensor, ranked_reward_scores: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    preferred, dispreferred = torch.triu_indices(
        policy_similarities.size(1),
        policy_similarities.size(1),
        offset=1,
        device=policy_similarities.device,
    )
    reward_margin = (
        ranked_reward_scores[:, preferred] - ranked_reward_scores[:, dispreferred]
    ).clamp_min(0)
    policy_margin = policy_similarities[:, preferred] - policy_similarities[:, dispreferred]
    return reward_margin, policy_margin


def pairwise_preference_loss(
    policy_similarities: torch.Tensor, ranked_reward_scores: torch.Tensor
) -> torch.Tensor:
    """Sum weighted logistic losses over pairs, then average over anchors.

    ``policy_similarities`` contains beta-scaled scores, shaped [batch, candidates].
    Rewards are sorted in descending order. Reward ties contribute zero loss.
    """
    if policy_similarities.ndim != 2 or policy_similarities.shape != ranked_reward_scores.shape:
        raise ValueError("Policy and reward scores must have shape [batch, candidates]")
    if policy_similarities.size(0) < 1 or policy_similarities.size(1) < 2:
        raise ValueError("SPA requires a nonempty batch with at least two candidates")
    if (
        not torch.isfinite(policy_similarities).all()
        or not torch.isfinite(ranked_reward_scores).all()
    ):
        raise ValueError("Policy and reward scores must be finite")
    if torch.any(ranked_reward_scores[:, :-1] - ranked_reward_scores[:, 1:] < -1e-7):
        raise ValueError("Reward scores must be sorted in descending order")
    reward_margin, policy_margin = _pairwise_margins(policy_similarities, ranked_reward_scores)
    return (-reward_margin * F.logsigmoid(policy_margin)).sum(dim=1).mean()


def preference_metrics(
    policy_similarities: torch.Tensor, ranked_reward_scores: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Compute raw-cosine diagnostics; reward ties are excluded from pair means."""
    with torch.no_grad():
        reward_margin, policy_margin = _pairwise_margins(policy_similarities, ranked_reward_scores)
        decisive = reward_margin > 0
        pair_count = decisive.sum()
        denominator = pair_count.clamp_min(1).float()
        correct = ((policy_margin > 0) & decisive).sum()
        return {
            "preference_pair_count": pair_count,
            "preference_accuracy": correct.float() / denominator,
            "top1_accuracy": policy_similarities.argmax(dim=1)
            .eq(ranked_reward_scores.argmax(dim=1))
            .float()
            .mean(),
            "mean_reward_margin": reward_margin.masked_fill(~decisive, 0).sum() / denominator,
            "mean_policy_similarity_margin": (
                policy_margin.masked_fill(~decisive, 0).sum() / denominator
            ),
            "mean_policy_similarity": policy_similarities.mean(),
        }
