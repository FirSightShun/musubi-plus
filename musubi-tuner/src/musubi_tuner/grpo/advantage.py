"""MO-GRPO advantage computation.

Each reward is normalised independently within the group (mean=0, std=1),
then weighted and summed. This prevents high-variance rewards from
dominating the advantage signal.
"""
from __future__ import annotations

import torch


def compute_advantages(
    scores: dict[str, torch.Tensor],
    weights: dict[str, float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MO-GRPO advantages from per-reward score tensors.

    Args:
        scores:  Mapping from reward name to a 1-D tensor of shape ``[B*G]``.
                 Scores for the same group of G images should be adjacent.
        weights: Mapping from reward name to scalar weight (should sum to ~1).
        eps:     Small constant for numerical stability in normalisation.

    Returns:
        Advantage tensor of shape ``[B*G]``.
    """
    device = next(iter(scores.values())).device
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Sum of reward weights is zero")

    advantage = torch.zeros(next(iter(scores.values())).shape, device=device, dtype=torch.float32)

    for name, raw in scores.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        r = raw.to(device=device, dtype=torch.float32)
        r_norm = (r - r.mean()) / (r.std(unbiased=False) + eps)
        advantage = advantage + w * r_norm

    return advantage


def compute_group_advantages(
    scores: dict[str, torch.Tensor],
    weights: dict[str, float],
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute MO-GRPO advantages with group-wise normalisation.

    Unlike ``compute_advantages`` which normalises over the entire batch,
    this function reshapes ``[B*G]`` into ``[B, G]`` and normalises each
    of the B groups independently — consistent with the GRPO paper.

    Args:
        scores:     Mapping from reward name to tensor of shape ``[B*G]``.
        weights:    Mapping from reward name to scalar weight.
        group_size: G, number of samples per prompt.
        eps:        Numerical stability constant.

    Returns:
        Advantage tensor of shape ``[B*G]``.
    """
    n = next(iter(scores.values())).shape[0]
    if n % group_size != 0:
        raise ValueError(f"Total samples {n} is not divisible by group_size {group_size}")

    device = next(iter(scores.values())).device
    b = n // group_size

    advantage = torch.zeros(n, device=device, dtype=torch.float32)

    for name, raw in scores.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        r = raw.to(device=device, dtype=torch.float32).reshape(b, group_size)  # [B, G]
        mean = r.mean(dim=1, keepdim=True)  # [B, 1]
        std = r.std(dim=1, keepdim=True, unbiased=False)  # [B, 1]
        r_norm = (r - mean) / (std + eps)  # [B, G]
        advantage = advantage + w * r_norm.reshape(n)

    return advantage
