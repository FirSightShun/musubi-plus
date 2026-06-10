"""Base reward interface and registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from PIL import Image

_REWARD_REGISTRY: dict[str, type["BaseReward"]] = {}


def register(name: str):
    """Decorator to register a reward class under a string key."""

    def decorator(cls: type[BaseReward]) -> type[BaseReward]:
        if name in _REWARD_REGISTRY:
            raise ValueError(f"Reward '{name}' is already registered")
        _REWARD_REGISTRY[name] = cls
        return cls

    return decorator


def build_rewards(reward_configs) -> list[tuple["BaseReward", float]]:
    """Instantiate reward objects from a list of RewardConfig.

    Returns a list of (reward_instance, weight) pairs.
    """
    from musubi_tuner.grpo.config import RewardConfig  # local import to avoid circular

    result: list[tuple[BaseReward, float]] = []
    for cfg in reward_configs:
        if isinstance(cfg, dict):
            cfg = RewardConfig.from_dict(cfg)
        cls = _REWARD_REGISTRY.get(cfg.name)
        if cls is None:
            raise ValueError(f"Unknown reward '{cfg.name}'. Available: {list(_REWARD_REGISTRY)}")
        result.append((cls(cfg.params), cfg.weight))
    return result


class BaseReward(ABC):
    """Abstract base class for all reward functions.

    Subclasses should be decorated with ``@register("name")`` so they can be
    referenced by name in the TOML config.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self._loaded = False

    def load(self, device: torch.device) -> None:  # noqa: D401
        """Lazy-load any heavyweight model assets onto *device*."""

    def offload(self) -> None:
        """Move model assets back to CPU to free GPU memory between steps.

        Override in subclasses that hold nn.Module instances. Default no-op.
        """

    @abstractmethod
    def score(
        self,
        images: list[Image.Image],
        prompts: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a 1-D float tensor of shape [N], higher is better."""
        ...

    def __call__(
        self,
        images: list[Image.Image],
        prompts: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.score(images, prompts, **kwargs)
