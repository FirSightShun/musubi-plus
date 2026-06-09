"""ImageReward aesthetic preference reward."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


@register("image_reward")
class ImageRewardReward(BaseReward):
    """ImageReward human-preference model.

    Requires: ``pip install image-reward``
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model_name = params.get("model", "ImageReward-v1.0")
        self._rm = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            import ImageReward as ir
        except ImportError as e:
            raise ImportError("ImageReward reward requires 'image-reward' package: pip install image-reward") from e
        self._rm = ir.load(self._model_name, device=str(device))
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        device = getattr(self, "_device", torch.device("cpu"))
        self.load(device)
        scores = [self._rm.score(prompt, img) for prompt, img in zip(prompts, images)]
        return torch.tensor(scores, dtype=torch.float32)
