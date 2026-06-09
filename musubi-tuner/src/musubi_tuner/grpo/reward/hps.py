"""HPSv2.1 aesthetic preference reward."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


@register("hps_v2")
class HPSv2Reward(BaseReward):
    """Human Preference Score v2.1.

    Requires: ``pip install hpsv2``
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model = None
        self._processor = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            import hpsv2
        except ImportError as e:
            raise ImportError("HPSv2 reward requires 'hpsv2' package: pip install hpsv2") from e
        self._hpsv2 = hpsv2
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        self.load(self._device if hasattr(self, "_device") else torch.device("cpu"))
        scores = []
        for img, prompt in zip(images, prompts):
            s = self._hpsv2.score(img, prompt, hps_version="v2.1")
            scores.append(float(s) if not isinstance(s, float) else s)
        return torch.tensor(scores, dtype=torch.float32)
