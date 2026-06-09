"""ΔE00 (CIEDE2000) colour fidelity reward."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import BaseReward, register


@register("delta_e00")
class DeltaE00Reward(BaseReward):
    """Colour fidelity reward based on the CIEDE2000 colour-difference formula.

    Lower ΔE00 means colours are more similar; reward = -mean(ΔE00).

    Config params:
        clip_max: clip ΔE00 values above this threshold before averaging
                  (reduces impact of occluded / background regions). Default: 20.0
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._clip_max = float(params.get("clip_max", 20.0))

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            import colour  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "delta_e00 reward requires 'colour-science': pip install colour-science"
            ) from e
        self._loaded = True

    def score(
        self,
        images: list[Image.Image],
        prompts: list[str],
        reference_images: list[Image.Image | None] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.load(getattr(self, "_device", torch.device("cpu")))
        import colour

        scores = []
        refs = reference_images or [None] * len(images)
        for img, ref in zip(images, refs):
            if ref is None:
                scores.append(0.0)
                continue
            # Resize ref to match generated image size
            if img.size != ref.size:
                ref = ref.resize(img.size, Image.LANCZOS)
            img_arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            ref_arr = np.array(ref.convert("RGB")).astype(np.float32) / 255.0
            img_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(img_arr))
            ref_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(ref_arr))
            delta_e = colour.delta_E(img_lab, ref_lab, method="CIE 2000")
            if self._clip_max > 0:
                delta_e = np.clip(delta_e, 0, self._clip_max)
            scores.append(-float(delta_e.mean()))
        return torch.tensor(scores, dtype=torch.float32)
