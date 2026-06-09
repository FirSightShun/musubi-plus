"""PickScore preference reward."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


@register("pickscore")
class PickScoreReward(BaseReward):
    """PickScore image-text alignment reward.

    Requires: ``pip install transformers``
    Model: ``yuvalkirstain/PickScore_v1``
    """

    _DEFAULT_MODEL = "yuvalkirstain/PickScore_v1"
    _DEFAULT_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model_id = params.get("model", self._DEFAULT_MODEL)
        self._processor_id = params.get("processor", self._DEFAULT_PROCESSOR)
        self._model = None
        self._processor = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._processor_id)
        self._model = AutoModel.from_pretrained(self._model_id).eval().to(device)
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        device = getattr(self, "_device", torch.device("cpu"))
        self.load(device)
        inputs = self._processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            image_embs = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
            text_embs = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            logit_scale = self._model.logit_scale.exp()
            scores = (logit_scale * (image_embs * text_embs).sum(dim=-1)).squeeze()
        return scores.cpu().float()
