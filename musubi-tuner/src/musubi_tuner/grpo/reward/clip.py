"""CLIP text-image alignment reward."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


@register("clip")
class CLIPReward(BaseReward):
    """CLIP cosine similarity between image and prompt text.

    Default model: ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``
    Requires: ``pip install open_clip_torch``
    """

    _DEFAULT_MODEL = "ViT-H-14"
    _DEFAULT_PRETRAINED = "laion2b_s32b_b79k"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model_name = params.get("model", self._DEFAULT_MODEL)
        self._pretrained = params.get("pretrained", self._DEFAULT_PRETRAINED)
        self._model = None
        self._preprocess = None
        self._tokenize = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            import open_clip
        except ImportError as e:
            raise ImportError("CLIP reward requires 'open_clip_torch': pip install open_clip_torch") from e
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name, pretrained=self._pretrained, device=device
        )
        self._model.eval()
        self._tokenize = open_clip.get_tokenizer(self._model_name)
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        device = getattr(self, "_device", torch.device("cpu"))
        self.load(device)
        img_tensors = torch.stack([self._preprocess(img) for img in images]).to(device)
        text_tokens = self._tokenize(prompts).to(device)
        with torch.no_grad():
            img_feats = self._model.encode_image(img_tensors)
            txt_feats = self._model.encode_text(text_tokens)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            scores = (img_feats * txt_feats).sum(dim=-1)
        return scores.cpu().float()
