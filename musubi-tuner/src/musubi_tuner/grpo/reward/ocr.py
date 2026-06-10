"""OCR text accuracy reward using EasyOCR."""
from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


def _extract_target_text(prompt: str) -> str:
    """Extract quoted target text from prompt, e.g. 'write "Hello" on a sign'."""
    m = re.search(r'[""「](.+?)[""」]', prompt)
    return m.group(1) if m else prompt


@register("ocr")
class OCRReward(BaseReward):
    """EasyOCR-based text accuracy reward.

    Scores how accurately the generated image renders the target text
    (extracted from quoted text in the prompt, or the full prompt).

    Metric: character-level F1 between OCR result and target text.
    Requires: ``pip install easyocr``
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._lang = params.get("lang", ["en"])
        if isinstance(self._lang, str):
            self._lang = [self._lang]
        self._reader = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            import easyocr
        except ImportError as e:
            raise ImportError("OCR reward requires 'easyocr': pip install easyocr") from e
        use_gpu = device.type == "cuda"
        self._reader = easyocr.Reader(self._lang, gpu=use_gpu, verbose=False)
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        self.load(getattr(self, "_device", torch.device("cpu")))
        import numpy as np

        scores = []
        for img, prompt in zip(images, prompts):
            target = _extract_target_text(prompt).lower()
            img_arr = np.array(img.convert("RGB"))
            result = self._reader.readtext(img_arr)
            recognized = " ".join(item[1].lower() for item in (result or []))
            scores.append(_char_f1(recognized, target))
        return torch.tensor(scores, dtype=torch.float32)


def _char_f1(pred: str, target: str) -> float:
    """Character-level F1 score."""
    if not target:
        return 1.0 if not pred else 0.0
    pred_chars = list(pred.replace(" ", ""))
    target_chars = list(target.replace(" ", ""))
    if not pred_chars:
        return 0.0
    common = sum(min(pred_chars.count(c), target_chars.count(c)) for c in set(target_chars))
    precision = common / len(pred_chars) if pred_chars else 0.0
    recall = common / len(target_chars) if target_chars else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
