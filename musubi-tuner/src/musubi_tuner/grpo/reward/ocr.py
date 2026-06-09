"""OCR text accuracy reward using PaddleOCR."""
from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register


def _extract_target_text(prompt: str) -> str:
    """Extract quoted target text from prompt, e.g. 'write "Hello" on a sign'."""
    m = re.search(r'["“「](.+?)["”」]', prompt)
    return m.group(1) if m else prompt


@register("ocr")
class OCRReward(BaseReward):
    """PaddleOCR-based text accuracy reward.

    Scores how accurately the generated image renders the target text
    (extracted from quoted text in the prompt, or the full prompt).

    Metric: character-level F1 between OCR result and target text.
    Requires: ``pip install paddlepaddle paddleocr`` (CPU or GPU)
    """

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._lang = params.get("lang", "en")
        self._ocr = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError("OCR reward requires 'paddleocr': pip install paddlepaddle paddleocr") from e
        use_gpu = device.type == "cuda"
        self._ocr = PaddleOCR(use_angle_cls=True, lang=self._lang, use_gpu=use_gpu, show_log=False)
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        self.load(getattr(self, "_device", torch.device("cpu")))
        scores = []
        for img, prompt in zip(images, prompts):
            target = _extract_target_text(prompt).lower()
            result = self._ocr.ocr(img, cls=True)
            recognized = " ".join(
                line[1][0].lower() for block in (result or []) for line in (block or [])
            )
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
