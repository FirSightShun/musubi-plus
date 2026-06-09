"""VLM semantic scoring reward using Qwen2-VL."""
from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register

_DEFAULT_TEMPLATE = (
    "Please rate this image on a scale of 1 to 10 based on how well it matches the description: "
    '"{prompt}". Reply with a single integer only.'
)


@register("vlm")
class VLMReward(BaseReward):
    """VLM semantic scoring via Qwen2-VL-Instruct.

    Config params:
        model: HuggingFace model ID (default: ``Qwen/Qwen2-VL-2B-Instruct``)
        prompt_template: f-string with ``{prompt}`` placeholder for the scoring instruction
        min_score: lower bound for normalisation (default 1)
        max_score: upper bound for normalisation (default 10)
    """

    _DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        self._model_id = params.get("model", self._DEFAULT_MODEL)
        self._template = params.get("prompt_template", _DEFAULT_TEMPLATE)
        self._min = float(params.get("min_score", 1))
        self._max = float(params.get("max_score", 10))
        self._model = None
        self._processor = None

    def load(self, device: torch.device) -> None:
        if self._loaded:
            return
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = (
            Qwen2VLForConditionalGeneration.from_pretrained(self._model_id, torch_dtype=torch.float16)
            .eval()
            .to(device)
        )
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        device = getattr(self, "_device", torch.device("cpu"))
        self.load(device)
        scores = []
        for img, prompt in zip(images, prompts):
            instruction = self._template.format(prompt=prompt)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[img], return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = self._model.generate(**inputs, max_new_tokens=8)
            generated = self._processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            scores.append(self._parse_score(generated[0] if generated else ""))
        return torch.tensor(scores, dtype=torch.float32)

    def _parse_score(self, text: str) -> float:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if not nums:
            return (self._min + self._max) / 2
        val = float(nums[0])
        # Normalise to [0, 1]
        return (val - self._min) / (self._max - self._min + 1e-8)
