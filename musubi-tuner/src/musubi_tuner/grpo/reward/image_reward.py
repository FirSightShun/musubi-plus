"""ImageReward aesthetic preference reward."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .base import BaseReward, register

# apply_chunking_to_forward was removed in transformers >= 4.48; patch it back
# before ImageReward's BLIP sub-model tries to import it at module load time.
import transformers.modeling_utils as _mu
if not hasattr(_mu, "apply_chunking_to_forward"):
    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size and chunk_size > 0:
            tensor_shape = input_tensors[0].shape[chunk_dim]
            if tensor_shape % chunk_size == 0:
                return torch.cat(
                    [forward_fn(*[torch.narrow(t, chunk_dim, i, chunk_size) for t in input_tensors])
                     for i in range(0, tensor_shape, chunk_size)],
                    dim=chunk_dim,
                )
        return forward_fn(*input_tensors)
    _mu.apply_chunking_to_forward = _apply_chunking_to_forward


@register("image_reward")
class ImageRewardReward(BaseReward):
    """ImageReward human-preference model.

    Requires: ``pip install image-reward``
    Also needs the openai CLIP package:
        pip install git+https://github.com/openai/CLIP.git
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
            raise ImportError(
                "ImageReward reward requires:\n"
                "  pip install image-reward --no-build-isolation\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            ) from e
        self._rm = ir.load(self._model_name, device=str(device))
        self._device = device
        self._loaded = True

    def score(self, images: list[Image.Image], prompts: list[str], **kwargs: Any) -> torch.Tensor:
        device = getattr(self, "_device", torch.device("cpu"))
        self.load(device)
        scores = [self._rm.score(prompt, img) for prompt, img in zip(prompts, images)]
        return torch.tensor(scores, dtype=torch.float32)
