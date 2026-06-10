"""GRPO trainer: online RL fine-tuning for Flow Matching image/video models.

Design principles:
- Composes (holds) a NetworkTrainer instance; does not inherit from it.
- Phase 1 (no_grad): online rollout via base_trainer.do_inference → rewards → advantages.
- Phase 2 (with grad): Flow Matching GRPO loss + KL penalty.
- Architecture-agnostic: all architecture-specific operations delegate to base_trainer.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from .advantage import compute_group_advantages
from .config import GRPOConfig
from .reward import BaseReward, build_rewards

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Online GRPO trainer wrapping any musubi-tuner NetworkTrainer subclass.

    Args:
        base_trainer: An already-initialised NetworkTrainer instance.
        config: GRPO hyper-parameter configuration.
        accelerator: HuggingFace Accelerate accelerator.
        args: Parsed argparse namespace passed to the base trainer.
        transformer: The trainable DiT (with LoRA weights merged / enabled).
        vae: Loaded VAE (kept on CPU; moved to device when needed).
        network: LoRA network module.
        dit_dtype: dtype used for the DiT forward pass.
        network_dtype: dtype for trainable network parameters.
    """

    def __init__(
        self,
        base_trainer,
        config: GRPOConfig,
        accelerator,
        args,
        transformer: torch.nn.Module,
        vae,
        network: torch.nn.Module,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
    ) -> None:
        self.base = base_trainer
        self.config = config
        self.accelerator = accelerator
        self.args = args
        self.transformer = transformer
        self.vae = vae
        self.network = network
        self.dit_dtype = dit_dtype
        self.network_dtype = network_dtype

        # Freeze a snapshot of the transformer as reference policy for KL.
        # Only create if kl_coeff > 0 — deepcopy of 39 GB+ transformers causes OOM otherwise.
        self.ref_transformer = self._build_ref(transformer) if config.kl_coeff > 0 else None

        # Build reward instances: list of (BaseReward, weight)
        self._reward_list: list[tuple[BaseReward, float]] = build_rewards(config.rewards)
        self._reward_weights: dict[str, float] = {
            f"r{i}_{rw.params.get('name', type(rw).__name__)}": w
            for i, (rw, w) in enumerate(self._reward_list)
        }
        # Use the reward class registered name as key for pretty logging
        self._named_rewards: list[tuple[str, BaseReward, float]] = []
        from musubi_tuner.grpo.reward.base import _REWARD_REGISTRY

        inv = {v: k for k, v in _REWARD_REGISTRY.items()}
        for rw, w in self._reward_list:
            self._named_rewards.append((inv.get(type(rw), type(rw).__name__), rw, w))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        sample_parameters: list[dict],
        reference_images: Optional[list[Optional[Image.Image]]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one GRPO training step.

        Args:
            sample_parameters: List of dicts produced by
                ``base_trainer.process_sample_prompts()``.  Each dict contains
                ``prompt``, ``llm_embeds``, ``llm_mask``, ``clipL_embeds`` etc.
                The step will sample ``config.group_size`` images per entry.
            reference_images: Optional PIL images for ΔE00 reward, aligned
                with ``sample_parameters``.

        Returns:
            (loss, log_dict) where ``loss`` is a scalar tensor with gradient.
        """
        device = self.accelerator.device
        G = self.config.group_size

        # ── Phase 1: online rollout (no_grad) ──────────────────────────────
        with torch.no_grad():
            all_images: list[Image.Image] = []
            all_latents: list[torch.Tensor] = []  # [C, F, H, W] each
            all_prompts: list[str] = []
            all_ref_images: list[Optional[Image.Image]] = []
            all_param_indices: list[int] = []  # which sample_parameter each image belongs to

            for idx, sp in enumerate(sample_parameters):
                for _ in range(G):
                    generator = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31, (1,)).item())
                    video, latents = self._rollout_one(sp, generator)
                    pil = _video_to_pil(video)
                    all_images.append(pil)
                    if latents is not None:
                        all_latents.append(latents)
                    all_prompts.append(sp.get("prompt", ""))
                    all_ref_images.append(reference_images[idx] if reference_images else None)
                    all_param_indices.append(idx)

            # ── Reward scoring ──────────────────────────────────────────────
            scores: dict[str, torch.Tensor] = {}
            for name, rw, _ in self._named_rewards:
                rw.load(device)
                try:
                    s = rw.score(all_images, all_prompts, reference_images=all_ref_images)
                    scores[name] = s.cpu()
                except Exception as e:
                    logger.warning(f"Reward '{name}' failed: {e}. Zeroing out.")
                    scores[name] = torch.zeros(len(all_images))

            # ── MO-GRPO advantages ──────────────────────────────────────────
            weight_map = {name: w for name, _rw, w in self._named_rewards}
            adv = compute_group_advantages(scores, weight_map, G)  # [B*G]
            adv = adv.to(device=device, dtype=torch.float32)

        # ── Phase 2: GRPO loss (with grad) ─────────────────────────────────
        loss, log_dict = self._grpo_loss(
            all_latents if all_latents else None,
            all_images,
            adv,
            sample_parameters,
            all_param_indices,
        )

        # Append reward stats to log
        for name, s in scores.items():
            log_dict[f"reward/{name}"] = s.mean().item()
        log_dict["reward/advantage_mean"] = adv.mean().item()
        log_dict["reward/advantage_std"] = adv.std().item()

        return loss, log_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ref(self, transformer: torch.nn.Module) -> torch.nn.Module:
        """Deep-copy transformer and freeze all parameters as reference policy."""
        ref = copy.deepcopy(transformer)
        ref.requires_grad_(False)
        ref.eval()
        return ref

    def _rollout_one(
        self,
        sample_parameter: dict,
        generator: torch.Generator,
    ) -> tuple[np.ndarray, Optional[torch.Tensor]]:
        """Run one denoising inference, returning (decoded_video, final_latents).

        Calls base_trainer.do_inference() which is architecture-specific.
        Returns decoded video as numpy array [1, C, F, H, W] in [0, 1] and
        the pre-decode latents tensor (or None if not available).
        """
        cfg = self.config
        device = self.accelerator.device

        sp = dict(sample_parameter)
        sp.setdefault("sample_steps", cfg.num_inference_steps)
        sp.setdefault("width", cfg.width)
        sp.setdefault("height", cfg.height)
        sp.setdefault("frame_count", cfg.frame_count)
        sp.setdefault("guidance_scale", cfg.guidance_scale)
        sp.setdefault("discrete_flow_shift", cfg.discrete_flow_shift)

        width = (sp["width"] // 8) * 8
        height = (sp["height"] // 8) * 8

        transformer = self.accelerator.unwrap_model(self.transformer)
        was_train = transformer.training
        transformer.eval()

        try:
            video = self.base.do_inference(
                self.accelerator,
                self.args,
                sp,
                self.vae,
                self.dit_dtype,
                transformer,
                sp["discrete_flow_shift"],
                sp["sample_steps"],
                width,
                height,
                sp["frame_count"],
                generator,
                do_classifier_free_guidance=False,
                guidance_scale=sp["guidance_scale"],
                cfg_scale=None,
            )
        finally:
            transformer.train(was_train)
            # Move vae back to CPU after inference (do_inference moves it to device)
            self.vae.to("cpu")

        if video is None:
            video = np.zeros((1, 3, 1, height, width), dtype=np.float32)

        return video, None  # latents not exposed by do_inference

    def _grpo_loss(
        self,
        latents_list: Optional[list[torch.Tensor]],
        images: list[Image.Image],
        advantages: torch.Tensor,
        sample_parameters: list[dict],
        param_indices: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO loss over the rollout batch.

        Flow:
        1. VAE-encode generated images → scaled training latents x0.
        2. Sample timestep t ~ Uniform[0, 1].
        3. x_t = (1-t) * x0 + t * eps.
        4. v_target = eps - x0.
        5. Forward pass: v_theta = base_trainer.call_dit(...).
        6. L = A * ||v_theta - v_target||^2  + beta * ||v_theta - v_ref||^2.
        """
        device = self.accelerator.device
        bsz = len(images)
        log = {}

        # ── 1. Encode images to latents ─────────────────────────────────────
        latents = self._encode_images_to_latents(images)  # [B, C, F, H, W]
        latents = latents.to(device=device, dtype=self.dit_dtype)
        latents = self.base.scale_shift_latents(latents)  # architecture-specific scaling

        # ── 2-3. Noise + x_t ───────────────────────────────────────────────
        noise = torch.randn_like(latents)
        t = torch.rand(bsz, device=device)  # [B] uniform in (0, 1)

        # Broadcast t to latent shape for x_t computation
        t_view = t.view(bsz, *([1] * (latents.ndim - 1)))
        noisy = (1 - t_view) * latents + t_view * noise  # x_t

        # ── 4. v_target ─────────────────────────────────────────────────────
        # target = noise - latents (standard Flow Matching target)
        target = noise - latents  # [B, C, F, H, W]

        # Build batch dict for call_dit (maps GRPO sample_parameter keys to batch keys)
        # Pass latents so architectures that read batch["latents"] (e.g. qwen_image) work correctly.
        batch_for_dit = self._build_batch_dict(sample_parameters, param_indices, bsz, device, latents=latents)

        # ── 5a. Advantage-weighted loss (with grad) ─────────────────────────
        model_pred, _ = self.base.call_dit(
            self.args,
            self.accelerator,
            self.transformer,
            latents,
            batch_for_dit,
            noise,
            noisy,
            t,
            self.network_dtype,
        )

        loss_per_elem = torch.nn.functional.mse_loss(
            model_pred.to(self.network_dtype),
            target.to(self.network_dtype),
            reduction="none",
        )
        loss_per_sample = loss_per_elem.mean(dim=list(range(1, loss_per_elem.ndim)))  # [B]
        adv_term = (advantages * loss_per_sample).mean()

        # ── 5b. KL penalty (no grad on ref) ────────────────────────────────
        kl_loss = torch.tensor(0.0, device=device)
        if self.config.kl_coeff > 0:
            with torch.no_grad():
                ref_pred, _ = self.base.call_dit(
                    self.args,
                    self.accelerator,
                    self.ref_transformer,
                    latents,
                    batch_for_dit,
                    noise,
                    noisy,
                    t,
                    self.network_dtype,
                )
            kl_elem = torch.nn.functional.mse_loss(
                model_pred.to(self.network_dtype),
                ref_pred.detach().to(self.network_dtype),
                reduction="none",
            )
            kl_loss = kl_elem.mean() * self.config.kl_coeff

        total_loss = adv_term + kl_loss
        log["loss/advantage_weighted"] = adv_term.item()
        log["loss/kl"] = kl_loss.item()
        log["loss/total"] = total_loss.item()

        return total_loss, log

    def _encode_images_to_latents(self, images: list[Image.Image]) -> torch.Tensor:
        """VAE-encode PIL images to latent tensors.

        Returns a [B, C, F, H, W] tensor in VAE latent space (pre-scale_shift_latents).
        Supports two VAE families:
        - diffusers-style: has .config.scaling_factor, encode() returns latent_dist
        - qwen_image-style: has .latents_mean/.latents_std, exposes encode_pixels_to_latents()
        """
        device = self.accelerator.device
        self.vae.to(device)
        self.vae.eval()

        frames = []
        for img in images:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            frames.append(t)

        imgs_t = torch.stack(frames).to(device=device, dtype=self.vae.dtype)  # [B, 3, H, W]

        with torch.no_grad():
            if hasattr(self.vae, "latents_mean"):
                # qwen_image VAE: encode_pixels_to_latents handles [-1,1] norm and mean/std scaling
                # Input expected in [0, 1]; unsqueeze temporal dim handled internally
                latents = self.vae.encode_pixels_to_latents(imgs_t)  # [B, C, 1, H, W]
            else:
                # diffusers-style VAE
                imgs_t = imgs_t * 2.0 - 1.0          # [0,1] → [-1,1]
                imgs_t = imgs_t.unsqueeze(2)           # [B, C, H, W] → [B, C, 1, H, W]
                latent_dist = self.vae.encode(imgs_t)
                if hasattr(latent_dist, "latent_dist"):
                    latents = latent_dist.latent_dist.sample()
                elif hasattr(latent_dist, "sample"):
                    latents = latent_dist.sample()
                else:
                    latents = latent_dist

                if hasattr(self.vae, "config"):
                    if getattr(self.vae.config, "shift_factor", None):
                        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    elif getattr(self.vae.config, "scaling_factor", None):
                        latents = latents * self.vae.config.scaling_factor

        self.vae.to("cpu")
        return latents.float()

    def _build_batch_dict(
        self,
        sample_parameters: list[dict],
        param_indices: list[int],
        bsz: int,
        device: torch.device,
        latents: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build a batch dict compatible with base_trainer.call_dit().

        Automatically detects the embed format from the first sample parameter:
        - HunyuanVideo / hv:  llm_embeds, llm_mask, clipL_embeds → llm, llm_mask, clipL
        - qwen_image:         vl_embed (list of variable-len tensors)
        """
        first_sp = sample_parameters[param_indices[0]]

        def _pad_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
            max_len = max(t.shape[1] if t.ndim > 1 else 1 for t in tensors)
            padded = []
            for t in tensors:
                if t.ndim > 1 and t.shape[1] < max_len:
                    pad = torch.zeros(*t.shape[:-2], max_len - t.shape[1], *t.shape[2:], device=t.device, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0)

        if "vl_embed" in first_sp:
            # qwen_image: vl_embed stored as [1, L, D] (with batch dim); call_dit expects list of [L, D]
            vl_list = [sample_parameters[idx]["vl_embed"].squeeze(0).to(device) for idx in param_indices]
            batch: dict = {"vl_embed": vl_list}
        else:
            # HunyuanVideo-style
            llm_list = [sample_parameters[idx]["llm_embeds"].to(device) for idx in param_indices]
            mask_list = [sample_parameters[idx]["llm_mask"].to(device) for idx in param_indices]
            clip_list = [sample_parameters[idx]["clipL_embeds"].to(device) for idx in param_indices]
            batch = {
                "llm": _pad_stack(llm_list),
                "llm_mask": _pad_stack(mask_list),
                "clipL": _pad_stack(clip_list),
            }

        # qwen_image's call_dit reads batch["latents"] directly
        if latents is not None:
            batch["latents"] = latents

        return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_to_pil(video: Any) -> Image.Image:
    """Convert a decoded video tensor/array (float32, [1, C, F, H, W]) to PIL."""
    if video is None:
        return Image.new("RGB", (256, 256))

    if isinstance(video, torch.Tensor):
        arr = video.cpu().float().numpy()
    else:
        arr = np.asarray(video, dtype=np.float32)

    # Handle [1, C, F, H, W] or [C, F, H, W] or [C, H, W]
    while arr.ndim > 4:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[:, 0]  # take first frame: [C, H, W]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)  # [H, W, C]

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return Image.fromarray(arr.squeeze(-1) if arr.ndim == 3 else arr, "L").convert("RGB")
    return Image.fromarray(arr[..., :3], "RGB")
