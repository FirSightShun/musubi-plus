"""DiffusionNFT trainer: online RL fine-tuning via forward-process policy optimization.

Reference: "DiffusionNFT: Negative-aware FineTuning for Diffusion Models"
           NVIDIA Research, ICLR 2026 Oral — https://arxiv.org/abs/2509.16117

Design:
- Composes (holds) a NetworkTrainer instance; does not inherit from it.
- Phase 1 (no_grad): rollout with old LoRA policy → rewards → advantages → r ∈ [0,1].
- Phase 2 (with_grad): Forward-process NFT loss with implicit positive/negative targets.
- Old policy maintained as CPU state dict; swapped in/out for forward passes.
- Architecture-agnostic: all arch-specific ops delegate to base_trainer.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image

from musubi_tuner.grpo.advantage import compute_group_advantages
from musubi_tuner.grpo.reward import BaseReward, build_rewards
from musubi_tuner.grpo.reward.base import _REWARD_REGISTRY

from .config import NFTConfig

logger = logging.getLogger(__name__)


class NFTTrainer:
    """Online DiffusionNFT trainer wrapping any musubi-tuner NetworkTrainer subclass.

    Args:
        base_trainer: An already-initialised NetworkTrainer instance.
        config: NFT hyper-parameter configuration.
        accelerator: HuggingFace Accelerate accelerator.
        args: Parsed argparse namespace passed to the base trainer.
        transformer: The DiT with LoRA applied (accelerator.prepare()'d).
        vae: Loaded VAE (kept on CPU; moved to device when needed).
        network: LoRA network module (accelerator.prepare()'d).
        dit_dtype: dtype for DiT forward passes.
        network_dtype: dtype for trainable network parameters.
    """

    def __init__(
        self,
        base_trainer,
        config: NFTConfig,
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
        self._step_count = 0

        # Build reward instances: list of (name, BaseReward, weight)
        self._reward_list: list[tuple[BaseReward, float]] = build_rewards(config.rewards)
        inv = {v: k for k, v in _REWARD_REGISTRY.items()}
        self._named_rewards: list[tuple[str, BaseReward, float]] = [
            (inv.get(type(rw), type(rw).__name__), rw, w)
            for rw, w in self._reward_list
        ]

        # Snapshot initial LoRA weights on CPU as the "old policy" starting point.
        # These are updated every old_policy_update_every steps via _update_old_policy().
        unwrapped_net = accelerator.unwrap_model(network)
        self._old_state: dict[str, torch.Tensor] = {
            n: p.detach().cpu().clone()
            for n, p in unwrapped_net.named_parameters()
            if p.requires_grad
        }

        # Deepcopy the transformer (with initial LoRA weights) as a frozen reference
        # for KL regularization — prevents training from drifting too far from the
        # initial checkpoint.  Skipped when kl_coeff == 0 to save ~1 GB memory.
        self.ref_transformer: Optional[torch.nn.Module] = None
        if config.kl_coeff > 0:
            self.ref_transformer = copy.deepcopy(accelerator.unwrap_model(transformer))
            self.ref_transformer.requires_grad_(False)
            self.ref_transformer.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        sample_parameters: list[dict],
        reference_images: Optional[list[Optional[Image.Image]]] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run one NFT training step.

        Args:
            sample_parameters: List of dicts from base_trainer.process_sample_prompts().
            reference_images: Optional PIL images for reference-based rewards (e.g. delta_e00).

        Returns:
            (loss, log_dict) where loss is a scalar tensor with gradient.
        """
        device = self.accelerator.device
        G = self.config.group_size

        # ── Phase 1: rollout with old policy (no_grad) ─────────────────────
        with torch.no_grad():
            all_images: list[Image.Image] = []
            all_prompts: list[str] = []
            all_ref_images: list[Optional[Image.Image]] = []
            all_param_indices: list[int] = []

            for idx, sp in enumerate(sample_parameters):
                for _ in range(G):
                    generator = torch.Generator(device=device).manual_seed(
                        torch.randint(0, 2**31, (1,)).item()
                    )
                    video = self._rollout_one_old_policy(sp, generator)
                    all_images.append(_video_to_pil(video))
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
                finally:
                    rw.offload()

            # ── Per-group advantage → r ∈ [0, 1] ───────────────────────────
            weight_map = {name: w for name, _rw, w in self._named_rewards}
            adv = compute_group_advantages(scores, weight_map, G)  # [B*G]
            adv = adv.to(device=device, dtype=torch.float32)

        # ── Phase 2: NFT loss (with grad) ──────────────────────────────────
        loss, log_dict = self._nft_loss(all_images, adv, sample_parameters, all_param_indices)

        # Append reward stats to log
        for name, s in scores.items():
            log_dict[f"reward/{name}"] = s.mean().item()
        log_dict["reward/advantage_mean"] = adv.mean().item()
        log_dict["reward/advantage_std"] = adv.std().item()

        # ── Update old policy ───────────────────────────────────────────────
        self._step_count += 1
        if self._step_count % self.config.old_policy_update_every == 0:
            self._update_old_policy()

        return loss, log_dict

    # ------------------------------------------------------------------
    # Internal: old policy management
    # ------------------------------------------------------------------

    def _with_old_policy(self, fn: Callable[[], Any]) -> Any:
        """Temporarily swap LoRA weights to old_state, call fn(), then restore."""
        net = self.accelerator.unwrap_model(self.network)
        # Backup current params (in-place, stays on device)
        current: dict[str, torch.Tensor] = {
            n: p.data.clone() for n, p in net.named_parameters() if p.requires_grad
        }
        # Load old params onto device
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data.copy_(self._old_state[n].to(p.device))
        try:
            result = fn()
        finally:
            # Always restore current params, even if fn() raises
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.data.copy_(current[n])
        return result

    def _update_old_policy(self) -> None:
        """EMA-update old_state toward current LoRA params.

        decay=0.0 → full copy (PPO-style hard reset each epoch).
        decay>0.0 → exponential moving average (slower drift).
        """
        decay = self.config.old_policy_decay
        net = self.accelerator.unwrap_model(self.network)
        for n, p in net.named_parameters():
            if p.requires_grad:
                old = self._old_state[n].to(p.device)
                self._old_state[n] = (decay * old + (1.0 - decay) * p.detach()).cpu()

    # ------------------------------------------------------------------
    # Internal: rollout
    # ------------------------------------------------------------------

    def _rollout_one_old_policy(
        self, sample_parameter: dict, generator: torch.Generator
    ) -> Optional[np.ndarray]:
        """Inference with the old policy (LoRA weight swap)."""
        cfg = self.config
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

        def _infer():
            return self.base.do_inference(
                self.accelerator, self.args, sp, self.vae, self.dit_dtype,
                transformer, sp["discrete_flow_shift"], sp["sample_steps"],
                width, height, sp["frame_count"], generator,
                do_classifier_free_guidance=False,
                guidance_scale=sp["guidance_scale"], cfg_scale=None,
            )

        try:
            video = self._with_old_policy(_infer)
        finally:
            transformer.train(was_train)
            self.vae.to("cpu")

        return video

    # ------------------------------------------------------------------
    # Internal: NFT loss
    # ------------------------------------------------------------------

    def _nft_loss(
        self,
        images: list[Image.Image],
        advantages: torch.Tensor,   # [B*G], float32, on device
        sample_parameters: list[dict],
        param_indices: list[int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the NFT loss.

        Steps:
        1. VAE-encode generated images → scaled latents x0.
        2. Sample t ~ U(0, 1) per sample.
        3. Construct noisy latents x_t = (1-t)*x0 + t*ε.
        4. Three forward passes: v_old (no_grad, old LoRA), v_θ (with_grad, current LoRA),
           v_ref (no_grad, frozen ref_transformer).
        5. Implicit positive/negative velocity:
               v_pos = β*v_θ + (1-β)*v_old
               v_neg = (1+β)*v_old - β*v_θ
        6. x0 reconstructions and adaptive-weight MSE loss.
        7. Map advantages to r ∈ [0,1]; combined NFT + KL objective.
        """
        device = self.accelerator.device
        bsz = len(images)
        cfg = self.config
        log: dict[str, float] = {}

        # ── 1. Encode images to latents ─────────────────────────────────────
        latents = self._encode_images_to_latents(images)            # [B, C, F, H, W]
        latents = latents.to(device=device, dtype=self.dit_dtype)
        latents = self.base.scale_shift_latents(latents)

        # ── 2–3. Sample t and build noisy latents ───────────────────────────
        noise = torch.randn_like(latents)
        t = torch.rand(bsz, device=device)                          # [B]
        t_view = t.view(bsz, *([1] * (latents.ndim - 1)))          # [B, 1, ...]
        noisy = (1.0 - t_view) * latents + t_view * noise           # x_t

        batch_for_dit = self._build_batch_dict(sample_parameters, param_indices, bsz, device, latents=latents)

        # ── 4a. v_old — old policy, no grad ─────────────────────────────────
        def _call_old():
            tr = self.accelerator.unwrap_model(self.transformer)
            tr.eval()
            out, _ = self.base.call_dit(
                self.args, self.accelerator, tr,
                latents, batch_for_dit, noise, noisy, t, self.network_dtype,
            )
            return out.detach().to(torch.float32)

        with torch.no_grad():
            v_old = self._with_old_policy(_call_old)                # [B, C, F, H, W], fp32

        # ── 4b. v_θ — current policy, with grad ────────────────────────────
        v_theta, _ = self.base.call_dit(
            self.args, self.accelerator, self.transformer,
            latents, batch_for_dit, noise, noisy, t, self.network_dtype,
        )
        v_theta = v_theta.to(torch.float32)                          # [B, C, F, H, W]

        # ── 4c. v_ref — frozen reference, no grad ───────────────────────────
        v_ref = None
        if self.ref_transformer is not None:
            with torch.no_grad():
                v_ref, _ = self.base.call_dit(
                    self.args, self.accelerator, self.ref_transformer,
                    latents, batch_for_dit, noise, noisy, t, self.network_dtype,
                )
            v_ref = v_ref.detach().to(torch.float32)

        # ── 5. Implicit positive and negative velocity ──────────────────────
        beta = cfg.beta
        v_pos = beta * v_theta + (1.0 - beta) * v_old
        v_neg = (1.0 + beta) * v_old - beta * v_theta               # reflected across v_old

        # ── 6. x0 reconstructions (flow-matching: x0 = x_t - t * v) ────────
        latents_f32 = latents.to(torch.float32)
        noisy_f32 = noisy.to(torch.float32)

        x0_pos = noisy_f32 - t_view * v_pos                         # [B, C, F, H, W]
        x0_neg = noisy_f32 - t_view * v_neg

        spatial_dims = list(range(1, latents_f32.ndim))

        # Adaptive per-sample normalization (prevents large-error samples from dominating)
        w_pos = (x0_pos - latents_f32).abs().mean(dim=spatial_dims, keepdim=True).detach().clamp(min=1e-5)
        w_neg = (x0_neg - latents_f32).abs().mean(dim=spatial_dims, keepdim=True).detach().clamp(min=1e-5)

        pos_loss = ((x0_pos - latents_f32) ** 2 / w_pos).mean(dim=spatial_dims)  # [B]
        neg_loss = ((x0_neg - latents_f32) ** 2 / w_neg).mean(dim=spatial_dims)  # [B]

        # ── 7. Advantage → r ∈ [0, 1], combined objective ───────────────────
        adv_clipped = advantages.clamp(-cfg.adv_clip_max, cfg.adv_clip_max)
        r = adv_clipped / cfg.adv_clip_max / 2.0 + 0.5              # [B], in [0, 1]

        nft_loss = (r * pos_loss + (1.0 - r) * neg_loss).mean() / beta

        # KL regularization
        kl_loss = torch.tensor(0.0, device=device)
        if v_ref is not None:
            kl_loss = ((v_theta - v_ref) ** 2).mean() * cfg.kl_coeff

        total_loss = nft_loss + kl_loss
        log["loss/nft"] = nft_loss.item()
        log["loss/kl"] = kl_loss.item()
        log["loss/total"] = total_loss.item()

        return total_loss, log

    # ------------------------------------------------------------------
    # Internal: VAE encoding and batch construction
    # (identical logic to GRPOTrainer — duplicated to keep modules independent)
    # ------------------------------------------------------------------

    def _encode_images_to_latents(self, images: list[Image.Image]) -> torch.Tensor:
        """VAE-encode PIL images → [B, C, F, H, W] float32 latents."""
        device = self.accelerator.device
        self.vae.to(device)
        self.vae.eval()

        frames = []
        for img in images:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # [3, H, W]

        imgs_t = torch.stack(frames).to(device=device, dtype=self.vae.dtype)  # [B, 3, H, W]

        vae_chunk = 4
        with torch.no_grad():
            chunks: list[torch.Tensor] = []
            for start in range(0, len(images), vae_chunk):
                chunk_imgs = imgs_t[start:start + vae_chunk]
                if hasattr(self.vae, "latents_mean"):
                    chunk_lat = self.vae.encode_pixels_to_latents(chunk_imgs)
                else:
                    chunk_imgs = chunk_imgs * 2.0 - 1.0
                    chunk_imgs = chunk_imgs.unsqueeze(2)
                    latent_dist = self.vae.encode(chunk_imgs)
                    if hasattr(latent_dist, "latent_dist"):
                        chunk_lat = latent_dist.latent_dist.sample()
                    elif hasattr(latent_dist, "sample"):
                        chunk_lat = latent_dist.sample()
                    else:
                        chunk_lat = latent_dist
                    if hasattr(self.vae, "config"):
                        if getattr(self.vae.config, "shift_factor", None):
                            chunk_lat = (chunk_lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                        elif getattr(self.vae.config, "scaling_factor", None):
                            chunk_lat = chunk_lat * self.vae.config.scaling_factor
                chunks.append(chunk_lat)
            latents = torch.cat(chunks, dim=0)

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
        """Build call_dit-compatible batch dict from sample_parameters."""
        first_sp = sample_parameters[param_indices[0]]

        def _pad_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
            max_len = max(t.shape[1] if t.ndim > 1 else 1 for t in tensors)
            padded = []
            for t in tensors:
                if t.ndim > 1 and t.shape[1] < max_len:
                    pad = torch.zeros(
                        *t.shape[:-2], max_len - t.shape[1], *t.shape[2:],
                        device=t.device, dtype=t.dtype,
                    )
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            return torch.stack(padded, dim=0)

        if "vl_embed" in first_sp:
            vl_list = [
                sample_parameters[idx]["vl_embed"].squeeze(0).to(device)
                for idx in param_indices
            ]
            batch: dict = {"vl_embed": vl_list}
        else:
            batch = {
                "llm": _pad_stack([sample_parameters[idx]["llm_embeds"].to(device) for idx in param_indices]),
                "llm_mask": _pad_stack([sample_parameters[idx]["llm_mask"].to(device) for idx in param_indices]),
                "clipL": _pad_stack([sample_parameters[idx]["clipL_embeds"].to(device) for idx in param_indices]),
            }

        if latents is not None:
            batch["latents"] = latents

        return batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _video_to_pil(video: Any) -> Image.Image:
    """Convert decoded video [1, C, F, H, W] float32 to PIL RGB."""
    if video is None:
        return Image.new("RGB", (256, 256))
    if isinstance(video, torch.Tensor):
        arr = video.cpu().float().numpy()
    else:
        arr = np.asarray(video, dtype=np.float32)
    while arr.ndim > 4:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[:, 0]  # first frame → [C, H, W]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)  # [H, W, C]
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return Image.fromarray(arr.squeeze(-1) if arr.ndim == 3 else arr, "L").convert("RGB")
    return Image.fromarray(arr[..., :3], "RGB")
