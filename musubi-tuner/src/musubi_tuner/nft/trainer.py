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

        # Save initial LoRA weights as a frozen reference for KL regularization.
        # Uses the same weight-swap pattern as _with_old_policy to avoid deepcopy
        # of the transformer (which fails when blocks_to_swap ModelOffloader is active).
        # Skipped when kl_coeff == 0 to save memory.
        self._ref_state: Optional[dict[str, torch.Tensor]] = None
        if config.kl_coeff > 0:
            self._ref_state = {
                n: p.detach().cpu().clone()
                for n, p in unwrapped_net.named_parameters()
                if p.requires_grad
            }

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
        device = self.accelerator.device
        # Backup current params, cloned to CPU to avoid holding GPU memory
        current: dict[str, torch.Tensor] = {
            n: p.data.clone().cpu() for n, p in net.named_parameters() if p.requires_grad
        }
        # Force all LoRA params to GPU with old weights.
        # This prevents device mismatch when blocks_to_swap offloads some blocks to CPU:
        # the block offloader moves base weights, but LoRA weights must be on GPU first.
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data = self._old_state[n].to(device)
        try:
            result = fn()
        finally:
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.data = current[n].to(device)
        return result

    def _with_ref_policy(self, fn: Callable[[], Any]) -> Any:
        """Temporarily swap LoRA weights to _ref_state (initial checkpoint), call fn(), then restore."""
        net = self.accelerator.unwrap_model(self.network)
        device = self.accelerator.device
        current: dict[str, torch.Tensor] = {
            n: p.data.clone().cpu() for n, p in net.named_parameters() if p.requires_grad
        }
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data = self._ref_state[n].to(device)
        try:
            result = fn()
        finally:
            for n, p in net.named_parameters():
                if p.requires_grad:
                    p.data = current[n].to(device)
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
        """Compute the NFT loss, optionally in micro-batches (phase2_chunk_size).

        Steps:
        1. VAE-encode generated images → scaled latents x0.
        2. Sample t ~ U(0, 1) per sample.
        3. Construct noisy latents x_t = (1-t)*x0 + t*ε.
        4. Three forward passes per chunk: v_old (no_grad, old LoRA), v_ref (no_grad, initial
           LoRA), v_θ (with_grad, current LoRA).
        5. Implicit positive/negative velocity; adaptive-weight MSE; advantage-weighted loss.
        6. When chunked (phase2_chunk_size > 0): backward() is called immediately after each
           chunk to free that chunk's graph before the next chunk is computed. Peak activation
           memory scales with chunk_size rather than full group_size. Returns a detached zero
           tensor so the outer accelerator.backward() in the training loop is a no-op.
           When un-chunked (phase2_chunk_size == 0): accumulate loss and return tensor as usual.
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

        # ── 4–7. (Chunked) forward passes ────────────────────────────────────
        chunk = cfg.phase2_chunk_size if cfg.phase2_chunk_size > 0 else bsz
        chunked_backward = cfg.phase2_chunk_size > 0

        beta = cfg.beta

        # Accumulators: float for the chunked path (graph freed per chunk),
        # tensor for the non-chunked path (single backward at the end).
        nft_log = 0.0
        kl_log = 0.0
        nft_sum = torch.tensor(0.0, device=device)
        kl_num = torch.tensor(0.0, device=device)
        kl_denom = 0

        for cs in range(0, bsz, chunk):
            ce = min(cs + chunk, bsz)
            c = ce - cs
            sl = slice(cs, ce)

            lat_c = latents[sl]
            noisy_c = noisy[sl]
            noise_c = noise[sl]
            t_c = t[sl]
            t_view_c = t_view[sl]
            adv_c = advantages[sl]
            pidx_c = param_indices[cs:ce]

            batch_c = self._build_batch_dict(sample_parameters, pidx_c, c, device, latents=lat_c)

            # 4a. v_old — old policy, no grad
            def _call_old(lat=lat_c, bat=batch_c, noi=noise_c, noisy=noisy_c, tc=t_c):
                tr = self.accelerator.unwrap_model(self.transformer)
                was_training = tr.training
                tr.eval()
                try:
                    out, _ = self.base.call_dit(
                        self.args, self.accelerator, tr,
                        lat, bat, noi, noisy, tc, self.network_dtype,
                    )
                finally:
                    tr.train(was_training)
                return out.detach().to(torch.float32)

            with torch.no_grad():
                v_old_c = self._with_old_policy(_call_old)

            # 4b. v_ref — initial LoRA policy, no grad (before v_θ to free memory first)
            v_ref_c = None
            if self._ref_state is not None:
                def _call_ref(lat=lat_c, bat=batch_c, noi=noise_c, noisy=noisy_c, tc=t_c):
                    tr = self.accelerator.unwrap_model(self.transformer)
                    was_training = tr.training
                    tr.eval()
                    try:
                        out, _ = self.base.call_dit(
                            self.args, self.accelerator, tr,
                            lat, bat, noi, noisy, tc, self.network_dtype,
                        )
                    finally:
                        tr.train(was_training)
                    return out.detach().to(torch.float32)

                with torch.no_grad():
                    v_ref_c = self._with_ref_policy(_call_ref)

            # Free no_grad intermediates before the gradient-tracked forward pass
            torch.cuda.empty_cache()

            # 4c. v_θ — current policy, with grad (transformer must be in training mode)
            v_theta_c, _ = self.base.call_dit(
                self.args, self.accelerator, self.transformer,
                lat_c, batch_c, noise_c, noisy_c, t_c, self.network_dtype,
            )
            v_theta_c = v_theta_c.to(torch.float32)

            # 5. Implicit positive and negative velocity
            v_pos_c = beta * v_theta_c + (1.0 - beta) * v_old_c
            v_neg_c = (1.0 + beta) * v_old_c - beta * v_theta_c

            # 6. x0 reconstructions (flow-matching: x0 = x_t - t * v)
            lat_f32_c = lat_c.to(torch.float32)
            noisy_f32_c = noisy_c.to(torch.float32)

            x0_pos_c = noisy_f32_c - t_view_c * v_pos_c
            x0_neg_c = noisy_f32_c - t_view_c * v_neg_c

            spatial_dims = list(range(1, lat_f32_c.ndim))
            w_pos_c = (x0_pos_c - lat_f32_c).abs().mean(dim=spatial_dims, keepdim=True).detach().clamp(min=1e-5)
            w_neg_c = (x0_neg_c - lat_f32_c).abs().mean(dim=spatial_dims, keepdim=True).detach().clamp(min=1e-5)

            pos_loss_c = ((x0_pos_c - lat_f32_c) ** 2 / w_pos_c).mean(dim=spatial_dims)  # [c]
            neg_loss_c = ((x0_neg_c - lat_f32_c) ** 2 / w_neg_c).mean(dim=spatial_dims)  # [c]

            # 7. Advantage → r ∈ [0, 1]
            adv_clipped_c = adv_c.clamp(-cfg.adv_clip_max, cfg.adv_clip_max)
            r_c = adv_clipped_c / cfg.adv_clip_max / 2.0 + 0.5

            if chunked_backward:
                # Scale by c/bsz so summing chunks = original mean-over-B
                nft_c = (r_c * pos_loss_c + (1.0 - r_c) * neg_loss_c).sum() / beta / bsz
                kl_c = torch.zeros([], device=device)
                if v_ref_c is not None:
                    kl_c = ((v_theta_c - v_ref_c) ** 2).mean() * cfg.kl_coeff * c / bsz

                chunk_total = nft_c + kl_c
                nft_log += nft_c.detach().item()
                kl_log += kl_c.detach().item()

                # Backward immediately — frees this chunk's computation graph
                self.accelerator.backward(chunk_total)
                del v_theta_c, v_pos_c, v_neg_c, x0_pos_c, x0_neg_c, nft_c, kl_c, chunk_total
                torch.cuda.empty_cache()
            else:
                nft_sum = nft_sum + (r_c * pos_loss_c + (1.0 - r_c) * neg_loss_c).sum() / beta
                if v_ref_c is not None:
                    kl_num = kl_num + ((v_theta_c - v_ref_c) ** 2).sum()
                    kl_denom += v_theta_c.numel()

        if chunked_backward:
            # Gradients already in .grad buffers. Return detached zero so the
            # outer accelerator.backward() in the training loop is a no-op.
            log["loss/nft"] = nft_log
            log["loss/kl"] = kl_log
            log["loss/total"] = nft_log + kl_log
            return torch.zeros([], device=device), log

        # Non-chunked path: return loss tensor for the outer backward()
        nft_loss = nft_sum / bsz
        kl_loss = torch.tensor(0.0, device=device)
        if kl_denom > 0:
            kl_loss = (kl_num / kl_denom) * cfg.kl_coeff

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
