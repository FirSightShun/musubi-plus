"""nft_train_network.py — Entry script for DiffusionNFT online RL fine-tuning.

Usage:
    accelerate launch --mixed_precision bf16 \\
        src/musubi_tuner/nft_train_network.py \\
        --nft_config nft_config.toml \\
        --prompt_file prompts.jsonl \\
        --dit path/to/dit \\
        --vae path/to/vae \\
        --network_module networks.lora \\
        --network_dim 32 \\
        [any other arch-specific flags]

Reference: "DiffusionNFT: Negative-aware FineTuning for Diffusion Models"
           NVIDIA Research, ICLR 2026 — https://arxiv.org/abs/2509.16117
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARCH_TRAINERS: dict[str, str] = {
    "hv": "musubi_tuner.hv_train_network",
    "hv_1_5": "musubi_tuner.hv_1_5_train_network",
    "wan": "musubi_tuner.wan_train_network",
    "fpack": "musubi_tuner.fpack_train_network",
    "flux_2": "musubi_tuner.flux_2_train_network",
    "flux_kontext": "musubi_tuner.flux_kontext_train_network",
    "qwen_image": "musubi_tuner.qwen_image_train_network",
    "kandinsky5": "musubi_tuner.kandinsky5_train_network",
    "zimage": "musubi_tuner.zimage_train_network",
}


def _import_trainer(architecture: str):
    module_path = ARCH_TRAINERS.get(architecture)
    if module_path is None:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {list(ARCH_TRAINERS)}")
    mod = importlib.import_module(module_path)

    import inspect
    from musubi_tuner.hv_train_network import NetworkTrainer as _BaseTrainer

    arch_classes = [
        cls
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, _BaseTrainer) and cls is not _BaseTrainer and cls.__module__ == mod.__name__
    ]
    trainer_cls = arch_classes[0] if arch_classes else getattr(mod, "NetworkTrainer", _BaseTrainer)

    return trainer_cls, getattr(mod, "setup_parser_common", None), mod


def main():
    # ── 1. Pre-parse to find nft_config and architecture ─────────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--nft_config", type=str, required=True)
    pre_parser.add_argument("--prompt_file", type=str, required=True)
    pre_parser.add_argument("--nft_architecture", type=str, default=None)
    pre_parser.add_argument("--nft_steps", type=int, default=None)
    pre_parser.add_argument("--nft_batch_size", type=int, default=1)
    pre_args, remaining = pre_parser.parse_known_args()

    from musubi_tuner.nft.config import NFTConfig

    nft_config = NFTConfig.from_toml(pre_args.nft_config)
    architecture = pre_args.nft_architecture or nft_config.architecture

    # ── 2. Import arch-specific trainer and build full parser ────────────
    TrainerClass, _, arch_mod = _import_trainer(architecture)

    base_parser = arch_mod.setup_parser_common()
    try:
        arch_extra = getattr(arch_mod, f"{architecture}_setup_parser", None) or getattr(arch_mod, "setup_parser", None)
        if arch_extra:
            base_parser = arch_extra(base_parser)
    except Exception:
        pass

    base_parser.add_argument("--nft_config", type=str, required=True)
    base_parser.add_argument("--prompt_file", type=str, required=True)
    base_parser.add_argument("--nft_architecture", type=str, default=None)
    base_parser.add_argument("--nft_steps", type=int, default=None)
    base_parser.add_argument("--nft_batch_size", type=int, default=1)

    args = base_parser.parse_args()

    if not hasattr(args, "dataset_config") or args.dataset_config is None:
        args.dataset_config = "__nft_placeholder__"

    # ── 3. Build trainer and run NFT loop ─────────────────────────────────
    base_trainer = TrainerClass()
    _nft_loop(base_trainer, args, nft_config, pre_args)


def _nft_loop(base_trainer, args, nft_config, pre_args):
    """The NFT training loop."""
    from accelerate import Accelerator, InitProcessGroupKwargs
    from accelerate.utils import set_seed
    from datetime import timedelta

    from musubi_tuner.grpo.prompt_dataset import PromptDataset
    from musubi_tuner.nft.trainer import NFTTrainer

    # ── Accelerator ──────────────────────────────────────────────────────
    timeout = getattr(args, "ddp_timeout", None) or 3600
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout))

    mixed_precision = getattr(args, "mixed_precision", "bf16") or "bf16"
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        mixed_precision=mixed_precision,
        log_with=getattr(args, "log_with", None),
        project_dir=getattr(args, "logging_dir", None),
        kwargs_handlers=[kwargs],
    )

    if hasattr(args, "seed") and args.seed is not None:
        set_seed(args.seed)

    device = accelerator.device
    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mixed_precision, torch.float32)
    dit_dtype = weight_dtype
    network_dtype = weight_dtype

    # ── Load models ──────────────────────────────────────────────────────
    accelerator.print(f"Loading models for architecture: {nft_config.architecture}")

    if not hasattr(args, "dit") or args.dit is None:
        raise ValueError("--dit is required")
    if not hasattr(args, "vae") or args.vae is None:
        raise ValueError("--vae is required")

    attn_mode = getattr(args, "attn_mode", None) or "torch"
    for flag, mode in (
        ("flash_attn", "flash"),
        ("flash3", "flash"),
        ("sageattn", "sageattn"),
        ("xformers", "xformers"),
    ):
        if getattr(args, flag, False):
            attn_mode = mode
            break

    if hasattr(args, "model_version") and not hasattr(args, "is_edit"):
        try:
            from musubi_tuner.qwen_image import qwen_image_utils
            qwen_image_utils.resolve_model_version_args(args)
        except Exception:
            pass

    base_trainer.handle_model_specific_args(args)

    vae = base_trainer.load_vae(args, weight_dtype, args.vae)
    vae.eval()
    vae.requires_grad_(False)

    dit_weight_dtype = None
    if getattr(args, "fp8_base", False):
        dit_weight_dtype = torch.float8_e4m3fn
    elif getattr(args, "fp16_base", False):
        dit_weight_dtype = torch.float16

    _load_dit_dtype = None if getattr(args, "fp8_scaled", False) else (dit_weight_dtype or dit_dtype)
    transformer = base_trainer.load_transformer(
        accelerator, args, args.dit, attn_mode, getattr(args, "split_attn", False), "cpu", _load_dit_dtype
    )
    transformer.requires_grad_(False)
    transformer.eval()

    blocks_to_swap = getattr(args, "blocks_to_swap", 0) or 0
    if blocks_to_swap > 0:
        transformer.enable_block_swap(
            blocks_to_swap, device,
            supports_backward=True,
            use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
        )
        transformer.move_to_device_except_swap_blocks(device)
    else:
        transformer.to(device, dtype=_load_dit_dtype)

    # ── LoRA network ─────────────────────────────────────────────────────
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    if getattr(args, "network_args", None):
        for net_arg in args.network_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    if hasattr(network_module, "create_arch_network"):
        network = network_module.create_arch_network(
            1.0, args.network_dim,
            getattr(args, "network_alpha", None) or args.network_dim,
            vae, None, transformer,
            neuron_dropout=getattr(args, "network_dropout", None),
            **net_kwargs,
        )
    else:
        network = network_module.create_network(
            1.0, args.network_dim,
            getattr(args, "network_alpha", None) or args.network_dim,
            vae, None, transformer, **net_kwargs,
        )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.to(device, dtype=network_dtype)

    # ── Load initial LoRA weights if provided ────────────────────────────
    if getattr(args, "network_weights", None):
        accelerator.print(f"Loading LoRA weights: {args.network_weights}")
        network.load_weights(args.network_weights)

    # ── Optimizer ────────────────────────────────────────────────────────
    trainable_params = network.get_trainable_params()
    lr = getattr(args, "learning_rate", 1e-4) or 1e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── Process prompts ──────────────────────────────────────────────────
    has_te = (
        (hasattr(args, "text_encoder1") and args.text_encoder1 is not None)
        or (hasattr(args, "text_encoder") and args.text_encoder is not None)
        or (hasattr(args, "text_encoder_path") and args.text_encoder_path is not None)
    )
    if not has_te:
        raise ValueError("A text encoder argument is required (--text_encoder / --text_encoder1)")

    sample_parameters = base_trainer.process_sample_prompts(args, accelerator, pre_args.prompt_file)

    # Move embeddings to CPU to free GPU memory
    for sp in sample_parameters:
        for k, v in list(sp.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                sp[k] = v.cpu()
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor) and v[0].is_cuda:
                sp[k] = [t.cpu() for t in v]

    # ── Accelerator prepare ──────────────────────────────────────────────
    if blocks_to_swap > 0:
        transformer_prepared = accelerator.prepare(transformer, device_placement=[False])
        accelerator.unwrap_model(transformer_prepared).prepare_block_swap_before_forward()
    else:
        transformer_prepared = accelerator.prepare(transformer)

    network_prepared, optimizer_prepared = accelerator.prepare(network, optimizer)

    if not getattr(args, "gradient_checkpointing", False):
        transformer_prepared.eval()
    else:
        unwrapped = accelerator.unwrap_model(transformer_prepared)
        if hasattr(unwrapped, "enable_gradient_checkpointing"):
            cpu_offload = getattr(args, "gradient_checkpointing_cpu_offload", False)
            unwrapped.enable_gradient_checkpointing(cpu_offload)
        if hasattr(network, "enable_gradient_checkpointing"):
            network.enable_gradient_checkpointing()

    accelerator.unwrap_model(network_prepared).prepare_grad_etc(transformer_prepared)

    # ── Prompt dataset ───────────────────────────────────────────────────
    prompt_dataset = PromptDataset(pre_args.prompt_file)
    batch_size = pre_args.nft_batch_size
    max_steps = pre_args.nft_steps or getattr(args, "max_train_steps", 100) or 100

    # ── NFTTrainer ───────────────────────────────────────────────────────
    nft_trainer = NFTTrainer(
        base_trainer=base_trainer,
        config=nft_config,
        accelerator=accelerator,
        args=args,
        transformer=transformer_prepared,
        vae=vae,
        network=network_prepared,
        dit_dtype=dit_dtype,
        network_dtype=network_dtype,
    )

    if nft_trainer.ref_transformer is not None:
        nft_trainer.ref_transformer.to(device, dtype=dit_dtype)

    accelerator.print(f"Starting NFT training: {max_steps} steps, group_size={nft_config.group_size}")
    accelerator.print(f"Rewards: {[(n, w) for n, _r, w in nft_trainer._named_rewards]}")

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process, desc="NFT steps")

    output_dir = getattr(args, "output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    # ── Training loop ────────────────────────────────────────────────────
    for global_step in range(max_steps):
        start = (global_step * batch_size * num_processes + process_index * batch_size) % len(sample_parameters)
        batch_params = [sample_parameters[(start + i) % len(sample_parameters)] for i in range(batch_size)]

        ref_imgs = []
        for i in range(batch_size):
            item = prompt_dataset[(start + i) % len(prompt_dataset)]
            if item.reference_image_path:
                try:
                    from PIL import Image as _PIL
                    ref_imgs.append(_PIL.open(item.reference_image_path).convert("RGB"))
                except Exception:
                    ref_imgs.append(None)
            else:
                ref_imgs.append(None)
        reference_images = ref_imgs if any(r is not None for r in ref_imgs) else None

        with accelerator.accumulate(network_prepared):
            loss, log_dict = nft_trainer.step(batch_params, reference_images=reference_images)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                max_grad_norm = getattr(args, "max_grad_norm", 1.0) or 1.0
                if max_grad_norm > 0:
                    params_to_clip = accelerator.unwrap_model(network_prepared).get_trainable_params()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

            optimizer_prepared.step()
            optimizer_prepared.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{log_dict.get('loss/total', 0):.4f}",
                nft=f"{log_dict.get('loss/nft', 0):.4f}",
            )

            if accelerator.is_main_process:
                adv_mean = log_dict.get("reward/advantage_mean", float("nan"))
                reward_keys = ["delta_e00", "pickscore", "hps_v2", "image_reward", "clip", "vlm"]
                reward_str = "  ".join(
                    f"{k}={log_dict[f'reward/{k}']:.4f}"
                    for k in reward_keys
                    if f"reward/{k}" in log_dict
                )
                accelerator.print(
                    f"[step {global_step + 1:5d}] "
                    f"loss={log_dict.get('loss/total', 0):+.4f}  "
                    f"nft={log_dict.get('loss/nft', 0):+.4f}  "
                    f"kl={log_dict.get('loss/kl', 0):.6f}  "
                    f"{reward_str}  adv={adv_mean:+.4f}"
                )
                if len(accelerator.trackers) > 0:
                    accelerator.log(log_dict, step=global_step)

            save_every = getattr(args, "save_every_n_steps", None)
            if save_every and (global_step + 1) % save_every == 0:
                ckpt_name = f"{getattr(args, 'output_name', 'nft')}_{global_step + 1:06d}.safetensors"
                ckpt_path = os.path.join(output_dir, ckpt_name)
                accelerator.print(f"\nSaving checkpoint: {ckpt_path}")
                accelerator.unwrap_model(network_prepared).save_weights(ckpt_path, network_dtype, {})

    # ── Final save ───────────────────────────────────────────────────────
    if accelerator.is_main_process:
        ckpt_name = f"{getattr(args, 'output_name', 'nft')}_final.safetensors"
        ckpt_path = os.path.join(output_dir, ckpt_name)
        accelerator.print(f"\nSaving final checkpoint: {ckpt_path}")
        accelerator.unwrap_model(network_prepared).save_weights(ckpt_path, network_dtype, {})

    accelerator.end_training()
    accelerator.print("NFT training complete.")


if __name__ == "__main__":
    main()
