"""grpo_train_network.py — Entry script for GRPO online RL fine-tuning.

Usage:
    accelerate launch --mixed_precision bf16 \\
        src/musubi_tuner/grpo_train_network.py \\
        --grpo_config grpo_config.toml \\
        --prompt_file prompts.jsonl \\
        --dit path/to/dit \\
        --vae path/to/vae \\
        --network_module networks.lora \\
        --network_dim 32 \\
        [any other hv_train_network.py flags]

The script dynamically imports the NetworkTrainer for the target architecture
(``--architecture`` in the TOML, or ``--grpo_architecture`` CLI flag), runs the
standard setup (model loading, LoRA init, optimizer), then replaces the
standard training loop with the GRPO loop.
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

# Architecture → trainer module path mapping
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

    # Find the concrete trainer class: prefer the arch-specific subclass over the base NetworkTrainer.
    # Each train_network module has a main() that instantiates the actual class — scan for it.
    import inspect
    from musubi_tuner.hv_train_network import NetworkTrainer as _BaseTrainer

    # Look for all subclasses of NetworkTrainer defined directly in this module
    arch_classes = [
        cls
        for _, cls in inspect.getmembers(mod, inspect.isclass)
        if issubclass(cls, _BaseTrainer) and cls is not _BaseTrainer and cls.__module__ == mod.__name__
    ]
    trainer_cls = arch_classes[0] if arch_classes else getattr(mod, "NetworkTrainer", _BaseTrainer)

    return trainer_cls, getattr(mod, "setup_parser_common", None), mod


def main():
    # ── 1. Parse GRPO-specific args (pre-pass to find grpo_config and arch) ──
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--grpo_config", type=str, required=True, help="Path to GRPO TOML config file")
    pre_parser.add_argument("--prompt_file", type=str, required=True, help="Path to prompt JSONL or .txt file")
    pre_parser.add_argument("--grpo_architecture", type=str, default=None, help="Override architecture from TOML config")
    pre_parser.add_argument("--grpo_steps", type=int, default=None, help="Number of GRPO training steps (overrides max_train_steps)")
    pre_parser.add_argument("--grpo_batch_size", type=int, default=1, help="Number of prompts per GRPO step")
    pre_args, remaining = pre_parser.parse_known_args()

    from musubi_tuner.grpo.config import GRPOConfig

    grpo_config = GRPOConfig.from_toml(pre_args.grpo_config)
    architecture = pre_args.grpo_architecture or grpo_config.architecture

    # ── 2. Import architecture-specific trainer and build full parser ────────
    TrainerClass, arch_setup_parser_fn, arch_mod = _import_trainer(architecture)

    # Build the full argument parser using the architecture's own setup function
    base_parser = arch_mod.setup_parser_common()
    try:
        # Some architectures have an arch-specific extra parser function
        arch_extra = getattr(arch_mod, f"{architecture}_setup_parser", None) or getattr(arch_mod, "setup_parser", None)
        if arch_extra:
            base_parser = arch_extra(base_parser)
    except Exception:
        pass

    # Add GRPO args to the base parser
    base_parser.add_argument("--grpo_config", type=str, required=True)
    base_parser.add_argument("--prompt_file", type=str, required=True)
    base_parser.add_argument("--grpo_architecture", type=str, default=None)
    base_parser.add_argument("--grpo_steps", type=int, default=None)
    base_parser.add_argument("--grpo_batch_size", type=int, default=1)

    args = base_parser.parse_args()

    # Apply dataset_config placeholder if not provided (GRPO doesn't use cached datasets)
    if not hasattr(args, "dataset_config") or args.dataset_config is None:
        # Create a minimal placeholder to pass the NetworkTrainer's validation
        args.dataset_config = "__grpo_placeholder__"

    # ── 3. Build base trainer and run GRPO loop ──────────────────────────────
    base_trainer = TrainerClass()
    _grpo_loop(base_trainer, args, grpo_config, pre_args)


def _grpo_loop(base_trainer, args, grpo_config, pre_args):
    """The actual GRPO training loop using direct model setup."""
    import importlib
    from accelerate import Accelerator, InitProcessGroupKwargs
    from accelerate.utils import set_seed
    from datetime import timedelta

    from musubi_tuner.grpo.prompt_dataset import PromptDataset
    from musubi_tuner.grpo.trainer import GRPOTrainer

    # ── Accelerator ─────────────────────────────────────────────────────────
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

    # ── Load models using base_trainer helpers ──────────────────────────────
    accelerator.print(f"Loading models for architecture: {grpo_config.architecture}")

    if not hasattr(args, "dit") or args.dit is None:
        raise ValueError("--dit is required")
    if not hasattr(args, "vae") or args.vae is None:
        raise ValueError("--vae is required")

    # Determine attention mode — delegate to base_trainer if it exposes a helper,
    # otherwise scan CLI flags. "sdpa" maps to "torch" for architectures that
    # use the hunyuan attention MEMORY_LAYOUT (qwen_image, hv, wan).
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

    # qwen_image: resolve_model_version_args must run before handle_model_specific_args
    # to populate args.is_edit / args.is_layered from --model_version.
    if hasattr(args, "model_version") and not hasattr(args, "is_edit"):
        try:
            from musubi_tuner.qwen_image import qwen_image_utils
            qwen_image_utils.resolve_model_version_args(args)
        except Exception:
            pass

    # handle_model_specific_args must come first — some architectures set attrs
    # (e.g. qwen_image sets args.is_layered) that load_vae / load_transformer read.
    base_trainer.handle_model_specific_args(args)

    vae_dtype = weight_dtype
    vae = base_trainer.load_vae(args, vae_dtype, args.vae)
    vae.eval()
    vae.requires_grad_(False)

    # Determine DiT weight dtype
    dit_weight_dtype = None
    if getattr(args, "fp8_base", False):
        dit_weight_dtype = torch.float8_e4m3fn
    elif getattr(args, "fp16_base", False):
        dit_weight_dtype = torch.float16

    # fp8_scaled means qwen_image handles fp8 internally and requires dit_weight_dtype=None
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

    # ── LoRA network ─────────────────────────────────────────────────────────
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    network_module = importlib.import_module(args.network_module)

    net_kwargs = {}
    if getattr(args, "network_args", None):
        for net_arg in args.network_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value

    if hasattr(network_module, "create_arch_network"):
        network = network_module.create_arch_network(
            1.0,
            args.network_dim,
            getattr(args, "network_alpha", None) or args.network_dim,
            vae,
            None,
            transformer,
            neuron_dropout=getattr(args, "network_dropout", None),
            **net_kwargs,
        )
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            getattr(args, "network_alpha", None) or args.network_dim,
            vae,
            None,
            transformer,
            **net_kwargs,
        )
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    network.to(device, dtype=network_dtype)

    # ── Optimizer ────────────────────────────────────────────────────────────
    trainable_params = network.get_trainable_params()
    lr = getattr(args, "learning_rate", 1e-4) or 1e-4
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # ── Process sample prompts (text encoding) ────────────────────────────────
    # Check for text encoder argument (different architectures use different names)
    has_te = (
        (hasattr(args, "text_encoder1") and args.text_encoder1 is not None)  # HunyuanVideo, Wan, etc.
        or (hasattr(args, "text_encoder") and args.text_encoder is not None)  # qwen_image
        or (hasattr(args, "text_encoder_path") and args.text_encoder_path is not None)  # some variants
    )
    if not has_te:
        raise ValueError("A text encoder argument is required for GRPO online prompt encoding (--text_encoder / --text_encoder1)")

    sample_parameters = base_trainer.process_sample_prompts(args, accelerator, pre_args.prompt_file)

    # Move all pre-encoded embeddings to CPU to free GPU memory for training.
    # With many prompts (e.g. 500), the VL embeds (which include image tokens) can
    # occupy 30+ GB on GPU. _build_batch_dict moves them back to device per-step.
    for sp in sample_parameters:
        for k, v in list(sp.items()):
            if isinstance(v, torch.Tensor) and v.is_cuda:
                sp[k] = v.cpu()
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor) and v[0].is_cuda:
                sp[k] = [t.cpu() for t in v]

    # ── Prepare with accelerator ──────────────────────────────────────────────
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

    # ── Prompt dataset loader ──────────────────────────────────────────────
    prompt_dataset = PromptDataset(pre_args.prompt_file)
    batch_size = pre_args.grpo_batch_size
    max_steps = pre_args.grpo_steps or getattr(args, "max_train_steps", 100) or 100

    # ── Build GRPOTrainer ──────────────────────────────────────────────────
    grpo_trainer = GRPOTrainer(
        base_trainer=base_trainer,
        config=grpo_config,
        accelerator=accelerator,
        args=args,
        transformer=transformer_prepared,
        vae=vae,
        network=network_prepared,
        dit_dtype=dit_dtype,
        network_dtype=network_dtype,
    )

    # Move ref transformer to same device (only if KL penalty is enabled)
    if grpo_trainer.ref_transformer is not None:
        grpo_trainer.ref_transformer.to(device, dtype=dit_dtype)

    accelerator.print(f"Starting GRPO training: {max_steps} steps, group_size={grpo_config.group_size}")
    accelerator.print(f"Rewards: {[(n, w) for n, _r, w in grpo_trainer._named_rewards]}")

    progress_bar = tqdm(range(max_steps), disable=not accelerator.is_local_main_process, desc="GRPO steps")

    output_dir = getattr(args, "output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    for global_step in range(max_steps):
        # Each rank samples a DIFFERENT prompt so DDP gradient averaging
        # accumulates signal from num_processes distinct prompts per step,
        # giving effective batch_size = batch_size * num_processes.
        start = (global_step * batch_size * num_processes + process_index * batch_size) % len(sample_parameters)
        batch_params = [sample_parameters[(start + i) % len(sample_parameters)] for i in range(batch_size)]

        # Load reference images for delta_e00 reward (None if not specified in prompt file)
        ref_imgs = []
        for i in range(batch_size):
            item = prompt_dataset[(start + i) % len(prompt_dataset)]
            if item.reference_image_path:
                try:
                    from PIL import Image as _PIL_Image
                    ref_imgs.append(_PIL_Image.open(item.reference_image_path).convert("RGB"))
                except Exception:
                    ref_imgs.append(None)
            else:
                ref_imgs.append(None)
        reference_images = ref_imgs if any(r is not None for r in ref_imgs) else None

        with accelerator.accumulate(network_prepared):
            loss, log_dict = grpo_trainer.step(batch_params, reference_images=reference_images)

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
            progress_bar.set_postfix(loss=f"{log_dict.get('loss/total', 0):.4f}")

            if accelerator.is_main_process:
                if len(accelerator.trackers) > 0:
                    accelerator.log(log_dict, step=global_step)

            # Save checkpoint
            save_every = getattr(args, "save_every_n_steps", None)
            if save_every and (global_step + 1) % save_every == 0:
                ckpt_name = f"{getattr(args, 'output_name', 'grpo')}_{global_step + 1:06d}.safetensors"
                ckpt_path = os.path.join(output_dir, ckpt_name)
                accelerator.print(f"\nSaving checkpoint: {ckpt_path}")
                accelerator.unwrap_model(network_prepared).save_weights(ckpt_path, network_dtype, {})

    # ── Final save ──────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        ckpt_name = f"{getattr(args, 'output_name', 'grpo')}_final.safetensors"
        ckpt_path = os.path.join(output_dir, ckpt_name)
        accelerator.print(f"\nSaving final checkpoint: {ckpt_path}")
        accelerator.unwrap_model(network_prepared).save_weights(ckpt_path, network_dtype, {})

    accelerator.end_training()
    accelerator.print("GRPO training complete.")


if __name__ == "__main__":
    main()
