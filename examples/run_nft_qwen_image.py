#!/usr/bin/env python3
"""
Example: DiffusionNFT training launcher for qwen_image architecture.

Demonstrates:
- Multi-GPU training with accelerate
- All 6 reward functions (delta_e00, pickscore, hps_v2, image_reward, clip, vlm)
- Old policy EMA update schedule
- Resuming from an existing LoRA checkpoint
- Offline mode for air-gapped environments

Reference: "DiffusionNFT: Negative-aware FineTuning for Diffusion Models"
           NVIDIA Research, ICLR 2026 — https://arxiv.org/abs/2509.16117

Usage:
    # Edit the CONFIG section below, then:
    python examples/run_nft_qwen_image.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "/data/your_name/nft_run1"
NFT_STEPS   = 500
NUM_GPUS    = 5
GPU_IDS     = "0,1,2,3,4"

TRAIN_WIDTH  = 512
TRAIN_HEIGHT = 512

# Paths to model weights
DIT_PATH = "/path/to/qwen_image_edit_2511_bf16.safetensors"
VAE_PATH = "/path/to/qwen_image_vae.safetensors"
TE_PATH  = "/path/to/qwen_2.5_vl_7b.safetensors"

# Optional: resume from existing LoRA checkpoint. None = random init.
LORA_INIT = None
# LORA_INIT = "/data/your_name/grpo_run1/checkpoints/grpo_final.safetensors"

PROMPT_FILE = "/data/your_name/prompts.json"

PYTHON_BIN     = "python"
ACCELERATE_BIN = "accelerate"

EXTRA_ENV: dict[str, str] = {
    # "TRANSFORMERS_OFFLINE": "1",
    # "HF_DATASETS_OFFLINE":  "1",
}

# ── NFT HYPER-PARAMETERS ───────────────────────────────────────────────────────

NFT_CONFIG = {
    "group_size":              16,      # samples per prompt per step
    "num_inference_steps":     10,
    "guidance_scale":          1.0,
    "discrete_flow_shift":     2.2,
    "beta":                    1.0,     # interpolation strength (1.0 for aesthetics, 0.1 for OCR)
    "kl_coeff":                0.0001,  # KL regularization weight
    "adv_clip_max":            5.0,
    "old_policy_update_every": 1,       # update old policy every N steps
    "old_policy_decay":        0.0,     # EMA decay: 0 = full copy each update
}

# ── REWARD WEIGHTS (must sum to 1.0) ──────────────────────────────────────────

REWARDS = [
    {"name": "delta_e00",    "weight": 0.40, "params": {"clip_max": 15.0}},
    {"name": "pickscore",    "weight": 0.15},
    {"name": "hps_v2",       "weight": 0.15},
    {"name": "image_reward", "weight": 0.10},
    {"name": "clip",         "weight": 0.10},
    {
        "name":   "vlm",
        "weight": 0.10,
        "params": {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "prompt_template": (
                'Rate this image from 1 to 10 based on how well it matches: "{prompt}". '
                "Reply with a single integer only."
            ),
            "min_score": 1,
            "max_score": 10,
        },
    },
]

# ── LoRA ───────────────────────────────────────────────────────────────────────

NETWORK_MODULE = "networks.lora_qwen_image"
NETWORK_DIM    = 64
LEARNING_RATE  = "3e-5"
SAVE_EVERY_N   = 50
OUTPUT_NAME    = "nft"


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_toml() -> str:
    lines = ["[nft]"]
    lines += [
        f'architecture        = "qwen_image"',
        f'group_size          = {NFT_CONFIG["group_size"]}',
        f'num_inference_steps = {NFT_CONFIG["num_inference_steps"]}',
        f"width               = {TRAIN_WIDTH}",
        f"height              = {TRAIN_HEIGHT}",
        f"frame_count         = 1",
        f'guidance_scale      = {NFT_CONFIG["guidance_scale"]}',
        f'discrete_flow_shift = {NFT_CONFIG["discrete_flow_shift"]}',
        f'beta                = {NFT_CONFIG["beta"]}',
        f'kl_coeff            = {NFT_CONFIG["kl_coeff"]}',
        f'adv_clip_max        = {NFT_CONFIG["adv_clip_max"]}',
        f'old_policy_update_every = {NFT_CONFIG["old_policy_update_every"]}',
        f'old_policy_decay        = {NFT_CONFIG["old_policy_decay"]}',
    ]
    for rw in REWARDS:
        lines.append("")
        lines.append("[[nft.reward]]")
        lines.append(f'name   = "{rw["name"]}"')
        lines.append(f'weight = {rw["weight"]}')
        if "params" in rw:
            lines.append("[nft.reward.params]")
            for k, v in rw["params"].items():
                lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if LORA_INIT and not Path(LORA_INIT).exists():
        sys.exit(f"[ERROR] LORA_INIT not found: {LORA_INIT}")

    prompt_dst = Path(OUTPUT_DIR) / Path(PROMPT_FILE).name
    if not prompt_dst.exists():
        shutil.copy(PROMPT_FILE, prompt_dst)
        print(f"Copied prompt file → {prompt_dst}")

    config_path = Path(OUTPUT_DIR) / "nft_config.toml"
    config_path.write_text(_build_toml())
    print(f"Wrote config → {config_path}")

    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    cmd = [
        ACCELERATE_BIN, "launch",
        "--multi_gpu",
        "--num_processes", str(NUM_GPUS),
        "--gpu_ids", GPU_IDS,
        "--mixed_precision", "bf16",
        "src/musubi_tuner/nft_train_network.py",
        "--nft_config",    str(config_path),
        "--prompt_file",   str(prompt_dst),
        "--dit",           DIT_PATH,
        "--vae",           VAE_PATH,
        "--text_encoder",  TE_PATH,
        "--network_module", NETWORK_MODULE,
        "--network_dim",   str(NETWORK_DIM),
        "--learning_rate", LEARNING_RATE,
        "--nft_steps",     str(NFT_STEPS),
        "--nft_batch_size", "1",
        "--model_version", "edit-2511",
        "--save_every_n_steps", str(SAVE_EVERY_N),
        "--output_dir",    str(ckpt_dir),
        "--output_name",   OUTPUT_NAME,
        "--gradient_checkpointing",
    ]
    if LORA_INIT:
        cmd += ["--network_weights", LORA_INIT]

    log_path = Path(OUTPUT_DIR) / "train.log"
    print(f"\nLaunching {NUM_GPUS}-GPU NFT training → log: {log_path}")
    print("CMD:", " ".join(cmd))

    musubi_dir = str(Path(__file__).resolve().parent.parent / "musubi-tuner")
    env = {**os.environ, **EXTRA_ENV}

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            [PYTHON_BIN, *cmd] if PYTHON_BIN != "python" else cmd,
            cwd=musubi_dir,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    print(f"Training PID: {proc.pid}  (tail -f {log_path})")
    proc.wait()
    print(f"Training finished with exit code {proc.returncode}")


if __name__ == "__main__":
    main()
