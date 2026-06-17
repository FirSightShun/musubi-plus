#!/usr/bin/env python3
"""
Example: GRPO training launcher for qwen_image architecture.

Demonstrates:
- Multi-GPU training with accelerate
- All 6 reward functions (delta_e00, pickscore, hps_v2, image_reward, clip, vlm)
- Resuming from an existing LoRA checkpoint
- Offline mode for air-gapped environments

Usage:
    # Edit the CONFIG section below, then:
    python examples/run_grpo_qwen_image.py

The script writes a grpo_config.toml, copies the prompt file, and launches
accelerate. Training output (checkpoints + train.log) goes to OUTPUT_DIR.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = "/data/your_name/grpo_run1"   # where checkpoints and logs are saved
GRPO_STEPS  = 500                            # total training steps
NUM_GPUS    = 5
GPU_IDS     = "0,1,2,3,4"                   # comma-separated GPU indices

TRAIN_WIDTH  = 512
TRAIN_HEIGHT = 512

# Paths to model weights
DIT_PATH  = "/path/to/qwen_image_edit_2511_bf16.safetensors"
VAE_PATH  = "/path/to/qwen_image_vae.safetensors"
TE_PATH   = "/path/to/qwen_2.5_vl_7b.safetensors"

# Optional: resume from an existing LoRA checkpoint.
# Set to None to start from scratch (random LoRA init).
LORA_INIT = None
# LORA_INIT = "/data/your_name/grpo_run0/checkpoints/grpo_000500.safetensors"

# Path to prompt file (.json array or .jsonl).
# Each entry: {"prompt": "...", "reference": "/path/to/ref.png"}
# "reference" is optional; required only for the delta_e00 reward.
PROMPT_FILE = "/data/your_name/prompts.json"

# Python / accelerate binaries (defaults to PATH if not set)
PYTHON_BIN      = "python"
ACCELERATE_BIN  = "accelerate"

# Extra environment variables (useful for air-gapped servers)
EXTRA_ENV: dict[str, str] = {
    # "TRANSFORMERS_OFFLINE": "1",
    # "HF_DATASETS_OFFLINE":  "1",
}

# ── REWARD WEIGHTS (must sum to 1.0) ──────────────────────────────────────────
#
# Enabled rewards here: delta_e00, pickscore, hps_v2, image_reward, clip, vlm.
# Remove entries or set weight=0 to disable individual rewards.
#
# delta_e00 requires reference images in the prompt file.
# vlm requires a local copy of Qwen/Qwen2-VL-2B-Instruct (or any Qwen2-VL model).

REWARDS = [
    # Task-specific: background colour preservation (CIEDE2000)
    {"name": "delta_e00", "weight": 0.40, "params": {"clip_max": 15.0}},
    # Human preference / aesthetic
    {"name": "pickscore",    "weight": 0.15},
    {"name": "hps_v2",       "weight": 0.15},
    {"name": "image_reward", "weight": 0.10},
    # Text-image alignment
    {"name": "clip", "weight": 0.10},
    # VLM semantic quality score
    {
        "name":   "vlm",
        "weight": 0.10,
        "params": {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            # Customise the scoring instruction for your task:
            "prompt_template": (
                "Rate this image from 1 to 10 based on how well it matches: "
                '"{prompt}". Reply with a single integer only.'
            ),
            "min_score": 1,
            "max_score": 10,
        },
    },
]

# ── GRPO hyper-parameters ──────────────────────────────────────────────────────
GRPO_CONFIG = {
    "group_size":          16,    # samples per prompt for advantage estimation
    "num_inference_steps": 20,
    "guidance_scale":      1.0,
    "discrete_flow_shift": 2.2,
    "kl_coeff":            0.0,   # KL penalty weight (0 = disabled)
    "clip_eps":            0.0,   # PPO-style clip epsilon (0 = disabled)
    "phase2_chunk_size":   2,     # micro-batch size for Phase 2 (reduce for OOM)
}

# ── LoRA architecture ──────────────────────────────────────────────────────────
NETWORK_MODULE = "networks.lora_qwen_image"
NETWORK_DIM    = 64
LEARNING_RATE  = "3e-5"
SAVE_EVERY_N   = 50     # save checkpoint every N steps
OUTPUT_NAME    = "grpo"


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_toml() -> str:
    lines = ["[grpo]"]
    lines += [
        f'architecture        = "qwen_image"',
        f'group_size          = {GRPO_CONFIG["group_size"]}',
        f'num_inference_steps = {GRPO_CONFIG["num_inference_steps"]}',
        f"width               = {TRAIN_WIDTH}",
        f"height              = {TRAIN_HEIGHT}",
        f"frame_count         = 1",
        f'guidance_scale      = {GRPO_CONFIG["guidance_scale"]}',
        f'discrete_flow_shift = {GRPO_CONFIG["discrete_flow_shift"]}',
        f'kl_coeff            = {GRPO_CONFIG["kl_coeff"]}',
        f'clip_eps            = {GRPO_CONFIG["clip_eps"]}',
        f'phase2_chunk_size   = {GRPO_CONFIG["phase2_chunk_size"]}',
    ]
    for rw in REWARDS:
        lines.append("")
        lines.append("[[grpo.reward]]")
        lines.append(f'name   = "{rw["name"]}"')
        lines.append(f'weight = {rw["weight"]}')
        if "params" in rw:
            lines.append("[grpo.reward.params]")
            for k, v in rw["params"].items():
                lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify LORA_INIT exists when set
    if LORA_INIT and not Path(LORA_INIT).exists():
        sys.exit(f"[ERROR] LORA_INIT not found: {LORA_INIT}")

    # Copy prompt file into output dir so the run is self-contained
    prompt_dst = Path(OUTPUT_DIR) / Path(PROMPT_FILE).name
    if not prompt_dst.exists():
        shutil.copy(PROMPT_FILE, prompt_dst)
        print(f"Copied prompt file → {prompt_dst}")

    # Write TOML config
    config_path = Path(OUTPUT_DIR) / "grpo_config.toml"
    config_path.write_text(_build_toml())
    print(f"Wrote config → {config_path}")

    # Build accelerate launch command
    ckpt_dir = Path(OUTPUT_DIR) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    cmd = [
        ACCELERATE_BIN, "launch",
        "--multi_gpu",
        "--num_processes", str(NUM_GPUS),
        "--gpu_ids", GPU_IDS,
        "--mixed_precision", "bf16",
        "src/musubi_tuner/grpo_train_network.py",
        "--grpo_config",   str(config_path),
        "--prompt_file",   str(prompt_dst),
        "--dit",           DIT_PATH,
        "--vae",           VAE_PATH,
        "--text_encoder",  TE_PATH,
        "--network_module", NETWORK_MODULE,
        "--network_dim",   str(NETWORK_DIM),
        "--learning_rate", LEARNING_RATE,
        "--grpo_steps",    str(GRPO_STEPS),
        "--grpo_batch_size", "1",
        "--model_version", "edit-2511",
        "--save_every_n_steps", str(SAVE_EVERY_N),
        "--output_dir",    str(ckpt_dir),
        "--output_name",   OUTPUT_NAME,
        "--gradient_checkpointing",
    ]
    if LORA_INIT:
        cmd += ["--network_weights", LORA_INIT]

    log_path = Path(OUTPUT_DIR) / "train.log"
    print(f"\nLaunching {NUM_GPUS}-GPU training → log: {log_path}")
    print("CMD:", " ".join(cmd))

    # Resolve musubi-tuner working directory relative to this script
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
