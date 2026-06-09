# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

musubi-plus extends [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) with RL-based training improvements for image/video generation models. Two features are being built:

1. **Off-Policy Sample-Weight** (complete) — offline per-sample weighted loss via `sample_weights.json`
2. **GRPO** (in progress) — online policy gradient RL training loop

All actual training code lives in `musubi-tuner/`. The repo root holds docs and this framework layer.

## Environment Setup

```bash
cd musubi-tuner
uv sync --extra cu128   # torch must be pinned to 2.7.1+cu128
```

Other CUDA options: `cu124` (torch 2.5.1+), `cu130` (torch 2.9.1+). Do not mix extras.

## Linting

```bash
cd musubi-tuner
uv run ruff check .          # lint
uv run ruff format .         # format
```

Line length is 132. Several upstream vendor files in `src/musubi_tuner/wan/`, `flux/`, `frame_pack/` are excluded from ruff entirely — do not add ruff rules to those files.

## Training Workflow

Training requires two pre-processing steps before launching the trainer:

```bash
# 1. Cache VAE latents
python src/musubi_tuner/cache_latents.py --dataset_config path/to/config.toml ...

# 2. Cache text encoder outputs
python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/config.toml ...

# 3. Train (HunyuanVideo example)
accelerate launch --mixed_precision bf16 src/musubi_tuner/hv_train_network.py \
    --dit path/to/dit --dataset_config path/to/config.toml \
    --network_module networks.lora --network_dim 32 \
    --sample_weight_file sample_weights.json   # musubi-plus extension
```

Each architecture has its own `{arch}_cache_latents.py`, `{arch}_cache_text_encoder_outputs.py`, and `{arch}_train_network.py`. Current architectures: `hv` (HunyuanVideo), `hv_1_5`, `wan`, `fpack` (FramePack), `flux_2`, `flux_kontext`, `kandinsky5`, `zimage`, `qwen_image`.

Dataset config uses TOML format. See `musubi-tuner/docs/dataset_config.md` for schema.

## Code Architecture

### musubi-plus Modifications (3 files, 8 locations)

All musubi-plus changes are confined to `musubi-tuner/src/musubi_tuner/`:

| File | What changed |
|---|---|
| `dataset/image_video_dataset.py` | `ItemInfo.sample_weight: float = 1.0` field; weight loading in `ImageDataset.prepare_for_training()`; weight injection into batch tensor in `BucketBatchManager.get_batch()` |
| `dataset/config_utils.py` | `ImageDatasetParams.sample_weight_file` field + schema entry |
| `hv_train_network.py` | Weighted loss branch (`if "sample_weight" in batch`); `--sample_weight_file` and `--sample_weight_multiplier` argparse args |

All changes are **backward-compatible**: omitting `--sample_weight_file` restores original behaviour.

### musubi-tuner Core Flow

```
TOML config → BlueprintGenerator/ConfigSanitizer → ImageDataset/VideoDataset
                                                         ↓
                                              BucketBatchManager (bucketed by resolution)
                                                         ↓
                                              train loop in {arch}_train_network.py
                                                         ↓
                                              LoRA network (networks/lora_{arch}.py)
```

- **Dataset layer** (`dataset/`): Reads pre-cached latent `.safetensors` + text encoder `.safetensors` files. Raw pixels are never loaded during training.
- **Network layer** (`networks/`): LoRA adapters are architecture-specific. `network_arch.py` defines the shared interface.
- **Train scripts** (`hv_train_network.py`, `wan_train_network.py`, etc.): Each wraps the shared train loop with architecture-specific model loading. The shared logic in `hv_train_network.py` is the most complete reference implementation.
- **Memory management**: Key flags are `--blocks_to_swap` (CPU offload), `--fp8_base`, `--fp8_llm`, `--gradient_checkpointing`.

### GRPO Work (next feature)

GRPO requires adding an online RL loop on top of the existing train scripts. Reference research is in `doc/GRPO_MultiReward_AntiHacking_Report.html` and `doc/grpo_reward_model_report.html`. Key design decisions still open: which GRPO variant (DanceGRPO / Flow-GRPO / Adv-GRPO), reward model selection, and where in the training loop to inject the policy gradient update.

## Documentation

- `doc/off_policy_sample_weight_method.md` — full design + code walkthrough for the sample-weight feature
- `musubi-tuner/docs/` — upstream docs for dataset config, architecture-specific training guides, advanced options
- `musubi-tuner/.ai/context/overview.md` — upstream developer context (installation, commands, architecture summary)
