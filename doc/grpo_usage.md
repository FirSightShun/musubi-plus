# GRPO 训练使用指南

本文档是 musubi-plus GRPO 功能的完整使用参考。理论背景见 [grpo_method.md](grpo_method.md)。

---

## 目录

1. [快速开始](#1-快速开始)
2. [依赖安装](#2-依赖安装)
3. [Prompt 文件格式](#3-prompt-文件格式)
4. [TOML 配置参考](#4-toml-配置参考)
5. [Reward 参考](#5-reward-参考)
6. [启动命令参考](#6-启动命令参考)
7. [架构适配说明](#7-架构适配说明)
8. [检查点与日志](#8-检查点与日志)
9. [资源估算](#9-资源估算)
10. [常见问题](#10-常见问题)

---

## 1. 快速开始

以下是用 qwen_image 架构、CLIP 单奖励跑 3 步的最简示例（已验证可运行）：

**grpo_config.toml**
```toml
[grpo]
architecture        = "qwen_image"
group_size          = 2
num_inference_steps = 5
width               = 256
height              = 256
frame_count         = 1
guidance_scale      = 1.0
discrete_flow_shift = 2.2
kl_coeff            = 0.01

[[grpo.reward]]
name   = "clip"
weight = 1.0
```

**prompts.jsonl**
```jsonl
{"prompt": "a red apple on a white table"}
{"prompt": "a blue sky with white clouds"}
```

**启动**
```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config  grpo_config.toml \
    --prompt_file  prompts.jsonl \
    --dit          path/to/dit.safetensors \
    --vae          path/to/vae.safetensors \
    --text_encoder path/to/text_encoder.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim  16 \
    --learning_rate 1e-4 \
    --grpo_steps   100 \
    --model_version edit-2511 \
    --output_dir   output/ \
    --output_name  grpo_run
```

---

## 2. 依赖安装

### 基础环境

```bash
cd musubi-tuner
uv sync --extra cu128   # torch 2.7.1+cu128（推荐）
# 或
uv sync --extra cu130   # torch 2.9.1+cu130
```

### Reward 依赖（按需安装）

| Reward | 安装命令 |
|---|---|
| `clip` | `pip install open_clip_torch` |
| `hps_v2` | `pip install hpsv2` |
| `pickscore` | `pip install transformers` |
| `image_reward` | `pip install image-reward` |
| `ocr` | `pip install paddlepaddle paddleocr` |
| `vlm` | `pip install transformers accelerate qwen-vl-utils` |
| `delta_e00` | `pip install colour-science` |

只需安装实际使用的 reward 对应的依赖，未使用的 reward 不需要安装。

### Python 3.10 兼容

Python 3.10 没有内置 `tomllib`，需要额外安装：

```bash
pip install tomli
```

---

## 3. Prompt 文件格式

### JSONL 格式（推荐）

每行一个 JSON 对象，`prompt` 必填，`reference` 可选（供 `delta_e00` 奖励使用）：

```jsonl
{"prompt": "a red apple on a white table"}
{"prompt": "text says \"Hello World\" on a street sign", "reference": "refs/sign_001.png"}
{"prompt": "futuristic city skyline at night"}
```

### JSON 数组格式

部分架构（如 qwen_image）的 `process_sample_prompts` 也接受 JSON 数组：

```json
[
  {"prompt": "a red apple on a white table"},
  {"prompt": "a blue sky with white clouds"}
]
```

### 纯文本格式

每行一条 prompt，不支持参考图：

```
a red apple on a white table
a blue sky with white clouds
```

---

## 4. TOML 配置参考

### `[grpo]` 表

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `architecture` | string | `"hv"` | 目标架构，见[架构列表](#架构列表) |
| `group_size` | int | `4` | 每条 prompt 采样 G 张图像 |
| `num_inference_steps` | int | `20` | 去噪步数（越少越快，质量越低） |
| `width` | int | `256` | 生成图像宽度（像素） |
| `height` | int | `256` | 生成图像高度（像素） |
| `frame_count` | int | `1` | 视频帧数（图像任务固定为 1） |
| `guidance_scale` | float | `1.0` | CFG 引导强度（大多数在线 RL 设为 1.0 关闭 CFG） |
| `discrete_flow_shift` | float | `14.5` | Flow Matching 时间步偏移，架构相关 |
| `kl_coeff` | float | `0.01` | KL 惩罚系数 β，防策略漂移 |
| `clip_eps` | float | `0.0` | PPO-style 梯度裁剪 ε，0 表示禁用 |

### `[[grpo.reward]]` 块（可重复）

每个 `[[grpo.reward]]` 块定义一个奖励函数：

```toml
[[grpo.reward]]
name   = "clip"      # 奖励名称（见 Reward 参考）
weight = 1.0         # 该奖励的权重

[grpo.reward.params]  # 可选：传给该奖励的额外参数
model = "ViT-L-14"
```

权重不需要归一化（代码内部会按比例处理），但推荐所有权重加和约为 1。

### 架构列表

| `architecture` 值 | 训练脚本 | 文本编码器参数 |
|---|---|---|
| `hv` | `hv_train_network.py` | `--text_encoder1`, `--text_encoder2` |
| `hv_1_5` | `hv_1_5_train_network.py` | `--text_encoder1`, `--text_encoder2` |
| `wan` | `wan_train_network.py` | `--text_encoder1` |
| `fpack` | `fpack_train_network.py` | `--text_encoder1` |
| `flux_2` | `flux_2_train_network.py` | `--text_encoder1`, `--text_encoder2` |
| `flux_kontext` | `flux_kontext_train_network.py` | `--text_encoder1`, `--text_encoder2` |
| `qwen_image` | `qwen_image_train_network.py` | `--text_encoder` |
| `kandinsky5` | `kandinsky5_train_network.py` | `--text_encoder1` |
| `zimage` | `zimage_train_network.py` | `--text_encoder1` |

### `discrete_flow_shift` 推荐值

| 架构 | 推荐值 |
|---|---|
| HunyuanVideo | `14.5` |
| Wan | `3.0` |
| qwen_image | `2.2` |
| Flux | `3.0` |

### 完整配置示例（多奖励）

```toml
[grpo]
architecture        = "qwen_image"
group_size          = 8
num_inference_steps = 20
width               = 512
height              = 512
frame_count         = 1
guidance_scale      = 1.0
discrete_flow_shift = 2.2
kl_coeff            = 0.01
clip_eps            = 0.0

[[grpo.reward]]
name   = "hps_v2"
weight = 0.3

[[grpo.reward]]
name   = "clip"
weight = 0.3

[[grpo.reward]]
name   = "image_reward"
weight = 0.2

[[grpo.reward]]
name   = "delta_e00"
weight = 0.2
[grpo.reward.params]
clip_max = 10.0
```

---

## 5. Reward 参考

### `clip` — CLIP 文图对齐

CLIP ViT-H-14 计算图像与 prompt 的余弦相似度。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | `"ViT-H-14"` | open_clip 模型名 |
| `pretrained` | `"laion2b_s32b_b79k"` | 预训练权重 |

```toml
[[grpo.reward]]
name   = "clip"
weight = 0.5
```

### `hps_v2` — Human Preference Score v2.1

基于人类偏好数据训练的美学打分模型。

无额外参数。

```toml
[[grpo.reward]]
name   = "hps_v2"
weight = 0.25
```

### `pickscore` — PickScore

基于 Pick-a-Pic 数据集训练的图文对齐偏好模型。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | `"yuvalkirstain/PickScore_v1"` | HuggingFace 模型 ID |
| `processor` | `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"` | 图像处理器 |

```toml
[[grpo.reward]]
name   = "pickscore"
weight = 0.25
```

### `image_reward` — ImageReward

专为文生图对齐训练的偏好模型。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | `"ImageReward-v1.0"` | 模型名 |

```toml
[[grpo.reward]]
name   = "image_reward"
weight = 0.2
```

### `ocr` — PaddleOCR 文字准确率

适用于需要在图像中渲染特定文字的任务。从 prompt 中提取引号内的目标文字，用字符级 F1 评分。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lang` | `"en"` | OCR 语言（`"en"`、`"ch"` 等） |

**Prompt 格式**：目标文字用引号括起来，例如：

```
write "Hello World" on a sign board
在招牌上写「你好世界」
```

```toml
[[grpo.reward]]
name   = "ocr"
weight = 0.5
[grpo.reward.params]
lang = "en"
```

### `vlm` — VLM 语义评分

使用 Qwen2-VL 按自定义 prompt 模板打分（1-10 整数）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model` | `"Qwen/Qwen2-VL-2B-Instruct"` | HuggingFace 模型 ID |
| `prompt_template` | 英文默认模板 | 评分指令，`{prompt}` 占位符替换为实际 prompt |
| `min_score` | `1` | 归一化下界 |
| `max_score` | `10` | 归一化上界 |

```toml
[[grpo.reward]]
name   = "vlm"
weight = 0.2
[grpo.reward.params]
model = "Qwen/Qwen2-VL-2B-Instruct"
prompt_template = "请从 1-10 打分评价这张图片与描述「{prompt}」的一致性，只输出数字。"
min_score = 1
max_score = 10
```

### `delta_e00` — CIEDE2000 色彩保真度

计算生成图像与参考图像的感知色差（ΔE00），奖励 = −mean(ΔE00)。ΔE00 < 1.0 表示人眼难以察觉的差异。

**需要** prompt 文件中提供 `reference` 字段；缺少参考图的样本该步奖励自动为 0。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `clip_max` | `20.0` | 截断异常大的色差值（遮挡/背景区域） |

```toml
[[grpo.reward]]
name   = "delta_e00"
weight = 0.3
[grpo.reward.params]
clip_max = 10.0
```

Prompt 文件配套格式：

```jsonl
{"prompt": "sunset over the ocean", "reference": "refs/sunset_001.png"}
```

---

## 6. 启动命令参考

### 核心 GRPO 参数

| 参数 | 说明 |
|---|---|
| `--grpo_config` | TOML 配置文件路径（必填） |
| `--prompt_file` | Prompt 文件路径，支持 `.jsonl`、`.json`、`.txt`（必填） |
| `--grpo_steps` | 训练总步数（覆盖 `--max_train_steps`） |
| `--grpo_batch_size` | 每步处理的 prompt 数量（默认 1） |
| `--grpo_architecture` | 覆盖 TOML 中的 `architecture` 字段 |

### qwen_image 架构

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config  grpo_config.toml \
    --prompt_file  prompts.jsonl \
    --dit          path/to/qwen_image_dit.safetensors \
    --vae          path/to/qwen_image_vae.safetensors \
    --text_encoder path/to/qwen_2.5_vl_7b.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim  32 \
    --network_alpha 16 \
    --learning_rate 1e-4 \
    --model_version edit-2511 \
    --grpo_steps   500 \
    --grpo_batch_size 1 \
    --save_every_n_steps 100 \
    --output_dir   output/ \
    --output_name  grpo_qwen
```

### HunyuanVideo 架构

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config   grpo_config.toml \
    --prompt_file   prompts.jsonl \
    --dit           path/to/hunyuan_dit.safetensors \
    --vae           path/to/hunyuan_vae.safetensors \
    --text_encoder1 path/to/llava_llama3.safetensors \
    --text_encoder2 path/to/clip_l.safetensors \
    --network_module networks.lora \
    --network_dim   32 \
    --learning_rate 1e-4 \
    --grpo_steps    500 \
    --output_dir    output/ \
    --output_name   grpo_hv
```

### Wan 架构

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config   grpo_config.toml \
    --prompt_file   prompts.jsonl \
    --dit           path/to/wan_dit.safetensors \
    --vae           path/to/wan_vae.safetensors \
    --text_encoder1 path/to/umt5_xxl.safetensors \
    --network_module networks.lora \
    --network_dim   32 \
    --learning_rate 1e-4 \
    --grpo_steps    500 \
    --output_dir    output/ \
    --output_name   grpo_wan
```

### 通用可选参数

| 参数 | 说明 |
|---|---|
| `--fp8_base` | DiT 权重以 FP8 加载（显存减半，推荐大模型） |
| `--blocks_to_swap N` | CPU offload N 个 transformer block（显存不足时使用） |
| `--gradient_checkpointing` | 梯度检查点（显存换时间） |
| `--seed N` | 固定随机种子 |
| `--max_grad_norm 1.0` | 梯度裁剪 |
| `--network_dropout 0.1` | LoRA dropout |

---

## 7. 架构适配说明

### qwen_image

- 文本编码器参数为 `--text_encoder`（不是 `--text_encoder1`）
- 必须提供 `--model_version`（如 `edit-2511`），否则模型会以非编辑模式初始化
- `discrete_flow_shift` 推荐 `2.2`（不同于 HunyuanVideo 的 14.5）
- 不支持 `sdpa` 注意力模式，默认使用 `torch`；如需加速可用 `--flash_attn`

### HunyuanVideo

- 需要两个文本编码器：`--text_encoder1`（LLaVA-LLaMA3）+ `--text_encoder2`（CLIP-L）
- `discrete_flow_shift` 推荐 `14.5`
- 支持 `--flash_attn`、`--sageattn`、`--xformers`

### Wan

- 仅需 `--text_encoder1`（UMT5-XXL）
- `discrete_flow_shift` 推荐 `3.0`
- 视频任务设 `frame_count > 1`（如 `frame_count = 17`）

---

## 8. 检查点与日志

### 检查点保存

每隔 `--save_every_n_steps` 步自动保存：

```
output/grpo_run_000100.safetensors
output/grpo_run_000200.safetensors
...
output/grpo_run_final.safetensors
```

检查点为标准 LoRA safetensors 格式，与 musubi-tuner 原生训练产生的格式完全兼容，可直接用于推理。

### 日志指标

每步记录以下指标（通过 `--log_with wandb` 等传给 accelerate）：

| 键 | 说明 |
|---|---|
| `loss/total` | 总损失 = 优势加权项 + KL 项 |
| `loss/advantage_weighted` | 优势加权 MSE 项 |
| `loss/kl` | KL 惩罚项 |
| `reward/<name>` | 各奖励的 batch 均值 |
| `reward/advantage_mean` | 优势均值（应在 0 附近） |
| `reward/advantage_std` | 优势标准差 |

---

## 9. 资源估算

每步显存开销 ≈ 标准 SFT 训练 × (G + 1)，其中 G 为 `group_size`：

- Phase 1（G 次推理）：逐次进行，峰值显存 = 单次推理
- Phase 2（反向传播）：batch size = G，与 `group_size` 线性相关

**典型配置（qwen_image, 256×256）：**

| group_size | num_inference_steps | 每步耗时（A100） | 显存占用 |
|---|---|---|---|
| 2 | 5 | ~4 分钟 | ~24 GB |
| 4 | 10 | ~12 分钟 | ~24 GB |
| 8 | 20 | ~40 分钟 | ~32 GB |

**降低资源消耗的方法：**

- 减小 `group_size`（最小为 2，否则组内归一化无意义）
- 减少 `num_inference_steps`（5~10 步已能产生有意义的奖励信号）
- 减小 `width`/`height`（256×256 已够 reward 判断质量）
- 开启 `--fp8_base`（DiT 权重 FP8，不影响 LoRA 梯度）
- 开启 `--blocks_to_swap`（CPU offload，速度换显存）

---

## 10. 常见问题

### `ModuleNotFoundError: tomllib`

Python 3.10 无内置 `tomllib`：

```bash
pip install tomli
```

### `KeyError: 'sdpa'` 启动报错

`sdpa` 不是 hunyuan 系架构（hv、qwen_image、wan）支持的注意力模式。默认已使用 `torch`，不需要额外设置。若要加速，改用 `--flash_attn`。

### `AttributeError: 'Namespace' has no attribute 'is_edit'`（qwen_image）

启动时缺少 `--model_version` 参数。qwen_image 需要从 `--model_version` 推断编辑/分层模式：

```bash
--model_version edit-2511
```

### `RuntimeError: Sizes of tensors must match` 在 call_dit 内

多见于 qwen_image 的 `vl_embed` 维度问题，通常是升级代码后缓存不一致。清除 `__pycache__` 重启即可。

### 训练收敛极慢

- 检查 `reward/advantage_std` 是否接近 0——若 group_size 内所有图像奖励完全相同（如全黑图像 CLIP 分相同），advantage 退化为 0，梯度为 0
- 提高 `num_inference_steps` 以生成质量差异更大的样本
- 检查 VAE 编解码是否正常（可临时保存推理图像验证）

### 奖励模型加载缓慢

首次使用会从 HuggingFace 下载模型权重，可预先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 显存不足（OOM）

优先级：
1. 减小 `group_size` 至 2
2. 减小 `width`/`height`
3. 开启 `--fp8_base`
4. 开启 `--blocks_to_swap 20`
5. 开启 `--gradient_checkpointing`
