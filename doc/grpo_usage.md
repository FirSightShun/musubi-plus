# GRPO 训练使用指南

本文档是 musubi-plus GRPO 功能的完整参数参考。理论背景见 [GRPO-Method.md](GRPO-Method.md)。

---

## 目录

1. [快速开始](#1-快速开始)
2. [依赖安装](#2-依赖安装)
3. [Prompt 文件格式](#3-prompt-文件格式)
4. [TOML 配置参考](#4-toml-配置参考)
5. [Reward 参考](#5-reward-参考)
6. [命令行参数参考](#6-命令行参数参考)
7. [完整启动示例](#7-完整启动示例)
8. [检查点与日志](#8-检查点与日志)
9. [资源估算](#9-资源估算)
10. [常见问题](#10-常见问题)

---

## 1. 快速开始

以下是 qwen_image 架构、CLIP 单奖励、3 步的最简验证示例（已完整运行通过）：

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
    --grpo_config   grpo_config.toml \
    --prompt_file   prompts.jsonl \
    --dit           /path/to/qwen_image_edit_2511_bf16.safetensors \
    --vae           /path/to/qwen_image_vae.safetensors \
    --text_encoder  /path/to/qwen_2.5_vl_7b.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim   16 \
    --learning_rate 1e-4 \
    --grpo_steps    3 \
    --model_version edit-2511 \
    --output_dir    output/ \
    --output_name   grpo_test
```

预期输出：
```
Starting GRPO training: 3 steps, group_size=2
Rewards: [('clip', 1.0)]
GRPO steps: 100%|██████████| 3/3 [12:40<00:00, loss=0.1904]
Saving final checkpoint: output/grpo_test_final.safetensors
GRPO training complete.
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

### Python 3.10 兼容

Python 3.10 没有内置 `tomllib`，需要额外安装：

```bash
pip install tomli
```

### Reward 依赖（按需安装）

只需安装实际启用的 reward 所对应的包，未用到的 reward 不需要安装。

| Reward 名称 | 安装命令 | 模型来源 |
|---|---|---|
| `clip` | `pip install open_clip_torch` | 首次运行时自动从 HuggingFace 下载 |
| `hps_v2` | `pip install hpsv2` | 首次运行时自动下载 |
| `pickscore` | `pip install transformers` | `yuvalkirstain/PickScore_v1` |
| `image_reward` | `pip install image-reward` | `ImageReward-v1.0` |
| `ocr` | `pip install paddlepaddle paddleocr` | 内置，无需额外下载 |
| `vlm` | `pip install transformers accelerate qwen-vl-utils` | `Qwen/Qwen2-VL-2B-Instruct` |
| `delta_e00` | `pip install colour-science` | 无模型，纯计算 |

如果网络受限，可以提前设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 3. Prompt 文件格式

### JSONL 格式（推荐）

每行一个独立的 JSON 对象。`prompt` 为必填字段，`reference` 为可选字段（仅 `delta_e00` 奖励使用）：

```jsonl
{"prompt": "a red apple on a white table"}
{"prompt": "a blue sky with white clouds"}
{"prompt": "street sign saying \"EXIT\"", "reference": "refs/sign_001.png"}
```

- `reference` 的路径相对于当前工作目录
- 缺少 `reference` 字段的条目在使用 `delta_e00` 时该步奖励自动计为 0，不报错

### JSON 数组格式

兼容格式，适用于 qwen_image 等架构的 `process_sample_prompts` 读取逻辑：

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

TOML 文件通过 `--grpo_config` 传入，控制 GRPO 算法的所有超参数和奖励配置。文件格式：

```toml
[grpo]
# ... 算法参数 ...

[[grpo.reward]]
name   = "clip"
weight = 1.0
```

### 4.1 采样参数

---

#### `architecture`

- **类型**：string
- **默认值**：`"hv"`
- **说明**：目标生成模型的架构名称，决定使用哪个 NetworkTrainer 子类。也可以通过命令行 `--grpo_architecture` 覆盖此字段。

| 值 | 对应模型 | 文本编码器参数 |
|---|---|---|
| `hv` | HunyuanVideo | `--text_encoder1`, `--text_encoder2` |
| `hv_1_5` | HunyuanVideo 1.5 | `--text_encoder1`, `--text_encoder2` |
| `wan` | Wan | `--text_encoder1` |
| `fpack` | FramePack | `--text_encoder1` |
| `flux_2` | FLUX.2 | `--text_encoder1`, `--text_encoder2` |
| `flux_kontext` | FLUX.1-Kontext | `--text_encoder1`, `--text_encoder2` |
| `qwen_image` | Qwen-Image-Edit | `--text_encoder` |
| `kandinsky5` | Kandinsky 5 | `--text_encoder1` |
| `zimage` | ZImage | `--text_encoder1` |

```toml
architecture = "qwen_image"
```

---

#### `group_size`

- **类型**：int
- **默认值**：`4`
- **说明**：每条 prompt 在线采样生成的图像数量 G。GRPO 以组内均值为基准计算相对优势，G 越大组内统计越稳定，但计算开销线性增加。**最小值为 2**，G=1 时组内标准差为 0，优势退化为全零，梯度消失。

```toml
# 快速验证流程
group_size = 2

# 正式训练（统计稳定性更好）
group_size = 8
```

---

#### `num_inference_steps`

- **类型**：int
- **默认值**：`20`
- **说明**：每次生成图像的去噪步数。步数越少越快，但生成质量越低，奖励信号的区分度也可能下降。实验表明 5~10 步已能产生足够区分度的图像用于奖励打分，正式训练推荐 20~50 步。

```toml
# 调试/验证流程用（快速）
num_inference_steps = 5

# 正式训练（质量较好）
num_inference_steps = 20
```

---

#### `width` / `height`

- **类型**：int
- **默认值**：`256`
- **说明**：生成图像的宽高，单位像素。实际使用时会对齐到 8 的倍数。在线 RL 训练时不需要生成高分辨率图像——奖励模型（CLIP、HPSv2 等）内部都会 resize 到固定尺寸，256×256 对于奖励打分已经足够。

```toml
# 验证和大多数奖励打分场景
width  = 256
height = 256

# 视频类任务（宽高比需与模型训练时一致）
width  = 480
height = 832
```

---

#### `frame_count`

- **类型**：int
- **默认值**：`1`
- **说明**：生成的视频帧数。图像任务固定为 `1`。视频任务设为实际帧数，需与所用架构的约束匹配（如 Wan 通常为 17 或 81 帧）。

```toml
frame_count = 1    # 图像任务
frame_count = 17   # 短视频（Wan 架构）
```

---

#### `guidance_scale`

- **类型**：float
- **默认值**：`1.0`
- **说明**：Classifier-Free Guidance（CFG）强度。在线 RL 训练中通常设为 `1.0` 以关闭 CFG——CFG 会使奖励分布偏移（引导后的图像分布与无引导时不同），影响优势信号的一致性；同时关闭 CFG 可减半推理计算量。

```toml
guidance_scale = 1.0   # 推荐：关闭 CFG
guidance_scale = 4.5   # 若生成质量对奖励区分度影响较大，可尝试开启
```

---

#### `discrete_flow_shift`

- **类型**：float
- **默认值**：`14.5`
- **说明**：Flow Matching 的时间步偏移参数，影响去噪过程的时间步分布。**该值与模型架构强绑定**，使用错误的值会导致生成图像质量极差，奖励信号无意义。

| 架构 | 推荐值 |
|---|---|
| HunyuanVideo (`hv`, `hv_1_5`) | `14.5` |
| Wan (`wan`) | `3.0` |
| FramePack (`fpack`) | `10.0` |
| Qwen-Image-Edit (`qwen_image`) | `2.2` |
| FLUX (`flux_2`, `flux_kontext`) | `3.0` |

```toml
discrete_flow_shift = 2.2   # qwen_image 架构
```

---

### 4.2 损失参数

---

#### `kl_coeff`

- **类型**：float
- **默认值**：`0.01`
- **说明**：KL 惩罚系数 β。训练开始时冻结一个参考策略快照，KL 项约束当前策略不偏离参考策略过远，防止策略崩溃或遗忘预训练能力。设为 `0.0` 可完全禁用 KL 惩罚（等同于纯 GRPO 无约束）。

```toml
kl_coeff = 0.0    # 禁用 KL 惩罚（纯 GRPO，策略更自由但可能不稳定）
kl_coeff = 0.001  # 极弱约束（奖励信号可信度高时）
kl_coeff = 0.01   # 推荐默认值
kl_coeff = 0.1    # 强约束（防止策略漂移，适合短期训练）
```

---

#### `clip_eps`

- **类型**：float
- **默认值**：`0.0`
- **说明**：PPO-style 梯度裁剪的 ε 参数。`0.0` 表示禁用。启用后限制单步策略更新幅度，公式为 `clip(ratio, 1-ε, 1+ε) * advantage`。目前在 Flow Matching 上的效果不如 KL 惩罚稳定，建议保持默认 `0.0`。

```toml
clip_eps = 0.0   # 禁用（推荐）
clip_eps = 0.2   # 标准 PPO 值（实验性）
```

---

### 4.3 奖励配置

`[[grpo.reward]]` 是可重复的数组块，每块定义一个奖励函数。

```toml
[[grpo.reward]]
name   = "奖励名称"   # 必填，见第 5 节
weight = 1.0          # 该奖励在优势函数中的权重

[grpo.reward.params]  # 可选，传给该奖励类的额外参数
key = value
```

**权重说明**：各奖励权重不需要归一化（代码中 MO-GRPO 会先将每个奖励在 group 内标准化，再乘以 weight 聚合）。推荐将所有权重加和约为 1.0 以便直观理解各奖励的实际贡献比例。

**多奖励示例**：

```toml
# 美学对齐为主，色彩保真为辅
[[grpo.reward]]
name   = "hps_v2"
weight = 0.4

[[grpo.reward]]
name   = "clip"
weight = 0.3

[[grpo.reward]]
name   = "delta_e00"
weight = 0.3
[grpo.reward.params]
clip_max = 10.0
```

---

## 5. Reward 参考

### `clip` — CLIP 文图对齐

使用 CLIP ViT-H-14 计算生成图像与 prompt 文本的余弦相似度。适合对"图像是否符合描述"有通用要求的场景。

**安装**：`pip install open_clip_torch`

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `"ViT-H-14"` | open_clip 模型标识符。其他可用值：`"ViT-L-14"`（更小更快）、`"ViT-bigG-14"`（最大最准）等 |
| `pretrained` | string | `"laion2b_s32b_b79k"` | 预训练权重名称，需与 `model` 匹配。首次使用会自动下载 |

```toml
# 使用默认 ViT-H-14（均衡选择）
[[grpo.reward]]
name   = "clip"
weight = 0.5

# 使用更小的 ViT-L-14（推理更快）
[[grpo.reward]]
name   = "clip"
weight = 0.5
[grpo.reward.params]
model      = "ViT-L-14"
pretrained = "laion2b_s32b_b32k"
```

---

### `hps_v2` — Human Preference Score v2.1

基于大规模人类偏好比较数据训练的美学评分模型，比 CLIP 更接近人类对图像质量的直觉判断。

**安装**：`pip install hpsv2`

无额外参数。模型权重首次使用时自动下载。

```toml
[[grpo.reward]]
name   = "hps_v2"
weight = 0.3
```

---

### `pickscore` — PickScore

基于 Pick-a-Pic 数据集（真实用户在文生图系统中的选图偏好）训练的奖励模型，对人类喜好的拟合度较好。

**安装**：`pip install transformers`

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `"yuvalkirstain/PickScore_v1"` | HuggingFace 模型 ID |
| `processor` | string | `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"` | 图像特征提取器，通常无需修改 |

```toml
[[grpo.reward]]
name   = "pickscore"
weight = 0.25
```

---

### `image_reward` — ImageReward

专为文生图设计的人类偏好模型，在语义一致性和视觉质量上综合打分。

**安装**：`pip install image-reward`

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `"ImageReward-v1.0"` | 模型版本，目前只有一个版本 |

```toml
[[grpo.reward]]
name   = "image_reward"
weight = 0.2
```

---

### `ocr` — PaddleOCR 文字准确率

使用 OCR 评估图像中的文字渲染准确度。适用于"在图像中生成指定文字"类任务，如商品标签、广告牌、UI 截图等。

**安装**：`pip install paddlepaddle paddleocr`

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `lang` | string | `"en"` | OCR 识别语言。`"en"` 英文，`"ch"` 中文，`"japan"` 日文，`"korean"` 韩文等 |

**目标文字提取规则**：自动从 prompt 中提取引号内的内容作为目标文字。支持英文双引号 `"..."` 、全角引号 `"..."` 和中文书名号 `「...」`。未找到引号时使用完整 prompt 作为目标。

```toml
[[grpo.reward]]
name   = "ocr"
weight = 0.6
[grpo.reward.params]
lang = "en"
```

对应 prompt 格式（目标文字用引号包裹）：
```jsonl
{"prompt": "a road sign saying \"STOP\" on a highway"}
{"prompt": "a shop window with the text \"SALE 50% OFF\""}
{"prompt": "一块牌子上写着「禁止入内」"}
```

评分方式为字符级 F1：`F1 = 2 * precision * recall / (precision + recall)`，完全正确为 1.0，完全不匹配为 0.0。

---

### `vlm` — Qwen2-VL 语义评分

使用视觉语言模型按照自定义评分指令对图像打分（输出 1–10 整数，归一化到 [0, 1]）。适合需要精细语义理解的场景，如"图像是否展示了正确的空间关系"、"食物是否看起来美味"等 CLIP 难以捕捉的细节。

**安装**：`pip install transformers accelerate qwen-vl-utils`

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | `"Qwen/Qwen2-VL-2B-Instruct"` | HuggingFace 模型 ID。2B 版本占用约 6GB 显存；更准确但更重的选项：`"Qwen/Qwen2-VL-7B-Instruct"` |
| `prompt_template` | string | 默认英文指令（见下） | 评分指令模板，`{prompt}` 占位符会被替换为实际 prompt 文本 |
| `min_score` | float | `1` | VLM 输出的分数下界（用于归一化） |
| `max_score` | float | `10` | VLM 输出的分数上界（用于归一化） |

默认 `prompt_template`：
```
Please rate this image on a scale of 1 to 10 based on how well it matches the description: "{prompt}". Reply with a single integer only.
```

```toml
# 使用中文评分指令
[[grpo.reward]]
name   = "vlm"
weight = 0.3
[grpo.reward.params]
model           = "Qwen/Qwen2-VL-2B-Instruct"
prompt_template = "请从 1-10 打分评价这张图片与描述「{prompt}」的匹配程度，只输出一个数字。"
min_score       = 1
max_score       = 10

# 评估食物的视觉吸引力（与 prompt 无关的维度）
[[grpo.reward]]
name   = "vlm"
weight = 0.2
[grpo.reward.params]
prompt_template = "Rate the visual appeal of this food image from 1 to 10. Output only the number."
```

---

### `delta_e00` — CIEDE2000 色彩保真度

计算生成图像与参考图像的感知色差（ΔE00，CIEDE2000 标准），奖励 = −mean(ΔE00)。适合需要精确色彩还原的任务，如产品图像编辑、颜色主题迁移等。

**ΔE00 数值参考**：< 1.0 人眼基本无法区分，1–2 微弱差异，2–10 明显差异，> 10 颜色大幅不同。

**安装**：`pip install colour-science`

**前提**：prompt 文件中必须有 `reference` 字段指向参考图像路径；缺少参考图的条目该奖励自动为 0。

| 参数名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `clip_max` | float | `20.0` | 计算均值前截断的 ΔE00 最大值。目的是消除遮挡区域、边缘等局部极端色差对均值的干扰。较小的值（如 5.0）对色彩更敏感；较大的值（如 20.0）更宽容 |

```toml
[[grpo.reward]]
name   = "delta_e00"
weight = 0.4
[grpo.reward.params]
clip_max = 10.0
```

对应 prompt 文件：
```jsonl
{"prompt": "a red sports car parked on a street", "reference": "data/refs/car_001.png"}
{"prompt": "a living room with blue sofa",        "reference": "data/refs/room_042.png"}
```

---

## 6. 命令行参数参考

所有参数都在 `accelerate launch` 启动的脚本后传入。参数分为两类：GRPO 专属参数（本框架新增），以及继承自基础训练脚本的参数。

### 6.1 GRPO 专属参数

这五个参数是本框架新增的，所有架构共用。

---

#### `--grpo_config`（必填）

- **类型**：string
- **说明**：GRPO TOML 配置文件的路径。包含算法超参数和奖励配置，见[第 4 节](#4-toml-配置参考)。

```bash
--grpo_config /path/to/grpo_config.toml
```

---

#### `--prompt_file`（必填）

- **类型**：string
- **说明**：Prompt 文件路径。支持 `.jsonl`（推荐）、`.json`（数组格式）、`.txt`（纯文本，每行一条）。见[第 3 节](#3-prompt-文件格式)。

```bash
--prompt_file prompts.jsonl
```

---

#### `--grpo_steps`

- **类型**：int
- **默认值**：未指定时从 `--max_train_steps` 读取（后者默认 1600）
- **说明**：GRPO 训练总步数。每步包含一次完整的采样→打分→反向传播流程。

```bash
--grpo_steps 3      # 快速验证
--grpo_steps 500    # 正式训练（轻量配置）
--grpo_steps 2000   # 正式训练（完整配置）
```

---

#### `--grpo_batch_size`

- **类型**：int
- **默认值**：`1`
- **说明**：每个 GRPO step 处理的 prompt 数量。总采样量 = `grpo_batch_size × group_size`。增大 batch_size 可以同时处理多条 prompt，提高显存利用率，但显存开销也线性增加（Phase 2 的 batch 大小为 `grpo_batch_size × group_size`）。

```bash
--grpo_batch_size 1   # 每步 1 条 prompt（显存最省）
--grpo_batch_size 4   # 每步 4 条 prompt（显存充足时）
```

---

#### `--grpo_architecture`

- **类型**：string
- **默认值**：读取 TOML 文件中的 `architecture` 字段
- **说明**：在命令行覆盖 TOML 中的架构设置。适合用同一份 TOML 切换不同架构测试时使用。

```bash
--grpo_architecture wan   # 覆盖 TOML 中的 architecture = "hv"
```

---

### 6.2 模型加载参数

这些参数继承自各架构的基础训练脚本。

---

#### `--dit`（必填）

- **类型**：string
- **说明**：DiT（扩散 Transformer）模型权重文件路径，`.safetensors` 格式。

```bash
--dit /data/models/qwen_image_edit_2511_bf16.safetensors
```

---

#### `--vae`（必填）

- **类型**：string
- **说明**：VAE 模型权重文件路径，`.safetensors` 格式。

```bash
--vae /data/models/qwen_image_vae.safetensors
```

---

#### `--text_encoder` / `--text_encoder1` / `--text_encoder2`（必填）

- **类型**：string
- **说明**：文本编码器权重路径。**参数名因架构不同而不同**，不能混用。

| 架构 | 需要哪些参数 | 说明 |
|---|---|---|
| `qwen_image` | `--text_encoder` | Qwen2.5-VL 多模态模型 |
| `hv`, `hv_1_5` | `--text_encoder1` + `--text_encoder2` | LLaVA-LLaMA3 + CLIP-L |
| `wan` | `--text_encoder1` | UMT5-XXL |
| `fpack` | `--text_encoder1` | LLaVA-LLaMA3 |
| `flux_2`, `flux_kontext` | `--text_encoder1` + `--text_encoder2` | CLIP-L + T5-XXL |

```bash
# qwen_image
--text_encoder /data/models/qwen_2.5_vl_7b.safetensors

# HunyuanVideo
--text_encoder1 /data/models/llava_llama3.safetensors \
--text_encoder2 /data/models/clip_l.safetensors
```

---

#### `--model_version`（qwen_image 必填）

- **类型**：string
- **默认值**：未指定时默认 `"original"`
- **说明**：仅 `qwen_image` 架构使用。指定 Qwen-Image 模型版本，决定是否启用编辑模式（`is_edit`）和分层模式（`is_layered`）。**不指定时默认以非编辑模式初始化，无法用于编辑任务。**

| 值 | 说明 |
|---|---|
| `"original"` | 原始非编辑版本（`is_edit=False`） |
| `"layered"` | 分层模式（`is_edit=False, is_layered=True`） |
| `"edit"` | 最初的编辑版本 |
| `"edit-2509"` | 2509 版本编辑模型（Qwen-Image-Edit-Plus） |
| `"edit-2511"` | 2511 版本编辑模型（当前推荐） |

```bash
--model_version edit-2511
```

---

### 6.3 LoRA 网络参数

---

#### `--network_module`（必填）

- **类型**：string
- **说明**：LoRA 网络模块的 Python 路径。各架构有对应的专属 LoRA 模块：

| 架构 | 推荐模块 |
|---|---|
| `qwen_image` | `networks.lora_qwen_image` |
| `hv`, `hv_1_5` | `networks.lora` |
| `wan` | `networks.lora` |
| `fpack` | `networks.lora` |
| `flux_2`, `flux_kontext` | `networks.lora` |

```bash
--network_module networks.lora_qwen_image
```

---

#### `--network_dim`（建议指定）

- **类型**：int
- **默认值**：`None`（需要显式指定）
- **说明**：LoRA 矩阵的秩（rank）。越大参数量越多表达能力越强，但过拟合风险也越大。GRPO 训练中模型只训练少量步数，dim 不必太大。

```bash
--network_dim 8     # 最轻量，参数量极小
--network_dim 16    # 推荐（快速验证和轻量正式训练）
--network_dim 32    # 正式训练标准配置
--network_dim 64    # 高容量（长期训练或复杂任务）
```

---

#### `--network_alpha`

- **类型**：float
- **默认值**：`1`
- **说明**：LoRA 输出的缩放系数 α。实际缩放比例 = `alpha / dim`。常见做法是设为 `dim / 2`（比例 = 0.5）或与 `dim` 相等（比例 = 1.0）。较小的 alpha 使训练初期的 LoRA 贡献较弱，有助于稳定早期训练。

```bash
--network_alpha 16   # alpha=dim（缩放比例=1.0，较强）
--network_alpha 8    # alpha=dim/2（缩放比例=0.5，推荐）
--network_alpha 1    # alpha=1/dim（极弱，训练初期几乎不改变模型）
```

---

#### `--network_dropout`

- **类型**：float
- **默认值**：`None`（无 dropout）
- **说明**：LoRA 神经元 dropout 概率（0 到 1 之间）。每个训练步随机将该比例的 LoRA 神经元置零。适度 dropout 可以防止 LoRA 过拟合，在 GRPO 短期训练中通常不需要。

```bash
--network_dropout 0.0    # 禁用（默认）
--network_dropout 0.1    # 轻微 dropout
```

---

#### `--network_args`

- **类型**：string（可多次指定）
- **默认值**：`None`
- **说明**：传给 LoRA 网络的额外参数，格式为 `key=value`，空格分隔多个参数。具体支持的参数取决于所用 `--network_module`。

```bash
# 指定 LoHA 的参数
--network_args "conv_dim=4" "conv_alpha=2"
```

---

#### `--network_weights`

- **类型**：string
- **默认值**：`None`
- **说明**：从已有 LoRA 权重文件继续训练（热启动），传入 `.safetensors` 文件路径。不指定则从零初始化。

```bash
--network_weights output/grpo_run_000500.safetensors
```

---

### 6.4 优化器与学习率参数

---

#### `--learning_rate`

- **类型**：float
- **默认值**：`2e-6`
- **说明**：AdamW 优化器学习率。GRPO 训练中 LoRA 参数每步更新幅度较大（因为 advantage 会放大有效 loss），学习率可以比 SFT 训练更小。

```bash
--learning_rate 1e-5   # 较大（早期探索阶段）
--learning_rate 1e-4   # 中等（常用值）
--learning_rate 5e-5   # 保守（稳定但收敛较慢）
```

---

#### `--optimizer_type`

- **类型**：string
- **默认值**：`""` （默认 AdamW）
- **说明**：优化器类型。支持 `AdamW`、`AdamW8bit`（bitsandbytes，8bit 量化，显存更省）、`AdaFactor`，以及任意完整类路径（如 `bitsandbytes.optim.PagedAdEMAMix8bit`）。

```bash
--optimizer_type AdamW           # 默认，全精度
--optimizer_type AdamW8bit       # 8bit 量化，节省约一半优化器显存
```

---

#### `--optimizer_args`

- **类型**：string（可多次指定）
- **默认值**：`None`
- **说明**：传给优化器的额外参数，格式为 `key=value`。

```bash
--optimizer_args "weight_decay=0.01" "betas=0.9,0.95"
```

---

#### `--max_grad_norm`

- **类型**：float
- **默认值**：`1.0`
- **说明**：梯度裁剪的最大 L2 范数。GRPO 优势可能导致偶发大梯度，裁剪有助于稳定训练。设为 `0` 禁用裁剪。

```bash
--max_grad_norm 1.0    # 推荐（标准裁剪）
--max_grad_norm 0.5    # 更强的裁剪（训练不稳定时）
--max_grad_norm 0      # 禁用
```

---

#### `--lr_scheduler`

- **类型**：string
- **默认值**：`"constant"`
- **说明**：学习率调度策略。

| 值 | 说明 |
|---|---|
| `constant` | 固定学习率（默认） |
| `linear` | 从初始值线性衰减到 0 |
| `cosine` | 余弦衰减 |
| `cosine_with_restarts` | 带重启的余弦衰减，需配合 `--lr_scheduler_num_cycles` |
| `constant_with_warmup` | 先 warmup 再保持恒定 |
| `polynomial` | 多项式衰减，需配合 `--lr_scheduler_power` |

```bash
--lr_scheduler constant                  # 推荐（GRPO 步数通常不多）
--lr_scheduler cosine                    # 长期训练可用
--lr_scheduler constant_with_warmup \
    --lr_warmup_steps 50                 # 前 50 步 warmup
```

---

#### `--lr_warmup_steps`

- **类型**：int 或 float（小于 1 时作为总步数的比例）
- **默认值**：`0`
- **说明**：学习率预热步数。前 N 步学习率从 0 线性增加到 `--learning_rate`，有助于训练初期的稳定性。

```bash
--lr_warmup_steps 0     # 不预热（默认）
--lr_warmup_steps 50    # 固定 50 步预热
--lr_warmup_steps 0.05  # 总步数的 5% 预热
```

---

### 6.5 训练精度与显存参数

---

#### `--mixed_precision`

- **类型**：`"no"` / `"fp16"` / `"bf16"`
- **默认值**：`"bf16"`（在命令行 `--mixed_precision bf16` 指定给 accelerate）
- **说明**：混合精度训练模式。`bf16` 兼顾精度和速度，是大多数现代 GPU（A100、H100、RTX 4090 等）的推荐选项。`fp16` 数值范围更小，可能出现溢出，不推荐。

```bash
# 传给 accelerate launch，而非训练脚本
accelerate launch --mixed_precision bf16 ...
```

---

#### `--fp8_base`

- **类型**：flag（无值）
- **默认值**：关闭
- **说明**：以 FP8 精度加载 DiT 基础模型权重，可将 DiT 显存占用减半（如从 40GB 降至 20GB）。LoRA 权重本身仍以 `bf16` 精度训练，不影响梯度质量。

```bash
--fp8_base   # 对大模型（qwen_image 39GB DiT）强烈推荐
```

---

#### `--blocks_to_swap`

- **类型**：int
- **默认值**：`0`（不启用）
- **说明**：将 DiT 的前 N 个 Transformer block 的权重保存在 CPU 内存，计算时按需换入 GPU，计算完毕后换回 CPU。显存不足时的救命参数，代价是每步推理额外增加 CPU-GPU 数据传输时间（大约每 10 个 block 增加 10~20% 耗时）。

```bash
--blocks_to_swap 0    # 不启用（默认）
--blocks_to_swap 20   # 换出 20 个 block（中等节省）
```

---

#### `--use_pinned_memory_for_block_swap`

- **类型**：flag（无值）
- **默认值**：关闭
- **说明**：配合 `--blocks_to_swap` 使用。使用 pin memory（锁页内存）存放需要换入换出的 block 权重，可以加速 CPU-GPU 数据传输。在 Linux 上推荐开启，在 Windows 上可能增加共享显存占用。

```bash
--blocks_to_swap 20 --use_pinned_memory_for_block_swap
```

---

#### `--gradient_checkpointing`

- **类型**：flag（无值）
- **默认值**：关闭
- **说明**：梯度检查点（Activation Checkpointing）。反向传播时不保留所有中间激活值，需要时重新计算，以额外计算量换取显存节省（约减少 30~50% 激活显存，但增加约 30% 训练耗时）。

```bash
--gradient_checkpointing
```

---

#### `--gradient_accumulation_steps`

- **类型**：int
- **默认值**：`1`
- **说明**：梯度累积步数。每 N 步才执行一次参数更新（相当于有效 batch size × N）。在 GRPO 中实际意义有限（每步已经处理 `group_size` 张图像），通常保持默认值。

```bash
--gradient_accumulation_steps 1   # 推荐默认
```

---

### 6.6 注意力加速参数

以下参数控制 attention 的计算实现，影响速度和显存。**qwen_image 等 hunyuan 系架构不支持 `--sdpa`**，默认使用标准 PyTorch 实现（torch）。

---

#### `--flash_attn`

- **类型**：flag
- **说明**：使用 FlashAttention 2，需要安装 `flash-attn` 包。速度提升约 1.5~2×，显存大幅减少（与序列长度成正比）。需要 NVIDIA A 系列或 H 系列 GPU（Ampere+）。

```bash
--flash_attn
```

---

#### `--sage_attn`

- **类型**：flag
- **说明**：使用 SageAttention，需要安装 `sageattention` 包。在部分架构上比 FlashAttention 更快，对 GH100 等新架构支持更好。

```bash
--sage_attn
```

---

#### `--xformers`

- **类型**：flag
- **说明**：使用 xformers 高效注意力实现，需要安装 `xformers` 包。与 FlashAttention 相比更易安装，但速度提升幅度略小。

```bash
--xformers
```

---

#### `--split_attn`

- **类型**：flag
- **说明**：将注意力计算切分为多个子批次（每次 batch=1）执行，减少峰值显存，但无加速效果。适合 GPU 显存非常有限的场景。

```bash
--split_attn
```

---

### 6.7 输出与日志参数

---

#### `--output_dir`

- **类型**：string
- **默认值**：`None`
- **说明**：检查点输出目录，目录不存在时自动创建。

```bash
--output_dir output/grpo_qwen
```

---

#### `--output_name`

- **类型**：string
- **默认值**：`None`
- **说明**：输出文件的基础名称（不含扩展名）。检查点文件名为 `{output_name}_{step:06d}.safetensors`，最终文件为 `{output_name}_final.safetensors`。

```bash
--output_name grpo_qwen_clip    # 输出 grpo_qwen_clip_final.safetensors
```

---

#### `--save_every_n_steps`

- **类型**：int
- **默认值**：`None`（不中途保存）
- **说明**：每隔 N 步保存一个中间检查点。GRPO 训练耗时较长，建议启用，避免中断后从头开始。

```bash
--save_every_n_steps 100   # 每 100 步保存一次
```

---

#### `--seed`

- **类型**：int
- **默认值**：`None`（随机）
- **说明**：固定随机种子，用于可复现的对比实验。

```bash
--seed 42
```

---

#### `--log_with`

- **类型**：`"tensorboard"` / `"wandb"` / `"all"`
- **默认值**：`None`（不记录）
- **说明**：日志记录工具。需配合 `--logging_dir` 指定日志目录（TensorBoard）或提前登录 wandb（`wandb login`）。

```bash
# TensorBoard
--log_with tensorboard --logging_dir logs/grpo_run

# Weights & Biases
--log_with wandb
```

---

#### `--logging_dir`

- **类型**：string
- **默认值**：`None`
- **说明**：TensorBoard 日志输出目录。

```bash
--logging_dir logs/grpo_run
```

---

## 7. 完整启动示例

### qwen_image — 轻量验证配置

用于验证流程是否跑通，最小资源消耗。

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config   grpo_config.toml \
    --prompt_file   prompts.jsonl \
    --dit           /data/models/qwen_image_edit_2511_bf16.safetensors \
    --vae           /data/models/qwen_image_vae.safetensors \
    --text_encoder  /data/models/qwen_2.5_vl_7b.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim   16 \
    --network_alpha 8 \
    --learning_rate 1e-4 \
    --grpo_steps    3 \
    --grpo_batch_size 1 \
    --model_version edit-2511 \
    --fp8_base \
    --output_dir    output/ \
    --output_name   grpo_test
```

对应 grpo_config.toml：
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

---

### qwen_image — 正式训练配置（多奖励）

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config   grpo_config_full.toml \
    --prompt_file   prompts_full.jsonl \
    --dit           /data/models/qwen_image_edit_2511_bf16.safetensors \
    --vae           /data/models/qwen_image_vae.safetensors \
    --text_encoder  /data/models/qwen_2.5_vl_7b.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim   32 \
    --network_alpha 16 \
    --learning_rate 5e-5 \
    --lr_scheduler  cosine \
    --lr_warmup_steps 50 \
    --grpo_steps    1000 \
    --grpo_batch_size 1 \
    --model_version edit-2511 \
    --fp8_base \
    --gradient_checkpointing \
    --max_grad_norm 1.0 \
    --seed          42 \
    --save_every_n_steps 200 \
    --output_dir    output/grpo_qwen_full \
    --output_name   grpo_qwen \
    --log_with      tensorboard \
    --logging_dir   logs/grpo_qwen
```

对应 grpo_config_full.toml：
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
weight = 0.35

[[grpo.reward]]
name   = "clip"
weight = 0.35

[[grpo.reward]]
name   = "image_reward"
weight = 0.3
```

---

### HunyuanVideo — 正式训练配置

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config    grpo_hv.toml \
    --prompt_file    prompts.jsonl \
    --dit            /data/models/hunyuan_dit.safetensors \
    --vae            /data/models/hunyuan_vae.safetensors \
    --text_encoder1  /data/models/llava_llama3.safetensors \
    --text_encoder2  /data/models/clip_l.safetensors \
    --network_module networks.lora \
    --network_dim    32 \
    --network_alpha  16 \
    --learning_rate  5e-5 \
    --grpo_steps     1000 \
    --fp8_base \
    --flash_attn \
    --seed           42 \
    --save_every_n_steps 200 \
    --output_dir     output/grpo_hv \
    --output_name    grpo_hv
```

对应 grpo_hv.toml：
```toml
[grpo]
architecture        = "hv"
group_size          = 4
num_inference_steps = 20
width               = 512
height              = 512
frame_count         = 1
guidance_scale      = 1.0
discrete_flow_shift = 14.5
kl_coeff            = 0.01

[[grpo.reward]]
name   = "hps_v2"
weight = 0.5

[[grpo.reward]]
name   = "clip"
weight = 0.5
```

---

### Wan — 视频训练配置

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config    grpo_wan.toml \
    --prompt_file    prompts.jsonl \
    --dit            /data/models/wan_dit.safetensors \
    --vae            /data/models/wan_vae.safetensors \
    --text_encoder1  /data/models/umt5_xxl.safetensors \
    --network_module networks.lora \
    --network_dim    32 \
    --network_alpha  16 \
    --learning_rate  5e-5 \
    --grpo_steps     500 \
    --fp8_base \
    --seed           42 \
    --save_every_n_steps 100 \
    --output_dir     output/grpo_wan \
    --output_name    grpo_wan
```

对应 grpo_wan.toml：
```toml
[grpo]
architecture        = "wan"
group_size          = 4
num_inference_steps = 20
width               = 480
height              = 832
frame_count         = 17
guidance_scale      = 1.0
discrete_flow_shift = 3.0
kl_coeff            = 0.01

[[grpo.reward]]
name   = "clip"
weight = 1.0
```

---

## 8. 检查点与日志

### 检查点格式

输出为标准 LoRA safetensors 文件，与 musubi-tuner 原生 SFT 训练产生的格式完全兼容，可直接加载用于推理：

```
output/
├── grpo_run_000100.safetensors   # save_every_n_steps 中间检查点
├── grpo_run_000200.safetensors
├── ...
└── grpo_run_final.safetensors    # 训练结束后自动保存
```

### TensorBoard 日志指标

通过 `--log_with tensorboard --logging_dir logs/` 启动后，可用 `tensorboard --logdir logs/` 查看：

| 指标键 | 含义 | 期望趋势 |
|---|---|---|
| `loss/total` | 总损失 = 优势加权项 + KL 项 | 无固定趋势（优势有正有负） |
| `loss/advantage_weighted` | 优势加权 MSE 项 | 应逐渐减小 |
| `loss/kl` | KL 惩罚项 | 应保持较小（< 0.1），否则 kl_coeff 过大 |
| `reward/clip` | 本步 CLIP 奖励的 batch 均值 | 期望上升 |
| `reward/hps_v2` | 本步 HPSv2 奖励的 batch 均值 | 期望上升 |
| `reward/advantage_mean` | 优势均值 | 应在 0 附近波动 |
| `reward/advantage_std` | 优势标准差 | 应 > 0；接近 0 说明奖励无区分度 |

---

## 9. 资源估算

每步资源开销主要来自两部分：Phase 1（G 次去噪推理）和 Phase 2（1 次带梯度前向+反向）。

**显存占用估算**（以 qwen_image 39GB DiT 为基准）：

| 配置 | DiT 精度 | group_size | 分辨率 | 估算显存 |
|---|---|---|---|---|
| 最小验证 | FP8 | 2 | 256×256 | ~20 GB |
| 轻量训练 | FP8 | 4 | 256×256 | ~24 GB |
| 标准训练 | FP8 | 8 | 512×512 | ~32 GB |
| 大批量 | bf16 | 8 | 512×512 | ~60 GB |

**每步耗时估算**（A100 80GB，qwen_image，256×256）：

| group_size | num_inference_steps | 每步耗时 |
|---|---|---|
| 2 | 5 | ~4 分钟 |
| 2 | 20 | ~12 分钟 |
| 4 | 10 | ~12 分钟 |
| 8 | 20 | ~40 分钟 |

**降低资源消耗的优先级**：

1. `--fp8_base` — DiT 显存减半，对质量影响极小（**最优先**）
2. 减小 `group_size` 至 2 — 采样量减半，但优势估计方差增大
3. 减小 `width`/`height` 至 256×256 — 推理和 Phase 2 显存均减小
4. 减少 `num_inference_steps` 至 5~10 — 推理时间线性减少
5. `--gradient_checkpointing` — Phase 2 显存节省 30~50%，但增加约 30% 耗时
6. `--blocks_to_swap 20` — DiT 部分 block 换出 CPU，显存进一步减少

---

## 10. 常见问题

### `ModuleNotFoundError: No module named 'tomllib'`

Python 3.10 无内置 `tomllib`（3.11 才引入），需要安装兼容包：

```bash
pip install tomli
```

---

### `KeyError: 'sdpa'` 或 `ValueError: invalid attn_mode`

`sdpa` 不是 hunyuan 系架构（`hv`、`qwen_image`、`wan`）支持的注意力模式。代码默认已使用 `torch`，不需要额外设置。若想加速，使用 `--flash_attn` 而非任何 `--sdpa` 相关选项。

---

### `AttributeError: 'Namespace' object has no attribute 'is_edit'`（qwen_image）

qwen_image 架构需要从 `--model_version` 推断 `is_edit`/`is_layered` 属性，未指定时默认 `"original"`（非编辑模式）。使用编辑类模型必须指定：

```bash
--model_version edit-2511
```

有效值：`original`、`layered`、`edit`、`edit-2509`、`edit-2511`。

---

### `reward/advantage_std ≈ 0`，loss 几乎不变

group 内所有图像的奖励分数几乎相同，优势归一化后接近 0，梯度消失。常见原因：

- `num_inference_steps` 过少（< 5），生成图像都是噪声，奖励无差异 → 增加到 10 以上
- 奖励模型对当前生成阶段不敏感 → 换用更有区分度的奖励（如 `hps_v2`）
- `group_size = 1` → 改为至少 2

---

### 训练中 loss 出现 `nan` 或 `inf`

可能原因及处理：

1. 学习率过大 → 降低 `--learning_rate`（尝试减小 10 倍）
2. 梯度爆炸 → 减小 `--max_grad_norm`（从 1.0 降到 0.3）
3. FP16 溢出 → 改用 bf16（`--mixed_precision bf16`）
4. 奖励值异常（如 delta_e00 参考图尺寸不匹配）→ 检查奖励输出

---

### `RuntimeError: Sizes of tensors must match` 在 `call_dit` 内

通常是 `vl_embed` 维度问题（qwen_image 特有）。清除 Python 缓存后重启：

```bash
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

---

### 奖励模型每步都重新加载，速度很慢

当前实现中奖励模型在 Phase 1 结束后会卸回 CPU（`rw.load(device)` 是懒加载，但 CLIP 等重模型每步重新移到 GPU 有开销）。如果只使用一种奖励且显存充裕，可以预先将奖励模型保留在 GPU：在奖励模块的 `load()` 方法中去掉 `self.vae.to("cpu")` 风格的卸载逻辑（需要修改对应奖励文件）。

---

### OOM（显存不足）

按优先级依次尝试：

1. `--fp8_base`（DiT 显存减半）
2. 减小 `group_size` 为 2
3. 减小 `width = 256, height = 256`
4. `--gradient_checkpointing`
5. `--blocks_to_swap 20 --use_pinned_memory_for_block_swap`
6. `--optimizer_type AdamW8bit`（优化器状态显存减半）
