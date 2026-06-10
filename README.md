# musubi-plus

基于 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 的强化学习增强训练框架，将两种互补的 RL 思路引入图像/视频生成模型微调：

| 方法 | 核心思路 | 状态 |
|---|---|---|
| **Off-Policy Sample-Weight** | 离线评估样本难度，训练时直接加权 loss | ✅ 已完成 |
| **GRPO** | 在线策略梯度，用多维 Reward 模型引导生成质量 | ✅ 已完成 |

---

## Feature 1：Off-Policy Sample-Weight Method

权重在训练前由外部策略离线计算（任意难度指标），训练时固定注入，**评估策略与训练策略完全解耦**。

```
原版 musubi-tuner:  loss = loss.mean()
musubi-plus:        loss = (loss * sample_weight).mean()
```

**优势：** 零磁盘膨胀、浮点精度权重、训练速度不变、改 JSON 即可切换策略。

> 注意：当前仅支持 `ImageDataset`（图像任务），VideoDataset 如需支持需自行扩展 `dataset/image_video_dataset.py`。

使用方法：

```bash
cd musubi-tuner
accelerate launch src/musubi_tuner/qwen_image_train_network.py \
    --sample_weight_file sample_weights.json \
    --sample_weight_multiplier 1.0 \
    ...
```

详细设计与框架改动见 [doc/off_policy_sample_weight_method.md](doc/off_policy_sample_weight_method.md)。

---

## Feature 2：GRPO ✅

将 MO-GRPO（Multi-Objective GRPO + Flow Matching）引入 musubi-tuner，实现在线 RL 训练循环。

**核心特性：**
- **多 Reward 并联 + 防 Hacking**：每个 Reward 在 group 内独立归一化后再加权聚合，高方差 Reward 不能主导优势函数
- **7 种内置 Reward**：HPSv2.1 / PickScore / ImageReward / CLIP 对齐 / OCR 文字准确率 / Qwen2-VL 语义评分 / CIEDE2000 色彩保真度（ΔE00）
- **KL 惩罚**：训练开始时冻结参考策略，防止策略漂移
- **架构无关**：通过 `--grpo_architecture` 支持所有 musubi-tuner 架构（hv / wan / fpack / flux / qwen_image 等）

```toml
# grpo_config.toml
[grpo]
architecture        = "qwen_image"
group_size          = 8
num_inference_steps = 20
kl_coeff            = 0.01

[[grpo.reward]]
name   = "hps_v2"
weight = 0.25

[[grpo.reward]]
name   = "clip"
weight = 0.25

[[grpo.reward]]
name   = "delta_e00"
weight = 0.5
[grpo.reward.params]
clip_max = 10.0
```

```bash
# qwen_image 架构示例（注意文本编码器参数名为 --text_encoder）
accelerate launch --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config grpo_config.toml \
    --prompt_file prompts.jsonl \
    --dit path/to/dit \
    --vae path/to/vae \
    --text_encoder path/to/text_encoder \
    --network_module networks.lora_qwen_image \
    --network_dim 32 \
    --model_version edit-2511 \
    --grpo_steps 1000 \
    --output_dir output/grpo_run \
    --output_name grpo_qwen

# HunyuanVideo 需要两个编码器：--text_encoder1（LLaVA-LLaMA3）+ --text_encoder2（CLIP-L）
# Wan 只需要 --text_encoder1（UMT5-XXL）
```

使用指南（参数说明 / Reward 配置 / 架构适配 / 常见问题）见 [doc/grpo_usage.md](doc/grpo_usage.md)。  
设计文档见 [doc/grpo_method.md](doc/grpo_method.md)。

---

## 两种方法的定位

| 维度 | Off-Policy Sample-Weight | GRPO |
|---|---|---|
| 反馈时机 | 训练前离线评估 | 训练中在线采样 |
| 实现复杂度 | 低（3 文件 8 处改动） | 高（需完整 RL 训练循环） |
| 适用场景 | 有明确质量指标、追求轻量改造 | 追求生成质量上限、资源充足 |
| 权重更新 | 静态（手动重跑评估脚本） | 动态（每步自动更新策略） |

两者可**组合使用**：用 Off-Policy 权重做课程采样，再用 GRPO 做在线策略优化。

---

## 环境配置

```bash
cd musubi-tuner
uv sync --extra cu128   # torch 2.7.1+cu128（推荐）
# 其他 CUDA 版本：cu124（torch 2.5.1+）、cu130（torch 2.9.1+）
# 不同 extra 之间不能混用
```

---

## 许可证

musubi-tuner 原始代码遵循上游许可证（Apache 2.0）。本扩展的修改部分同样遵循 Apache 2.0。
