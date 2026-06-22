# DiffusionNFT 使用文档

**参考论文**: [DiffusionNFT: Negative-aware FineTuning for Diffusion Models](https://arxiv.org/abs/2509.16117)  
NVIDIA Research, ICLR 2026 Oral

---

## 算法原理

DiffusionNFT 是一种基于**前向扩散过程**的在线 RL 微调方法。与 GRPO 在反向去噪轨迹上优化不同，NFT 在前向加噪过程中优化速度场，天然兼容任何黑盒采样器，无需存储完整轨迹。

每个训练步骤分为两个阶段：

**Phase 1 — Rollout（无梯度）**

用旧策略（old LoRA）执行推理，生成 `group_size` 张图像，通过奖励函数打分，计算组内优势 `adv`，映射到 `r ∈ [0, 1]`：

```
r = clip(adv, -adv_clip_max, adv_clip_max) / adv_clip_max / 2.0 + 0.5
```

r > 0.5 → 偏向正样本方向（优于组均值）；r < 0.5 → 偏向负样本方向；r = 0.5 → 两者平衡。

**Phase 2 — NFT Loss（有梯度）**

将生成图像 VAE 编码为 `x0`，采样 `t ~ U(0,1)`，构造加噪潜变量 `x_t = (1-t)*x0 + t*ε`。

三次 DiT 前向：
- `v_old`：旧 LoRA，无梯度
- `v_ref`：初始 LoRA 快照，无梯度（用于 KL 正则）
- `v_θ`：当前 LoRA，有梯度

隐式正/负速度构造：
```
v_pos = β * v_θ + (1 - β) * v_old      # 正样本方向（优化目标）
v_neg = (1 + β) * v_old - β * v_θ      # 负样本方向（抑制目标）
```

自适应权重重建 MSE：
```python
x0_pos = x_t - t * v_pos
x0_neg = x_t - t * v_neg

w_pos = |x0_pos - x0|.mean(spatial).detach().clamp(1e-5)   # 归一化权重
w_neg = |x0_neg - x0|.mean(spatial).detach().clamp(1e-5)

pos_loss = ((x0_pos - x0)² / w_pos).mean(spatial)   # [B]
neg_loss = ((x0_neg - x0)² / w_neg).mean(spatial)   # [B]
```

优势加权总损失：
```python
nft_loss = (r * pos_loss + (1 - r) * neg_loss).mean() / β
kl_loss  = ((v_θ - v_ref)²).mean()
total    = nft_loss + kl_coeff * kl_loss
```

---

## 配置文件（TOML）

NFT 的配置通过 `--nft_config` 传入 TOML 文件。主配置节为 `[nft]`，奖励函数为 `[[nft.reward]]`（与 GRPO 格式相同）。

```toml
[nft]
architecture        = "qwen_image"    # 使用的模型架构
group_size          = 16              # 每个 prompt 每步生成的图像数量
num_inference_steps = 20              # 推理步数
width               = 768             # 生成分辨率（宽）
height              = 768             # 生成分辨率（高）
frame_count         = 1               # 帧数（图像任务 = 1）
guidance_scale      = 1.0             # CFG 系数（1.0 = 无 CFG）
discrete_flow_shift = 2.2             # Flow Matching 偏移参数

beta                = 1.0             # 正/负速度插值强度
kl_coeff            = 0.0001          # KL 正则化权重
adv_clip_max        = 5.0             # 优势裁剪上界

old_policy_update_every = 1           # 每 N 步更新一次旧策略
old_policy_decay        = 0.0         # 旧策略 EMA 衰减系数

phase2_chunk_size   = 0               # Phase 2 显存分块大小（0 = 不分块）

[[nft.reward]]
name   = "delta_e00"
weight = 0.7
[nft.reward.params]
clip_max = 15.0

[[nft.reward]]
name   = "clip"
weight = 0.3
```

---

## 所有参数详解

### 采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `architecture` | `"qwen_image"` | 模型架构标识，决定加载哪个 arch trainer。可选值：`hv`, `hv_1_5`, `wan`, `fpack`, `flux_2`, `flux_kontext`, `qwen_image`, `kandinsky5`, `zimage` |
| `group_size` | `16` | 每个 prompt 每步生成的图像数量。越大优势估计越准确，但显存消耗线性增加。推荐配合 `phase2_chunk_size` 使用 |
| `num_inference_steps` | `10` | 每张图的去噪步数。步数越多质量越高，但 Phase 1 耗时线性增加。图像任务一般 10–20 步足够 |
| `width` / `height` | `512` | Phase 1 rollout 的图像分辨率。分辨率越高显存消耗越大（Phase 2 激活内存约正比于 `H×W`）|
| `frame_count` | `1` | 视频帧数，图像任务保持 `1` |
| `guidance_scale` | `1.0` | Classifier-Free Guidance 系数。`1.0` 表示不使用 CFG（qwen_image 编辑模型通常不需要 CFG）|
| `discrete_flow_shift` | `2.2` | Flow Matching 时间步偏移，模型相关，保持默认即可 |

### NFT 损失超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | `1.0` | 正/负速度插值强度。`β=1.0` 适合感知质量任务（色彩、美学）；`β=0.1` 适合精细结构任务（OCR、文字）。β 越大正负方向差异越大，梯度越强但可能不稳定 |
| `kl_coeff` | `0.0001` | KL 正则化权重，防止 LoRA 权重远离初始点。设为 `0` 则跳过 v_ref 前向（节省一次 DiT 调用）。推荐范围：`0.00001`–`0.001` |
| `adv_clip_max` | `5.0` | 优势裁剪上界。优势先裁剪到 `[-adv_clip_max, adv_clip_max]`，再映射到 `[0,1]`。值过小会压缩有效信号；值过大会放大异常样本影响。通常不需要调整 |

### 旧策略维护

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `old_policy_update_every` | `1` | 每 N 步更新一次旧策略。默认每步都更新（类 PPO 风格）。增大此值可拉大新旧策略差距，但更新不及时 |
| `old_policy_decay` | `0.0` | EMA 衰减系数，控制旧策略追踪当前策略的速度：`old = decay * old + (1 - decay) * current`。`0.0` = 每次完整复制（推荐初始值）；`0.9` = EMA 半衰期约 9 步；`0.99` = 半衰期约 69 步 |

**关键注意事项**：`old_policy_decay` 是最容易出问题的参数。

- `decay = 0.0`（默认）：旧策略每步完整复制当前策略。在同一步内，旧策略 = 当前策略（更新发生在 step 末尾），所以组内 rollout 完全由当前策略执行，奖励方差仅来自随机性。这是最稳定的设置。
- `decay = 0.5`：EMA 半衰期约 1.4 步，旧策略几乎等于当前策略 → 组内奖励方差极小 → 优势 ≈ 0 → 有效 RL 信号消失（r ≈ 0.5，损失退化为平衡 pos/neg）。**不推荐使用 0.5**。
- `decay = 0.9`：半衰期约 9 步，是引入非零优势的最低合理值，适合需要"记住过去版本"的场景。

实践建议：**先用 `decay=0.0` 验证训练是否正常**，确认 `advantage_std > 0` 后再考虑引入 EMA。

### 显存管理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `phase2_chunk_size` | `0` | Phase 2 显存分块大小。`0` = 整个 group 一次性前向（简单，适合 group_size 小的情况）；`N > 0` = 将 group_size 张图分成每批 N 张处理，每批立即调用 `backward()` 释放计算图 |

**核心原理**：开启分块后，`_nft_loss` 在每个 chunk 内立即调用 `accelerator.backward(chunk_loss)` 释放当前 chunk 的计算图，然后返回 `torch.zeros([])`（`requires_grad=False`），训练循环中的外层 `backward()` 因此跳过：

```python
# nft_train_network.py（训练循环）
loss, log_dict = nft_trainer.step(batch_params, ...)
if loss.requires_grad:                  # phase2_chunk_size=0 时为 True
    accelerator.backward(loss)          # phase2_chunk_size>0 时跳过（已在内部 backward）
```

**显存估算（7B 模型，768×768，开启 gradient_checkpointing）**：
- 每张图梯度激活约 2 GiB
- `group_size=32`，不分块：32 × 2 GiB = 64 GiB 激活 + 14 GiB 模型 → OOM（H100 79 GiB）
- `phase2_chunk_size=8`：8 × 2 GiB = 16 GiB 激活 + 14 GiB 模型 → ~30 GiB，安全

**注意**：仅分块而不立即 backward 无效——累积 loss 后统一调用 backward 仍会同时持有所有 chunk 的计算图。

---

## 奖励函数

奖励函数与 GRPO 完全共享，支持以下类型：

| name | 说明 | 适用场景 |
|------|------|---------|
| `delta_e00` | CIEDE2000 色差（越小越好，自动取负） | 颜色保真、抠图 |
| `clip` | CLIP ViT-H-14 图文相似度 | 文本对齐 |
| `pickscore` | PickScore 人类偏好 | 美学质量 |
| `hps_v2` | HPSv2.1 人类偏好 | 美学质量 |
| `image_reward` | ImageReward | 美学质量 |
| `vlm` | Qwen2-VL 语义评分 | 复杂语义理解 |

多奖励时，各奖励在组内独立归一化后加权求和。

---

## 启动命令

```bash
accelerate launch \
    --multi_gpu \
    --num_processes 5 \
    --gpu_ids 2,3,4,5,6 \
    --mixed_precision bf16 \
    src/musubi_tuner/nft_train_network.py \
    --nft_config    /path/to/nft_config.toml \
    --prompt_file   /path/to/prompts.jsonl \
    --dit           /path/to/dit.safetensors \
    --vae           /path/to/vae.safetensors \
    --text_encoder  /path/to/text_encoder.safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim   64 \
    --network_weights /path/to/init_lora.safetensors \
    --learning_rate 3e-5 \
    --nft_steps     1000 \
    --nft_batch_size 1 \
    --model_version edit-2511 \
    --save_every_n_steps 50 \
    --output_dir    /path/to/output \
    --output_name   nft_run \
    --gradient_checkpointing
```

### 专用 CLI 参数

| 参数 | 说明 |
|------|------|
| `--nft_config` | NFT TOML 配置文件路径（必须） |
| `--prompt_file` | Prompt 文件（JSONL 或 txt，必须） |
| `--nft_steps` | 训练步数（覆盖 TOML；默认 100）|
| `--nft_batch_size` | 每步处理的 prompt 数量（默认 1）|
| `--nft_architecture` | 覆盖 TOML 中的 `architecture` 字段 |

其余参数（`--dit`, `--vae`, `--text_encoder`, `--network_module`, `--gradient_checkpointing` 等）与对应架构的常规训练脚本完全一致。

---

## 核心实现代码

### 旧策略权重交换

old policy 以 CPU state dict 形式存储，前向时临时交换到 GPU，完成后立即还原。无需 PEFT，无需额外模型副本：

```python
def _with_old_policy(self, fn):
    net = self.accelerator.unwrap_model(self.network)
    device = self.accelerator.device
    # 备份当前权重到 CPU
    current = {n: p.data.clone().cpu() for n, p in net.named_parameters() if p.requires_grad}
    # 换入旧权重
    for n, p in net.named_parameters():
        if p.requires_grad:
            p.data = self._old_state[n].to(device)
    try:
        result = fn()
    finally:
        # 还原当前权重
        for n, p in net.named_parameters():
            if p.requires_grad:
                p.data = current[n].to(device)
    return result
```

同一套机制用于 `_with_ref_policy`（初始权重快照，用于 KL 正则）。

### 旧策略 EMA 更新

```python
def _update_old_policy(self):
    decay = self.config.old_policy_decay
    net = self.accelerator.unwrap_model(self.network)
    for n, p in net.named_parameters():
        if p.requires_grad:
            old = self._old_state[n].to(p.device)
            self._old_state[n] = (decay * old + (1.0 - decay) * p.detach()).cpu()
```

`decay=0.0` → 完整复制（每步后 old = current）；`decay=0.9` → EMA 慢追踪。

### NFT Loss（分块 backward 路径）

```python
chunk = cfg.phase2_chunk_size if cfg.phase2_chunk_size > 0 else bsz
chunked_backward = cfg.phase2_chunk_size > 0

for cs in range(0, bsz, chunk):
    ce = min(cs + chunk, bsz)
    c = ce - cs
    sl = slice(cs, ce)

    # v_old（无梯度，旧 LoRA）
    with torch.no_grad():
        v_old_c = self._with_old_policy(_call_old)

    # v_ref（无梯度，初始 LoRA）
    if self._ref_state is not None:
        with torch.no_grad():
            v_ref_c = self._with_ref_policy(_call_ref)
    torch.cuda.empty_cache()

    # v_θ（有梯度，当前 LoRA）
    v_theta_c, _ = self.base.call_dit(...)
    v_theta_c = v_theta_c.to(torch.float32)

    # 隐式正负速度
    v_pos_c = beta * v_theta_c + (1.0 - beta) * v_old_c
    v_neg_c = (1.0 + beta) * v_old_c - beta * v_theta_c

    # x0 重建
    x0_pos_c = noisy_c - t_view_c * v_pos_c
    x0_neg_c = noisy_c - t_view_c * v_neg_c

    # 自适应权重
    w_pos_c = (x0_pos_c - lat_c).abs().mean(spatial, keepdim=True).detach().clamp(1e-5)
    w_neg_c = (x0_neg_c - lat_c).abs().mean(spatial, keepdim=True).detach().clamp(1e-5)

    pos_loss_c = ((x0_pos_c - lat_c) ** 2 / w_pos_c).mean(spatial)   # [c]
    neg_loss_c = ((x0_neg_c - lat_c) ** 2 / w_neg_c).mean(spatial)   # [c]

    # 优势 → r
    adv_clipped_c = adv_c.clamp(-cfg.adv_clip_max, cfg.adv_clip_max)
    r_c = adv_clipped_c / cfg.adv_clip_max / 2.0 + 0.5

    if chunked_backward:
        nft_c = (r_c * pos_loss_c + (1.0 - r_c) * neg_loss_c).sum() / beta / bsz
        kl_c  = ((v_theta_c - v_ref_c) ** 2).mean() * cfg.kl_coeff * c / bsz
        chunk_total = nft_c + kl_c

        # 立即 backward，释放本 chunk 计算图
        self.accelerator.backward(chunk_total)
        del v_theta_c, v_pos_c, v_neg_c, x0_pos_c, x0_neg_c, nft_c, kl_c, chunk_total
        torch.cuda.empty_cache()

if chunked_backward:
    # 梯度已写入 .grad，返回 detached zero，外层 backward 跳过
    return torch.zeros([], device=device), log
```

---

## 训练日志解读

每步输出示例：
```
[step   100] loss=+0.0231  nft=+0.0229  kl=0.000002  delta_e00=-8.61  clip=0.3412  adv=+0.0015
```

| 字段 | 含义 | 健康范围 |
|------|------|---------|
| `loss` | NFT loss + KL loss 总和 | 通常 0.01–0.1 |
| `nft` | NFT 重建损失 | 同上 |
| `kl` | KL 正则化项 | 初期极小（< 0.001），随训练缓慢增大 |
| `delta_e00` | 色差奖励均值（越负越好） | 随训练负值应减小（向 0 趋近）|
| `adv` | 组内优势均值 | 应接近 0；std 应 > 0 |

**adv ≈ 0 且 adv_std ≈ 0**：组内奖励方差极小，RL 信号消失。常见原因：
1. `old_policy_decay` 过大（如 0.5），旧策略与当前策略几乎相同
2. `group_size` 太小（< 4），统计估计不稳定
3. 奖励函数对所有样本输出相同分数

---

## 推荐配置

**图像编辑任务（颜色保真 + 文本对齐）**

```toml
[nft]
architecture        = "qwen_image"
group_size          = 16
num_inference_steps = 20
width               = 768
height              = 768
guidance_scale      = 1.0
discrete_flow_shift = 2.2
beta                = 1.0
kl_coeff            = 0.0001
adv_clip_max        = 5.0
old_policy_update_every = 1
old_policy_decay        = 0.0
phase2_chunk_size   = 4       # group_size=16, H100 80G

[[nft.reward]]
name   = "delta_e00"
weight = 0.7

[[nft.reward]]
name   = "clip"
weight = 0.3
```

**高分辨率 / 大 group_size（group_size=32，768×768，H100）**

```toml
group_size        = 32
phase2_chunk_size = 8      # 8×2 GiB ≈ 16 GiB 激活，安全
```

**精细结构任务（OCR、文字）**

```toml
beta     = 0.1    # 更小的 β，正负方向差异更平缓，梯度更稳定
kl_coeff = 0.001  # 稍大的 KL 约束，防止 LoRA 过拟合
```
