# GRPO 训练框架：musubi-plus 扩展方案

**2026-06-09**

---

## 1. 问题定义

### 1.1 Off-Policy 方案的局限

Off-Policy Sample-Weight 方案（已实现）建立了一条清晰的路径：**离线评估样本质量 → 训练时加权 loss**。它的根本局限在于，权重是静态的——它能引导模型学哪些样本，但不能主动要求模型生成"更好"的内容。

更进一步的问题是：**如何让模型在训练中持续朝"人类期望的方向"进化**？这需要：

1. 在训练中在线生成内容
2. 用奖励信号评估生成质量
3. 将奖励反馈直接塑造模型参数

这是强化学习的经典设定，而 GRPO（Group Relative Policy Optimization）是目前最适配扩散/流模型的 RL 变体。

### 1.2 直接套用 RL 的挑战

| 挑战 | 说明 |
|---|---|
| **连续动作空间** | 扩散模型的"动作"是每步预测的连续速度场，不是离散 token |
| **长链依赖** | 一次生成 = 数十步去噪，每步都影响最终结果 |
| **Reward Hacking** | 单一奖励模型容易被过度拟合，偏离真实质量 |
| **多目标冲突** | 不同奖励（美学 vs 文本对齐 vs 色彩保真）量纲不同，直接相加会让高方差 reward 主导 |

### 1.3 设计目标

- **Flow Matching 原生**：损失函数直接在速度场空间计算，不引入近似
- **多奖励，防 Hacking**：每个 reward 独立归一化后再聚合，高方差 reward 不能主导优势函数
- **与现有框架解耦**：不修改任何现有文件，作为独立模块插入
- **架构无关**：支持 musubi-tuner 所有架构（hv、wan、fpack、flux、qwen_image 等）
- **Reward 可配置**：深度模型 / 规则 / 任意指标均可注册为 reward，TOML 声明使用哪些

---

## 2. 方案设计

### 2.1 核心思路

Off-Policy Sample-Weight 与 GRPO 在数学上是同一结构：

```
Off-Policy:   loss = w_offline · ‖v_θ(x_t) − v*‖²     w 来自离线 JSON
GRPO:         loss = A_online  · ‖v_θ(x_t) − v*‖²     A 来自在线采样 + reward
```

区别只是**权重的来源**：离线静态 vs 在线动态。GRPO 是 Off-Policy Sample-Weight 的在线泛化。

### 2.2 训练流水线

![GRPO Pipeline](grpo_pipeline.svg)

> 流水线分两个阶段：**①~③ no_grad**（在线采样 + reward 打分）生成优势信号；**④~⑤ with grad**（优势加权 loss + 反向传播）更新模型。

### 2.3 Group 采样与 GRPO 优势

GRPO 的"group"是：对同一条 prompt，用当前策略采样 G 张图像。奖励的基准是组内均值，而非绝对分数：

```
advantage_i = reward_i − mean({reward_1, ..., reward_G})
```

这使得不同 prompt 的难度差异自动被消除——容易的 prompt 和困难的 prompt 在同一优化目标下。

### 2.4 MO-GRPO 多奖励归一化

多个奖励直接相加的问题：高方差 reward（如 OCR 分数）会在归一化后产生更大的梯度，实际上劫持了训练。MO-GRPO 的解法是**先归一化，再聚合**：

```
Ã_k^(i) = (r_k^(i) − μ_k) / (σ_k + ε)    # 每个 reward 组内独立归一化
A^(i)   = Σ_k  w_k · Ã_k^(i)              # 归一化后再加权求和
```

其中 μ_k、σ_k 在同一 group 的 G 个样本上计算。每个 reward 在聚合前的贡献量级相同，权重 w_k 真实反映人的偏好比例，而不受 reward 值域影响。

### 2.5 Flow Matching GRPO Loss

对于 Flow Matching 模型（musubi-tuner 全系架构），前向过程是：

```
x_t = (1 − t) · x_0 + t · ε,  ε ~ N(0, I)
v_target = ε − x_0
```

标准训练 loss 为 `L_FM = ‖v_θ(x_t, t, c) − v_target‖²`。GRPO 在此基础上乘以优势并加 KL 惩罚：

```
L_GRPO = E_{i,t} [ A^(i) · ‖v_θ(x_t^(i), t, c) − v_target^(i)‖² ]
       + β · E_t [ ‖v_θ(x_t, t, c) − v_ref(x_t, t, c)‖² ]
```

- `A^(i)` 为 MO-GRPO 优势（第 2.4 节），正值 = 鼓励，负值 = 抑制
- 第二项为 KL 惩罚，`v_ref` 是训练开始时冻结的参考策略，防止策略漂移过远
- `x_t^(i)` 是对采样图像加噪得到的中间状态，**不参与梯度**（在线采样阶段已 `no_grad`）

---

## 3. Reward 系统设计

### 3.1 BaseReward 接口

所有 reward 实现同一接口，通过注册器动态加载：

```python
class BaseReward(ABC):
    def __init__(self, params: dict): ...
    def load(self, device: torch.device): ...   # 懒加载模型

    @abstractmethod
    def score(
        self,
        images: list[Image.Image],
        prompts: list[str],
        **kwargs,
    ) -> torch.Tensor:
        """返回 [N] 张量，越高越好。"""
        ...
```

注册方式（装饰器）：

```python
@register("hps_v2")
class HPSv2Reward(BaseReward):
    ...
```

TOML 中写 `name = "hps_v2"` 即可自动匹配。

### 3.2 内置 Reward 列表

| name | 类型 | 来源 | 输入 |
|---|---|---|---|
| `hps_v2` | 偏好模型 | HPSv2.1 | image + prompt |
| `pickscore` | 偏好模型 | PickScore | image + prompt |
| `image_reward` | 偏好模型 | ImageReward | image + prompt |
| `clip` | 对齐指标 | CLIP ViT-H-14 | image + prompt |
| `ocr` | 规则指标 | PaddleOCR | image + prompt（含目标文字） |
| `vlm` | VLM 打分 | Qwen2-VL-2B | image + prompt（自定义评分指令） |
| `delta_e00` | 规则指标 | CIEDE2000 | image + reference_image |

### 3.3 ΔE00 色彩保真度 Reward

ΔE00（CIEDE2000）是感知均匀色差标准，1.0 表示人眼刚好可感知的差异。实现步骤：

```python
# 1. 转换至 Lab 色彩空间
img_lab = rgb_to_lab(generated_image)      # [H, W, 3]
ref_lab = rgb_to_lab(reference_image)      # [H, W, 3]

# 2. 逐像素计算 ΔE00（colour-science 库）
delta_e = colour.delta_E(img_lab, ref_lab, method="CIE 2000")  # [H, W]

# 3. 转为奖励（低色差 = 高奖励）
reward = -delta_e.mean()    # 取均值后取反
```

`clip_max` 参数可截断异常大的色差值（如背景区域遮挡导致的极端值）。

### 3.4 VLM Reward 配置

VLM reward 通过 prompt 模板灵活定义评分维度：

```toml
[[grpo.reward]]
name = "vlm"
weight = 0.2
[grpo.reward.params]
model = "Qwen/Qwen2-VL-2B-Instruct"
prompt_template = "请从 1-10 打分评价这张图片与描述「{prompt}」的一致性，只输出数字。"
```

---

## 4. 框架设计

### 4.1 文件结构

**新增文件，不修改任何现有文件。**

```
musubi-tuner/src/musubi_tuner/
├── grpo/
│   ├── __init__.py
│   ├── config.py              # GRPOConfig / RewardConfig（dataclass，TOML 加载）
│   ├── trainer.py             # GRPOTrainer：主训练循环
│   ├── advantage.py           # MO-GRPO 优势计算
│   ├── prompt_dataset.py      # Prompt + 参考图数据集（JSONL / txt）
│   └── reward/
│       ├── __init__.py
│       ├── base.py            # BaseReward ABC + @register 装饰器
│       ├── hps.py             # HPSv2
│       ├── pickscore.py       # PickScore
│       ├── image_reward.py    # ImageReward
│       ├── clip.py            # CLIP
│       ├── ocr.py             # PaddleOCR
│       ├── vlm.py             # Qwen2-VL
│       └── delta_e.py         # ΔE00
└── grpo_train_network.py      # 入口脚本
```

### 4.2 GRPOTrainer 设计

`GRPOTrainer` 不继承 `NetworkTrainer`，而是**持有**它——组合优于继承：

```python
class GRPOTrainer:
    def __init__(self, base_trainer: NetworkTrainer, config: GRPOConfig):
        self.base   = base_trainer
        self.ref    = self._freeze(base_trainer.transformer)  # 冻结参考策略
        self.rewards = build_rewards(config.rewards)

    def step(self, prompts: list[str]) -> torch.Tensor:
        # Phase 1: 在线采样（no_grad）
        with torch.no_grad():
            images = self._rollout(prompts)                   # [B·G, H, W, 3]
            scores = self._score(images, prompts)             # {name: [B·G]}
            adv    = compute_advantages(scores, self.weights) # [B·G]

        # Phase 2: 计算 loss（with grad）
        return self._grpo_loss(images, adv, prompts)

    def _rollout(self, prompts):
        # 复用 base_trainer.sample_image_inference()
        ...

    def _grpo_loss(self, images, adv, prompts):
        # 对采样图像加噪 → 预测速度场 → 优势加权 MSE + KL 惩罚
        ...
```

### 4.3 架构无关性

`grpo_train_network.py` 通过 `--architecture` 动态导入对应 Trainer：

```python
ARCH_TRAINERS = {
    "hv":          "musubi_tuner.hv_train_network.NetworkTrainer",
    "wan":         "musubi_tuner.wan_train_network.NetworkTrainer",
    "qwen_image":  "musubi_tuner.qwen_image_train_network.NetworkTrainer",
    "fpack":       "musubi_tuner.fpack_train_network.NetworkTrainer",
    # ...
}
```

模型加载（DiT、VAE、文本编码器）全部复用各架构现有的 `load_transformer()` / `load_vae()`，GRPO 层不感知架构细节。

### 4.4 Prompt 数据集格式

```jsonl
{"prompt": "a red apple on a white table", "reference": "data/refs/apple_001.png"}
{"prompt": "futuristic city at night"}
```

`reference` 字段可选，仅 `delta_e00` 等需要参考图的 reward 会用到。

---

## 5. 配置与使用

### 5.1 TOML 配置示例

```toml
[grpo]
architecture        = "qwen_image"
group_size          = 8          # G：每条 prompt 采样几张
num_inference_steps = 20         # 去噪步数
kl_coeff            = 0.01       # KL 惩罚系数 β
clip_eps            = 0.2        # PPO-style 梯度裁剪 ε

[[grpo.reward]]
name   = "hps_v2"
weight = 0.25

[[grpo.reward]]
name   = "clip"
weight = 0.25

[[grpo.reward]]
name   = "image_reward"
weight = 0.2

[[grpo.reward]]
name   = "delta_e00"
weight = 0.3
[grpo.reward.params]
reference_dir = "data/references"
clip_max      = 10.0
```

### 5.2 启动命令

```bash
accelerate launch --mixed_precision bf16 \
    src/musubi_tuner/grpo_train_network.py \
    --grpo_config  grpo_config.toml \
    --prompt_file  prompts.jsonl \
    --dit          path/to/dit \
    --vae          path/to/vae \
    --network_module networks.lora \
    --network_dim  32
```

大部分参数与现有训练脚本相同（`--fp8_base`、`--blocks_to_swap`、`--seed` 等均可复用）。

---

## 6. 防 Reward Hacking 策略

| 机制 | 作用 | 实现位置 |
|---|---|---|
| MO-GRPO 归一化 | 防高方差 reward 劫持优势函数 | `advantage.py` |
| Reward 截断 | 消除极端打分噪声（如遮挡导致的 ΔE00 异常值） | `BaseReward.score()` post-processing |
| KL 惩罚 | 防策略偏离参考模型过远，保留预训练能力 | `trainer._grpo_loss()` 第二项 |
| PPO clipping（可选） | 限制单步策略更新幅度 | `trainer._grpo_loss()`，由 `clip_eps` 控制 |
| 多 reward 集成 | 单一 reward 被 hack 时，其他 reward 提供纠正信号 | `advantage.py` 加权聚合 |

---

## 7. 与 Off-Policy Sample-Weight 的对比

| 维度 | Off-Policy Sample-Weight | GRPO |
|---|---|---|
| 权重来源 | 训练前离线计算 | 训练中在线采样 |
| 反馈延迟 | 静态（手动重跑评估） | 即时（每 step 更新） |
| 数据来源 | 训练集中的真实样本 | 模型当前策略采样的合成样本 |
| 奖励维度 | 单一难度指标 | 多维 reward 加权聚合 |
| 计算开销 | 极低（一次乘法） | 高（G × 完整去噪链 + reward 推理） |
| 适用阶段 | 监督微调（SFT）增强 | 对齐训练（RL fine-tuning） |
| 实现复杂度 | 低（3 文件 8 处改动） | 高（独立模块 ~12 个文件） |

两者**正交**，可以同时开启：用 Off-Policy 权重做课程采样，再用 GRPO 做在线对齐。

---

## 8. 注意事项

1. **计算开销**：每个训练 step 需要 G 次完整推理（默认 G=8，20 步去噪），显存和时间开销远高于 SFT。建议先用小 `group_size` 和少 `num_inference_steps` 验证流程。

2. **Reward 冷启动**：部分 reward 模型（VLM、HPSv2）在第一次调用时需要加载大模型，建议在训练开始前 warmup。

3. **参考策略更新**：`v_ref` 当前为训练开始时的快照（固定）。如训练轮数很长，可考虑定期软更新（Polyak averaging）。

4. **ΔE00 的参考图对齐**：参考图需与 prompt 严格对应（通过文件名匹配），`prompt_dataset.py` 负责此映射，参考图缺失时该样本的 `delta_e00` 奖励自动降权为 0。

5. **多卡同步**：每个进程独立采样并打分，advantages 在 `accelerator.gather()` 后跨卡归一化，保证 group 统计量在全局 batch 上计算。
