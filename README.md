# musubi-plus: Sample-Weight Optimized Training Framework

基于 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 扩展的困难样本学习框架，支持 **per-sample weighted loss**。

---

## TODO

1. ~~add sample-weight method to the project~~ ✓ 已完成
2. add GRPO support to the project

---

## 核心特性

### 样本加权损失（Sample-Weighted Loss）

传统做法是复制困难样本（数据膨胀），本方案改为在 loss 计算时直接加权：

```
原版 musubi-tuner:  loss = loss.mean()  # 所有样本等权
musubi-plus:        loss = (loss * sample_weight).mean()  # 按样本难度加权
```

**优势：**
- 磁盘占用不变（无需复制数据）
- 权重连续可调（任意浮点数，非整数倍复制）
- 训练速度不变（仅 loss 计算多一次乘法）
- 完全向后兼容（不传 `--sample_weight_file` 时行为与原版一致）

---

## 文件结构

```
musubi-plus/
└── musubi-tuner/                   # 修改后的训练框架
    ├── src/musubi_tuner/
    │   ├── dataset/
    │   │   ├── image_video_dataset.py    # +4 处改动（ItemInfo、加载、batch）
    │   │   └── config_utils.py            # +2 处改动（schema、参数）
    │   └── hv_train_network.py            # +2 处改动（loss 计算、argparse）
    ├── qwen_image_train_network.py        # Qwen-Image-Edit 训练入口
    ├── qwen_image_cache_latents.py        # VAE latent 缓存
    └── ...                                # 其他模型的训练脚本
```

---

## 使用方法

### 1. 生成样本权重文件

用任何指标评估样本难度，映射为权重，输出 JSON：

```json
{
  "sample_stem_001": 2.5,
  "sample_stem_002": 1.0,
  "sample_stem_003": 7.3
}
```

**权重生成示例（Python）：**

```python
import json

def score_to_weight(score: float) -> float:
    """将难度得分映射为训练权重（示例：线性映射）"""
    threshold = 3.0
    alpha = 0.3
    excess = max(0.0, score - threshold)
    return 1.0 + alpha * excess

# 你的评估逻辑
weights = {}
for sample in your_samples:
    difficulty = evaluate_difficulty(sample)  # 任意指标
    weights[sample.stem] = score_to_weight(difficulty)

with open("sample_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
```

**常见映射函数：**

| 映射函数 | 公式 | 适用场景 |
|---|---|---|
| **线性** | `w = 1 + α × max(0, s − τ)` | 得分与难度线性相关 |
| **平方根** | `w = 1 + α × √max(0, s − τ)` | 得分差异大，需要平滑 |
| **对数** | `w = 1 + α × ln(1 + max(0, s − τ))` | 得分范围极宽 |
| **分位数** | `w = 1 + α × rank(s) / N` | 只关心相对排序 |

其中 `s` 是难度得分，`τ` 是阈值，`α` 是强度系数。

---

### 2. 训练时指定权重文件

```bash
accelerate launch qwen_image_train_network.py \
    --sample_weight_file sample_weights.json \
    --sample_weight_multiplier 1.0 \
    ...
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--sample_weight_file` | None | 权重 JSON 文件路径（不传则所有样本权重为 1.0） |
| `--sample_weight_multiplier` | 1.0 | 全局倍率（用于实验调参） |

---

## 框架修改详解

### (1) ItemInfo 扩展（image_video_dataset.py）

```python
class ItemInfo:
    def __init__(self, ...):
        ...
        self.sample_weight: float = 1.0  # 新增
```

### (2) 数据集加载权重（image_video_dataset.py）

```python
def prepare_for_training(self, num_timestep_buckets=None):
    sample_weights = {}
    if self.sample_weight_file:
        with open(self.sample_weight_file) as f:
            sample_weights = json.load(f)
    
    for cache_file in latent_cache_files:
        item_info = ItemInfo(item_key, ...)
        if sample_weights:
            item_info.sample_weight = sample_weights.get(item_key, 1.0)
```

### (3) Batch 注入权重（image_video_dataset.py）

```python
sample_weights = [
    getattr(item_info, "sample_weight", 1.0)
    for item_info in bucket[start:end]
]
batch_tensor_data["sample_weight"] = torch.tensor(
    sample_weights, dtype=torch.float32
)
```

### (4) Loss 计算加权（hv_train_network.py）

```python
if "sample_weight" in batch:
    # 空间维度求平均 → per-sample loss [B]
    loss = loss.mean(dim=list(range(1, loss.ndim)))
    # 乘以样本权重
    sample_w = batch["sample_weight"].to(loss.device, loss.dtype) \
               * args.sample_weight_multiplier
    loss = (loss * sample_w).mean()
else:
    loss = loss.mean()  # 向后兼容
```

---

## 与数据复制的对比

| 维度 | 数据复制 | 样本加权 loss |
|---|---|---|
| 磁盘占用 | N × 原始数据 | 1 × 原始数据 + 1 个 JSON |
| 每 epoch 步数 | N × 原始步数 | 等于原始步数 |
| 权重精度 | 整数（1×、2×） | 任意浮点数 |
| 训练速度 | 与数据量成正比变慢 | 不变 |
| 调参灵活性 | 需重新生成数据 | 改 JSON 即可 |

---

## 适用场景

任何图像/视频生成模型的微调任务，只要能找到某种"难度指标"：

| 任务 | 难度指标示例 |
|---|---|
| 图像修复 | PSNR / SSIM / LPIPS |
| 图像编辑 | CLIP 语义一致性 |
| 超分辨率 | HR-LR 重建误差 |
| 风格迁移 | 风格损失 + 内容损失 |
| 任意任务 | 人工标注难度等级 |

---

## 环境配置

```bash
cd musubi-tuner
uv sync --extra cu128
```

**关键：** torch 必须 pin 到 2.7.1+cu128（避免 CUDNN 兼容性问题）。

---

## 注意事项

1. **权重与 loss 绝对值：** 加权后的 loss 绝对值会比不加权时高，两者不可直接比较
2. **静态权重：** 当前方案是训练前计算一次权重。如果模型能力显著提升，可重新生成权重文件
3. **未覆盖的样本：** JSON 中没有出现的样本默认取 weight=1.0
4. **Video dataset：** 当前改动仅覆盖 ImageDataset，VideoDataset 需做类似扩展

---

## 许可证

musubi-tuner 原始代码遵循上游许可证（Apache 2.0）。本扩展的修改部分同样遵循 Apache 2.0。
