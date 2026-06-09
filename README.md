# musubi-plus

基于 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 的强化学习增强训练框架，将两种互补的 RL 思路引入图像/视频生成模型微调：

| 方法 | 核心思路 | 状态 |
|---|---|---|
| **Off-Policy Sample-Weight** | 离线评估样本难度，训练时直接加权 loss | ✅ 已完成 |
| **GRPO** | 在线策略梯度，用多维 Reward 模型引导生成质量 | 🚧 开发中 |

---

## Feature 1：Off-Policy Sample-Weight Method

权重在训练前由外部策略离线计算（任意难度指标），训练时固定注入，**评估策略与训练策略完全解耦**。

```
原版 musubi-tuner:  loss = loss.mean()
musubi-plus:        loss = (loss * sample_weight).mean()
```

**优势：** 零磁盘膨胀、浮点精度权重、训练速度不变、改 JSON 即可切换策略。

使用方法：

```bash
accelerate launch qwen_image_train_network.py \
    --sample_weight_file sample_weights.json \
    --sample_weight_multiplier 1.0 \
    ...
```

详细设计与框架改动见 [doc/off_policy_sample_weight_method.md](doc/off_policy_sample_weight_method.md)。

---

## Feature 2：GRPO（开发中）

将 GRPO（Group Relative Policy Optimization）引入 musubi-tuner，实现在线 RL 训练循环。调研覆盖当前主流变体：

- **DanceGRPO** — 统一支持 Diffusion / Rectified Flow，覆盖文生图、文生视频、图像编辑
- **Flow-GRPO** — 首个将在线策略梯度整合进 Flow Matching 的方法，支持 9 种 Reward 灵活组合
- **Adv-GRPO** — 引入对抗性奖励（DINOv2），缓解 Reward Hacking，支持风格迁移
- **MO-GRPO** — 多目标设置下防 Reward Hacking，先归一化再聚合
- **PREF-GRPO** — 以成对偏好胜率替代绝对分数，从奖励建模层面消除打分偏差

Reward Model 体系调研见 [doc/grpo_reward_model_report.html](doc/grpo_reward_model_report.html)，防 Reward Hacking 方案对比见 [doc/GRPO_MultiReward_AntiHacking_Report.html](doc/GRPO_MultiReward_AntiHacking_Report.html)。

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
uv sync --extra cu128
```

> torch 必须 pin 到 2.7.1+cu128，避免 CUDNN 兼容性问题。

---

## 许可证

musubi-tuner 原始代码遵循上游许可证（Apache 2.0）。本扩展的修改部分同样遵循 Apache 2.0。
