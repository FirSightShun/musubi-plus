# 图像生成强化学习调研报告

> 调研时间：2026-05-28 ~ 2026-05-29  
> 覆盖方向：GRPO 变体算法 · 开源 Reward Model 生态 · 多 Reward 防 Hacking 方法 · 图像编辑专用 Reward Model 训练

---

## 目录

1. [GRPO 变体算法详解](#1-grpo-变体算法详解)
2. [开源 Reward Model 全景](#2-开源-reward-model-全景)
3. [多 Reward 防 Reward Hacking 方法体系](#3-多-reward-防-reward-hacking-方法体系)
4. [图像编辑专用 Reward Model 训练方案](#4-图像编辑专用-reward-model-训练方案)
5. [参考文献](#5-参考文献)

---

## 1. GRPO 变体算法详解

### 1.1 DanceGRPO：视觉生成大一统强化学习框架

港大 & 字节跳动 | arXiv:2505.07818 | GitHub ⭐1.6k  
**特性：全量开源 · Diffusion + Flow · T2I + T2V + I2V**

首个将 GRPO 统一应用于视觉生成的框架，同时支持 Diffusion 和 Rectified Flow 两大生成范式，覆盖文生图、文生视频、图生视频、图像编辑四项任务。

**所用 Reward Models：**
- HPS-v2.1（偏好）— 人类偏好评分
- PickScore（偏好）— 人类偏好奖励
- CLIP ViT-H-14（对齐）— 文本-图像对齐评分
- VideoReward（视频）— 视频质量评估
- Qwen2-VL-2B（VLM）— 视频评估

**支持基础模型：** SD v1.4 · FLUX.1-dev · HunyuanVideo · SkyReels-I2V · Qwen-Image/Edit · Wan-2.1

---

### 1.2 Flow-GRPO：首个基于在线 RL 的 Flow Matching 训练方法

arXiv:2505.05470 | 全量开源 · Flow Matching · 9 种 Reward

首个将在线策略梯度 RL 整合进 Flow Matching 模型的方法，在 SD3.5-M 上实现了组合生成、文本渲染和人类偏好对齐三大任务的显著提升。

**支持的 9 种 Reward Models：**

| Reward | 类型 | 说明 |
|---|---|---|
| ImageReward | 偏好 | 通用 T2I 偏好，捕捉对齐+保真+安全 |
| PickScore | 偏好 | 人类偏好奖励模型 |
| UnifiedReward | 偏好 | 多模态统一奖励，SOTA |
| GenEval | 对齐 | 组合提示词评估 |
| OCR | 对齐 | 文本渲染准确率（PaddleOCR） |
| Aesthetic | 质量 | CLIP 线性回归器美学评分 |
| DeQA | 质量 | MLLM 图像质量评估 |
| QwenVL | VLM | 提示工程奖励 |
| JPEG Compressibility | 质量 | 图像大小代理指标 |

**多奖励组合示例：** `pickscore:0.5 + ocr:0.2 + aesthetic:0.3`

---

### 1.3 Adv-GRPO：对抗性奖励机制

NUS Show Lab & 字节 | arXiv:2511.20256 | CVPR 2026 | GitHub ⭐83

核心创新：引入对抗性奖励框架，用视觉基础模型（DINOv2）作为奖励信号，从三个维度改进 T2I 生成——①缓解 Reward Hacking；②密集视觉先验信号；③基于 RL 的风格定制。

**奖励流程：** 参考图像（正样本）→ 判别器训练 → 判别器作为 RM → GRPO Loss 优化生成器

**所用 Reward Models：** PickScore（传统偏好基线）· DINOv2（对抗视觉奖励）· DINO_OCR · DINO_GENEVAL

---

### 1.4 AR-GRPO：自回归图像生成的在线 RL 训练

快手 Klear 团队 | arXiv:2508.06924 | GitHub ⭐52  
**特性：全量开源 · 自回归 · C2I + T2I**

首次将 GRPO 在线 RL 应用于自回归图像生成模型（LlamaGen），在 Class-to-Image 和 Text-to-Image 任务上均取得显著改进。

**支持的 7+ 种 Reward Models：** CLIP ViT-L-14 · HPSv2 · PickScore · ImageReward · Aesthetic Score · DeQA · UniReward · Qwen2.5-VL-3B

---

### 1.5 Pref-GRPO：成对偏好奖励的稳定 T2I 强化学习

arXiv:2508.20751 | GitHub ⭐268 | NeurIPS 2025 关联

首个基于成对偏好奖励（Pairwise Preference Reward）的 GRPO 方法，替代传统逐点评分，有效缓解 reward hacking。已被阿里巴巴和通义实验室外部验证。

**支持的 Reward Models：**

| Reward | 类型 |
|---|---|
| UnifiedReward-Think | 偏好（带 CoT 推理） |
| UnifiedReward-Flex | 偏好（个性化） |
| UnifiedReward-Edit | 偏好（图像编辑） |
| HPSv2 / HPSv3 | 偏好 |
| PickScore / CLIP / Aesthetic | 质量 |
| VideoAlign | 视频对齐 |

**支持基础模型：** FLUX.1-dev · FLUX.2-Klein · FLUX.1-Kontext-dev · Qwen-Image/Edit · Z-Image · Wan2.1/2.2

---

## 2. 开源 Reward Model 全景

### 2.1 开源 Reward Model 对照表

| Reward Model | 类型 | 架构 | 训练数据 | 开源状态 | 被引用于 |
|---|---|---|---|---|---|
| UnifiedReward 系列 | 偏好+评分 | Qwen2.5-VL/Qwen3-VL (2b~72b) | ~700K 多模态偏好数据 | ✅ 全量开源 (⭐778) | Pref-GRPO, Flow-GRPO, AR-GRPO, Meta, NVIDIA, Apple 等 |
| ImageReward | 偏好 | BLIP + MLP Head | 137K 专家比较对 | ✅ 全量开源 | Flow-GRPO, AR-GRPO |
| HPSv2 / HPSv2.1 | 偏好 | CLIP ViT-H-14 (fine-tuned) | HPDv2 大规模人类偏好 | ✅ 全量开源 | DanceGRPO, AR-GRPO, Pref-GRPO |
| PickScore | 偏好 | CLIP ViT-L-14 (fine-tuned) | Pick-a-Pic 人类偏好数据 | ✅ 开源 | DanceGRPO, Flow-GRPO, Adv-GRPO, AR-GRPO, Pref-GRPO |
| CLIP Score | 对齐 | CLIP ViT-H-14 / ViT-L-14 | LAION-2B | ✅ 开源 | DanceGRPO, AR-GRPO, Pref-GRPO |
| Aesthetic Scorer | 质量 | CLIP + 线性回归器 | AVA 美学评分数据 | ✅ 开源 | Flow-GRPO, AR-GRPO, Pref-GRPO |
| DeQA-Score | 质量 | MLLM (LLaVA 架构) | 多质量数据集 | ✅ 全量开源 | Flow-GRPO, AR-GRPO |
| DINOv2 | 对抗 | ViT-L/14（自监督） | LVD-142M（自监督） | ✅ 开源 | Adv-GRPO |
| VideoReward | 视频 | VideoAlign 架构 | 视频偏好数据 | ✅ 开源 | DanceGRPO |
| Qwen2-VL / Qwen2.5-VL | VLM | Qwen-VL 系列 (2B~72B) | 多模态预训练 | ✅ 开源 | DanceGRPO, Flow-GRPO, AR-GRPO |
| GenEval | 对齐 | 基于检测器的评估 | 组合提示词数据 | ✅ 开源 | Flow-GRPO, Adv-GRPO |
| OCR Reward | 对齐 | PaddleOCR | — | ✅ 开源 | Flow-GRPO, Adv-GRPO |

### 2.2 GRPO 变体 × Reward Model 支持矩阵

| Reward Model | DanceGRPO | Flow-GRPO | Adv-GRPO | AR-GRPO | Pref-GRPO |
|---|---|---|---|---|---|
| UnifiedReward 系列 | — | ✓ | — | ✓ | ✓ |
| ImageReward | — | ✓ | — | ✓ | — |
| HPSv2 / HPSv2.1 | ✓ | — | — | ✓ | ✓ |
| PickScore | ✓ | ✓ | ✓ | ✓ | ✓ |
| CLIP Score | ✓ | — | — | ✓ | ✓ |
| Aesthetic Scorer | — | ✓ | — | ✓ | ✓ |
| DeQA-Score | — | ✓ | — | ✓ | — |
| DINOv2 | — | — | ✓ | — | — |
| VideoReward | ✓ | — | — | — | ✓ |
| Qwen-VL 系列 | ✓ | ✓ | — | ✓ | — |
| GenEval | — | ✓ | — | — | — |
| OCR Reward | — | ✓ | — | — | — |

### 2.3 关键趋势

**UnifiedReward 系列成为新标杆**  
覆盖面最广的统一奖励模型，支持图像/视频的理解与生成评估，提供 Think（CoT 推理）、Flex（个性化）、Edit（编辑）三种变体。已获 NeurIPS 2025 接收，被 Meta/NVIDIA/Apple/字节/腾讯/快手等广泛引用。

**从逐点评分到成对偏好**  
传统 GRPO 使用逐点评分（Pointwise Score），容易导致 reward hacking。Pref-GRPO 创新性地提出成对偏好奖励（Pairwise Preference Reward），用 win rate 替代绝对分数，训练更稳定，已被阿里巴巴和通义实验室独立验证。

**VLM 作为 Reward 正在兴起**  
从 Qwen2-VL、Qwen2.5-VL 到 UnifiedReward（基于 Qwen3-VL），视觉语言模型（VLM）作为 Reward Model 成为明确趋势，可通过 prompt engineering 灵活定义奖励维度。

**实操推荐**

| 使用场景 | 推荐 Reward Model | 推荐框架 |
|---|---|---|
| 通用文生图偏好对齐 | UnifiedReward-Think + PickScore | Pref-GRPO |
| 多任务统一训练（T2I+T2V+I2V） | HPS-v2.1 + CLIP | DanceGRPO |
| Flow Matching 模型优化 | ImageReward + GenEval + OCR | Flow-GRPO |
| 缓解 Reward Hacking | DINOv2 对抗性奖励 | Adv-GRPO |
| 自回归图像生成 | UniReward + HPSv2 + Aesthetic | AR-GRPO |
| 快速上手 / 资源有限 | PickScore + Aesthetic | DanceGRPO (FLUX LoRA) |

---

## 3. 多 Reward 防 Reward Hacking 方法体系

### 3.1 问题定义

**Reward Hacking** 是指智能体过度拟合到错误指定的奖励模型，优化代理奖励但偏离真实目标的现象。在 GRPO 的多 Reward 设置中表现为两种典型模式：

**模式 A — 高方差奖励主导（MO-GRPO 揭示）**  
GRPO 先聚合所有 Reward 再归一化，导致方差最大的 Reward 主导优势函数。低方差但同样重要的 Reward 被系统性忽略，策略只优化单一目标而牺牲其他目标。

**模式 B — 虚假优势（PREF-GRPO 揭示）**  
点式 Reward 模型对同组相似图像打分过于接近（标准差极小）。归一化后，微小分数差异被过度放大，产生巨大但虚假的优势值，导致策略过度更新和奖励欺骗。

**Reward Hacking 典型表现：**

| 场景 | Hacking 行为 | 后果 |
|---|---|---|
| 机器翻译（En→Ja） | 避免使用日语词汇来提升可读性分数 | 68.7% 输出为非日语文本（GRPO），MO-GRPO 仅 5.6% |
| 文生图（HPS 优化） | 过度饱和（颜色异常鲜艳） | 奖励分数增长，图像质量实际下降 |
| 文生图（UR 优化） | 图像偏暗（亮度异常偏低） | 分数增长但视觉退化 |
| 指令跟随（RM+Length） | 优先优化 RM 分数，完全牺牲长度目标 | 两个对抗性目标无法均衡 |

---

### 3.2 方法一：MO-GRPO（权重归一化层）

arXiv:2509.22047 | 2025.09

**核心洞察：** 原始 GRPO 优势函数计算顺序为「先聚合后归一化」，MO-GRPO 将顺序改为「先归一化后聚合」，从根本上平衡各 Reward 的贡献。

**优势函数对比：**

```
GRPO（先聚合后归一化）：
A_g = (Σ R_i(q,o_g) − mean_o(Σ R_i)) / std_o(Σ R_i)

MO-GRPO（先归一化后聚合）：
A_g^MO = Σ [(R_i(q,o_g) − mean_o(R_i)) / std_o(R_i)] / √K
```

| 性质 | GRPO | MO-GRPO |
|---|---|---|
| 高方差奖励主导 | 存在 | 消除 |
| 正仿射变换不变性 | 不满足 | 满足 |
| 需要手动调权重 | 是 | 否（自动重加权） |

**实验结果：**

| 任务 | 指标 | GRPO | MO-GRPO |
|---|---|---|---|
| En→Ja 翻译 | GPT-Eval | 35.6% | 68.8% |
| En→Zh 翻译 | 非中文输出比例↓ | 68.7% | 5.6% |

**核心结论：** 仅需改变优势函数的计算顺序（先归一化后聚合），即可自动平衡多 Reward 贡献，改动最小，即插即用。

---

### 3.3 方法二：PREF-GRPO（奖励范式变革层）

复旦 & 腾讯 | 2025.09

**核心洞察：** 点式 Reward 对同组相似图像打分过于接近，归一化后微小差异被过度放大，产生**虚假优势**。PREF-GRPO 用**成对偏好胜率**替代绝对分数。

**成对偏好胜率公式：**
```
R_i^pref = 1/(G−1) × Σ_{j≠i} I[PPRM(x_i) ≻ PPRM(x_j)]
```

| 维度 | 绝对分数奖励 | 成对偏好奖励 |
|---|---|---|
| 方差特性 | 相似图像分数接近，方差极小 | 高质量→1，低质量→0，方差自然增大 |
| 优势估计 | 微小差异被过度放大→虚假优势 | 区分度更高，优势估计更稳健 |
| 对噪声鲁棒性 | 高度敏感 | 依赖相对排序，大幅减轻偏差放大 |

**核心结论：** 从奖励建模范式层面解决问题——将"最大化绝对分数"转变为"拟合成对偏好"，从根本上增大奖励方差，消除虚假优势的土壤。

---

### 3.4 方法三：Adv-GRPO（对抗博弈层）

CVPR 2026 | NUS Show Lab & ByteDance

**核心洞察：** 预训练 Reward 模型存在固有偏差（PickScore 降低图像质量，OCR Reward 降低美学）。Adv-GRPO 将图像生成建模为**对抗博弈**，用视觉基础模型（DINO）提供密集信号。

**对抗性奖励机制：** 参考图像（正样本）→ 判别器训练 → 判别器作为 RM → GRPO Loss 优化生成器

| 对比 | 图像质量胜率 | 美学胜率 |
|---|---|---|
| Adv-GRPO vs Flow-GRPO | 70.0% | 72.4% |
| Adv-GRPO vs SD3 | 70.0% | 72.4% |

**核心结论：** 学习型奖励以参考图像为正样本监督，直接通过视觉输出引导生成器。DINO 等视觉基础模型提供密集信号，取代易被 hack 的标量奖励。

---

### 3.5 工程实践层面常用策略

| 策略 | 具体做法 | 效果 | 局限 |
|---|---|---|---|
| 多 Reward 简单加权 | R = αR₁ + βR₂ | 部分缓解 Hacking | 仍基于点式分数，虚假优势根因未解决 |
| KL 散度惩罚 | β·KL(π_θ, π_ref) | 防止策略过度更新 | 限制探索能力，不解决多 Reward 不均衡 |
| 奖励模型集成 | 多个 RM 投票/平均 | 降低单个 RM 偏差 | 计算开销大，仍基于点式分数范式 |
| 训练调度策略 | 早停 / 系数衰减 / 课程学习 | 减缓 Hacking 发生速度 | 治标不治本 |

### 3.6 方法综合对比与选型建议

| 维度 | MO-GRPO | PREF-GRPO | Adv-GRPO |
|---|---|---|---|
| 防 Hacking 层级 | 权重归一化 | 奖励范式变革 | 对抗博弈机制 |
| 实现复杂度 | 低（改动最小） | 中（需 PPRM） | 高（训练判别器） |
| 额外计算开销 | 几乎无 | O(G²) 成对比较 | 判别器训练开销 |
| 仿射变换不变性 | ✓ | N/A | N/A |
| 需手动调参 | 否 | 需选 PPRM | 需选参考图像+判别器 |
| 图像领域专用 | 否（通用） | 是 | 是 |
| 风格定制能力 | 否 | 否 | 是 |

防 Hacking 效果等级：**简单加权 < KL 惩罚 < RM 集成 < MO-GRPO < PREF-GRPO < Adv-GRPO**

**选型建议：**
- **快速修复 / 通用场景：** MO-GRPO。改动最小，即插即用，适用于任何多 Reward GRPO 场景
- **图像生成专用 / 追求稳定训练：** PREF-GRPO。针对 T2I 虚假优势设计，需配备 PPRM（如 UnifiedReward-Think）
- **追求最强防 Hacking / 需要风格定制：** Adv-GRPO。需额外训练判别器 + 参考图像集
- **工程落地组合拳：** MO-GRPO + KL 惩罚 + 早停（兼顾效果和实现成本）

---

## 4. 图像编辑专用 Reward Model 训练方案

> 目标：基于 Bradley-Terry 损失函数，为图像编辑模型训练一个可独立评分的 Critic Model

### 4.1 Bradley-Terry 损失基础

BT 模型将偏好比较建模为概率分布。给定两个输出 y_w（chosen）和 y_l（rejected），BT 损失为：

```
L_BT = −E_{(x,y_w,y_l)} [ log σ(r(x,y_w) − r(x,y_l)) ]
```

在图像编辑场景，输入为三元组 (I_src, P, I_edit)：

```
L_BT = −E [ log σ(r(I_src, P, I_edit,w) − r(I_src, P, I_edit,l)) ]
```

**BT 损失核心价值：** 获得可复用的标量评分函数，用于数据筛选、在线 RL（PPO/GRPO）、模型评估等多场景。DPO 虽也基于 BT 模型，但绕过了显式 RM 训练，无法输出独立分数。

---

### 4.2 技术方案全景对比

| 方案 | 损失函数 | 模型架构 | 数据量 | GPU 资源 | 训练显式 RM | 推荐度 |
|---|---|---|---|---|---|---|
| EditReward | 多维度不确定性排序损失（BT 变体） | VLM 7B + MLP Head | 200K 偏好对 | 8×A800 | ✅ | ⭐⭐⭐⭐⭐ |
| ImageReward + ReFL | BT 损失（标准） | BLIP + MLP Head | 137K 偏好对 | 4×A100 | ✅ | ⭐⭐⭐⭐ |
| SPIE (RLAIF) | DPO（BT 隐式） | SD v1.5 U-Net | ~5 张参考图 | 1~2×A100 | ❌ | ⭐⭐⭐ |
| HPSv2/v3 | BT 损失 + 概率建模 | CLIP/BLIP2 | 800K+ 偏好对 | 4~8×A100 | ✅ | ⭐⭐⭐ |
| Classification RM | 交叉熵（二分类） | VLM 7B + [CLS] Head | 同 BT 方案 | 同 BT 方案 | ✅（分类器） | ⭐⭐⭐ |

**快速决策：**
- 需要显式 Critic Model → **EditReward 方案**（专为图像编辑，SOTA，200K 开源数据）
- 资源有限且只需对齐 → **SPIE 方案**（RLAIF，5 张图 + 10 步，但无显式 RM）
- 需要成熟的文生图 RM → **ImageReward 方案**（NeurIPS 2023，开箱即用，但非编辑专用）

---

### 4.3 方案一：EditReward（当前 SOTA）

arXiv:2509.26346 | github.com/TIGER-AI-Lab/EditReward

**模型架构：**

| 组件 | 型号 | 参数量 |
|---|---|---|
| 骨干 VLM（最优） | MiMo-VL-7B-SFT-2508 | 7B |
| 骨干 VLM（备选） | Qwen2.5-VL-7B-Instruct | 7B |
| 骨干 VLM（轻量） | Qwen2.5-VL-3B-Instruct | 3B |
| 奖励头 | Multi-Dim MLP Head | ~数 M |

**损失函数：** 多维度不确定性感知排序损失（BT 扩展）。输出两维度（指令遵循 + 视觉质量）独立高斯分布，以聚合均值 + 不确定性 σ² 计算偏好概率。

**训练超参数：** lr = 2×10⁻⁶ · Cosine Scheduler · Warmup 0.05 · Batch 16 · 2 Epochs · 图像 448×448 · 全量解冻

**核心评估结果：**

| 方法 | GenAI-Bench | AURORA-Bench |
|---|---|---|
| GPT-4o | 53.54 | 50.81 |
| GPT-5 | 59.61 | 47.27 |
| EditReward (Qwen2.5-VL-7B) | **63.97** | **59.50** |
| EditReward (MiMo-VL-7B) | **65.72** | **63.62** |

**核心发现：** Top 20K 高质量筛选数据训练效果超全量 46K — 数据质量远比数量重要。

---

### 4.4 方案二：ImageReward + ReFL

NeurIPS 2023 | arXiv:2304.05977 | github.com/zai-org/ImageReward

架构：BLIP + MLP Head，输出近似标准正态分布 N(0,1) 的标量奖励分数，标准 BT 损失训练，137K 专家比较对。

ReFL（Reward Feedback Learning）直接用 ImageReward 梯度信号微调 Stable Diffusion，胜率 58.4% vs 原始 SD。有后续升级版 VisionReward（2024.12）。

**局限：** 面向文生图而非编辑；BLIP 骨干较老，对复杂编辑指令理解弱；单维度评分。

---

### 4.5 方案三：SPIE（RLAIF + DPO）

ICCV 2025W | arXiv:2504.12833

通过 DPO 数学变换将 BT 中的奖励函数表达为策略与参考策略的比值，**消除训练显式 RM 的需要**。

**AI 反馈评分组件：**

| 评分组件 | 模型 | 评分维度 |
|---|---|---|
| 结构评分 | Depth Anything V2 | 输入-编辑图像深度图 L1 距离 |
| 语义评分 | DreamSim + grounded-SAM2 | 编辑区域语义对齐 + 非编辑区保持 |

**极简训练需求：** 参考图像 5 张 · 训练步数 10 步 · 人类标注零成本

**注意：** 此方案无显式 RM，不能作为 Critic 复用、不能输出标量分数用于数据筛选，仅适合直接对齐编辑模型的场景。

---

### 4.6 方案四：HPSv2/v3

arXiv:2306.09341 | HPD v2 数据集：800K+ 偏好对（目前最大规模）

架构：CLIP/BLIP2 骨干 + BT 损失 + 概率建模。数据量最大，泛化性最强，但面向文生图而非编辑场景，对编辑指令理解有限。

---

### 4.7 方案五：Classification RM

将 (chosen, rejected) 对输入，输出 chosen 被偏好的概率（二分类，交叉熵损失）。

| 维度 | BT Reward Model | Classification RM |
|---|---|---|
| 输出 | 标量奖励分数 r(x,y) | 偏好概率 p(chosen) |
| 输入 | 单个 (x,y) 对 | (x, y_w, y_l) 三元组 |
| 后续用途 | PPO 训练、数据筛选 | 偏好判断、排序 |
| 灵活性 | 可独立评分任意样本 | 需成对比较 |

**结论：** Classification RM 性能不输 BT RM，且实现更简单。但**不能输出独立标量分数，不适合作为 PPO 的 Critic Model**。

---

### 4.8 数据准备

**偏好数据集构建 Pipeline（5 步）：**

1. **收集源数据** — (源图像, 编辑指令) 对；来源：EmuEdit / MagicBrush / AnyEdit + 内部数据
2. **多模型生成候选** — 用多个 SOTA 模型（Step1X-Edit, Flux-Kontext, Qwen-Image-Edit 等）对每对生成 10~12 个候选
3. **人工偏好标注** — 4-point Likert 量表（1=Poor, 4=Excellent），标注指令遵循 + 视觉质量双维度
4. **构建偏好对** — 同一指令下两两比较，构建 (chosen, rejected) 对
5. **数据质量控制** — 过滤标注者间一致性低的数据；处理平局样本

**各方案数据量需求：**

| 方案 | 偏好对数量 | 标注维度 | 标注来源 | 数据是否开源 |
|---|---|---|---|---|
| EditReward | 200K | 2 维（IF + VQ） | 人工（4-point Likert） | ✅ EditReward-Data |
| ImageReward | 137K | 1 维 | 专家比较 | ✅ ImageRewardDB |
| HPSv2 | 800K+ | 1 维 | 众包标注 | ✅ HPD v2 |
| SPIE | ~无需 | 2 维（结构+语义） | AI 反馈 | N/A |

**规模建议：** 最小可行 10K~20K · 推荐 50K~100K · 理想 200K+  
**关键发现：** 数据质量远比数量重要（EditReward 验证：Top 20K 筛选数据 > 全量 46K）

**低成本数据构建策略：**

| 策略 | 成本 | 周期 | 说明 |
|---|---|---|---|
| 利用开源数据集 | 最低 | 1~2 周 | EditReward-Data（200K）或 ImageRewardDB（137K）+ 少量领域补充 |
| AI 标注 + 人工校验 | 中等 | 2~3 周 | GPT-4o/VLM 初筛，仅对不确定样本人工校验，降低标注成本 70~80% |
| 全人工标注 | 最高 | 3~5 周 | 最高质量，按 EditReward 流程，3~5 名标注者 |

---

### 4.9 训练框架选型

| 维度 | HuggingFace TRL | OpenRLHF | 自建（PyTorch + HF） |
|---|---|---|---|
| 定位 | 研究友好，轻量级 | 生产级，高性能 | 完全可控，最灵活 |
| RM 训练 | ✅ RewardTrainer | ✅ train_rm.sh | ✅ 手动实现 |
| BT 损失 | ✅ 内置支持 | ✅ 通过 RM 训练 | ✅ 自行编写 |
| 视觉模型支持 | ⚠️ 需适配 | ✅ v0.10 原生支持 VLM | ✅ 完全自由 |
| 大规模（70B+） | ❌ | ✅ 核心优势 | ⚠️ |
| 上手难度 | ⭐ 低 | ⭐⭐ 中 | ⭐⭐⭐ 高 |
| 推荐场景 | 快速实验、7B 以下 | 7B~70B+ 生产训练 | 特殊架构/自定义损失 |

---

### 4.10 训练资源估算

**显存需求（7B VLM，全量微调）：**

| 组成 | 7B 全量微调 | 7B LoRA (r=16) |
|---|---|---|
| 模型参数 | ~28 GB (FP32) | ~14 GB (BF16) |
| 梯度 | ~28 GB | ~0.5 GB |
| 优化器状态（Adam） | ~56 GB | ~1 GB |
| 激活值 | ~10~20 GB | ~10~20 GB |
| **合计** | **~120~130 GB** | **~26~36 GB** |

**各方案 GPU 资源需求：**

| 方案 | GPU 配置 | 训练时长 | 月成本（参考） |
|---|---|---|---|
| EditReward (7B 全量) | 8×A800-80G | ~1~2 天 | ¥15,000~30,000 |
| EditReward (7B LoRA) | 2~4×A100-80G | ~1~3 天 | ¥3,000~8,000 |
| ImageReward (BLIP) | 2~4×A100-40G | ~0.5~1 天 | ¥1,500~4,000 |
| SPIE (SD v1.5) | 1~2×A100-40G | 分钟级（10 步） | ¥100~500 |

**资源缩减策略：** LoRA（显存降至 1/4~1/5）· 梯度检查点（节省 30~40% 激活显存）· DeepSpeed ZeRO-3（多卡参数分片）

---

### 4.11 推荐实施路线

**Phase 1：Baseline 搭建（1~2 周）**  
用 ImageReward 预训练模型作为初始 Critic，验证 RM-in-the-loop 流程。TRL RewardTrainer + 标准 BT 损失，在 EditReward-Data 子集（20K）上微调。资源：2×A100-80G + LoRA，成本 < ¥3,000。

**Phase 2：编辑专用 RM 训练（2~3 周）**  
基于 EditReward 架构（Qwen2.5-VL-7B + 多维度 MLP Head），使用 EditReward-Data 全量 200K 或自建数据，训练多维度不确定性感知排序损失（BT 变体）。资源：8×A800 全量 或 4×A100 + LoRA。

**Phase 3：RM 驱动的编辑模型对齐（2~4 周）**  
① 数据筛选：用 RM 从大规模数据中筛选高质量子集  
② 在线 RL：用 OpenRLHF 的 PPO/GRPO 以 RM 为奖励信号训练编辑模型  
③ 持续迭代：RM 评估 → 发现 Bad Case → 补充数据 → 重新训练 RM

**关键风险：**
- 7B VLM 全量微调显存需求大（~120GB+），必须多卡 + ZeRO-3 或 LoRA
- 标注质量直接影响 RM 性能，标注者间一致性 α < 0.5 的数据建议丢弃
- RM 可能存在 reward hacking 风险，需在 held-out 数据上持续监控
- BT 模型假设偏好可传递，对存在循环偏好的数据需特殊处理

---

## 5. 参考文献

1. **MO-GRPO** — "Mitigating Reward Hacking of Group Relative Policy Optimization in Multi-Objective Settings", arXiv:2509.22047, 2025.09
2. **PREF-GRPO** — 复旦大学 & 腾讯，"基于成对偏好奖励的 GRPO 方法缓解文生图 Reward Hacking", 2025.09; arXiv:2508.20751
3. **Adv-GRPO** — "The Image as Its Own Reward: Reinforcement Learning with Visual Foundation Models at Scale", CVPR 2026, NUS Show Lab & ByteDance; arXiv:2511.20256
4. **DanceGRPO** — arXiv:2505.07818, 港大 & 字节跳动, 2025.05
5. **Flow-GRPO** — arXiv:2505.05470, 2025.05
6. **AR-GRPO** — arXiv:2508.06924, 快手 Klear 团队, 2025.08
7. **GRPO 原始** — DeepSeek-R1, "Group Relative Policy Optimization", 2025
8. **Dr. GRPO** — "Dr. GRPO: The Graceful GRPO", 2025
9. **UnifiedReward** — GitHub ⭐778, NeurIPS 2025
10. **EditReward** — "EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing", arXiv:2509.26346, github.com/TIGER-AI-Lab/EditReward
11. **ImageReward** — "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation", NeurIPS 2023, arXiv:2304.05977
12. **SPIE** — "Semantic and Structural Post-Training of Image Editing Diffusion Models", ICCV 2025W, arXiv:2504.12833
13. **HPSv2** — "Human Preference Score v2", arXiv:2306.09341
14. **OpenRLHF** — arXiv:2501.03262
15. **TRL** — huggingface.co/docs/trl
