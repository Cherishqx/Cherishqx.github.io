# 通用 VLA 架构模板

> 参考：https://github\.com/sou350121/VLA\-Handbook/blob/main/theory/vla\-core/vla\_arch\.md

![Image](/具身智能学习/src/通用vla框架.png)

## Vision Encoder

**ViT：**

> ViT 本身**不是一个训练目标模型**，而是一个视觉 backbone：
>
> - 把图像切成 patch（如 16×16）
> - 每个 patch 当 token
> - 用 Transformer 做全局建模
>
> 👉 类似 NLP 里的 Transformer encoder

**SigLIP：**

> SigLIP 属于 CLIP 家族，但做了关键改进：
>
> - 图像 encoder \+ 文本 encoder
> - 学习 image\-text matching
> - 用 sigmoid loss（不是 softmax contrastive）
>
> 👉 本质：**对齐语义空间**

**DINOv2：视觉 Transformer 的自监督学习（现在有v3了）**

> DINOv2 = **self\-supervised ViT 表征学习**
>
> 训练方式：
>
> - teacher\-student self\-distillation（无标签）
> - 强数据增强一致性
> - 不依赖文本
>
> 👉 学的是：**视觉结构本身**

## Language Encoder

LLM/Gemma/Qwen，还挺多的

## State Encoder

MLP/Tokenizer

**疑问：**只用选MLP和Tokenizer中的一个吗？不应该先Tokenizer再MLP吗？

**回答：**机器人的状态是连续状态向量，可以直接投影到 LLM 的 embedding 空间，不需要先经过 Tokenizer。也有少数工作会对状态进行离散化处理，且主要用于与离散的 action tokenization 保持统一的训练框架。

## Multimodel Fusion

* **Cross\-Attention**
* **Interleave**

含义 A：层间的交错堆叠（Architecture\-level）

指 MoE 层与普通 Transformer Block 交替排列，而非把所有 MoE 堆在一起。

也就是说，网络结构可能是：`Transformer → MoE → Transformer → MoE → ...` 这样的交错方式，而不是前面全是 Transformer、后面全是 MoE。

含义 B：序列的交错排列（Data/Sequence\-level）

指多模态数据按时间步交错排列成一个统一序列。

即序列形式为：`[语音_t, 图像_t, 文本_t, 动作_t, 语音_t+1, 图像_t+1, ...]`，不同模态在时序上交错出现，而非按模态分块排列。

* **MoE（Mixture of Experts，混合专家模型）**

核心思想：用多个小型「专家网络」替代传统 Transformer 中庞大的密集前馈层（FFN），再通过一个「门控网络（Router/Gating Network）」动态决定每个 token 该由哪几个专家处理。

## Action Head

- **Token Head: **RT\-2, Fast           \-\-\-离散化为256bins
- **Diffusion Head: **RDT, Octo       \-\-\-DDPM去噪
- **Flow Head:** pi0, WALL\-OSS        \-\-\-Flow Matching
