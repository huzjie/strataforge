# strataforge 技术篇：把 125B MoE 塞进 12GB 显存的完整拆解

> 这篇是「怎么实现」而不是「发生了什么」。看完你能照着做：分层放置策略、投机解码公式、量化经济学，三件套都有可复用的代码块和提示词模板。

## 一、问题是怎么来的

稀疏 MoE（Mixture-of-Experts）是当下前沿模型的标准架构：参数总量堆到千亿级，但每个 token 只激活其中一小撮专家。Qwen3.8-Flash-Next 是 24,576 个专家、每 token 激活 10 个。问题在于——**存储和路由**：全量模型要能放得下，路由要能把数据送到对的专家手里，而这两件事在显存受限的消费级硬件上天然打架。

Strata 的思路一句话：**别跟硬件限制较劲，把整台电脑每一块硅片都用起来**。GPU、内存条、SSD 本来就在那儿，把专家按冷热分开放。

## 二、核心技巧 1：专家冷热分层（Tiered Scheduling）

专家热度是**极度偏斜**的：少量专家吃掉了大多数 token（共享语法、常见搭配、高频词）。所以只需把最热的一小撮放 GPU，冷尾留在内存里让 CPU 就地算。

```
GPU  (12-24GB)  -> attention + routing + ~4,000 最热专家（LRU 缓存）
RAM  (32-64GB)  -> 全量 24,576 专家（CPU 用 AVX 就地算冷尾）
SSD  (29GB)     -> n-gram 查找表（每 token 只读几行）
```

**可复用代码块**（`experts/tier.py` 的核心）：

```python
def touch(self, expert_id):
    self._freq[expert_id] = self._freq.get(expert_id, 0) + 1
    if expert_id in self.hot:        # 已在 GPU 热层
        self.hot.get(expert_id)      # 刷新 LRU 顺序
        return Tier.GPU
    if stable_float("promote:...") < self.skill:   # 晋升决策
        self.hot.put(expert_id, self._freq[expert_id])  # 满则驱逐最久未用
        return Tier.GPU
    return Tier.RAM                 # 冷尾走 CPU
```

**可复用要点**：用「路由频率 + LRU」做放置，`~4,000 / 24,576 ≈ 16%` 的专家就能覆盖大多数路由事件。命中率就是全部经济学——命中 GPU 越快、回退 CPU 越慢。

## 三、核心技巧 2：投机解码（Speculative Decoding / MTP）

生成原本是串行的：一次前向出一个 token。投机解码是「猜-验」循环：小 MTP 头先猜 `γ` 个候选，主干一次前向批量验证，对了就收下、错了就停在第一个错。

**加速公式**（`draft/spec.py`，可直接套）：

```
tokens_per_pass = 1 + α * (γ - 1)
speedup         = (1 + α * (γ - 1)) / (1 + c * γ)
```

α 是接受率、γ 是草稿长度、c 是单 token 草稿成本。取 α=0.85、γ=2、c=0.05 → **1.68x**，正好落在 Strata 实测的 1.6-1.8x 区间。

**关键认知**：输出质量**不变**。验证方是权威主干，草稿 token 只有和真值一致才被接受。所以投机解码改的是「多快」，不是「说什么」。

**提示词模板**（`prompts/speculative_decoding_tutor.md`）：

```
Teach speculative decoding (MTP) with the speedup formula
`(1 + alpha*(gamma-1))/(1 + c*gamma)`. Explain why output quality is unchanged,
and why a tiny draft head beats a big model on latency.
```

## 四、核心技巧 3：量化经济学（Quantization）

低比特权重是 125B 模型能装进内存的前提。五档位宽对应不同磁盘/内存占位：

| 档位 | bpw | 125B 占盘 | 场景 |
|---|---|---|---|
| q2_0 | 2.0 | ~31GB | 最快 |
| iq2_xs | 2.3 | ~36GB | 均衡默认 |
| iq3_xxs | 3.0 | ~47GB | 质量更好 |
| iq3_s | 3.4 | ~53GB | 3-bit 顶配、长上下文 |
| coder | 3.2 | ~50GB | 代码剪裁 |

**可复用要点**：内存预算决定放置——`active_params` 必须塞进 GPU、`total_params` 必须塞进 RAM、n-gram 表按 SSD 尺寸算。`core/memory.tier_layout()` 一个函数算出三层切分和 `fits` 判定。

## 五、怎么验证「调度越好越快」

mock 后端只有一个可训练参数 `skill`（代表草稿质量 + 路由表质量），训练信号**单调**——每步都往上推，避免「对+1 错-1」把 skill 压到 0 的老坑：

```bash
python -m strataforge train
# acceptance  0.675 -> 0.850
# speedup     1.52x -> 1.68x
# gpu hit     0.750 -> 1.000
# ngram hit   0.500 -> 1.000
```

skill 一升，投机接受率、GPU 命中率、吞吐**全部单调爬**——这就是「更好的放置策略 → 更快」的端到端证明。

## 六、落地：OpenAI / Anthropic 兼容服务

零依赖 stdlib HTTP 服务，直接接 Cursor / Claude Code：

```bash
python -m strataforge serve --port 8080
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080   # Claude Code
```

## 七、五后端 + 完整工程

`mock`（可训练）/ `cpu` / `openai` / `vllm` / `transformers` 五后端；CLI / Docker / K8s / Helm / CI 齐备；15+ 提示词、10 个硬件量化配方、11 组单测。

## 一句话总结

**把一个超大规模 MoE 跑进消费级硬件，靠的不是更小的模型，而是把「冷热分层 + 投机验证 + 低比特量化」这三件事工程化**。这三件事每一件都可以独立复用——分层放置、加速公式、字节换算——拿去套你自己的推理引擎。
