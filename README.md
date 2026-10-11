# strataforge

**把 125B 稀疏 MoE 跑进一台 12GB 显存的游戏 PC** —— GPU / RAM / SSD 三级专家分层调度 + 投机解码的零依赖参考实现。

[![CI](https://github.com/huzjie/strataforge/actions/workflows/ci.yml/badge.svg)](https://github.com/huzjie/strataforge/actions)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Zero-dep](https://img.shields.io/badge/deps-zero-lightgrey)

## 这是什么

Qwen3.8-Flash-Next 是一个 **125B 参数、24,576 个专家**的稀疏 MoE，每个 token 只激活约 10 个专家。Strata 证明了它能在 **12GB 显存 + 64GB 内存**的普通游戏机上跑出 90+ token/s。核心不是魔法，而是三件事：

1. **专家分层调度** —— 把整台电脑当一条内存层次：GPU 放注意力 + 路由 + 最热的 ~4,000 个专家，RAM 放全量 24,576 个专家（CPU 就地算冷尾），SSD 放 29GB 的 n-gram 查找表。
2. **投机解码** —— 小 MTP 头猜词，主干一次前向批量验证，1.6-1.8x 端到端加速，输出逐字节不变。
3. **低比特量化** —— Q2_0 / IQ2_XS / IQ3_XXS / IQ3_S / Coder 五档，驱动内存占位与速度估算。

这个仓库把这三件事拆成**可复用、可运行、零依赖**的纯 Python 代码，配一个确定性可训练 mock 后端，让你先把「分层放置策略 + 投机加速 + 量化经济学」跑通、验证、再对接真 ggml 内核。

## 快速开始

```bash
pip install -e .                       # 零依赖，纯 Python

python -m strataforge doctor           # 自检：5 后端 + 一次生成 + 内存占位
python -m strataforge train            # 训练 mock：skill 0.5 -> 1.0
python -m strataforge bench            # 吞吐 / prefill / 投机 / 命中率 / 内存
python -m strataforge serve --port 8080   # OpenAI + Anthropic 兼容服务
```

## 为什么值得看

- **零依赖核心**：托管 Python 3.13（无 PyYAML/numpy）也能跑，经济学完全可审计。
- **确定性可训练 mock**：`skill` 一升，投机接受率、GPU 命中率、吞吐全部单调爬升——「调度越好就越快」这件事能端到端验证。
- **真·经济学**：内存预算、层级占位、量化字节数用的都是 125B / 24,576 的真实数字，结论可放大到真实内核。

## 可复用干货

- 三层放置策略 + LRU 晋升/驱逐（`experts/tier.py`）
- 投机解码闭式加速公式（`draft/spec.py`）
- 五档量化位宽与字节换算（`core/quantize.py`）
- OpenAI/Anthropic 兼容 stdlib 服务（`serving/http.py`）
- 15+ 提示词、10 个硬件/量化配方（`prompts/`、`recipes/`）

详见 [docs/architecture.md](docs/architecture.md) 与 [docs/INTRO.md](docs/INTRO.md)。

## License

MIT
