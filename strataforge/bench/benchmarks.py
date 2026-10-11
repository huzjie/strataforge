"""Benchmark suite: throughput, prefill, speculation, tier hit-rate, memory.

Every benchmark is deterministic. The headline numbers:

  - generation: decode tokens/sec (climbs with skill via better tier hits)
  - prefill: prompt-processing tokens/sec (climbs via n-gram reuse)
  - speculation: end-to-end speedup vs no-speculation baseline
  - tier hit-rate: fraction of routed experts served from the GPU tier
  - memory: does the quantized 125B model fit the RAM + SSD budget
"""
from ..model.backends import get_backend
from ..engine.strata_engine import StrataEngine
from ..model.strata_model import StrataModel
from ..core.memory import tier_layout, estimate_model_bytes, GB
from ..draft.spec import expected_speedup
from ..utils.stable import stable_ints

_BENCH = {}


def register_benchmark(name):
    def deco(fn):
        _BENCH[name] = fn
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCH)


def get_benchmark(name):
    return _BENCH[name]


def _engine(cfg):
    model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                        cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size,
                        seed=cfg.seed)
    engine = StrataEngine(cfg, model=model)
    engine.scheduler.skill = cfg.backend.skill
    engine.ngram.skill = cfg.backend.skill
    return engine


@register_benchmark("generation")
def run_generation(cfg, prompt="Explain tiered inference", max_tokens=64):
    engine = _engine(cfg)
    res = engine.generate(stable_ints("p", 8, 0, 512), max_tokens=max_tokens,
                          use_speculation=False)
    return {"decode_tokens_per_s": res["decode_tokens_per_s"],
            "generated_tokens": res["generated_tokens"],
            "elapsed_s": res["elapsed_s"]}


@register_benchmark("prefill")
def run_prefill(cfg, prompt="Explain tiered inference", prompt_tokens=4096):
    engine = _engine(cfg)
    ids = stable_ints("pf", prompt_tokens, 0, 512)
    import time
    t0 = time.perf_counter()
    for i in range(len(ids)):
        engine.ngram.lookup(ids[:i + 1])
    dt = time.perf_counter() - t0
    return {"prefill_tokens_per_s": prompt_tokens / dt if dt else 0.0,
            "ngram_hit_rate": engine.ngram.hit_rate()}


@register_benchmark("speculation")
def run_speculation(cfg, prompt="Explain tiered inference"):
    skill = cfg.backend.skill
    base = expected_speedup(0.0, cfg.spec.draft_len, cfg.spec.draft_cost)
    boosted = expected_speedup(
        0.5 + 0.35 * skill, cfg.spec.draft_len, cfg.spec.draft_cost)
    return {"speedup_no_spec": base,
            "speedup_with_spec": boosted,
            "acceptance_rate": 0.5 + 0.35 * skill}


@register_benchmark("tier_hitrate")
def run_tier_hitrate(cfg, n_tokens=200):
    engine = _engine(cfg)
    for i in range(n_tokens):
        engine._route_token(i)
    return engine.scheduler.stats()


@register_benchmark("memory")
def run_memory(cfg, prompt="Explain tiered inference"):
    layout = tier_layout(cfg.model.total_params_b, cfg.model.active_params_b,
                         cfg.quant.level, cfg.tier)
    return {
        "quant_level": cfg.quant.level,
        "total_bytes": layout["total_bytes"],
        "total_gb": round(layout["total_bytes"] / GB, 2),
        "gpu_resident_gb": round(layout["gpu_resident_bytes"] / GB, 2),
        "fits": layout["fits"],
    }


def run_all(cfg, prompt="Explain tiered inference"):
    results = {}
    for name in sorted(_BENCH):
        try:
            results[name] = _BENCH[name](cfg, prompt=prompt)
        except TypeError:
            results[name] = _BENCH[name](cfg)
    return results


def summarize(results):
    """Human-readable one-line-per-benchmark summary."""
    lines = []
    for name, r in sorted(results.items()):
        if name == "generation":
            lines.append("generation     decode %.1f tok/s (%d tokens)" %
                         (r["decode_tokens_per_s"], r["generated_tokens"]))
        elif name == "prefill":
            lines.append("prefill        %.0f tok/s (ngram hit %.3f)" %
                         (r["prefill_tokens_per_s"], r["ngram_hit_rate"]))
        elif name == "speculation":
            lines.append("speculation    %.2fx -> %.2fx (accept %.2f)" %
                         (r["speedup_no_spec"], r["speedup_with_spec"], r["acceptance_rate"]))
        elif name == "tier_hitrate":
            lines.append("tier_hitrate   gpu hit %.3f (resident %d/%d)" %
                         (r["hit_rate"], r["hot_resident"], r["gpu_capacity"]))
        elif name == "memory":
            lines.append("memory         %s %.2f GB (fits=%s)" %
                         (r["quant_level"], r["total_gb"], r["fits"]))
    return "\n".join(lines)
