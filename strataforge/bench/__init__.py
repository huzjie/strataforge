"""Benchmark registry."""
from .benchmarks import (
    register_benchmark, list_benchmarks, get_benchmark,
    run_generation, run_prefill, run_speculation, run_tier_hitrate, run_memory,
    run_all, summarize,
)

__all__ = [
    "register_benchmark", "list_benchmarks", "get_benchmark",
    "run_generation", "run_prefill", "run_speculation", "run_tier_hitrate", "run_memory",
    "run_all", "summarize",
]
