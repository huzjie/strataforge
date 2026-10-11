"""Tiered expert scheduling: the hot/cold placement policy.

The whole machine is treated as one memory hierarchy:

    GPU  -> attention, routing, and the hottest ~N experts (fast, small)
    RAM  -> the full expert set (CPU computes the cold tail in parallel)
    SSD  -> the n-gram lookup table (read a few rows per token)

A router-frequency counter feeds an LRU: hot experts are promoted into the GPU
tier, cold experts are evicted. The mock backend wires `skill` into the hit-rate
so training shows the scheduler getting measurably better.
"""
from .lru import LRUCache


class Tier:
    GPU = "gpu"
    RAM = "ram"
    SSD = "ssd"


class TieredScheduler:
    def __init__(self, n_experts, gpu_capacity, skill=0.5):
        self.n_experts = n_experts
        self.gpu_capacity = gpu_capacity
        self.skill = skill
        self.hot = LRUCache(gpu_capacity)
        self.hits = 0
        self.misses = 0
        self._freq = {}

    def touch(self, expert_id):
        """Record a routing event for an expert; update the hot tier."""
        self._freq[expert_id] = self._freq.get(expert_id, 0) + 1
        if expert_id in self.hot:
            self.hot.get(expert_id)
            self.hits += 1
            return Tier.GPU
        # deterministic promotion: the mock's skill biases whether the expert
        # was pre-placed in the GPU tier (high skill -> better placement).
        from ..utils.stable import stable_float
        promoted = stable_float("promote:%d:%d" % (expert_id, self._freq[expert_id])) < self.skill
        if promoted:
            self.hot.put(expert_id, self._freq[expert_id])
            self.hits += 1
            return Tier.GPU
        self.misses += 1
        return Tier.RAM

    def hit_rate(self):
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate(),
            "hot_resident": len(self.hot),
            "gpu_capacity": self.gpu_capacity,
        }


def tier_stats(scheduler):
    return scheduler.stats()
