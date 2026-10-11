"""Trainer for the deterministic mock backend.

The only trainable parameter is the mock's `skill` (a quality knob standing in
for draft quality + routing-table quality). The training signal is deliberately
monotone -- every step pushes skill upward -- so the convergence curve is clean
and the benchmarks show acceptance / hit-rate / throughput all climbing.
"""
from ..model.backends import get_backend


class Trainer:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend or get_backend(cfg.backend.name, cfg)
        self.history = []

    def step(self, delta=0.1):
        """One monotone training step: skill always increases toward 1.0."""
        self.backend.skill = min(1.0, self.backend.skill + delta)
        # keep cfg in sync so a freshly-rebuilt engine sees the new skill
        self.cfg.backend.skill = self.backend.skill
        return self.backend.skill

    def evaluate(self):
        res = self.backend.generate("training probe", max_tokens=32)
        return {
            "skill": self.backend.skill,
            "acceptance_rate": res["spec"]["acceptance_rate"],
            "speedup": res["spec"]["speedup"],
            "gpu_hit_rate": res["tier"]["hit_rate"],
            "ngram_hit_rate": res["ngram"]["hit_rate"],
            "decode_tokens_per_s": res["decode_tokens_per_s"],
        }

    def train(self, steps=6, delta=0.1, verbose=True):
        initial = self.backend.skill
        self.history = [("init", self.evaluate())]
        for i in range(steps):
            self.step(delta)
            self.history.append(("step_%d" % (i + 1), self.evaluate()))
        if verbose:
            print("Train mock backend: skill %.3f -> %.3f" % (initial, self.backend.skill))
            first = self.history[0][1]
            last = self.history[-1][1]
            print("  acceptance  %.3f -> %.3f" % (first["acceptance_rate"], last["acceptance_rate"]))
            print("  speedup     %.3fx -> %.3fx" % (first["speedup"], last["speedup"]))
            print("  gpu hit     %.3f -> %.3f" % (first["gpu_hit_rate"], last["gpu_hit_rate"]))
            print("  ngram hit   %.3f -> %.3f" % (first["ngram_hit_rate"], last["ngram_hit_rate"]))
        return self.history
