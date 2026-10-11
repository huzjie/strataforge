"""strataforge example: tiered scheduling

Run from the project root:
    python examples/03_tiered_scheduling.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.engine.strata_engine import StrataEngine
from strataforge.model.strata_model import StrataModel

cfg = load_config()
model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                    cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size, seed=cfg.seed)
engine = StrataEngine(cfg, model=model)
engine.scheduler.skill = cfg.backend.skill
for i in range(200):
    engine._route_token(i)
s = engine.scheduler.stats()
print("GPU hot experts: %d / %d" % (s["hot_resident"], s["gpu_capacity"]))
print("hits=%d misses=%d  hit_rate=%.3f" % (s["hits"], s["misses"], s["hit_rate"]))
