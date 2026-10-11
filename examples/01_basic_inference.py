"""strataforge example: basic inference

Run from the project root:
    python examples/01_basic_inference.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.model.backends import get_backend

cfg = load_config()
backend = get_backend("mock", cfg)
res = backend.generate("Explain tiered inference in one line", max_tokens=32)
print("text:", res["text"])
print("skill: %.2f  speedup: %.2fx  gpu-hit: %.2f" %
      (res["skill"], res["spec"]["speedup"], res["tier"]["hit_rate"]))
