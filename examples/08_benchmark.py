"""strataforge example: benchmark

Run from the project root:
    python examples/08_benchmark.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.bench.benchmarks import run_all, summarize

cfg = load_config()
print(summarize(run_all(cfg)))
