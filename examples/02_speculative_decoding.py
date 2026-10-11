"""strataforge example: speculative decoding

Run from the project root:
    python examples/02_speculative_decoding.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.model.backends import get_backend
from strataforge.draft.spec import expected_speedup

cfg = load_config()
backend = get_backend("mock", cfg)
no_spec = backend.generate("speed probe", max_tokens=32, use_speculation=False)
spec = backend.generate("speed probe", max_tokens=32, use_speculation=True)
print("no-spec tokens/s: %.1f" % no_spec["decode_tokens_per_s"])
print("with-spec tokens/s: %.1f" % spec["decode_tokens_per_s"])
print("acceptance: %.2f  expected speedup: %.2fx" %
      (spec["spec"]["acceptance_rate"], spec["spec"]["speedup"]))
