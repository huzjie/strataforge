"""strataforge example: quantization tiers

Run from the project root:
    python examples/04_quantize.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.core.quantize import QUANT_BITS, quantize_bytes
from strataforge.core.memory import GB

total = int(125.0 * 1e9)
print("125B params at each quant level:")
for level, bpw in sorted(QUANT_BITS.items()):
    print("  %-10s %.1f bpw -> %.2f GB" % (level, bpw, quantize_bytes(level, total) / GB))
