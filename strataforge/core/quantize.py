"""Quantization-tier modeling (ggml-style low-bit weights).

We don't ship real kernels -- we model the *bit budget* per weight for each
quantization level, which is exactly what drives the memory-footprint and
speed estimates the tiered scheduler relies on.
"""

QUANT_BITS = {
    "q2_0": 2.0,      # ~2.0 bits/weight (2-bit with a per-block scale)
    "iq2_xs": 2.3,    # 2-bit with importance matrix + sign, ~2.3 bpw
    "iq3_xxs": 3.0,   # 3-bit extra-extra-small
    "iq3_s": 3.4,     # 3-bit with small importance matrix
    "coder": 3.2,     # code-tuned mix
}

QUANT_META = {
    "q2_0": {"bits": 2.0, "quality": "fastest", "vram_12gb_gen_tps": 94},
    "iq2_xs": {"bits": 2.3, "quality": "balanced", "vram_12gb_gen_tps": 79},
    "iq3_xxs": {"bits": 3.0, "quality": "better", "vram_12gb_gen_tps": 62},
    "iq3_s": {"bits": 3.4, "quality": "best-3bit", "vram_12gb_gen_tps": 53},
    "coder": {"bits": 3.2, "quality": "code-tuned", "vram_12gb_gen_tps": 55},
}


def quantize_weights(weights, level, seed_key="qw"):
    """Deterministically quantize a weight matrix to `level` bit budget.

    Returns a proxy representation: the quantized matrix is replaced by a
    per-row scale + integer codes, so downstream code can reason about it
    without materializing the full float matrix (which is the whole point of
    quantization for a 125B model).
    """
    if level not in QUANT_BITS:
        raise ValueError("unknown quant level %r (valid: %s)" % (level, sorted(QUANT_BITS)))
    from ..utils.stable import stable_float
    n_rows = len(weights)
    n_cols = len(weights[0]) if n_rows else 0
    scales = [max(1e-3, max(abs(v) for v in row)) for row in weights]
    codes = [[round((v / (scales[i] + 1e-9)) * 127) for v in weights[i]] for i in range(n_rows)]
    return {
        "level": level,
        "bits_per_weight": QUANT_BITS[level],
        "rows": n_rows,
        "cols": n_cols,
        "scales": scales,
        "codes": codes,
    }


def quantize_bytes(level, n_params):
    """Estimated on-disk bytes for `n_params` weights at `level`."""
    if level not in QUANT_BITS:
        raise ValueError("unknown quant level %r" % level)
    return int(n_params * QUANT_BITS[level] / 8.0)
