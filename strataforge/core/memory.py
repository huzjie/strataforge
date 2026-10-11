"""Memory-budget math for the unified multi-tier hierarchy."""

GB = 1024 ** 3


class MemoryBudget:
    def __init__(self, gpu_bytes, ram_bytes, ssd_bytes):
        self.gpu = gpu_bytes
        self.ram = ram_bytes
        self.ssd = ssd_bytes

    @classmethod
    def from_config(cls, tier):
        return cls(tier.gpu_bytes, tier.ram_bytes, tier.ssd_bytes)

    def summary(self):
        return {
            "gpu_gb": round(self.gpu / GB, 2),
            "ram_gb": round(self.ram / GB, 2),
            "ssd_gb": round(self.ssd / GB, 2),
        }


def estimate_model_bytes(total_params_b, quant_level, bits_per_weight=None):
    """Total bytes for a model of `total_params_b` billion params at `level`."""
    from .quantize import QUANT_BITS
    n_params = total_params_b * 1e9
    bpw = bits_per_weight if bits_per_weight is not None else QUANT_BITS[quant_level]
    return int(n_params * bpw / 8.0)


def tier_layout(total_params_b, active_params_b, quant_level, tier):
    """Compute the three-tier layout: (gpu_bytes, ram_bytes, ssd_bytes).

    `tier` is a TierConfig (gpu_bytes / ram_bytes / ssd_bytes / ngram_bytes).
    """
    from .quantize import QUANT_BITS
    total_bytes = estimate_model_bytes(total_params_b, quant_level)
    # only the active experts + attention/routing need to live on the GPU;
    # the full expert set lives in RAM; the n-gram table lives on SSD.
    active_bytes = int(active_params_b * 1e9 * QUANT_BITS[quant_level] / 8.0)
    ngram_bytes = int(tier.ngram_bytes)  # n-gram table is sized to SSD
    return {
        "total_bytes": total_bytes,
        "gpu_resident_bytes": min(active_bytes, tier.gpu_bytes),
        "ram_resident_bytes": total_bytes,
        "ssd_ngram_bytes": ngram_bytes,
        "fits": total_bytes <= tier.ram_bytes and ngram_bytes <= tier.ssd_bytes,
    }
