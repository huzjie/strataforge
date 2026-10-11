"""Core numerical primitives (zero-dependency)."""
from .tensor import Tensor, zeros, ones, matmul, softmax, layernorm
from .nn import Linear, SwiGLU
from .quantize import quantize_bytes, QUANT_BITS, quantize_weights
from .memory import MemoryBudget, estimate_model_bytes, tier_layout

__all__ = [
    "Tensor", "zeros", "ones", "matmul", "softmax", "layernorm",
    "Linear", "SwiGLU",
    "quantize_bytes", "QUANT_BITS", "quantize_weights",
    "MemoryBudget", "estimate_model_bytes", "tier_layout",
]
