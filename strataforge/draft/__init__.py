"""Speculative decoding (MTP draft + verification)."""
from .mtp import MTPDraftHead
from .spec import SpeculativeVerifier, expected_speedup

__all__ = ["MTPDraftHead", "SpeculativeVerifier", "expected_speedup"]
