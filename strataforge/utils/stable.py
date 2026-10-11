"""Deterministic pseudo-random helpers.

Every function is a pure function of its key: the same key always yields the
same value across runs and platforms. This is what makes the mock backend
reproducible and makes benchmarks comparable.
"""
import hashlib
import random


def _seed(key):
    """Deterministic 64-bit seed from an arbitrary key string."""
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def stable_seed(key):
    return _seed(key)


def stable_float(key, lo=0.0, hi=1.0):
    """Deterministic float in [lo, hi)."""
    rng = random.Random(_seed(key))
    return lo + (hi - lo) * rng.random()


def stable_ints(key, n, lo=0, hi=100):
    """Deterministic list of n ints in [lo, hi)."""
    rng = random.Random(_seed(key))
    return [rng.randrange(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    """Deterministic list of n floats in [lo, hi)."""
    rng = random.Random(_seed(key))
    return [lo + (hi - lo) * rng.random() for _ in range(n)]


def stable_choice(key, items):
    """Deterministic pick from a sequence."""
    rng = random.Random(_seed(key))
    return items[rng.randrange(len(items))]


def stable_bool(key, p=0.5):
    """Deterministic Bernoulli."""
    return stable_float(key) < p
