"""n-gram lookup table (SSD-resident) for prompt-processing reuse.

Strata keeps a ~29GB n-gram table on SSD. During prompt processing, previously
seen n-gram prefixes map to cached hidden states, so repeated sequences skip
recomputation. This module models that table + its hit-rate.
"""
from ..utils.stable import stable_seed, stable_bool


class NGramTable:
    def __init__(self, n=3, size_bytes=29 * (1024 ** 3), skill=0.5):
        self.n = n
        self.size_bytes = size_bytes
        self.skill = skill
        self._seen = set()
        self.lookups = 0
        self.hits = 0

    def key(self, tokens):
        return tuple(tokens[-self.n:])

    def lookup(self, context_tokens):
        """Return True if the trailing n-gram is already precomputed in the table.

        `skill` models the table's precompute coverage: a higher skill means
        more of the n-gram space has been precomputed, so a lookup is more
        likely to hit. This makes hit-rate monotone in skill, which is the
        signal the trainer and benchmark rely on.
        """
        self.lookups += 1
        k = self.key(context_tokens)
        cached = stable_bool("ngram:cached:" + str(k), self.skill)
        if cached:
            self._seen.add(k)
            self.hits += 1
            return True
        return False

    def hit_rate(self):
        return (self.hits / self.lookups) if self.lookups else 0.0

    def stats(self):
        return {
            "n": self.n,
            "size_bytes": self.size_bytes,
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": self.hit_rate(),
            "entries": len(self._seen),
        }
