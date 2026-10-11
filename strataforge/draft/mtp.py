"""MTP (multi-token prediction) draft head.

A tiny draft layer guesses the next few tokens; the main model then verifies
them in a single forward pass. The draft is cheap, so if its guesses are good
the whole generation speeds up.
"""
from ..utils.stable import stable_ints


class MTPDraftHead:
    def __init__(self, vocab_size, draft_len=2, seed=0):
        self.vocab_size = vocab_size
        self.draft_len = draft_len
        self.seed = seed

    def draft(self, context_ids):
        """Guess `draft_len` candidate tokens given the context.

        Deterministic: the same context yields the same draft. Correctness of a
        candidate is decided later by the verifier against the ground-truth
        sequence, which is what makes acceptance a function of skill.
        """
        return stable_ints("mtp:%d:%s" % (self.seed, str(context_ids[-8:])),
                           self.draft_len, 0, self.vocab_size)
