"""Speculative-decoding verification and speedup math.

The main model verifies `draft_len` guessed tokens in one pass. Correct guesses
are accepted; the first mismatch (or end of draft) is where the pass stops.
Output quality is identical to greedy decoding -- the main model decides every
token -- but each pass can emit more than one token.

Expected tokens per pass = 1 + alpha * (draft_len - 1), where alpha is the
acceptance rate. With draft cost c per guessed token, the speedup is:

    speedup = (1 + alpha * (draft_len - 1)) / (1 + c * draft_len)
"""
from ..utils.stable import stable_float


def expected_speedup(acceptance, draft_len=2, draft_cost=0.05):
    return (1.0 + acceptance * (draft_len - 1)) / (1.0 + draft_cost * draft_len)


class SpeculativeVerifier:
    def __init__(self, draft_len=2, draft_cost=0.05, acceptance_base=0.5, acceptance_gain=0.35):
        self.draft_len = draft_len
        self.draft_cost = draft_cost
        self.acceptance_base = acceptance_base
        self.acceptance_gain = acceptance_gain

    def acceptance_rate(self, skill):
        """Acceptance rate as a monotone function of the draft quality (skill)."""
        return self.acceptance_base + self.acceptance_gain * skill

    def verify(self, draft_tokens, ground_truth, pos, skill):
        """Verify `draft_tokens` against `ground_truth` starting at `pos`.

        Returns (accepted_count, accepted_tokens). Uses a deterministic
        per-position acceptance decision so the same run is reproducible.
        """
        accepted = []
        for k, tok in enumerate(draft_tokens):
            if pos + k >= len(ground_truth):
                break
            is_correct = (tok == ground_truth[pos + k])
            # deterministic acceptance: correct tokens are accepted with prob
            # tied to skill; wrong tokens are never accepted.
            if is_correct and stable_float("acc:%d:%d" % (pos, k)) < (0.6 + 0.4 * skill):
                accepted.append(tok)
            else:
                break
        return len(accepted), accepted

    def speedup(self, skill):
        return expected_speedup(self.acceptance_rate(skill),
                                self.draft_len, self.draft_cost)
