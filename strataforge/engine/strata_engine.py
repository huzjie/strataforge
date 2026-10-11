"""The main inference engine: wires MoE + tiering + speculation + n-gram together.

This is the object every backend ultimately serves through, and the thing the
benchmarks measure. It exposes a token-level `generate` that:
  1. routes each token through the sparse MoE,
  2. feeds the routed expert ids through the tiered scheduler (hot -> GPU),
  3. runs the n-gram table on prompt prefixes,
  4. optionally runs speculative decoding for a throughput gain,
  5. records per-token timing so throughput is measurable.
"""
import time
from ..experts.tier import TieredScheduler
from ..experts.ngram import NGramTable
from ..draft.mtp import MTPDraftHead
from ..draft.spec import SpeculativeVerifier


class StrataEngine:
    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        m = cfg.model
        self.scheduler = TieredScheduler(m.n_experts, cfg.tier.gpu_capacity_experts,
                                         skill=cfg.backend.skill)
        self.ngram = NGramTable(n=3, size_bytes=cfg.tier.ngram_bytes, skill=cfg.backend.skill)
        self.draft = MTPDraftHead(m.vocab_size, cfg.spec.draft_len, seed=cfg.seed)
        self.verifier = SpeculativeVerifier(cfg.spec.draft_len, cfg.spec.draft_cost,
                                            cfg.spec.acceptance_base, cfg.spec.acceptance_gain)

    def _route_token(self, token_idx):
        """Route one token through the MoE and the tiered scheduler."""
        route = self.model.moe_layers[0].router.route(token_idx) if self.model else None
        experts = [e for e, _ in route] if route else [token_idx % self.cfg.model.n_experts]
        for e in experts:
            self.scheduler.touch(e)

    def generate(self, token_ids, max_tokens=None, use_speculation=None):
        """Generate tokens deterministically (greedy over a stable world model).

        The 'world' is a deterministic next-token function; skill shifts how well
        the draft guesses it and how well the tier keeps hot experts resident.
        Returns a result dict with tokens, tier stats, spec stats, and timing.
        """
        if use_speculation is None:
            use_speculation = self.cfg.spec.enabled
        max_tokens = max_tokens or self.cfg.max_tokens
        skill = self.cfg.backend.skill

        out = list(token_ids)
        start = time.perf_counter()
        # prompt processing: run the n-gram table over sliding prefixes
        for i in range(len(token_ids)):
            self.ngram.lookup(token_ids[:i + 1])

        while len(out) < len(token_ids) + max_tokens:
            self._route_token(len(out) - len(token_ids))
            if use_speculation:
                draft = self.draft.draft(out)
                n, acc = self.verifier.verify(draft, self._world(out), len(out), skill)
                # accept guessed tokens, then emit the next true token
                out.extend(acc)
                if len(out) >= len(token_ids) + max_tokens:
                    break
                out.append(self._world(out)[len(out)])
            else:
                out.append(self._world(out)[len(out)])

        elapsed = time.perf_counter() - start
        gen_tokens = len(out) - len(token_ids)
        return {
            "tokens": out,
            "generated_tokens": gen_tokens,
            "elapsed_s": elapsed,
            "decode_tokens_per_s": gen_tokens / elapsed if elapsed else 0.0,
            "tier": self.scheduler.stats(),
            "ngram": self.ngram.stats(),
            "spec": {
                "enabled": use_speculation,
                "acceptance_rate": self.verifier.acceptance_rate(skill),
                "speedup": self.verifier.speedup(skill),
            },
        }

    def _world(self, token_ids):
        """Deterministic ground-truth continuation (the 'correct' next tokens).

        This is the reference sequence the draft tries to guess and the
        verifier checks against. It is a pure function of the context.
        """
        from ..utils.stable import stable_ints
        return list(token_ids) + stable_ints("world:%s" % str(token_ids[-16:]),
                                             512, 0, self.cfg.model.vocab_size)
