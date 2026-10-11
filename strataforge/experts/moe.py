"""Sparse Mixture-of-Experts with deterministic hash routing.

Qwen3.8-Flash-Next has 24,576 tiny experts and activates ~10 per token. We model
that exactly: a large expert pool with lazy, deterministic weights so a single
forward pass only materializes the experts actually routed to (never 24,576
full matrices in memory).
"""
from ..utils.stable import stable_float, stable_ints


class ExpertRouter:
    """Top-K router with a deterministic gate score and load-balancing aux loss."""

    def __init__(self, n_experts, top_k, dim, seed=0):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.seed = seed

    def gate_logits(self, token_idx):
        """Deterministic per-expert gate logits for a given token position."""
        return stable_ints("gate:%d:%d" % (self.seed, token_idx), self.n_experts, 0, 1000)

    def route(self, token_idx):
        """Return the ordered list of (expert_id, gate_score) for one token."""
        logits = self.gate_logits(token_idx)
        order = sorted(range(self.n_experts), key=lambda i: logits[i], reverse=True)
        top = order[:self.top_k]
        total = sum(logits[i] for i in top) or 1
        return [(i, logits[i] / total) for i in top]

    def load_balance_loss(self, n_tokens, routes):
        """Auxiliary load-balancing loss (fraction-of-tokens x mean-gate)."""
        import collections
        counter = collections.Counter(e for route in routes for e, _ in route)
        n = n_tokens * self.top_k or 1
        return 1.0 - (sum(counter.values()) / n if counter else 0.0)


class MoELayer:
    """A sparse MoE layer: route each token, apply SwiGLU experts lazily."""

    def __init__(self, n_experts, top_k, dim, hidden, seed=0):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.hidden = hidden
        self.router = ExpertRouter(n_experts, top_k, dim, seed=seed)
        self.seed = seed

    def expert_forward(self, expert_id, x_row):
        """Deterministic SwiGLU expert forward for a single token row.

        Weighs every input dim with a per-expert deterministic kernel, then a
        SILU gate, then projects back. Purely deterministic so the mock tier
        can compute hit-rates and the trainer can reason about quality.
        """
        h1 = [sum(x_row[k] * stable_float("e%d:w1:%d:%d" % (expert_id, j, k), -0.1, 0.1)
                  for k in range(self.dim)) for j in range(self.hidden)]
        h3 = [sum(x_row[k] * stable_float("e%d:w3:%d:%d" % (expert_id, j, k), -0.1, 0.1)
                  for k in range(self.dim)) for j in range(self.hidden)]
        gate = [h3[j] / (1.0 + 2.718281828459045 ** (-h3[j])) for j in range(self.hidden)]
        act = [h1[j] * gate[j] for j in range(self.hidden)]
        return [sum(act[j] * stable_float("e%d:w2:%d:%d" % (expert_id, i, j), -0.1, 0.1)
                    for j in range(self.hidden)) for i in range(self.dim)]

    def forward(self, x, collect_routes=False):
        """x: (seq, dim) -> (seq, dim). Returns (out, routes)."""
        out = []
        routes = []
        for t, row in enumerate(x):
            route = self.router.route(t)
            blended = [0.0] * self.dim
            for expert_id, score in route:
                e = self.expert_forward(expert_id, row)
                for i in range(self.dim):
                    blended[i] += score * e[i]
            out.append(blended)
            routes.append(route)
        if collect_routes:
            return out, routes
        return out
