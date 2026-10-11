"""The main model: attention + sparse MoE + output head (reference impl).

Dimensions are kept small in the reference build so everything runs on CPU in
seconds; the *economics* (125B / 24,576 experts / 10 active) are modeled by the
tier scheduler and memory budget, not by allocating real giant matrices.
"""
from ..core.nn import Linear, SwiGLU
from ..core.tensor import softmax, layernorm
from ..experts.moe import MoELayer
from ..utils.stable import stable_vector


class StrataModel:
    def __init__(self, dim=32, hidden=64, n_experts=24576, top_k=10, n_layers=4,
                 vocab_size=512, seed=0):
        self.dim = dim
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.seed = seed
        self.moe_layers = [MoELayer(n_experts, top_k, dim, hidden, seed=seed + i)
                           for i in range(n_layers)]
        self.head = Linear(dim, vocab_size, key="head")

    def embed(self, token_ids):
        return [stable_vector("emb:%d" % t, self.dim) for t in token_ids]

    def forward(self, token_ids, collect_routes=False):
        """Reference forward: embed -> (moe + norm) x layers -> logits."""
        x = self.embed(token_ids)
        all_routes = []
        for layer in self.moe_layers:
            if collect_routes:
                x, routes = layer.forward(x, collect_routes=True)
                all_routes.append(routes)
            else:
                x = layer.forward(x)
            x = layernorm(x)
        logits = self.head.forward(x)
        return logits, all_routes
