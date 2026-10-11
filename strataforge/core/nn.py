"""Neural-network building blocks with deterministic weights."""
from .tensor import matmul
from ..utils.stable import stable_vector


def _weight(rows, cols, seed_key):
    """Deterministic weight matrix (rows x cols), entries in [-1, 1]."""
    return [stable_vector("%s:%d" % (seed_key, r), cols) for r in range(rows)]


class Linear:
    def __init__(self, in_dim, out_dim, key="lin"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = _weight(out_dim, in_dim, key + ":w")
        self.b = stable_vector(key + ":b", out_dim)

    def forward(self, x):
        """x: (seq, in_dim) -> (seq, out_dim)."""
        out = []
        for row in x:
            out.append([sum(row[k] * self.w[j][k] for k in range(self.in_dim)) + self.b[j]
                        for j in range(self.out_dim)])
        return out


class SwiGLU:
    """SwiGLU expert: h = (x @ W1 * silu(x @ W3)) @ W2."""
    def __init__(self, dim, hidden, key="swiglu"):
        self.dim = dim
        self.hidden = hidden
        self.w1 = _weight(hidden, dim, key + ":w1")
        self.w3 = _weight(hidden, dim, key + ":w3")
        self.w2 = _weight(dim, hidden, key + ":w2")

    def forward(self, x):
        """x: (seq, dim) -> (seq, dim)."""
        out = []
        for row in x:
            h1 = [sum(row[k] * self.w1[j][k] for k in range(self.dim)) for j in range(self.hidden)]
            h3 = [sum(row[k] * self.w3[j][k] for k in range(self.dim)) for j in range(self.hidden)]
            # silu(x) = x * sigmoid(x)
            gate = [h3[j] / (1.0 + 2.718281828459045 ** (-h3[j])) for j in range(self.hidden)]
            act = [h1[j] * gate[j] for j in range(self.hidden)]
            out.append([sum(act[j] * self.w2[i][j] for j in range(self.hidden)) for i in range(self.dim)])
        return out


def silu(x):
    if isinstance(x, (list, tuple)):
        return [silu(v) for v in x]
    return x / (1.0 + 2.718281828459045 ** (-x))
