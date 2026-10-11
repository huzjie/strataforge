"""A tiny zero-dependency tensor library.

Just enough linear algebra to run a reference MoE forward pass and the
speculative-decoding verifier without numpy. All values are plain Python
floats so behaviour is fully deterministic.
"""


class Tensor:
    __slots__ = ("data", "shape", "_op")

    def __init__(self, data, shape=None):
        if data and not isinstance(data[0], (list, tuple)):
            data = [list(data)]
        self.data = [list(row) for row in data]
        if shape is None:
            rows = len(self.data)
            cols = len(self.data[0]) if rows else 0
            shape = (rows, cols)
        self.shape = tuple(shape)
        self._op = "leaf"

    @property
    def n_rows(self):
        return self.shape[0]

    @property
    def n_cols(self):
        return self.shape[1]

    @staticmethod
    def from_rows(rows):
        return Tensor(rows)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return "Tensor(shape=%r)" % (self.shape,)

    def to_list(self):
        return self.data


def zeros(rows, cols):
    return Tensor([[0.0] * cols for _ in range(rows)], (rows, cols))


def ones(rows, cols):
    return Tensor([[1.0] * cols for _ in range(rows)], (rows, cols))


def matmul(a, b):
    """Dense matrix multiply a (m x k) @ b (k x n) -> (m x n)."""
    m, k = a.shape
    kb, n = b.shape
    if k != kb:
        raise ValueError("matmul shape mismatch: %r vs %r" % (a.shape, b.shape))
    out = zeros(m, n)
    bd = b.data
    for i in range(m):
        ai = a.data[i]
        oi = out.data[i]
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += ai[p] * bd[p][j]
            oi[j] = acc
    return out


def add(a, b):
    if isinstance(b, (int, float)):
        return Tensor([[x + b for x in row] for row in a.data], a.shape)
    return Tensor([[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(a.data, b.data)], a.shape)


def mul(a, b):
    if isinstance(b, (int, float)):
        return Tensor([[x * b for x in row] for row in a.data], a.shape)
    return Tensor([[x * y for x, y in zip(r1, r2)] for r1, r2 in zip(a.data, b.data)], a.shape)


def softmax(x, dim=-1):
    """Softmax over the last dimension (a list of rows or a single row)."""
    if x and isinstance(x[0], (list, tuple)):
        return [softmax(list(row), dim=dim) for row in x]
    m = max(x)
    ex = [2.718281828459045 ** (v - m) for v in x]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def layernorm(x, eps=1e-5):
    """LayerNorm over the last dimension."""
    if x and isinstance(x[0], (list, tuple)):
        return [layernorm(list(row), eps=eps) for row in x]
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    inv = 1.0 / ((var + eps) ** 0.5)
    return [(v - mu) * inv for v in x]


def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])


def topk(x, k):
    order = sorted(range(len(x)), key=lambda i: x[i], reverse=True)
    return order[:k]
