import unittest
from strataforge.core.tensor import matmul, softmax, layernorm, Tensor


class TestTensor(unittest.TestCase):
    def test_matmul_shape(self):
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        c = matmul(a, b)
        self.assertEqual(c.shape, (2, 2))

    def test_softmax_sums_to_one(self):
        s = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(s), 1.0, places=6)

    def test_layernorm_zero_mean(self):
        y = layernorm([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(y) / len(y), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
