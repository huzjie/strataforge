import unittest
from strataforge.experts.moe import MoELayer, ExpertRouter


class TestMoE(unittest.TestCase):
    def test_route_topk(self):
        r = ExpertRouter(100, 10, 32, seed=1)
        route = r.route(0)
        self.assertEqual(len(route), 10)
        ids = [e for e, _ in route]
        self.assertEqual(len(set(ids)), 10)

    def test_forward_shape(self):
        moe = MoELayer(64, 4, 8, 16, seed=2)
        x = [[0.1] * 8 for _ in range(3)]
        out = moe.forward(x)
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[0]), 8)


if __name__ == "__main__":
    unittest.main()
