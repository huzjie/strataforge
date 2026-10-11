import unittest
from strataforge.experts.tier import TieredScheduler, Tier


class TestTier(unittest.TestCase):
    def test_hit_rate_bounds(self):
        s = TieredScheduler(1000, 100, skill=0.9)
        for i in range(50):
            s.touch(i % 10)
        self.assertTrue(0.0 <= s.hit_rate() <= 1.0)

    def test_skill_raises_hits(self):
        low = TieredScheduler(1000, 100, skill=0.1)
        high = TieredScheduler(1000, 100, skill=0.95)
        for i in range(100):
            low.touch(i % 20)
            high.touch(i % 20)
        self.assertGreaterEqual(high.hit_rate(), low.hit_rate())


if __name__ == "__main__":
    unittest.main()
