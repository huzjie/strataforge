import unittest
from strataforge.experts.ngram import NGramTable


class TestNGram(unittest.TestCase):
    def test_reuse_hits(self):
        t = NGramTable(n=3, skill=1.0)
        self.assertTrue(t.lookup([1, 2, 3]))
        self.assertTrue(t.lookup([1, 2, 3]))

    def test_hit_rate_monotone_in_skill(self):
        lo = NGramTable(n=3, skill=0.2)
        hi = NGramTable(n=3, skill=0.9)
        for i in range(200):
            lo.lookup([i % 7, (i + 1) % 7, (i + 2) % 7])
            hi.lookup([i % 7, (i + 1) % 7, (i + 2) % 7])
        self.assertGreater(hi.hit_rate(), lo.hit_rate())


if __name__ == "__main__":
    unittest.main()
