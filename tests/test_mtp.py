import unittest
from strataforge.draft.mtp import MTPDraftHead
from strataforge.draft.spec import SpeculativeVerifier, expected_speedup


class TestMTP(unittest.TestCase):
    def test_draft_len(self):
        d = MTPDraftHead(512, draft_len=2, seed=1)
        self.assertEqual(len(d.draft([1, 2, 3, 4, 5, 6, 7, 8])), 2)

    def test_speedup_monotone(self):
        self.assertLess(expected_speedup(0.5), expected_speedup(0.9))

    def test_verify_no_overshoot(self):
        v = SpeculativeVerifier(draft_len=2)
        n, acc = v.verify([7, 8], [1, 2, 7, 8, 9], 2, skill=1.0)
        self.assertLessEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
