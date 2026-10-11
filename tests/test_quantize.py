import unittest
from strataforge.core.quantize import QUANT_BITS, quantize_bytes, quantize_weights


class TestQuantize(unittest.TestCase):
    def test_bytes_scales_with_bits(self):
        self.assertLess(quantize_bytes("q2_0", 1000), quantize_bytes("iq3_s", 1000))

    def test_quantize_weights(self):
        w = [[0.5, -0.5], [1.0, -1.0]]
        q = quantize_weights(w, "q2_0")
        self.assertEqual(q["level"], "q2_0")
        self.assertEqual(q["rows"], 2)


if __name__ == "__main__":
    unittest.main()
