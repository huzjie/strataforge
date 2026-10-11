import unittest
import tempfile
import os
from strataforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.model.n_experts, 24576)
        self.assertEqual(c.model.top_k, 10)

    def test_yaml_roundtrip(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  n_experts: 1000\n  top_k: 4\nquant:\n  level: q2_0\n")
            path = f.name
        try:
            cfg = load_config(path)
            self.assertEqual(cfg.model.n_experts, 1000)
            self.assertEqual(cfg.quant.level, "q2_0")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
