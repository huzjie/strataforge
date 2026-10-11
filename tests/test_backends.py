import unittest
from strataforge.config import Config
from strataforge.model.backends import get_backend, list_backends


class TestBackends(unittest.TestCase):
    def test_all_registered(self):
        names = list_backends()
        for want in ("mock", "cpu", "openai", "vllm", "transformers"):
            self.assertIn(want, names)

    def test_mock_generate(self):
        cfg = Config()
        b = get_backend("mock", cfg)
        res = b.generate("hello", max_tokens=8)
        self.assertEqual(res["generated_tokens"], 8)
        self.assertIn("text", res)


if __name__ == "__main__":
    unittest.main()
