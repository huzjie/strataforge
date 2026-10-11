import unittest
from strataforge.config import Config
from strataforge.engine.strata_engine import StrataEngine
from strataforge.model.strata_model import StrataModel


class TestEngine(unittest.TestCase):
    def _cfg(self):
        cfg = Config()
        cfg.model.n_experts = 256
        cfg.model.top_k = 4
        return cfg

    def test_generate(self):
        cfg = self._cfg()
        model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                            cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size)
        engine = StrataEngine(cfg, model=model)
        res = engine.generate([1, 2, 3], max_tokens=16)
        self.assertEqual(res["generated_tokens"], 16)
        self.assertTrue(res["decode_tokens_per_s"] > 0)

    def test_spec_speedup(self):
        cfg = self._cfg()
        model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                            cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size)
        engine = StrataEngine(cfg, model=model)
        self.assertGreater(engine.verifier.speedup(0.9), engine.verifier.speedup(0.5))


if __name__ == "__main__":
    unittest.main()
