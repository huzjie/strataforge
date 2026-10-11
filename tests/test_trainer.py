import unittest
from strataforge.config import Config
from strataforge.model.backends import get_backend
from strataforge.train.trainer import Trainer


class TestTrainer(unittest.TestCase):
    def test_skill_monotone(self):
        cfg = Config()
        backend = get_backend("mock", cfg)
        t = Trainer(cfg, backend=backend)
        hist = t.train(steps=4, delta=0.1, verbose=False)
        skills = [h[1]["skill"] for h in hist]
        self.assertEqual(skills, sorted(skills))
        self.assertGreater(skills[-1], skills[0])

    def test_acceptance_climbs(self):
        cfg = Config()
        backend = get_backend("mock", cfg)
        t = Trainer(cfg, backend=backend)
        hist = t.train(steps=4, delta=0.1, verbose=False)
        acc = [h[1]["acceptance_rate"] for h in hist]
        self.assertGreater(acc[-1], acc[0])


if __name__ == "__main__":
    unittest.main()
