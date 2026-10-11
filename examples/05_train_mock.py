"""strataforge example: train the mock

Run from the project root:
    python examples/05_train_mock.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.model.backends import get_backend
from strataforge.train.trainer import Trainer

cfg = load_config()
backend = get_backend("mock", cfg)
t = Trainer(cfg, backend=backend)
t.train(steps=6, delta=0.1)
