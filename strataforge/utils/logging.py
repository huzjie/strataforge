"""Minimal logging helper."""
import logging


def get_logger(name="strataforge", level=None):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level or logging.INFO)
    return logger
