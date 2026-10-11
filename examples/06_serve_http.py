"""strataforge example: serve HTTP

Run from the project root:
    python examples/06_serve_http.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from strataforge.config import load_config
from strataforge.serving.http import serve

cfg = load_config()
serve(cfg, host="127.0.0.1", port=8080)
