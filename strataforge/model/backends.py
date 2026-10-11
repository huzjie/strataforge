"""Backend registry: mock (trainable) / cpu / openai / vllm / transformers.

Every backend exposes the same minimal surface used by the engine, trainer,
benchmarks and HTTP server:

    backend.skill                    # float quality knob (mock only, trainable)
    backend.generate(prompt, **kw)   # -> {"tokens": [...], "text": str, ...}
    backend.route(experts)           # -> tier placement list (optional)

The mock backend is fully deterministic and trainable: its `skill` controls
draft acceptance, tier hit-rate and n-gram reuse, so `train()` produces a clean
0.5 -> 1.0 convergence curve.
"""
import time

from ..engine.strata_engine import StrataEngine
from ..model.strata_model import StrataModel
from ..utils.stable import stable_ints, stable_vector

_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        cls.backend_name = name
        _BACKENDS[name] = cls
        return cls
    return deco


def list_backends():
    return sorted(_BACKENDS)


def get_backend(name, cfg):
    if name not in _BACKENDS:
        raise ValueError("unknown backend %r (available: %s)" % (name, list_backends()))
    return _BACKENDS[name](cfg)


def _ids(prompt):
    if isinstance(prompt, (list, tuple)):
        return list(prompt)
    return stable_ints("tok:" + str(prompt), 16, 0, 512)


@register_backend("mock")
class MockBackend:
    """Deterministic, trainable backend. The reference implementation target."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = float(cfg.backend.skill)
        self.model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                                 cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size,
                                 seed=cfg.seed)
        self.engine = StrataEngine(cfg, model=self.model)

    def generate(self, prompt, max_tokens=None, use_speculation=None):
        cfg = self.cfg
        # rebuild the engine each call so the current skill is reflected
        self.engine = StrataEngine(cfg, model=self.model)
        self.engine.scheduler.skill = self.skill
        self.engine.ngram.skill = self.skill
        res = self.engine.generate(_ids(prompt), max_tokens=max_tokens,
                                   use_speculation=use_speculation)
        res["text"] = "".join(chr(32 + (t % 95)) for t in res["tokens"])
        res["skill"] = self.skill
        return res

    def route(self, expert_ids):
        return [self.engine.scheduler.touch(e) for e in expert_ids]


@register_backend("cpu")
class CpuBackend:
    """Local CPU reference engine (same math as mock, non-trainable)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = float(cfg.backend.skill)
        self.model = StrataModel(cfg.model.dim, cfg.model.hidden, cfg.model.n_experts,
                                 cfg.model.top_k, cfg.model.n_layers, cfg.model.vocab_size,
                                 seed=cfg.seed)
        self.engine = StrataEngine(cfg, model=self.model)

    def generate(self, prompt, max_tokens=None, use_speculation=None):
        res = self.engine.generate(_ids(prompt), max_tokens=max_tokens,
                                   use_speculation=use_speculation)
        res["text"] = "".join(chr(32 + (t % 95)) for t in res["tokens"])
        res["skill"] = self.skill
        return res


@register_backend("openai")
class OpenAIBackend:
    """OpenAI-compatible remote backend (uses stdlib urllib, no deps)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = float(cfg.backend.skill)
        self.base_url = cfg.backend.base_url or "https://api.openai.com/v1"
        self.api_key = cfg.backend.api_key or ""
        self.model_name = cfg.backend.model or "gpt-4o-mini"

    def generate(self, prompt, max_tokens=None, use_speculation=None):
        import json
        import urllib.request
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        req = urllib.request.Request(
            "%s/chat/completions" % self.base_url.rstrip("/"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer %s" % self.api_key},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "tokens": _ids(text), "skill": self.skill, "raw": data}


@register_backend("vllm")
class VllmBackend:
    """vLLM OpenAI-compatible serving endpoint."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = float(cfg.backend.skill)
        self.base_url = cfg.backend.base_url or "http://127.0.0.1:8000/v1"
        self.model_name = cfg.backend.model or "qwen3.8-flash-next"

    def generate(self, prompt, max_tokens=None, use_speculation=None):
        import json
        import urllib.request
        body = {
            "model": self.model_name,
            "prompt": str(prompt),
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        req = urllib.request.Request(
            "%s/completions" % self.base_url.rstrip("/"),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["text"]
        return {"text": text, "tokens": _ids(text), "skill": self.skill}


@register_backend("transformers")
class TransformersBackend:
    """HuggingFace Transformers backend (lazy import, optional dependency)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.skill = float(cfg.backend.skill)
        self.model_name = cfg.backend.model or "Qwen/Qwen2.5-0.5B"
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline("text-generation", model=self.model_name, max_new_tokens=32)
        return self._pipe

    def generate(self, prompt, max_tokens=None, use_speculation=None):
        pipe = self._ensure()
        out = pipe(str(prompt), max_new_tokens=max_tokens or self.cfg.max_tokens)
        text = out[0]["generated_text"] if isinstance(out, list) else str(out)
        return {"text": text, "tokens": _ids(text), "skill": self.skill}
