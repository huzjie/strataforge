"""Configuration handling with a YAML fallback for zero-dependency installs."""
import os
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "qwen3.8-flash-next"
    n_experts: int = 24576
    top_k: int = 10
    dim: int = 32
    hidden: int = 64
    n_layers: int = 4
    vocab_size: int = 512
    total_params_b: float = 125.0
    active_params_b: float = 6.2


@dataclass
class TierConfig:
    gpu_capacity_experts: int = 4000
    gpu_bytes: float = 12.0 * (1024 ** 3)
    ram_bytes: float = 64.0 * (1024 ** 3)
    ssd_bytes: float = 29.0 * (1024 ** 3)
    ngram_bytes: float = 29.0 * (1024 ** 3)
    eviction: str = "lru"


@dataclass
class SpecConfig:
    enabled: bool = True
    draft_len: int = 2
    draft_cost: float = 0.05
    acceptance_base: float = 0.5
    acceptance_gain: float = 0.35


@dataclass
class QuantConfig:
    level: str = "iq2_xs"
    levels: dict = field(default_factory=lambda: {
        "q2_0": 2.0, "iq2_xs": 2.3, "iq3_xxs": 3.0, "iq3_s": 3.4, "coder": 3.2,
    })


@dataclass
class BackendConfig:
    name: str = "mock"
    skill: float = 0.5
    api_key: str = ""
    base_url: str = ""
    model: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tier: TierConfig = field(default_factory=TierConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    max_tokens: int = 64
    seed: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(
            model=ModelConfig(**{k: v for k, v in (d.get("model") or {}).items() if k in ModelConfig.__dataclass_fields__}),
            tier=TierConfig(**{k: v for k, v in (d.get("tier") or {}).items() if k in TierConfig.__dataclass_fields__}),
            spec=SpecConfig(**{k: v for k, v in (d.get("spec") or {}).items() if k in SpecConfig.__dataclass_fields__}),
            quant=QuantConfig(**{k: v for k, v in (d.get("quant") or {}).items() if k in QuantConfig.__dataclass_fields__}),
            backend=BackendConfig(**{k: v for k, v in (d.get("backend") or {}).items() if k in BackendConfig.__dataclass_fields__}),
            max_tokens=int(d.get("max_tokens", 64)),
            seed=int(d.get("seed", 0)),
        )


def _load_yaml(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        from .utils.yamlish import parse
        return parse(text)


def load_config(path=None):
    """Load config from a YAML path; fall back to package defaults if absent."""
    if path is None:
        candidate = Path(__file__).parent / "configs" / "default.yaml"
        if candidate.exists():
            path = str(candidate)
    if path and os.path.exists(path):
        return Config.from_dict(_load_yaml(path))
    return Config()
