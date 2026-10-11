"""CLI: doctor / train / bench / serve / quantize / run."""
import argparse
import sys

from ..config import load_config, Config
from ..model.backends import get_backend, list_backends
from ..train.trainer import Trainer
from ..bench.benchmarks import run_all, summarize
from ..core.quantize import QUANT_BITS, quantize_bytes
from ..core.memory import tier_layout, GB


def _cmd_doctor(cfg):
    print("strataforge doctor")
    print("  model      %s (%d experts / top-%d / %.1fB total / %.1fB active)" %
          (cfg.model.name, cfg.model.n_experts, cfg.model.top_k,
           cfg.model.total_params_b, cfg.model.active_params_b))
    print("  backends   %s" % ", ".join(list_backends()))
    for name in list_backends():
        try:
            b = get_backend(name, cfg)
            print("    %-14s OK" % name)
        except Exception as e:
            print("    %-14s FAIL: %s" % (name, e))
    backend = get_backend(cfg.backend.name, cfg)
    res = backend.generate("doctor probe", max_tokens=8)
    print("  generate   %d tokens (skill %.2f, spec %.2fx, gpu hit %.2f)" %
          (res["generated_tokens"], res["skill"], res["spec"]["speedup"],
           res["tier"]["hit_rate"]))
    layout = tier_layout(cfg.model.total_params_b, cfg.model.active_params_b,
                         cfg.quant.level, cfg.tier)
    print("  memory     %s -> %.2f GB, fits=%s" %
          (cfg.quant.level, layout["total_bytes"] / GB, layout["fits"]))
    print("doctor OK")


def _cmd_train(cfg, steps, delta):
    backend = get_backend(cfg.backend.name, cfg)
    t = Trainer(cfg, backend=backend)
    t.train(steps=steps, delta=delta)


def _cmd_bench(cfg):
    results = run_all(cfg)
    print("strataforge benchmark")
    print(summarize(results))


def _cmd_serve(cfg, host, port):
    from ..serving.http import serve
    serve(cfg, host=host, port=port)


def _cmd_quantize(cfg):
    print("quantization levels (bits/weight -> on-disk for %.1fB params):" % cfg.model.total_params_b)
    for level, bpw in sorted(QUANT_BITS.items()):
        print("  %-10s %.1f bpw  %.2f GB" %
              (level, bpw, quantize_bytes(level, int(cfg.model.total_params_b * 1e9)) / GB))


def _cmd_run(cfg, prompt, max_tokens, speculation):
    backend = get_backend(cfg.backend.name, cfg)
    res = backend.generate(prompt, max_tokens=max_tokens, use_speculation=speculation)
    print(res["text"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="strataforge", description="Tiered heterogeneous inference engine")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--backend", default=None, help="backend name (mock/cpu/openai/vllm/transformers)")
    sub = ap.add_subparsers(dest="command")

    p_doctor = sub.add_parser("doctor", help="self-check all backends + one generation")
    p_train = sub.add_parser("train", help="train the mock backend's skill")
    p_train.add_argument("--steps", type=int, default=6)
    p_train.add_argument("--delta", type=float, default=0.1)
    p_bench = sub.add_parser("bench", help="run the benchmark suite")
    p_serve = sub.add_parser("serve", help="start the HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_quant = sub.add_parser("quantize", help="show quantization tiers")
    p_run = sub.add_parser("run", help="one-shot generation")
    p_run.add_argument("--prompt", default="Explain tiered inference")
    p_run.add_argument("--max-tokens", type=int, default=64)
    p_run.add_argument("--no-speculation", action="store_true")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend.name = args.backend

    if args.command == "doctor":
        _cmd_doctor(cfg)
    elif args.command == "train":
        _cmd_train(cfg, args.steps, args.delta)
    elif args.command == "bench":
        _cmd_bench(cfg)
    elif args.command == "serve":
        _cmd_serve(cfg, args.host, args.port)
    elif args.command == "quantize":
        _cmd_quantize(cfg)
    elif args.command == "run":
        _cmd_run(cfg, args.prompt, args.max_tokens, not args.no_speculation)
    else:
        ap.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
