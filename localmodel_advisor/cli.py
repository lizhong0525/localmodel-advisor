"""Command-line interface for LocalModel Advisor."""

from __future__ import annotations

import argparse
import json
import sys

from .benchmarks import (
    BenchmarkError,
    find_matching,
    import_benchmarks,
    list_benchmarks,
)
from .estimate import (
    Architecture,
    EstimationError,
    QUANT_FORMATS,
    KV_DTYPES,
    estimate,
)
from .html_report import render_html

VERDICT_LABELS = {
    "fits_gpu_comfortably": "Fits in VRAM comfortably",
    "fits_gpu_tight": "Fits in VRAM (tight - little headroom)",
    "weights_fit_kv_overflow": "Weights fit; KV cache may overflow at this context",
    "partial_offload": "Requires partial offload to system RAM",
    "cpu_only": "CPU-only / mostly system RAM",
    "does_not_fit": "Does not fit in available memory",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localmodel-advisor",
        description="Transparent memory estimator for running local LLMs. Estimates, not benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    est = sub.add_parser("estimate", help="Estimate memory requirements")
    est.add_argument("--vram", type=float, required=True, help="GPU VRAM in GB (use 0 for CPU-only)")
    est.add_argument("--ram", type=float, required=True, help="System RAM in GB")
    est.add_argument("--params", type=float, required=True, help="Model parameters in billions (e.g. 7, 13, 70)")
    est.add_argument("--quant", required=True, help=f"Quantization: {', '.join(sorted(QUANT_FORMATS))}")
    est.add_argument("--context", type=int, required=True, help="Context length in tokens")
    est.add_argument("--backend", default="unknown", help="Inference backend (llama.cpp, ollama, vllm, transformers, lmstudio, exllamav2)")
    est.add_argument("--kv-dtype", default="fp16", help=f"KV cache dtype: {', '.join(sorted(KV_DTYPES))}")
    est.add_argument("--batch", type=int, default=1, help="Concurrent sequences (default 1)")
    est.add_argument("--layers", type=int, default=None, help="Transformer layers (from model config.json)")
    est.add_argument("--kv-heads", type=int, default=None, help="KV attention heads (num_key_value_heads)")
    est.add_argument("--head-dim", type=int, default=None, help="Attention head dimension")
    est.add_argument("--html", metavar="FILE", help="Also write an offline HTML report to FILE")
    est.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    est.add_argument("--no-benchmarks", action="store_true", help="Skip community benchmark lookup")

    imp = sub.add_parser("import", help="Import a benchmark JSON file into the local store")
    imp.add_argument("file", help="Path to benchmark JSON (array of records)")

    sub.add_parser("benchmarks", help="List imported community benchmarks")
    sub.add_parser("formats", help="List supported quantization formats")
    return parser


def _fmt_range(r) -> str:
    return f"{r.low_gib:7.2f} - {r.high_gib:7.2f} GiB (central: {r.mid_gib:.2f})"


def cmd_estimate(args) -> int:
    arch = Architecture(args.layers, args.kv_heads, args.head_dim)
    result = estimate(
        vram_gib=args.vram,
        ram_gib=args.ram,
        params_b=args.params,
        quant=args.quant,
        context_tokens=args.context,
        backend=args.backend,
        kv_dtype=args.kv_dtype,
        batch_size=args.batch,
        arch=arch if arch.is_complete() else None,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("=== LocalModel Advisor: memory estimate ===")
        print(f"Model: {args.params}B params, {args.quant}, context {args.context}, backend {result.backend}")
        print(f"Hardware: {args.vram} GB VRAM, {args.ram} GB RAM")
        print()
        print(f"Weights:   {_fmt_range(result.weight_mem)}")
        print(f"  basis:   {result.weight_mem.assumption}")
        print(f"KV cache:  {_fmt_range(result.kv_mem)}")
        print(f"  basis:   {result.kv_mem.assumption}")
        print(f"Backend overhead (heuristic): ~{result.backend_overhead_gib:.2f} GiB")
        print(f"TOTAL:     {_fmt_range(result.total_mem)}")
        print()
        print(f"Verdict: {VERDICT_LABELS.get(result.fit_verdict, result.fit_verdict)}")
        if result.bottlenecks:
            print("Likely bottlenecks:")
            for b in result.bottlenecks:
                print(f"  - {b}")
        print("Caveats:")
        for c in result.caveats:
            print(f"  - {c}")
        if result.references:
            print("References:")
            for k, v in result.references.items():
                print(f"  - {k}: {v}")

    if not args.no_benchmarks:
        records = list_benchmarks()
        matches = find_matching(records, args.params, args.quant, result.backend, args.vram)
        if matches:
            print()
            print(f"Community benchmarks matching this configuration ({len(matches)}):")
            for m in matches:
                ctx = f", ctx {m['context_tokens']}" if m.get("context_tokens") else ""
                print(
                    f"  - {m['tokens_per_second']} tok/s on {m['gpu']} "
                    f"({m['vram_gb']} GB VRAM{ctx}) [{m['backend']}, {m['model']}]"
                    + (f" -- {m['notes']}" if m.get("notes") else "")
                )
            print("  (measured results contributed by users; your hardware may differ)")

    if args.html:
        inputs = {
            "VRAM (GB)": args.vram, "RAM (GB)": args.ram,
            "Params (B)": args.params, "Quantization": args.quant,
            "Context": args.context, "Backend": args.backend,
            "KV dtype": args.kv_dtype, "Batch": args.batch,
        }
        if arch.is_complete():
            inputs.update({"Layers": args.layers, "KV heads": args.kv_heads, "Head dim": args.head_dim})
        try:
            with open(args.html, "w", encoding="utf-8") as fh:
                fh.write(render_html(result, inputs))
        except OSError as exc:
            print(f"error: cannot write HTML report to {args.html}: {exc}", file=sys.stderr)
            return 1
        print(f"\nHTML report written to {args.html}")
    return 0


def cmd_import(args) -> int:
    count, store = import_benchmarks(args.file)
    print(f"Imported {count} benchmark record(s) into {store}")
    return 0


def cmd_benchmarks(args) -> int:
    records = list_benchmarks()
    if not records:
        print("No benchmarks imported yet. Add some with: localmodel-advisor import <file.json>")
        return 0
    print(f"{len(records)} community benchmark record(s):")
    for r in records:
        ctx = f", ctx {r['context_tokens']}" if r.get("context_tokens") else ""
        print(f"  - {r.get('model', '?')} {r.get('quant', '?')} on {r.get('gpu', '?')}: "
              f"{r.get('tokens_per_second', '?')} tok/s [{r.get('backend', '?')}{ctx}]")
    return 0


def cmd_formats(args) -> int:
    print("Supported quantization formats:")
    for key in sorted(QUANT_FORMATS):
        f = QUANT_FORMATS[key]
        print(f"  {key:8s} {f.bytes_per_param:.2f} bytes/param  {f.note}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "estimate": cmd_estimate,
        "import": cmd_import,
        "benchmarks": cmd_benchmarks,
        "formats": cmd_formats,
    }
    try:
        return handlers[args.command](args)
    except (EstimationError, BenchmarkError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
