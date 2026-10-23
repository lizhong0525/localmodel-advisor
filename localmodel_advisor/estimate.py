"""Core memory-estimation formulas for local LLM inference.

Everything here is an estimate with an explicit uncertainty range. The
formulas used are documented inline and in README.md:

Weight memory:
    params (in billions) * bytes_per_parameter(quantization)
    plus a small container/metadata overhead factor.

KV-cache memory (exact, when architecture is known):
    2 (K and V) * n_layers * n_kv_heads * head_dim * kv_dtype_bytes
    * context_tokens * batch_size

KV-cache memory (heuristic, when architecture is unknown):
    We fall back to a per-parameter ratio observed for common dense
    transformers (e.g. Llama-2-7B with MHA is ~7.5e-5 bytes/param/token
    at FP16). Because real architectures vary (MHA vs GQA vs MLA), the
    heuristic reports a wide range and is flagged as uncertain.

No tokens-per-second numbers are produced here: throughput depends on
hardware, backend, and driver state, and must come from real benchmarks
(see the `benchmarks` module).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from typing import Optional

GiB = 1024 ** 3


class EstimationError(ValueError):
    """Raised when user input cannot be turned into an estimate."""


@dataclass(frozen=True)
class QuantFormat:
    """A quantization format and its effective bytes per parameter.

    `bytes_per_param` includes typical tensor overhead (scales, block
    metadata) for GGUF-style formats, so it is slightly above the raw
    bit width divided by 8.
    """

    name: str
    bytes_per_param: float
    overhead_factor: float  # multiplicative container/runtime overhead
    note: str
    reference: str


QUANT_FORMATS = {
    "fp32": QuantFormat(
        "fp32", 4.0, 1.00,
        "Full precision. Rarely used for inference.",
        "https://huggingface.co/docs/transformers/main/en/quantization/overview",
    ),
    "fp16": QuantFormat(
        "fp16", 2.0, 1.00,
        "Half precision weights.",
        "https://huggingface.co/docs/transformers/main/en/quantization/overview",
    ),
    "bf16": QuantFormat(
        "bf16", 2.0, 1.00,
        "bfloat16 weights; same size as fp16.",
        "https://huggingface.co/docs/transformers/main/en/quantization/overview",
    ),
    "q8_0": QuantFormat(
        "q8_0", 1.06, 1.02,
        "GGUF 8-bit quant (~8.5 bits/param incl. block scales).",
        "https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md",
    ),
    "q6_k": QuantFormat(
        "q6_k", 0.80, 1.02,
        "GGUF 6-bit K-quant (~6.56 bits/param effective).",
        "https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md",
    ),
    "q5_k_m": QuantFormat(
        "q5_k_m", 0.68, 1.02,
        "GGUF 5-bit K-quant medium (~5.5 bits/param effective).",
        "https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md",
    ),
    "q4_k_m": QuantFormat(
        "q4_k_m", 0.58, 1.02,
        "GGUF 4-bit K-quant medium (~4.8 bits/param effective). Popular default.",
        "https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md",
    ),
    "q4_0": QuantFormat(
        "q4_0", 0.56, 1.02,
        "GGUF legacy 4-bit quant (~4.5 bits/param effective).",
        "https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md",
    ),
    "int8": QuantFormat(
        "int8", 1.0, 1.02,
        "Generic 8-bit integer quantization (bitsandbytes-style).",
        "https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes",
    ),
    "int4": QuantFormat(
        "int4", 0.5, 1.05,
        "Generic 4-bit integer quantization (GPTQ/AWQ-style); overhead varies by format.",
        "https://huggingface.co/docs/transformers/main/en/quantization/overview",
    ),
}

# KV cache dtype widths in bytes.
KV_DTYPES = {"fp32": 4, "fp16": 2, "bf16": 2, "q8_0": 1, "q4_0": 0.5}

# Heuristic KV bytes per parameter per token (FP16 cache), centered on
# Llama-2-7B MHA (~7.5e-5). The range covers typical GQA models (lower)
# and wide MHA models (higher).
KV_HEURISTIC_RATIO = 7.5e-5
KV_HEURISTIC_LOW = 2e-5    # aggressive GQA / MLA
KV_HEURISTIC_HIGH = 1.5e-4  # plain MHA, many layers

BACKEND_DOCS = {
    "llama.cpp": "https://github.com/ggml-org/llama.cpp",
    "ollama": "https://github.com/ollama/ollama",
    "vllm": "https://docs.vllm.ai/",
    "transformers": "https://huggingface.co/docs/transformers/",
    "lmstudio": "https://lmstudio.ai/docs",
    "exllamav2": "https://github.com/turboderp-org/exllamav2",
}

# Backend runtime overhead heuristics (GiB), flagged as rough.
BACKEND_OVERHEAD_GIB = {
    "llama.cpp": 0.5,
    "ollama": 0.8,
    "vllm": 1.5,       # CUDA graphs, block tables
    "transformers": 1.0,
    "lmstudio": 1.0,
    "exllamav2": 0.8,
    "unknown": 1.0,
}


@dataclass
class MemoryRange:
    """A memory estimate with low/central/high bounds, in GiB."""

    low_gib: float
    mid_gib: float
    high_gib: float
    assumption: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Architecture:
    """Optional transformer architecture details for exact KV math."""

    n_layers: Optional[int] = None
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    def is_complete(self) -> bool:
        return all(v is not None for v in (self.n_layers, self.n_kv_heads, self.head_dim))


@dataclass
class EstimateResult:
    weight_mem: MemoryRange
    kv_mem: MemoryRange
    total_mem: MemoryRange
    backend_overhead_gib: float
    backend: str
    vram_gib: float
    ram_gib: float
    fit_verdict: str
    bottlenecks: list = field(default_factory=list)
    caveats: list = field(default_factory=list)
    references: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _positive(value: float, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise EstimationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(v):
        raise EstimationError(f"{name} must be a finite number, got {v}")
    if v <= 0:
        raise EstimationError(f"{name} must be positive, got {v}")
    return v


def estimate_weights(params_b: float, quant: str) -> MemoryRange:
    """Weight memory in GiB for `params_b` billion parameters."""
    params_b = _positive(params_b, "params_b")
    key = quant.lower().strip()
    if key not in QUANT_FORMATS:
        known = ", ".join(sorted(QUANT_FORMATS))
        raise EstimationError(f"unknown quantization {quant!r}; known formats: {known}")
    fmt = QUANT_FORMATS[key]
    raw = params_b * 1e9 * fmt.bytes_per_param / GiB
    mid = raw * fmt.overhead_factor
    # File formats are well understood, but runtime duplication (e.g.
    # mmap vs copy, scratch buffers) adds a few percent of uncertainty.
    return MemoryRange(
        low_gib=round(raw, 2),
        mid_gib=round(mid, 2),
        high_gib=round(mid * 1.05, 2),
        assumption=(
            f"{fmt.name}: {fmt.bytes_per_param} bytes/param effective "
            f"({fmt.note})"
        ),
    )


def estimate_kv_cache(
    context_tokens: int,
    params_b: float,
    arch: Optional[Architecture] = None,
    kv_dtype: str = "fp16",
    batch_size: int = 1,
) -> MemoryRange:
    """KV-cache memory in GiB.

    Exact when `arch` is complete (layers, kv heads, head dim); otherwise
    a per-parameter heuristic with a wide range, because the true value
    depends on attention architecture (MHA/GQA/MLA) which the user did
    not supply.
    """
    context_tokens = int(_positive(context_tokens, "context_tokens"))
    params_b = _positive(params_b, "params_b")
    batch_size = int(_positive(batch_size, "batch_size"))
    key = kv_dtype.lower().strip()
    if key not in KV_DTYPES:
        known = ", ".join(sorted(KV_DTYPES))
        raise EstimationError(f"unknown kv dtype {kv_dtype!r}; known: {known}")
    dtype_bytes = KV_DTYPES[key]

    if arch is not None and arch.is_complete():
        for name, v in (("layers", arch.n_layers), ("kv_heads", arch.n_kv_heads), ("head_dim", arch.head_dim)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise EstimationError(f"architecture field {name} must be a positive integer, got {v!r}")
        per_token = (
            2 * arch.n_layers * arch.n_kv_heads * arch.head_dim * dtype_bytes
        )
        total = per_token * context_tokens * batch_size / GiB
        return MemoryRange(
            low_gib=round(total * 0.95, 2),
            mid_gib=round(total, 2),
            high_gib=round(total * 1.10, 2),
            assumption=(
                f"exact formula: 2 * {arch.n_layers} layers * {arch.n_kv_heads} "
                f"kv_heads * {arch.head_dim} head_dim * {dtype_bytes}B ({key}); "
                "range covers allocator/paging overhead"
            ),
        )

    # Architecture unknown: heuristic based on per-parameter ratio.
    ratio_scale = dtype_bytes / 2.0  # ratios are calibrated for fp16
    per_token_mid = params_b * 1e9 * KV_HEURISTIC_RATIO * ratio_scale
    per_token_low = params_b * 1e9 * KV_HEURISTIC_LOW * ratio_scale
    per_token_high = params_b * 1e9 * KV_HEURISTIC_HIGH * ratio_scale
    scale = context_tokens * batch_size / GiB
    return MemoryRange(
        low_gib=round(per_token_low * scale, 2),
        mid_gib=round(per_token_mid * scale, 2),
        high_gib=round(per_token_high * scale, 2),
        assumption=(
            "HEURISTIC: architecture unknown; estimated "
            f"{KV_HEURISTIC_LOW}-{KV_HEURISTIC_HIGH} bytes/param/token "
            f"scaled for {key} cache. Real value depends on attention "
            "type (MHA/GQA/MLA) and layer count - supply architecture "
            "details for an exact figure."
        ),
    )


def backend_overhead(backend: str) -> float:
    key = backend.lower().strip()
    return BACKEND_OVERHEAD_GIB.get(key, BACKEND_OVERHEAD_GIB["unknown"])


def judge_fit(total: MemoryRange, weights: MemoryRange, vram_gib: float, ram_gib: float) -> tuple[str, list]:
    """Decide where the model fits and what the bottleneck is."""
    bottlenecks = []
    vram = _positive(vram_gib, "vram_gib") if vram_gib else 0.0
    ram = _positive(ram_gib, "ram_gib") if ram_gib else 0.0

    if vram and total.high_gib <= vram * 0.85:
        verdict = "fits_gpu_comfortably"
    elif vram and total.mid_gib <= vram:
        verdict = "fits_gpu_tight"
        bottlenecks.append(
            "VRAM headroom is thin; context growth, CUDA graphs, or OS "
            "display memory may push you over. Expect partial offload risk."
        )
    elif vram and weights.mid_gib <= vram and total.high_gib > vram:
        verdict = "weights_fit_kv_overflow"
        bottlenecks.append(
            "Weights fit in VRAM but KV cache may not at this context "
            "length. Reduce context, quantize the KV cache (q8_0), or "
            "accept KV offload (slower)."
        )
    elif vram and weights.mid_gib > vram and (vram + ram) >= total.mid_gib:
        verdict = "partial_offload"
        bottlenecks.append(
            "Weights exceed VRAM: expect layer offloading to system RAM. "
            "This is usually the dominant bottleneck - generation speed "
            "drops sharply once layers spill (measure with a real benchmark)."
        )
    elif (vram + ram) >= total.high_gib:
        verdict = "cpu_only"
        bottlenecks.append(
            "Model only fits in combined VRAM+RAM or RAM alone: CPU-class "
            "inference. Memory bandwidth, not compute, is the bottleneck."
        )
    else:
        verdict = "does_not_fit"
        bottlenecks.append(
            "Estimated total memory exceeds VRAM + system RAM. Use a "
            "smaller model, a stronger quantization, or more memory."
        )

    if ram and total.high_gib > ram and not vram:
        bottlenecks.append(
            "No GPU reported and estimate exceeds RAM: the OS will swap, "
            "which makes inference unusable in practice."
        )
    return verdict, bottlenecks


def estimate(
    vram_gib: float,
    ram_gib: float,
    params_b: float,
    quant: str,
    context_tokens: int,
    backend: str = "unknown",
    kv_dtype: str = "fp16",
    batch_size: int = 1,
    arch: Optional[Architecture] = None,
) -> EstimateResult:
    """Full pipeline: weights + KV + backend overhead + fit verdict."""
    backend_key = backend.lower().strip() or "unknown"
    weights = estimate_weights(params_b, quant)
    kv = estimate_kv_cache(context_tokens, params_b, arch, kv_dtype, batch_size)
    overhead = backend_overhead(backend_key)

    total = MemoryRange(
        low_gib=round(weights.low_gib + kv.low_gib + overhead * 0.5, 2),
        mid_gib=round(weights.mid_gib + kv.mid_gib + overhead, 2),
        high_gib=round(weights.high_gib + kv.high_gib + overhead * 1.5, 2),
        assumption="weights + KV cache + backend runtime overhead",
    )
    verdict, bottlenecks = judge_fit(total, weights, vram_gib, ram_gib)

    caveats = [
        "No tokens-per-second claim is made: throughput depends on your "
        "exact hardware, driver, and backend build. Import a real "
        "benchmark (import <file.json>) for measured numbers.",
        f"Backend overhead for '{backend_key}' is a rough heuristic "
        f"({overhead} GiB); check the backend docs for exact settings.",
    ]
    if arch is None or not arch.is_complete():
        caveats.append(
            "KV cache used the heuristic path. Provide --layers, "
            "--kv-heads and --head-dim from the model's config.json "
            "for an exact figure."
        )

    references = {"quantization": QUANT_FORMATS[quant.lower().strip()].reference}
    if backend_key in BACKEND_DOCS:
        references["backend"] = BACKEND_DOCS[backend_key]

    return EstimateResult(
        weight_mem=weights,
        kv_mem=kv,
        total_mem=total,
        backend_overhead_gib=overhead,
        backend=backend_key,
        vram_gib=vram_gib,
        ram_gib=ram_gib,
        fit_verdict=verdict,
        bottlenecks=bottlenecks,
        caveats=caveats,
        references=references,
    )
