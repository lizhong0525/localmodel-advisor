# LocalModel Advisor

A transparent, offline memory estimator for people who want to run local language models: given your GPU VRAM, system RAM, model size, quantization, context length, and backend, it tells you whether the model fits, how much memory weights and KV cache need (with uncertainty ranges), and what the likely bottleneck is.

It deliberately does **not** promise tokens-per-second numbers. Throughput depends on your exact hardware and backend build; the tool instead lets you import real measured benchmarks and matches them to your configuration.

Pure Python standard library. No dependencies, no network calls, no telemetry.

## Quickstart

```bash
python -m localmodel_advisor estimate --vram 12 --ram 32 --params 7 --quant q4_k_m --context 4096 --backend llama.cpp
```

Requires Python 3.10+. No installation step: run it from the repo directory, or copy the `localmodel_advisor/` package anywhere.

## Demo walkthrough (uses the included fixtures)

1. **Estimate a 7B Q4_K_M model on a 12 GB GPU:**

   ```bash
   python -m localmodel_advisor estimate --vram 12 --ram 32 --params 7 --quant q4_k_m --context 4096 --backend llama.cpp
   ```

   Output shows weight memory (~3.9 GiB), KV cache (heuristic range, since no architecture was given), a fit verdict, bottlenecks, and caveats.

2. **Get an exact KV-cache figure by supplying the architecture** (values from the model's `config.json`):

   ```bash
   python -m localmodel_advisor estimate --vram 12 --ram 32 --params 7 --quant q4_k_m --context 4096 \
       --layers 32 --kv-heads 32 --head-dim 128 --html report.html
   ```

   Also writes a self-contained `report.html` you can open offline.

3. **Import community benchmarks** from the included fixture and see them matched:

   ```bash
   python -m localmodel_advisor import tests/fixtures/benchmarks_valid.json
   python -m localmodel_advisor benchmarks
   python -m localmodel_advisor estimate --vram 12 --ram 32 --params 7 --quant q4_k_m --context 4096 --backend llama.cpp
   ```

   The estimate now also lists measured results from similar setups (e.g. "38.5 tok/s on NVIDIA RTX 3060 12GB").

## How the estimates work

- **Weights:** `params × bytes_per_param(quant)` with effective bits including GGUF block scales/metadata (e.g. Q4_K_M ≈ 0.58 B/param), plus container overhead. Uncertainty: ±~5%.
- **KV cache (exact):** `2 × layers × kv_heads × head_dim × dtype_bytes × context × batch`. Used when you pass `--layers/--kv-heads/--head-dim`.
- **KV cache (heuristic):** without architecture details, a per-parameter ratio calibrated on Llama-2-7B (~7.5e-5 B/param/token at FP16) with a wide range (2e-5–1.5e-4) covering GQA/MLA vs plain MHA. Clearly flagged as heuristic.
- **Backend overhead:** rough per-backend heuristic (e.g. vLLM's CUDA graphs and block tables cost more than llama.cpp). Flagged as approximate.
- **Fit verdicts:** comfortable / tight / KV-overflow / partial-offload / CPU-only / does-not-fit, each with an explanation of the dominant bottleneck.

## Features

- CLI: `estimate`, `import`, `benchmarks`, `formats`
- Weight + KV-cache memory with low/central/high ranges and the assumption behind each number
- Exact KV math when architecture is supplied; honest heuristic otherwise
- Bottleneck analysis (VRAM headroom, KV overflow, offload, memory bandwidth)
- Community benchmark import (JSON, validated) with hardware metadata; matching records shown alongside estimates
- Optional self-contained offline HTML report (`--html report.html`)
- Links to official quantization/backend documentation
- Fully offline; pure stdlib; runs on Windows/macOS/Linux

## Benchmark file format

A JSON array (or `{"benchmarks": [...]}`) of records:

```json
{
  "gpu": "NVIDIA RTX 3060 12GB",
  "vram_gb": 12,
  "ram_gb": 32,
  "model": "Llama-2-7B-chat",
  "params_b": 6.7,
  "quant": "q4_k_m",
  "backend": "llama.cpp",
  "context_tokens": 4096,
  "tokens_per_second": 38.5,
  "cpu": "Ryzen 5 5600X",
  "os": "Windows 11",
  "notes": "Full GPU offload",
  "date": "2024-11-02"
}
```

Required: `gpu`, `vram_gb`, `model`, `params_b`, `quant`, `backend`, `tokens_per_second`. Records are stored locally in `~/.localmodel_advisor/benchmarks.json` (override with `LOCALMODEL_ADVISOR_DB`).

## Known limitations

- KV heuristic range is wide; supply architecture details for a trustworthy figure.
- Backend overhead numbers are rough heuristics, not measured.
- No MoE-specific weight/KV handling yet (active-parameter vs total-parameter distinction).
- No multi-GPU modeling.
- Estimates assume a single model instance; speculative decoding / draft models are not modeled.

## Roadmap

1. Built-in architecture table for popular models (auto-fill layers/kv-heads/head-dim by model name).
2. Benchmark matching by GPU model and submission of aggregated stats (median tok/s per config).
3. MoE support (active vs total parameters) and multi-GPU tensor-parallel splits.

## Development

```bash
python -m pytest -q        # or: python -m unittest discover -s tests -v
```

## License

MIT — see [LICENSE](LICENSE).
