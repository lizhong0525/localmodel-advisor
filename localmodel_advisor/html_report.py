"""Optional offline HTML report generation (stdlib only)."""

from __future__ import annotations

import html
from datetime import datetime, timezone

from .estimate import EstimateResult

VERDICT_LABELS = {
    "fits_gpu_comfortably": ("Fits in VRAM comfortably", "#1a7f37"),
    "fits_gpu_tight": ("Fits in VRAM (tight)", "#9a6700"),
    "weights_fit_kv_overflow": ("Weights fit, KV cache may overflow", "#9a6700"),
    "partial_offload": ("Requires partial offload to RAM", "#cf222e"),
    "cpu_only": ("CPU-only / mostly RAM", "#cf222e"),
    "does_not_fit": ("Does not fit", "#cf222e"),
}


def render_html(result: EstimateResult, inputs: dict) -> str:
    """Render a self-contained, offline HTML report."""
    esc = html.escape
    label, color = VERDICT_LABELS.get(result.fit_verdict, (result.fit_verdict, "#57606a"))

    def mem_row(name: str, r, note: str = "") -> str:
        return (
            f"<tr><td>{esc(name)}</td>"
            f"<td>{r.low_gib:.2f}</td><td>{r.mid_gib:.2f}</td><td>{r.high_gib:.2f}</td>"
            f"<td class='note'>{esc(note or r.assumption)}</td></tr>"
        )

    bottlenecks = "".join(f"<li>{esc(b)}</li>" for b in result.bottlenecks) or "<li>None detected.</li>"
    caveats = "".join(f"<li>{esc(c)}</li>" for c in result.caveats)
    refs = "".join(
        f'<li>{esc(k)}: <a href="{esc(v)}">{esc(v)}</a></li>'
        for k, v in result.references.items()
    )
    input_rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in inputs.items())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LocalModel Advisor Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 2em auto; padding: 0 1em; color: #1f2328; }}
h1 {{ font-size: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; font-size: 0.92em; }}
th {{ background: #f6f8fa; }}
.verdict {{ display: inline-block; padding: 4px 12px; border-radius: 6px; color: #fff; background: {color}; font-weight: 600; }}
.note {{ color: #57606a; font-size: 0.85em; }}
section {{ margin-bottom: 1.6em; }}
footer {{ color: #57606a; font-size: 0.8em; margin-top: 2em; }}
</style>
</head>
<body>
<h1>LocalModel Advisor &mdash; Memory Estimate</h1>
<section>
<h2>Verdict</h2>
<p><span class="verdict">{esc(label)}</span></p>
<p>GPU VRAM: {result.vram_gib} GiB &middot; System RAM: {result.ram_gib} GiB &middot; Backend: {esc(result.backend)}</p>
</section>
<section>
<h2>Memory breakdown (GiB)</h2>
<table>
<tr><th>Component</th><th>Low</th><th>Estimate</th><th>High</th><th>Basis</th></tr>
{mem_row("Weights", result.weight_mem)}
{mem_row("KV cache", result.kv_mem)}
<tr><td>Backend overhead</td><td>{result.backend_overhead_gib * 0.5:.2f}</td><td>{result.backend_overhead_gib:.2f}</td><td>{result.backend_overhead_gib * 1.5:.2f}</td><td class="note">rough heuristic per backend</td></tr>
{mem_row("Total", result.total_mem)}
</table>
</section>
<section>
<h2>Likely bottlenecks</h2>
<ul>{bottlenecks}</ul>
</section>
<section>
<h2>Caveats (read these)</h2>
<ul>{caveats}</ul>
</section>
<section>
<h2>Inputs</h2>
<table>{input_rows}</table>
</section>
<section>
<h2>References</h2>
<ul>{refs}</ul>
</section>
<footer>Generated offline by LocalModel Advisor at {datetime.now(timezone.utc).isoformat(timespec="seconds")}. Estimates, not benchmarks.</footer>
</body>
</html>
"""
