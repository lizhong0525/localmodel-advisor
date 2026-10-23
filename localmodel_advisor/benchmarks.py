"""User-contributed benchmark records.

Benchmarks are the only legitimate source of tokens-per-second numbers.
Records are plain JSON with hardware metadata, stored in a local JSON
file (default: ~/.localmodel_advisor/benchmarks.json). No network, no
telemetry.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

REQUIRED_FIELDS = ("gpu", "vram_gb", "model", "params_b", "quant", "backend", "tokens_per_second")
OPTIONAL_FIELDS = ("ram_gb", "context_tokens", "kv_dtype", "cpu", "driver", "os", "notes", "date")


class BenchmarkError(ValueError):
    """Raised for malformed benchmark files or records."""


@dataclass
class BenchmarkRecord:
    gpu: str
    vram_gb: float
    model: str
    params_b: float
    quant: str
    backend: str
    tokens_per_second: float
    ram_gb: Optional[float] = None
    context_tokens: Optional[int] = None
    kv_dtype: Optional[str] = None
    cpu: Optional[str] = None
    driver: Optional[str] = None
    os: Optional[str] = None
    notes: Optional[str] = None
    date: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def default_store_path() -> Path:
    return Path(os.environ.get("LOCALMODEL_ADVISOR_DB", Path.home() / ".localmodel_advisor" / "benchmarks.json"))


def validate_record(raw: dict, index: int = 0) -> BenchmarkRecord:
    if not isinstance(raw, dict):
        raise BenchmarkError(f"record #{index}: expected an object, got {type(raw).__name__}")
    for field_name in REQUIRED_FIELDS:
        if field_name not in raw:
            raise BenchmarkError(f"record #{index}: missing required field {field_name!r}")
    for num_field in ("vram_gb", "params_b", "tokens_per_second"):
        value = raw[num_field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise BenchmarkError(f"record #{index}: {num_field!r} must be a positive number, got {value!r}")
    for str_field in ("gpu", "model", "quant", "backend"):
        if not isinstance(raw[str_field], str) or not raw[str_field].strip():
            raise BenchmarkError(f"record #{index}: {str_field!r} must be a non-empty string")
    if "ram_gb" in raw and raw["ram_gb"] is not None:
        if not isinstance(raw["ram_gb"], (int, float)) or raw["ram_gb"] <= 0:
            raise BenchmarkError(f"record #{index}: 'ram_gb' must be a positive number")
    if "context_tokens" in raw and raw["context_tokens"] is not None:
        if not isinstance(raw["context_tokens"], int) or raw["context_tokens"] <= 0:
            raise BenchmarkError(f"record #{index}: 'context_tokens' must be a positive integer")
    unknown = set(raw) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise BenchmarkError(f"record #{index}: unknown fields {sorted(unknown)} (allowed: {sorted(REQUIRED_FIELDS + OPTIONAL_FIELDS)})")
    return BenchmarkRecord(**{f: raw.get(f) for f in REQUIRED_FIELDS + OPTIONAL_FIELDS if raw.get(f) is not None})


def load_benchmark_file(path: str | Path) -> list[BenchmarkRecord]:
    """Parse a benchmark JSON file (list of records, or {"benchmarks": [...]})."""
    path = Path(path)
    if not path.exists():
        raise BenchmarkError(f"benchmark file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkError(f"cannot read {path}: {exc}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path} is not valid JSON: {exc}")
    if isinstance(data, dict) and "benchmarks" in data:
        data = data["benchmarks"]
    if not isinstance(data, list):
        raise BenchmarkError(f"{path}: expected a JSON array of records or an object with a 'benchmarks' key")
    if not data:
        raise BenchmarkError(f"{path}: file contains no benchmark records")
    return [validate_record(rec, i + 1) for i, rec in enumerate(data)]


def import_benchmarks(file_path: str | Path, store_path: str | Path | None = None) -> tuple[int, Path]:
    """Validate and append records from a file to the local store."""
    records = load_benchmark_file(file_path)
    store = Path(store_path) if store_path else default_store_path()
    existing = []
    if store.exists():
        try:
            existing = json.loads(store.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise BenchmarkError(f"existing store {store} is unreadable: {exc}")
        if not isinstance(existing, list):
            raise BenchmarkError(f"existing store {store} is corrupt (expected a JSON array)")
    existing.extend(rec.to_dict() for rec in records)
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(records), store


def list_benchmarks(store_path: str | Path | None = None) -> list[dict]:
    store = Path(store_path) if store_path else default_store_path()
    if not store.exists():
        return []
    try:
        data = json.loads(store.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BenchmarkError(f"store {store} is unreadable: {exc}")
    if not isinstance(data, list):
        raise BenchmarkError(f"store {store} is corrupt (expected a JSON array)")
    return data


def find_matching(records: list[dict], params_b: float, quant: str, backend: str, vram_gb: Optional[float] = None) -> list[dict]:
    """Find community records roughly matching a configuration.

    Malformed records (e.g. from a hand-edited store file) are skipped
    rather than crashing the estimate command.
    """
    matches = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        try:
            rec_params = float(rec.get("params_b", 0))
            rec_quant = str(rec.get("quant", ""))
            rec_backend = str(rec.get("backend", ""))
            rec_vram = float(rec.get("vram_gb", 0))
        except (TypeError, ValueError):
            continue
        if rec_quant.lower() != quant.lower():
            continue
        if rec_backend.lower() != backend.lower():
            continue
        if abs(rec_params - params_b) / max(params_b, 0.1) > 0.35:
            continue
        if vram_gb is not None and rec_vram < params_b:  # coarse sanity filter
            continue
        matches.append(rec)
    return matches
