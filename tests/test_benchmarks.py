"""Tests for benchmark import, validation, and matching."""

import json
import tempfile
import unittest
from pathlib import Path

from localmodel_advisor.benchmarks import (
    BenchmarkError,
    find_matching,
    import_benchmarks,
    list_benchmarks,
    load_benchmark_file,
    validate_record,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestValidateRecord(unittest.TestCase):
    def _good(self):
        return {
            "gpu": "RTX 3060", "vram_gb": 12, "model": "Llama-2-7B",
            "params_b": 7, "quant": "q4_k_m", "backend": "llama.cpp",
            "tokens_per_second": 38.5,
        }

    def test_valid_record(self):
        rec = validate_record(self._good())
        self.assertEqual(rec.gpu, "RTX 3060")
        self.assertAlmostEqual(rec.tokens_per_second, 38.5)

    def test_missing_required_field(self):
        raw = self._good()
        del raw["tokens_per_second"]
        with self.assertRaises(BenchmarkError) as ctx:
            validate_record(raw)
        self.assertIn("tokens_per_second", str(ctx.exception))

    def test_nonpositive_numbers(self):
        for field in ("vram_gb", "params_b", "tokens_per_second"):
            raw = self._good()
            raw[field] = 0
            with self.assertRaises(BenchmarkError, msg=field):
                validate_record(raw)
            raw[field] = -3
            with self.assertRaises(BenchmarkError, msg=field):
                validate_record(raw)

    def test_bool_rejected_as_number(self):
        raw = self._good()
        raw["vram_gb"] = True
        with self.assertRaises(BenchmarkError):
            validate_record(raw)

    def test_empty_string_field(self):
        raw = self._good()
        raw["gpu"] = "   "
        with self.assertRaises(BenchmarkError):
            validate_record(raw)

    def test_unknown_field_rejected(self):
        raw = self._good()
        raw["hack"] = 1
        with self.assertRaises(BenchmarkError) as ctx:
            validate_record(raw)
        self.assertIn("unknown fields", str(ctx.exception))

    def test_non_dict_rejected(self):
        with self.assertRaises(BenchmarkError):
            validate_record(["nope"])

    def test_optional_fields_kept(self):
        raw = self._good()
        raw.update({"cpu": "Ryzen 5", "notes": "full offload", "context_tokens": 4096})
        rec = validate_record(raw)
        self.assertEqual(rec.cpu, "Ryzen 5")
        self.assertEqual(rec.context_tokens, 4096)


class TestLoadFile(unittest.TestCase):
    def test_valid_fixture(self):
        records = load_benchmark_file(FIXTURES / "benchmarks_valid.json")
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].gpu, "NVIDIA RTX 3060 12GB")

    def test_every_malformed_record_fails(self):
        raw = json.loads((FIXTURES / "benchmarks_malformed.json").read_text())
        self.assertEqual(len(raw), 5)
        for i, rec in enumerate(raw):
            with self.assertRaises(BenchmarkError, msg=f"record {i} should fail"):
                validate_record(rec, i)

    def test_missing_file(self):
        with self.assertRaises(BenchmarkError) as ctx:
            load_benchmark_file(FIXTURES / "does_not_exist.json")
        self.assertIn("not found", str(ctx.exception))

    def test_broken_json(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.json"
            p.write_text("{not json", encoding="utf-8")
            with self.assertRaises(BenchmarkError) as ctx:
                load_benchmark_file(p)
            self.assertIn("not valid JSON", str(ctx.exception))

    def test_empty_array_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.json"
            p.write_text("[]", encoding="utf-8")
            with self.assertRaises(BenchmarkError):
                load_benchmark_file(p)

    def test_wrong_top_level_type(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "obj.json"
            p.write_text('{"foo": 1}', encoding="utf-8")
            with self.assertRaises(BenchmarkError):
                load_benchmark_file(p)

    def test_wrapped_format_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wrapped.json"
            p.write_text(json.dumps({"benchmarks": json.loads((FIXTURES / "benchmarks_valid.json").read_text())}), encoding="utf-8")
            records = load_benchmark_file(p)
            self.assertEqual(len(records), 3)


class TestStore(unittest.TestCase):
    def test_import_and_list_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            count, path = import_benchmarks(FIXTURES / "benchmarks_valid.json", store)
            self.assertEqual(count, 3)
            self.assertEqual(path, store)
            listed = list_benchmarks(store)
            self.assertEqual(len(listed), 3)
            # Import again: appends.
            import_benchmarks(FIXTURES / "benchmarks_valid.json", store)
            self.assertEqual(len(list_benchmarks(store)), 6)

    def test_list_empty_store(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(list_benchmarks(Path(td) / "none.json"), [])

    def test_corrupt_store_errors(self):
        with tempfile.TemporaryDirectory() as td:
            store = Path(td) / "store.json"
            store.write_text("not json", encoding="utf-8")
            with self.assertRaises(BenchmarkError):
                list_benchmarks(store)
            with self.assertRaises(BenchmarkError):
                import_benchmarks(FIXTURES / "benchmarks_valid.json", store)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.records = [r.to_dict() for r in load_benchmark_file(FIXTURES / "benchmarks_valid.json")]

    def test_finds_matching_config(self):
        m = find_matching(self.records, params_b=7, quant="q4_k_m", backend="llama.cpp")
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["gpu"], "NVIDIA RTX 3060 12GB")

    def test_different_quant_no_match(self):
        self.assertEqual(find_matching(self.records, 7, "fp16", "llama.cpp"), [])

    def test_distant_size_no_match(self):
        self.assertEqual(find_matching(self.records, 70, "q4_k_m", "llama.cpp"), [])

    def test_tolerates_missing_optional_fields(self):
        recs = [{"params_b": 7, "quant": "q4_k_m", "backend": "ollama"}]
        # tokens_per_second missing here is fine for matching; validate separately
        self.assertEqual(find_matching(recs, 7, "q4_k_m", "ollama"), recs)


if __name__ == "__main__":
    unittest.main()
