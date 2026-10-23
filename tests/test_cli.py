"""End-to-end CLI tests (no network, temp store via env var)."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from localmodel_advisor.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestEstimateCommand(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["LOCALMODEL_ADVISOR_DB"] = str(Path(self._tmp.name) / "store.json")

    def tearDown(self):
        os.environ.pop("LOCALMODEL_ADVISOR_DB", None)
        self._tmp.cleanup()

    def test_basic_estimate(self):
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096", "--backend", "llama.cpp",
        ])
        self.assertEqual(code, 0, err)
        self.assertIn("Weights:", out)
        self.assertIn("KV cache:", out)
        self.assertIn("Verdict:", out)
        self.assertIn("Fits in VRAM", out)

    def test_estimate_json_output(self):
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096", "--json",
        ])
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["fit_verdict"], "fits_gpu_comfortably")
        self.assertIn("weight_mem", data)

    def test_exact_kv_with_arch(self):
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "2048",
            "--layers", "32", "--kv-heads", "32", "--head-dim", "128",
        ])
        self.assertEqual(code, 0, err)
        self.assertIn("exact formula", out)

    def test_html_report_written(self):
        html_path = Path(self._tmp.name) / "report.html"
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096", "--html", str(html_path),
        ])
        self.assertEqual(code, 0, err)
        text = html_path.read_text(encoding="utf-8")
        self.assertIn("<html", text)
        self.assertIn("LocalModel Advisor", text)
        self.assertIn("KV cache", text)

    def test_html_unwritable_path_fails_cleanly(self):
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096",
            "--html", str(Path(self._tmp.name) / "missing-dir" / "r.html"),
        ])
        self.assertEqual(code, 1)
        self.assertIn("cannot write HTML", err)

    def test_bad_quant_exits_2(self):
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "nope", "--context", "4096",
        ])
        self.assertEqual(code, 2)
        self.assertIn("unknown quantization", err)

    def test_negative_vram_exits_2(self):
        code, out, err = run_cli([
            "estimate", "--vram", "-4", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096",
        ])
        self.assertEqual(code, 2)
        self.assertIn("error", err)


class TestBenchmarkCommands(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["LOCALMODEL_ADVISOR_DB"] = str(Path(self._tmp.name) / "store.json")

    def tearDown(self):
        os.environ.pop("LOCALMODEL_ADVISOR_DB", None)
        self._tmp.cleanup()

    def test_import_and_list(self):
        code, out, err = run_cli(["import", str(FIXTURES / "benchmarks_valid.json")])
        self.assertEqual(code, 0, err)
        self.assertIn("Imported 3", out)
        code, out, err = run_cli(["benchmarks"])
        self.assertEqual(code, 0, err)
        self.assertIn("RTX 3060", out)

    def test_import_malformed_file_fails(self):
        code, out, err = run_cli(["import", str(FIXTURES / "benchmarks_malformed.json")])
        self.assertEqual(code, 2)
        self.assertIn("missing required field", err)

    def test_list_empty(self):
        code, out, err = run_cli(["benchmarks"])
        self.assertEqual(code, 0, err)
        self.assertIn("No benchmarks imported", out)

    def test_estimate_shows_matching_benchmarks(self):
        run_cli(["import", str(FIXTURES / "benchmarks_valid.json")])
        code, out, err = run_cli([
            "estimate", "--vram", "12", "--ram", "32", "--params", "7",
            "--quant", "q4_k_m", "--context", "4096", "--backend", "llama.cpp",
        ])
        self.assertEqual(code, 0, err)
        self.assertIn("Community benchmarks", out)
        self.assertIn("38.5 tok/s", out)


class TestFormatsCommand(unittest.TestCase):
    def test_formats(self):
        code, out, err = run_cli(["formats"])
        self.assertEqual(code, 0, err)
        self.assertIn("q4_k_m", out)
        self.assertIn("bytes/param", out)


if __name__ == "__main__":
    unittest.main()
