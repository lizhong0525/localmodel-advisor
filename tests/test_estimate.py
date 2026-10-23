"""Tests for the core estimation formulas and edge cases."""

import unittest

from localmodel_advisor.estimate import (
    Architecture,
    EstimationError,
    GiB,
    estimate,
    estimate_kv_cache,
    estimate_weights,
    judge_fit,
)


class TestWeightEstimation(unittest.TestCase):
    def test_fp16_7b(self):
        r = estimate_weights(7, "fp16")
        # 7e9 * 2 bytes = 14e9 bytes = 13.04 GiB
        self.assertAlmostEqual(r.low_gib, 14e9 / GiB, places=2)
        self.assertGreater(r.high_gib, r.mid_gib)
        self.assertGreaterEqual(r.mid_gib, r.low_gib)

    def test_q4_k_m_includes_overhead(self):
        raw_q4 = estimate_weights(7, "fp16").low_gib / 4  # rough 4-bit floor
        r = estimate_weights(7, "q4_k_m")
        # q4_k_m is ~4.8 bits/param -> bigger than a naive 4x reduction of fp16
        self.assertGreater(r.mid_gib, raw_q4 * 0.9)
        self.assertLess(r.mid_gib, raw_q4 * 1.4)

    def test_quant_name_case_insensitive(self):
        self.assertEqual(estimate_weights(1, "Q4_K_M").mid_gib, estimate_weights(1, "q4_k_m").mid_gib)

    def test_unknown_quant_raises(self):
        with self.assertRaises(EstimationError) as ctx:
            estimate_weights(7, "q3_something")
        self.assertIn("unknown quantization", str(ctx.exception))

    def test_nonpositive_params_raise(self):
        for bad in (0, -1, -7.5):
            with self.assertRaises(EstimationError):
                estimate_weights(bad, "fp16")

    def test_nonnumeric_params_raise(self):
        with self.assertRaises(EstimationError):
            estimate_weights("seven", "fp16")

    def test_scales_linearly(self):
        self.assertAlmostEqual(
            estimate_weights(14, "q8_0").low_gib, estimate_weights(7, "q8_0").low_gib * 2, places=2
        )


class TestKVCacheExact(unittest.TestCase):
    def setUp(self):
        # Llama-2-7B architecture: 32 layers, 32 kv heads (MHA), head_dim 128
        self.arch = Architecture(n_layers=32, n_kv_heads=32, head_dim=128)

    def test_llama2_7b_reference(self):
        r = estimate_kv_cache(2048, 7, self.arch, "fp16", 1)
        # 2*32*32*128*2 bytes/token = 524288 B/token; *2048 = 1 GiB
        self.assertAlmostEqual(r.mid_gib, 1.0, places=2)

    def test_scales_with_context(self):
        r1 = estimate_kv_cache(1024, 7, self.arch, "fp16")
        r2 = estimate_kv_cache(4096, 7, self.arch, "fp16")
        self.assertAlmostEqual(r2.mid_gib, r1.mid_gib * 4, places=2)

    def test_scales_with_batch(self):
        r1 = estimate_kv_cache(2048, 7, self.arch, "fp16", batch_size=1)
        r4 = estimate_kv_cache(2048, 7, self.arch, "fp16", batch_size=4)
        self.assertAlmostEqual(r4.mid_gib, r1.mid_gib * 4, places=2)

    def test_q8_cache_halves_fp16(self):
        fp16 = estimate_kv_cache(2048, 7, self.arch, "fp16")
        q8 = estimate_kv_cache(2048, 7, self.arch, "q8_0")
        self.assertAlmostEqual(q8.mid_gib, fp16.mid_gib / 2, places=2)

    def test_gqa_reduces_cache(self):
        mha = estimate_kv_cache(2048, 7, Architecture(32, 32, 128), "fp16")
        gqa = estimate_kv_cache(2048, 7, Architecture(32, 8, 128), "fp16")
        self.assertAlmostEqual(gqa.mid_gib, mha.mid_gib / 4, places=2)

    def test_unknown_kv_dtype_raises(self):
        with self.assertRaises(EstimationError):
            estimate_kv_cache(2048, 7, self.arch, "int3")


class TestKVCacheHeuristic(unittest.TestCase):
    def test_heuristic_used_without_arch(self):
        r = estimate_kv_cache(2048, 7, None, "fp16")
        self.assertIn("HEURISTIC", r.assumption)
        # Wide range: high bound should be much larger than low bound.
        self.assertGreater(r.high_gib / max(r.low_gib, 1e-9), 3)

    def test_incomplete_arch_falls_back_to_heuristic(self):
        r = estimate_kv_cache(2048, 7, Architecture(n_layers=32), "fp16")
        self.assertIn("HEURISTIC", r.assumption)

    def test_heuristic_mid_is_plausible(self):
        # Llama-2-7B exact is 1.0 GiB at 2048 ctx; heuristic mid should be near it.
        r = estimate_kv_cache(2048, 7, None, "fp16")
        self.assertTrue(0.5 < r.mid_gib < 2.0)

    def test_bad_context_raises(self):
        for bad in (0, -100):
            with self.assertRaises(EstimationError):
                estimate_kv_cache(bad, 7, None, "fp16")


class TestFitJudgement(unittest.TestCase):
    def _total(self, mid, low=None, high=None):
        from localmodel_advisor.estimate import MemoryRange
        return MemoryRange(low if low is not None else mid * 0.9, mid, high if high is not None else mid * 1.1, "t")

    def test_comfortable_fit(self):
        verdict, bn = judge_fit(self._total(5), self._total(4), 12, 32)
        self.assertEqual(verdict, "fits_gpu_comfortably")
        self.assertEqual(bn, [])

    def test_tight_fit_warns(self):
        verdict, bn = judge_fit(self._total(10), self._total(8), 12, 32)
        self.assertEqual(verdict, "fits_gpu_tight")
        self.assertTrue(bn)

    def test_kv_overflow(self):
        verdict, bn = judge_fit(self._total(14, high=15), self._total(8), 12, 64)
        self.assertEqual(verdict, "weights_fit_kv_overflow")

    def test_partial_offload(self):
        verdict, bn = judge_fit(self._total(20), self._total(16), 12, 32)
        self.assertEqual(verdict, "partial_offload")
        self.assertTrue(any("offload" in b.lower() for b in bn))

    def test_does_not_fit(self):
        verdict, bn = judge_fit(self._total(100), self._total(90), 12, 32)
        self.assertEqual(verdict, "does_not_fit")


class TestFullEstimate(unittest.TestCase):
    def test_end_to_end(self):
        res = estimate(
            vram_gib=12, ram_gib=32, params_b=7, quant="q4_k_m",
            context_tokens=4096, backend="llama.cpp",
            arch=Architecture(32, 32, 128),
        )
        self.assertEqual(res.fit_verdict, "fits_gpu_comfortably")
        self.assertGreater(res.total_mem.mid_gib, res.weight_mem.mid_gib)
        self.assertIn("backend", res.references)
        self.assertIn("quantization", res.references)
        # The no-throughput-claim caveat must always be present.
        self.assertTrue(any("tokens-per-second" in c for c in res.caveats))

    def test_heuristic_caveat_added(self):
        res = estimate(vram_gib=12, ram_gib=32, params_b=7, quant="q4_k_m", context_tokens=4096)
        self.assertTrue(any("heuristic" in c.lower() for c in res.caveats))

    def test_70b_cpu_only(self):
        res = estimate(vram_gib=0, ram_gib=64, params_b=70, quant="q4_k_m", context_tokens=4096)
        self.assertIn(res.fit_verdict, ("cpu_only", "does_not_fit"))

    def test_unknown_backend_ok(self):
        res = estimate(vram_gib=12, ram_gib=32, params_b=7, quant="q4_k_m", context_tokens=2048, backend="my-custom-engine")
        self.assertEqual(res.backend, "my-custom-engine")
        self.assertGreater(res.backend_overhead_gib, 0)

    def test_to_dict_serializable(self):
        import json
        res = estimate(vram_gib=12, ram_gib=32, params_b=7, quant="q4_k_m", context_tokens=2048)
        json.dumps(res.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main()
