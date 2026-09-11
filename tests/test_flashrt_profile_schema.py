"""The FlashRT profile must survive the trip into a prompt block.

The writer (`flash_rt/kopt/device_profile.py`) and the reader
(`k_search/utils/cuda_gpu_info.py`) were written independently and disagreed on
fifteen of nineteen field names. Nothing failed: absent fields render as
"unknown", so a whole 150-round search ran against a hardware block that claimed
the shared-memory budget, the register budget and the achievable bandwidth were
all unmeasurable. These tests pin both halves of the fix — the translation, and
the reader saying so out loud when a translation is missing.

No GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.tasks.flashrt.profile import (  # noqa: E402
    is_flashrt_profile,
    materialize_ksearch_profile,
    to_ksearch_profile,
)
from k_search.utils.cuda_gpu_info import (  # noqa: E402
    TARGET_PROFILE_ENV_VAR,
    get_gpu_info,
    get_gpu_info_or_placeholder,
)

FLASHRT_PROFILE = {
    "schema": "flashrt.kopt.device_profile/1",
    "generated_at": "2026-09-02T13:53:46-0700",
    "target": "local-a100",
    "hardware": {
        "name": "NVIDIA PG509-210",
        "compute_capability": "8.0",
        "gpu_arch": "80",
        "sm_count": 108,
        "warp_size": 32,
        "shared_mem_per_sm_bytes": 167936,
        "shared_mem_per_block_optin_bytes": 166912,
        "regs_per_sm": 65536,
        "l2_bytes": 41943040,
        "memory_total_bytes": 85093777408,
        "memory_free_bytes": 84632469504,
        "clock_sm_max_mhz": 1410,
        "clock_sm_current_mhz": 210,
        "clock_mem_max_mhz": 1512,
        "measured": {
            "copy_bandwidth_gbps": 1592.4,
            "graph_node_replay_us": 1.839,
            "status": "measured",
        },
    },
    "build": {
        "status": "unavailable",
        "reason": "flash_rt not importable",
        "kernel_modules": {"flash_rt_kernels": "flash_rt_kernels.so", "flash_rt_fp4": None},
        "numeric_formats": {
            "bf16": {"reachable": True, "why": "Ampere and later"},
            "fp8_e4m3": {"reachable": False, "why": "sm_89+ only"},
        },
        "nvcc_flags_by_target": {"some_unrelated_obj": ["-O3"]},
    },
}


class TranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out = to_ksearch_profile(FLASHRT_PROFILE)
        self.hw = self.out["hardware"]

    def test_recognises_a_flashrt_profile(self) -> None:
        self.assertTrue(is_flashrt_profile(FLASHRT_PROFILE))
        self.assertFalse(is_flashrt_profile({"hardware": {}}))

    def test_byte_valued_fields_become_the_units_the_reader_names(self) -> None:
        self.assertEqual(self.hw["shared_memory_per_sm_kb"], 164.0)
        self.assertEqual(self.hw["shared_memory_per_block_optin_kb"], 163.0)
        self.assertEqual(self.hw["l2_cache_mb"], 40.0)
        self.assertAlmostEqual(self.hw["total_memory_gb"], 79.2498, places=3)

    def test_renamed_fields_survive(self) -> None:
        self.assertEqual(self.hw["registers_per_sm"], 65536)
        self.assertEqual(self.hw["device_name"], "NVIDIA PG509-210")
        self.assertEqual(self.hw["arch"], "sm_80")

    def test_measured_numbers_are_lifted_out_of_their_nesting(self) -> None:
        self.assertEqual(self.hw["measured_copy_bandwidth_gb_s"], 1592.4)
        self.assertEqual(self.hw["graph_node_replay_overhead_us"], 1.839)

    def test_unmeasured_profile_does_not_claim_a_bandwidth(self) -> None:
        src = json.loads(json.dumps(FLASHRT_PROFILE))
        src["hardware"]["measured"]["status"] = "not-measured"
        hw = to_ksearch_profile(src)["hardware"]
        self.assertNotIn("measured_copy_bandwidth_gb_s", hw)

    def test_sm_clock_is_the_boost_ceiling_not_the_idle_sample(self) -> None:
        # clock_sm_current_mhz is read off an idle GPU and is routinely ~210 MHz
        # on a part that boosts to 1410; reporting it invites nonsense reasoning.
        self.assertEqual(self.hw["sm_clock_mhz"], 1410)

    def test_dtype_reachability_becomes_the_two_lists(self) -> None:
        self.assertEqual(self.hw["supported_dtypes"], ["bf16"])
        self.assertEqual(self.hw["unsupported_dtypes"], ["fp8_e4m3"])

    def test_unrelated_nvcc_flag_dump_is_dropped(self) -> None:
        # ~30 lines of flags for CMake targets unrelated to the tunable. The
        # referee's own describe carries the flags that do matter.
        self.assertNotIn("nvcc_flags", self.out["build"])

    def test_null_kernel_modules_are_dropped(self) -> None:
        self.assertEqual(self.out["build"]["kernel_modules"],
                         {"flash_rt_kernels": "flash_rt_kernels.so"})


class RenderedBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        get_gpu_info.cache_clear()
        self._saved = os.environ.pop(TARGET_PROFILE_ENV_VAR, None)
        self.addCleanup(self._restore)
        self.dir = tempfile.mkdtemp()

    def _restore(self) -> None:
        get_gpu_info.cache_clear()
        os.environ.pop(TARGET_PROFILE_ENV_VAR, None)
        if self._saved is not None:
            os.environ[TARGET_PROFILE_ENV_VAR] = self._saved

    def _write(self, payload: dict) -> str:
        p = Path(self.dir) / "profile.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return str(p)

    def test_untranslated_profile_renders_mostly_unknown_and_says_so(self) -> None:
        block = get_gpu_info_or_placeholder(self._write(FLASHRT_PROFILE), "A100")
        self.assertIn("unknown", block)
        self.assertIn("expected hardware fields are absent", block)
        self.assertIn("flashrt.kopt.device_profile/1", block)

    def test_translated_profile_renders_every_number(self) -> None:
        path = materialize_ksearch_profile(self._write(FLASHRT_PROFILE), self.dir)
        get_gpu_info.cache_clear()
        block = get_gpu_info_or_placeholder(path, "A100")
        for expected in ("164.0 KB", "65536", "40.0 MB", "1592.4 GB/s", "1.839 us", "sm_80"):
            self.assertIn(expected, block)
        self.assertNotIn("expected hardware fields are absent", block)

    def test_translation_is_a_no_op_for_a_foreign_profile(self) -> None:
        path = self._write({"hardware": {"sm_count": 1}})
        self.assertEqual(materialize_ksearch_profile(path, self.dir), path)


if __name__ == "__main__":
    unittest.main()
