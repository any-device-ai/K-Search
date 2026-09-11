"""Target-profile rendering tests. No GPU, no network, no FlashRT."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.utils.cuda_gpu_info import (  # noqa: E402
    TARGET_PROFILE_ENV_VAR,
    get_gpu_info,
    get_gpu_info_or_placeholder,
    get_profile_device_name,
)


A100_PROFILE = {
    "schema_version": 1,
    "target": "local-a100",
    "hardware": {
        "device_name": "NVIDIA A100-SXM4-80GB",
        "compute_capability": "8.0",
        "arch": "sm_80",
        "sm_count": 108,
        "shared_memory_per_sm_kb": 164,
        "shared_memory_per_block_optin_kb": 163,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 40,
        "total_memory_gb": 80.0,
        "free_memory_gb": 78.4,
        "sm_clock_mhz": 1410,
        "memory_clock_mhz": 1593,
        "measured_copy_bandwidth_gb_s": 1670.0,
        "graph_node_replay_overhead_us": 1.35,
        "tensor_cores": "3rd gen (mma.sync m16n8k16)",
        "supported_dtypes": ["tf32", "bf16", "fp16", "int8"],
        "unsupported_dtypes": ["fp8_e4m3", "nvfp4"],
        "features": ["cp.async available", "no TMA"],
    },
    "build": {
        "available": True,
        "flashrt_arch": "rtx_sm80",
        "kernel_modules": ["norm", "gemm_bf16"],
        "reachable_formats": ["bf16", "fp16"],
        "unreachable_formats": ["fp8", "nvfp4"],
        "nvcc_flags": {"flashrt_kernels": "-O3 -gencode arch=compute_80,code=sm_80"},
    },
}


def _write_profile(obj: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="ksearch_profile_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path


class TargetProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        get_gpu_info.cache_clear()
        self._paths: list[str] = []
        self._saved_env = os.environ.pop(TARGET_PROFILE_ENV_VAR, None)

    def tearDown(self) -> None:
        get_gpu_info.cache_clear()
        for p in self._paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        os.environ.pop(TARGET_PROFILE_ENV_VAR, None)
        if self._saved_env is not None:
            os.environ[TARGET_PROFILE_ENV_VAR] = self._saved_env

    def _profile(self, obj: dict) -> str:
        path = _write_profile(obj)
        self._paths.append(path)
        return path

    # -- profile parsing + rendering ------------------------------------------------

    def test_profile_renders_hardware_and_build_layers(self) -> None:
        path = self._profile(A100_PROFILE)
        block = get_gpu_info(path, "A100")

        self.assertIn("NVIDIA A100-SXM4-80GB", block)
        self.assertIn("local-a100", block)
        self.assertIn("8.0", block)
        self.assertIn("sm_80", block)
        self.assertIn("108", block)
        self.assertIn("163 KB", block)
        # Measured numbers, not spec-sheet ones.
        self.assertIn("1670.0 GB/s", block)
        self.assertIn("1.35 us", block)
        # Unavailable formats must be stated plainly.
        self.assertIn("UNAVAILABLE dtypes", block)
        self.assertIn("nvfp4", block)
        # Build layer.
        self.assertIn("rtx_sm80", block)
        self.assertIn("-gencode arch=compute_80,code=sm_80", block)
        self.assertNotIn("WARNING", block)

    def test_profile_wins_over_static_table(self) -> None:
        path = self._profile(A100_PROFILE)
        block = get_gpu_info(path, "H100")
        self.assertIn("NVIDIA A100-SXM4-80GB", block)
        self.assertIn("sm_80", block)
        self.assertNotIn("sm_90a", block)

    def test_mismatch_between_profile_and_target_gpu_warns(self) -> None:
        path = self._profile(A100_PROFILE)
        block = get_gpu_info(path, "H100")
        self.assertIn("WARNING", block)
        self.assertIn("'H100'", block)
        self.assertIn("profile is authoritative", block)

    def test_matching_names_do_not_warn(self) -> None:
        path = self._profile(A100_PROFILE)
        for name in ("A100", "a100", "NVIDIA A100-SXM4-80GB", "A100-SXM"):
            get_gpu_info.cache_clear()
            self.assertNotIn("WARNING", get_gpu_info(path, name), f"unexpected warning for {name!r}")

    def test_empty_target_gpu_does_not_warn(self) -> None:
        path = self._profile(A100_PROFILE)
        self.assertNotIn("WARNING", get_gpu_info(path, ""))

    def test_flat_profile_is_accepted_as_hardware_only(self) -> None:
        path = self._profile({"device_name": "NVIDIA H100 80GB HBM3", "sm_count": 132})
        block = get_gpu_info(path, "H100")
        self.assertIn("NVIDIA H100 80GB HBM3", block)
        self.assertIn("132", block)
        # Everything the flat doc omitted degrades to "unknown", never to a guess.
        self.assertIn("unknown", block)

    def test_absent_build_layer_is_reported_as_absent(self) -> None:
        obj = {"hardware": dict(A100_PROFILE["hardware"]), "build": {"available": False, "reason": "flash_rt not importable"}}
        path = self._profile(obj)
        block = get_gpu_info(path, "A100")
        self.assertIn("absent", block)
        self.assertIn("flash_rt not importable", block)

    def test_unreadable_profile_falls_back_to_static_table(self) -> None:
        block = get_gpu_info("/nonexistent/path/profile.json", "A100")
        self.assertIn("static spec-sheet table", block)
        self.assertIn("sm_80", block)

    def test_malformed_profile_falls_back_to_static_table(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self._paths.append(path)
        block = get_gpu_info(path, "H100")
        self.assertIn("static spec-sheet table", block)

    def test_env_var_supplies_the_profile_path(self) -> None:
        path = self._profile(A100_PROFILE)
        os.environ[TARGET_PROFILE_ENV_VAR] = path
        get_gpu_info.cache_clear()
        block = get_gpu_info(None, "A100")
        self.assertIn("NVIDIA A100-SXM4-80GB", block)

    def test_get_profile_device_name(self) -> None:
        path = self._profile(A100_PROFILE)
        self.assertEqual(get_profile_device_name(path), "NVIDIA A100-SXM4-80GB")
        self.assertEqual(get_profile_device_name("/nope.json"), "")

    # -- static table fallback -------------------------------------------------------

    def test_static_table_entries_all_render(self) -> None:
        for name in ("H100", "H200", "A100", "B200", "L40S", "RTX4090", "RTX5090", "Thor", "Orin"):
            get_gpu_info.cache_clear()
            block = get_gpu_info(None, name)
            self.assertTrue(block.strip(), f"static table produced nothing for {name!r}")
            self.assertIn("static spec-sheet table", block)
            self.assertIn("NOT measured", block)

    def test_static_table_states_unavailable_formats(self) -> None:
        self.assertIn("nvfp4", get_gpu_info(None, "A100"))
        get_gpu_info.cache_clear()
        a100 = get_gpu_info(None, "A100")
        self.assertIn("UNAVAILABLE dtypes", a100)
        get_gpu_info.cache_clear()
        b200 = get_gpu_info(None, "B200")
        self.assertNotIn("UNAVAILABLE dtypes", b200)

    def test_static_table_aliases(self) -> None:
        for name, expected in (
            ("a100-sxm4-80gb", "sm_80"),
            ("NVIDIA H100 PCIe", "sm_90a"),
            ("rtx 4090", "sm_89"),
            ("Jetson Thor", "sm_110"),
            ("agx-orin", "sm_87"),
        ):
            get_gpu_info.cache_clear()
            block = get_gpu_info(None, name)
            self.assertIn(expected, block, f"{name!r} did not resolve to {expected}")

    # -- no source at all -------------------------------------------------------------

    def test_returns_empty_string_when_nothing_is_known(self) -> None:
        self.assertEqual(get_gpu_info(None, ""), "")
        get_gpu_info.cache_clear()
        self.assertEqual(get_gpu_info(None, "MI300X"), "")

    def test_placeholder_says_so_instead_of_guessing(self) -> None:
        text = get_gpu_info_or_placeholder(None, "MI300X")
        self.assertIn("unavailable", text.lower())
        self.assertIn("Do NOT assume", text)
        # It must not name any architecture from the static table.
        for token in ("sm_80", "sm_90a", "sm_89", "H100", "A100"):
            self.assertNotIn(token, text)

    def test_placeholder_defers_to_a_real_block_when_one_exists(self) -> None:
        text = get_gpu_info_or_placeholder(None, "A100")
        self.assertIn("sm_80", text)
        self.assertNotIn("Do NOT assume", text)


if __name__ == "__main__":
    unittest.main()
