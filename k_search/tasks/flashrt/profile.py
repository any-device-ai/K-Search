"""Translate a FlashRT target profile into the shape `cuda_gpu_info` reads.

The two sides were written independently and never checked against each other:
FlashRT's `flash_rt/kopt/device_profile.py` emits `flashrt.kopt.device_profile/1`
(SI-suffixed keys, `measured` nested), while `k_search/utils/cuda_gpu_info.py`
reads a flatter, unit-in-the-name schema. Fifteen of the nineteen fields the
reader looks for were absent, and a missing field renders as "unknown" without
complaint — so the world model spent an entire 150-round search asking for the
shared-memory budget, the register budget and the achievable bandwidth that were
sitting in the artifact the whole time.

The translation lives here, in the FlashRT-only corner of the tasks package,
rather than in either of the two general modules: `cuda_gpu_info` must not learn
one backend's vocabulary, and FlashRT's artifact is consumed by its own referee,
docs and roofline check, so it cannot be bent to a consumer's spelling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FLASHRT_PROFILE_SCHEMA = "flashrt.kopt.device_profile/1"


def is_flashrt_profile(profile: Any) -> bool:
    return (isinstance(profile, dict)
            and str(profile.get("schema", "")).startswith("flashrt.kopt.device_profile/"))


def _div(value: Any, denom: float) -> float | None:
    return round(float(value) / denom, 4) if isinstance(value, (int, float)) else None


def _pick(value: Any) -> Any:
    return value if isinstance(value, (int, float, str)) else None


# Phrased as `cuda_gpu_info`'s static table phrases it, so a measured profile and a
# fallback table entry describe the same device the same way. Only the capabilities
# that table already asserts appear here; anything else omits the key rather than guess.
_TENSOR_CORES_BY_COMPUTE_CAPABILITY = {
    "8.0": "3rd gen (mma.sync m16n8k16)",
    "8.7": "3rd gen (mma.sync m16n8k16)",
    "8.9": "4th gen (mma.sync, no wgmma)",
    "9.0": "4th gen (wgmma)",
    "10.0": "5th gen (tcgen05)",
    "11.0": "5th gen (Blackwell)",
    "12.0": "5th gen (mma.sync, no tcgen05)",
}


def _tensor_cores(hw: dict, compute_capability: Any) -> str | None:
    declared = _pick(hw.get("tensor_cores"))
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    if isinstance(compute_capability, (int, float)):
        cc = f"{float(compute_capability):.1f}"
    else:
        cc = str(compute_capability or "").strip()
    return _TENSOR_CORES_BY_COMPUTE_CAPABILITY.get(cc)


def to_ksearch_profile(profile: dict) -> dict:
    """FlashRT profile -> the schema `cuda_gpu_info._render_profile_block` reads."""
    hw = profile.get("hardware") or {}
    build = profile.get("build") or {}

    gpu_arch = hw.get("gpu_arch") or build.get("gpu_arch")
    # Carry the family into the name, as FlashRT's own renderer does. nvidia-smi
    # reports A100 SXM4 boards by their board code ("NVIDIA PG509-210"), which the
    # reader's name matcher cannot reconcile with a requested "A100" -- leaving a
    # WARNING that A100 does not match A100 on every prompt.
    name = _pick(hw.get("name"))
    family = _pick(hw.get("family"))
    if name and family and str(family).lower() not in str(name).lower():
        name = f"{name} ({family})"

    compute_capability = _pick(hw.get("compute_capability"))

    out_hw: dict[str, Any] = {
        "device_name": name,
        "compute_capability": compute_capability,
        "tensor_cores": _tensor_cores(hw, compute_capability),
        "arch": (f"sm_{gpu_arch}" if gpu_arch else None),
        "sm_count": _pick(hw.get("sm_count")),
        "warp_size": _pick(hw.get("warp_size")),
        "shared_memory_per_sm_kb": _div(hw.get("shared_mem_per_sm_bytes"), 1024),
        "shared_memory_per_block_optin_kb": _div(
            hw.get("shared_mem_per_block_optin_bytes"), 1024),
        "registers_per_sm": _pick(hw.get("regs_per_sm")),
        "l2_cache_mb": _div(hw.get("l2_bytes"), 1 << 20),
        "total_memory_gb": _div(hw.get("memory_total_bytes"), 1 << 30),
        "free_memory_gb": _div(hw.get("memory_free_bytes"), 1 << 30),
        # Max, not `clock_sm_current_mhz`: the current reading is sampled off an
        # idle GPU and is routinely ~210 MHz on a part that boosts to 1410.
        "sm_clock_mhz": _pick(hw.get("clock_sm_max_mhz")),
        "memory_clock_mhz": _pick(hw.get("clock_mem_max_mhz")),
    }

    measured = hw.get("measured") or {}
    if str(measured.get("status", "")) == "measured":
        out_hw["measured_copy_bandwidth_gb_s"] = _pick(measured.get("copy_bandwidth_gbps"))
        out_hw["graph_node_replay_overhead_us"] = _pick(measured.get("graph_node_replay_us"))

    formats = build.get("numeric_formats")
    if isinstance(formats, dict):
        out_hw["supported_dtypes"] = sorted(
            k for k, v in formats.items() if isinstance(v, dict) and v.get("reachable"))
        out_hw["unsupported_dtypes"] = sorted(
            k for k, v in formats.items() if isinstance(v, dict) and not v.get("reachable"))

    out_build: dict[str, Any] = {
        "available": str(build.get("status", "")) != "unavailable",
        "flashrt_arch": _pick(build.get("detect_arch")),
        "kernel_modules": {k: v for k, v in (build.get("kernel_modules") or {}).items() if v},
        "reason": _pick(build.get("reason")),
        # `nvcc_flags_by_target` is deliberately dropped. It renders ~30 lines of
        # flags for CMake targets that have nothing to do with the tunable under
        # search; the referee's `describe` already carries the right flags for the
        # one target that matters.
    }

    return {
        "target": profile.get("target"),
        "schema": profile.get("schema"),
        "generated_at": profile.get("generated_at"),
        "hardware": {k: v for k, v in out_hw.items() if v is not None},
        "build": {k: v for k, v in out_build.items() if v is not None},
    }


def materialize_ksearch_profile(profile_path: str, dest_dir: str | Path) -> str:
    """Write a translated copy next to the search artifacts; return its path.

    Returns `profile_path` unchanged when the artifact is not a FlashRT profile,
    so pointing this at an already-conforming file is a no-op.
    """
    src = Path(profile_path)
    profile = json.loads(src.read_text(encoding="utf-8"))
    if not is_flashrt_profile(profile):
        return str(profile_path)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{src.stem}.ksearch.json"
    out.write_text(json.dumps(to_ksearch_profile(profile), indent=2) + "\n",
                   encoding="utf-8")
    return str(out)
