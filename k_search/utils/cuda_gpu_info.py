"""CUDA / NVIDIA target hardware profile helpers.

These utilities are the CUDA counterpart of `metal_gpu_info` and must be safe to
import on a host with no GPU, no CUDA toolkit and no torch. They are used to give
CUDA prompts an accurate hardware profile.

Unlike the Metal path, **nothing here probes the local device**. The search host is
not necessarily the validation device (the referee may run over ssh or as a remote
service), so a local probe would describe the wrong GPU. A wrong-but-plausible
hardware block is worse than none: the model would confidently write kernels for
an architecture that is not the one being measured.

Source order (first hit wins):
  1. a target-profile JSON artifact generated *on the validation device* — authoritative
  2. a static spec-sheet table keyed on the `--target-gpu` name — approximate
  3. nothing: return `""` and let the caller say so in the prompt
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any


# Environment fallback for the profile artifact path, so call sites that mirror the
# MLX ones (`get_gpu_info()` with no arguments, deep inside prompt builders) still
# see the profile without threading a new parameter through the search loop.
TARGET_PROFILE_ENV_VAR = "KSEARCH_TARGET_PROFILE"


# Static, approximate spec-sheet facts. This is the fallback when no profile artifact
# was generated on the validation device; it is deliberately coarse and is labelled as
# such in the rendered block. Values that vary by SKU are marked "approx".
#
# Tuple layout is avoided here on purpose: the field set is wide enough that a dict
# keyed by field name stays readable and lets entries omit what they do not know.
_STATIC_GPU_TABLE: dict[str, dict[str, Any]] = {
    "H100": {
        "device_name": "NVIDIA H100 (SXM5)",
        "compute_capability": "9.0",
        "arch": "sm_90a",
        "sm_count": 132,
        "shared_memory_per_sm_kb": 228,
        "shared_memory_per_block_optin_kb": 227,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 50,
        "memory_gb": 80,
        "memory_type": "HBM3",
        "peak_bandwidth_gb_s": 3350,
        "tensor_cores": "4th gen (wgmma)",
        "dtypes": ["fp64", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"],
        "unavailable_dtypes": ["nvfp4"],
        "features": ["cp.async", "TMA", "thread-block clusters / DSMEM", "async wgmma"],
    },
    "H200": {
        "device_name": "NVIDIA H200 (SXM5)",
        "compute_capability": "9.0",
        "arch": "sm_90a",
        "sm_count": 132,
        "shared_memory_per_sm_kb": 228,
        "shared_memory_per_block_optin_kb": 227,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 50,
        "memory_gb": 141,
        "memory_type": "HBM3e",
        "peak_bandwidth_gb_s": 4800,
        "tensor_cores": "4th gen (wgmma)",
        "dtypes": ["fp64", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"],
        "unavailable_dtypes": ["nvfp4"],
        "features": ["cp.async", "TMA", "thread-block clusters / DSMEM", "async wgmma"],
    },
    "A100": {
        "device_name": "NVIDIA A100-SXM4-80GB",
        "compute_capability": "8.0",
        "arch": "sm_80",
        "sm_count": 108,
        "shared_memory_per_sm_kb": 164,
        "shared_memory_per_block_optin_kb": 163,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 40,
        "memory_gb": 80,
        "memory_type": "HBM2e",
        "peak_bandwidth_gb_s": 2039,
        "tensor_cores": "3rd gen (mma.sync m16n8k16)",
        "dtypes": ["fp64", "tf32", "bf16", "fp16", "int8"],
        "unavailable_dtypes": ["fp8_e4m3", "fp8_e5m2", "nvfp4"],
        "features": ["cp.async", "no TMA", "no wgmma", "no thread-block clusters"],
    },
    "B200": {
        "device_name": "NVIDIA B200",
        "compute_capability": "10.0",
        "arch": "sm_100a",
        "sm_count": 148,
        "shared_memory_per_sm_kb": 228,
        "shared_memory_per_block_optin_kb": 227,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 126,
        "memory_gb": 180,
        "memory_type": "HBM3e",
        "peak_bandwidth_gb_s": 8000,
        "tensor_cores": "5th gen (tcgen05)",
        "dtypes": ["fp64", "tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "nvfp4"],
        "unavailable_dtypes": [],
        "features": ["cp.async", "TMA", "thread-block clusters / DSMEM", "tcgen05 MMA"],
    },
    "L40S": {
        "device_name": "NVIDIA L40S",
        "compute_capability": "8.9",
        "arch": "sm_89",
        "sm_count": 142,
        "shared_memory_per_sm_kb": 128,
        "shared_memory_per_block_optin_kb": 99,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 96,
        "memory_gb": 48,
        "memory_type": "GDDR6 ECC",
        "peak_bandwidth_gb_s": 864,
        "tensor_cores": "4th gen (mma.sync, no wgmma)",
        "dtypes": ["tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"],
        "unavailable_dtypes": ["nvfp4"],
        "features": ["cp.async", "no TMA", "no wgmma", "weak fp64"],
    },
    "RTX4090": {
        "device_name": "NVIDIA GeForce RTX 4090",
        "compute_capability": "8.9",
        "arch": "sm_89",
        "sm_count": 128,
        "shared_memory_per_sm_kb": 128,
        "shared_memory_per_block_optin_kb": 99,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 72,
        "memory_gb": 24,
        "memory_type": "GDDR6X",
        "peak_bandwidth_gb_s": 1008,
        "tensor_cores": "4th gen (mma.sync, no wgmma)",
        "dtypes": ["tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"],
        "unavailable_dtypes": ["nvfp4"],
        "features": ["cp.async", "no TMA", "no wgmma", "weak fp64"],
    },
    "RTX5090": {
        "device_name": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "arch": "sm_120a",
        "sm_count": 170,
        "shared_memory_per_sm_kb": 128,
        "shared_memory_per_block_optin_kb": 99,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 88,
        "memory_gb": 32,
        "memory_type": "GDDR7",
        "peak_bandwidth_gb_s": 1792,
        "tensor_cores": "5th gen (mma.sync, no tcgen05)",
        "dtypes": ["tf32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "nvfp4", "int8"],
        "unavailable_dtypes": [],
        "features": ["cp.async", "TMA", "no wgmma", "no tcgen05", "weak fp64"],
    },
    "THOR": {
        "device_name": "NVIDIA Jetson AGX Thor (approx)",
        "compute_capability": "11.0",
        "arch": "sm_110",
        "sm_count": 20,
        "shared_memory_per_sm_kb": 228,
        "shared_memory_per_block_optin_kb": 227,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": None,
        "memory_gb": 128,
        "memory_type": "LPDDR5X (unified, shared with CPU)",
        "peak_bandwidth_gb_s": 273,
        "tensor_cores": "5th gen (Blackwell)",
        "dtypes": ["bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "nvfp4", "int8"],
        "unavailable_dtypes": [],
        "features": [
            "unified memory with the CPU — host/device copies are cheap but bandwidth is shared",
            "thermally constrained: clocks drift over a long search",
            "graph-node replay overhead is several µs — small launches are latency-bound",
        ],
    },
    "ORIN": {
        "device_name": "NVIDIA Jetson AGX Orin (approx)",
        "compute_capability": "8.7",
        "arch": "sm_87",
        "sm_count": 16,
        "shared_memory_per_sm_kb": 164,
        "shared_memory_per_block_optin_kb": 163,
        "registers_per_sm": 65536,
        "warp_size": 32,
        "l2_cache_mb": 4,
        "memory_gb": 64,
        "memory_type": "LPDDR5 (unified, shared with CPU)",
        "peak_bandwidth_gb_s": 205,
        "tensor_cores": "3rd gen (mma.sync m16n8k16)",
        "dtypes": ["tf32", "bf16", "fp16", "int8"],
        "unavailable_dtypes": ["fp8_e4m3", "fp8_e5m2", "nvfp4"],
        "features": [
            "unified memory with the CPU — host/device copies are cheap but bandwidth is shared",
            "cp.async",
            "no TMA",
            "no wgmma",
            "thermally constrained: clocks drift over a long search",
        ],
    },
}

# Aliases so common spellings of --target-gpu resolve to a table entry.
_STATIC_GPU_ALIASES: dict[str, str] = {
    "A100SXM": "A100",
    "A100SXM4": "A100",
    "A100PCIE": "A100",
    "A10080GB": "A100",
    "A10040GB": "A100",
    "H100SXM": "H100",
    "H100PCIE": "H100",
    "HOPPER": "H100",
    "AMPERE": "A100",
    "BLACKWELL": "B200",
    "4090": "RTX4090",
    "5090": "RTX5090",
    "GEFORCERTX4090": "RTX4090",
    "GEFORCERTX5090": "RTX5090",
    "JETSONTHOR": "THOR",
    "AGXTHOR": "THOR",
    "T5000": "THOR",
    "JETSONORIN": "ORIN",
    "AGXORIN": "ORIN",
    "ORINAGX": "ORIN",
}

_BAR = "═══════════════════════════════════════════════════════════════════════════"


def _header(title: str) -> str:
    """Box header padded to the same width as `_BAR`."""
    prefix = f"═══ {title} "
    pad = len(_BAR) - len(prefix)
    return prefix + ("═" * pad if pad > 0 else "")


def _normalize_gpu_name(name: str) -> str:
    """Uppercase and strip everything that is not alphanumeric."""
    return "".join(ch for ch in str(name or "").upper() if ch.isalnum())


def _lookup_static_entry(target_gpu: str) -> tuple[str, dict[str, Any]] | None:
    """Resolve a `--target-gpu` string to a static table entry, or None."""
    key = _normalize_gpu_name(target_gpu)
    if not key:
        return None
    if key in _STATIC_GPU_TABLE:
        return key, _STATIC_GPU_TABLE[key]
    alias = _STATIC_GPU_ALIASES.get(key)
    if alias and alias in _STATIC_GPU_TABLE:
        return alias, _STATIC_GPU_TABLE[alias]
    # Substring match both ways: "NVIDIAA100SXM480GB" contains "A100"; "A100X" is close enough.
    for table_key in sorted(_STATIC_GPU_TABLE.keys(), key=len, reverse=True):
        if table_key in key or key in table_key:
            return table_key, _STATIC_GPU_TABLE[table_key]
    return None


def _names_disagree(device_name: str, target_gpu: str) -> bool:
    """True when a profile's device name and the requested target_gpu look unrelated."""
    dev = _normalize_gpu_name(device_name)
    tgt = _normalize_gpu_name(target_gpu)
    if not dev or not tgt:
        return False
    if tgt in dev or dev in tgt:
        return False
    # Resolve both through the static table: "H100" and "NVIDIA H100 80GB HBM3" agree.
    dev_hit = _lookup_static_entry(device_name)
    tgt_hit = _lookup_static_entry(target_gpu)
    if dev_hit and tgt_hit:
        return dev_hit[0] != tgt_hit[0]
    return True


def _fmt(value: Any, unit: str = "") -> str:
    """Render a scalar field, collapsing missing values to 'unknown'."""
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value if str(v).strip()]
        return ", ".join(items) if items else "none"
    s = str(value).strip()
    if not s:
        return "unknown"
    return f"{s}{unit}"


def _read_profile(profile_path: str) -> dict[str, Any] | None:
    """Load and lightly validate a target-profile JSON artifact."""
    try:
        with open(profile_path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _profile_sections(profile: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a profile into (hardware, build).

    A profile is expected to nest the two layers under "hardware" and "build"
    (§4.1). A flat document is accepted as hardware-only, so a minimal hand-written
    artifact still works.
    """
    hw = profile.get("hardware")
    build = profile.get("build")
    if not isinstance(hw, dict):
        hw = {k: v for k, v in profile.items() if k not in ("build", "target", "schema_version")}
    if not isinstance(build, dict):
        build = {}
    return hw, build


# The hardware fields a profile is expected to carry. Used only to notice, and
# say out loud, that an artifact was written to a schema this reader does not
# speak — not to reject it.
_EXPECTED_HW_KEYS = (
    "compute_capability", "arch", "sm_count", "warp_size",
    "shared_memory_per_sm_kb", "shared_memory_per_block_optin_kb",
    "registers_per_sm", "l2_cache_mb", "total_memory_gb",
    "measured_copy_bandwidth_gb_s", "graph_node_replay_overhead_us",
)


def _render_profile_block(*, profile: dict[str, Any], profile_path: str, target_gpu: str) -> str:
    hw, build = _profile_sections(profile)
    device_name = str(hw.get("device_name") or hw.get("name") or "")
    target = str(profile.get("target") or "")

    lines: list[str] = [
        _header("TARGET GPU PROFILE (measured on the validation device)"),
    ]
    if target:
        lines.append(f"  Target:                {target}")
    lines += [
        f"  Device:                {_fmt(device_name)}",
        f"  Compute capability:    {_fmt(hw.get('compute_capability'))}  (arch {_fmt(hw.get('arch'))})",
        f"  SM count:              {_fmt(hw.get('sm_count'))}",
        f"  Shared mem / SM:       {_fmt(hw.get('shared_memory_per_sm_kb'), ' KB')}",
        f"  Shared mem / block:    {_fmt(hw.get('shared_memory_per_block_optin_kb'), ' KB')} (opt-in max)",
        f"  Registers / SM:        {_fmt(hw.get('registers_per_sm'))}",
        f"  Warp size:             {_fmt(hw.get('warp_size'))}",
        f"  L2 cache:              {_fmt(hw.get('l2_cache_mb'), ' MB')}",
        f"  Device memory:         {_fmt(hw.get('total_memory_gb'), ' GB')} total"
        + (
            f", {_fmt(hw.get('free_memory_gb'), ' GB')} free"
            if hw.get("free_memory_gb") is not None
            else ""
        ),
        f"  SM clock:              {_fmt(hw.get('sm_clock_mhz'), ' MHz')}",
        f"  Memory clock:          {_fmt(hw.get('memory_clock_mhz'), ' MHz')}",
        f"  Copy bandwidth:        {_fmt(hw.get('measured_copy_bandwidth_gb_s'), ' GB/s')} (measured, achievable)",
        f"  Graph replay overhead: {_fmt(hw.get('graph_node_replay_overhead_us'), ' us')} per node (measured)",
    ]
    if hw.get("tensor_cores") is not None:
        lines.append(f"  Tensor cores:          {_fmt(hw.get('tensor_cores'))}")
    if hw.get("supported_dtypes") is not None:
        lines.append(f"  Usable dtypes:         {_fmt(hw.get('supported_dtypes'))}")
    if hw.get("unsupported_dtypes"):
        lines.append(f"  UNAVAILABLE dtypes:    {_fmt(hw.get('unsupported_dtypes'))} — do not emit these")
    for note in list(hw.get("features") or []):
        lines.append(f"  Note:                  {note}")

    lines.append("  ── FlashRT build layer ──")
    if build and build.get("available", True):
        lines.append(f"  detect_arch():         {_fmt(build.get('flashrt_arch'))}")
        lines.append(f"  Kernel modules built:  {_fmt(build.get('kernel_modules'))}")
        lines.append(f"  Reachable formats:     {_fmt(build.get('reachable_formats'))}")
        if build.get("unreachable_formats"):
            lines.append(
                f"  UNREACHABLE formats:   {_fmt(build.get('unreachable_formats'))} — not built here"
            )
        flags = build.get("nvcc_flags")
        if isinstance(flags, dict) and flags:
            for tgt_name in sorted(flags.keys()):
                lines.append(f"  nvcc [{tgt_name}]: {_fmt(flags.get(tgt_name))}")
        elif flags:
            lines.append(f"  nvcc flags:            {_fmt(flags)}")
    else:
        reason = str(build.get("reason") or "flash_rt not importable when the profile was generated")
        lines.append(f"  (absent — {reason})")

    missing = [k for k in _EXPECTED_HW_KEYS if hw.get(k) is None]
    if len(missing) > len(_EXPECTED_HW_KEYS) // 3:
        # A field the reader does not find renders as "unknown" and is otherwise
        # silent, so a profile written to a different schema looks like a device
        # nobody measured. Saying so is the difference between a prompt bug that
        # is found in minutes and one that survives a whole search.
        schema = str(profile.get("schema") or profile.get("schema_version") or "unstated")
        lines.append(
            f"  WARNING: {len(missing)} of {len(_EXPECTED_HW_KEYS)} expected hardware "
            f"fields are absent from this profile (schema: {schema}). Missing: "
            + ", ".join(missing)
        )
        lines.append(
            "  WARNING: treat the 'unknown' entries above as a broken profile, not as "
            "a device whose properties are genuinely unmeasurable."
        )

    if _names_disagree(device_name, target_gpu):
        lines.append(
            f"  WARNING: requested --target-gpu {str(target_gpu).strip()!r} does not match the "
            f"profile device {device_name!r}."
        )
        lines.append(
            "  WARNING: the profile is authoritative — it was generated on the device that will "
            "measure your kernel. Write for the profile, not for the requested name."
        )

    lines.append(f"  Source:                {profile_path}")
    lines.append(_BAR)
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_static_block(*, key: str, entry: dict[str, Any], target_gpu: str) -> str:
    lines: list[str] = [
        _header("TARGET GPU (static spec-sheet table — NOT measured)"),
        f"  Requested:             {str(target_gpu).strip()}",
        f"  Device:                {_fmt(entry.get('device_name'))}",
        f"  Compute capability:    {_fmt(entry.get('compute_capability'))}  (arch {_fmt(entry.get('arch'))})",
        f"  SM count:              {_fmt(entry.get('sm_count'))}",
        f"  Shared mem / SM:       {_fmt(entry.get('shared_memory_per_sm_kb'), ' KB')}",
        f"  Shared mem / block:    {_fmt(entry.get('shared_memory_per_block_optin_kb'), ' KB')} (opt-in max)",
        f"  Registers / SM:        {_fmt(entry.get('registers_per_sm'))}",
        f"  Warp size:             {_fmt(entry.get('warp_size'))}",
        f"  L2 cache:              {_fmt(entry.get('l2_cache_mb'), ' MB')}",
        f"  Device memory:         {_fmt(entry.get('memory_gb'), ' GB')} {_fmt(entry.get('memory_type'))}",
        f"  Peak bandwidth:        {_fmt(entry.get('peak_bandwidth_gb_s'), ' GB/s')} (spec sheet, not achievable)",
        f"  Tensor cores:          {_fmt(entry.get('tensor_cores'))}",
        f"  Usable dtypes:         {_fmt(entry.get('dtypes'))}",
    ]
    if entry.get("unavailable_dtypes"):
        lines.append(
            f"  UNAVAILABLE dtypes:    {_fmt(entry.get('unavailable_dtypes'))} — do not emit these"
        )
    for note in list(entry.get("features") or []):
        lines.append(f"  Note:                  {note}")
    lines += [
        f"  Source:                static table entry {key!r} — approximate SKU-independent",
        "                         values. No measured bandwidth and no launch-overhead number.",
        "                         Generate a target profile on the validation device for those.",
        _BAR,
        "",
    ]
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=16)
def get_gpu_info(profile_path: str | None = None, target_gpu: str = "") -> str:
    """Return a formatted CUDA target hardware block, or "" when nothing is known.

    `profile_path` (a JSON artifact generated on the validation device) wins over the
    static table. When neither is available the caller is expected to say so in the
    prompt rather than substitute a guess.

    Results are cached per (profile_path, target_gpu); a profile artifact edited
    in-place within one process will not be re-read.
    """

    path = str(profile_path or "").strip() or str(os.getenv(TARGET_PROFILE_ENV_VAR, "") or "").strip()
    if path:
        profile = _read_profile(path)
        if profile is not None:
            try:
                return _render_profile_block(
                    profile=profile, profile_path=path, target_gpu=str(target_gpu or "")
                )
            except Exception:
                return ""

    hit = _lookup_static_entry(str(target_gpu or ""))
    if hit is not None:
        key, entry = hit
        try:
            return _render_static_block(key=key, entry=entry, target_gpu=str(target_gpu or ""))
        except Exception:
            return ""

    return ""


def get_profile_device_name(profile_path: str | None = None) -> str:
    """Return the profile's device name, or "" if unavailable."""
    path = str(profile_path or "").strip() or str(os.getenv(TARGET_PROFILE_ENV_VAR, "") or "").strip()
    if not path:
        return ""
    profile = _read_profile(path)
    if profile is None:
        return ""
    hw, _ = _profile_sections(profile)
    return str(hw.get("device_name") or hw.get("name") or "")


def get_gpu_info_or_placeholder(profile_path: str | None = None, target_gpu: str = "") -> str:
    """`get_gpu_info` with the "we do not know" text the prompts must show instead of a guess."""
    block = get_gpu_info(profile_path, target_gpu).strip()
    if block:
        return block
    return (
        "TARGET GPU PROFILE: unavailable.\n"
        "No target-profile artifact was supplied (--target-profile) and the requested\n"
        f"target GPU {str(target_gpu or '').strip()!r} is not in the static table.\n"
        "Do NOT assume a specific architecture, shared-memory budget, tensor-core\n"
        "generation or numeric format. Prefer portable CUDA and guard anything\n"
        "architecture-specific behind __CUDA_ARCH__."
    )
