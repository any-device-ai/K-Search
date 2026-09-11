"""Render the FlashRT referee's `describe` output into CUDA prompt blocks.

These are injected into the generic generator templates via `{definition}`,
`{per_task_requirement}` and `{code_format}`, exactly like the FlashInferBench
blocks in `k_search/tasks/flashinfer_bench/prompts.py`.

`describe` is documentation, not a specification (design §1.4): the referee, not
this file, decides whether a kernel is correct. Imprecision here costs rounds, not
safety — so every field is optional and rendering degrades to "not specified".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _s(v: Any, default: str = "") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s or default


def _render_tensor_row(t: Mapping[str, Any]) -> str:
    name = _s(t.get("name"), "?")
    role = _s(t.get("role"), "in")
    dtype = _s(t.get("dtype"), "?")
    shape = t.get("shape")
    shape_s = "x".join(str(x) for x in _as_list(shape)) if shape is not None else "?"
    row = f"  - {name} [{role}] {dtype} shape={shape_s}"
    layout = _s(t.get("layout"))
    if layout:
        row += f" layout={layout}"
    stride = t.get("strides")
    if stride is not None:
        row += f" strides={list(_as_list(stride))}"
    alias = _s(t.get("alias_of"))
    if alias:
        row += f" ALIASES {alias}"
    note = _s(t.get("note"))
    if note:
        row += f"  # {note}"
    return row


# The referee's `describe` JSON and this renderer were named independently and
# never checked against each other. The renderer looked for `text`/`tunable_id`/
# `signature`/`tensors`; the referee emits `rendered`/`id`/`c_signature`/`params`.
# Nothing failed -- the renderer just fell through to its degraded branch, so 1,430
# of an 8,360-character payload reached the model and N, K, the entry signature,
# the correctness gate, the shared-memory cap and the cost model were all silently
# dropped. A 150-round search spent its whole budget asking for the tensor shapes.
_DESCRIBE_ALIASES = {
    "text": "rendered",
    "tunable_id": "id",
    "signature": "c_signature",
}


def _normalise_describe(describe: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept either spelling. K-Search's own names win when both are present."""
    if not isinstance(describe, Mapping):
        return {}
    missing = {k: describe[v] for k, v in _DESCRIBE_ALIASES.items()
               if not describe.get(k) and describe.get(v)}
    return {**describe, **missing} if missing else describe


def render_describe(describe: Mapping[str, Any] | None, *, gpu_info: str = "") -> str:
    """Render one `describe` payload into the `{definition}` block.

    A payload may carry a pre-rendered `text` field; when present it is used
    verbatim as the body and only the hardware profile is appended, so the
    FlashRT side can improve the wording without a K-Search change.
    """
    d: Mapping[str, Any] = _normalise_describe(describe)
    parts: list[str] = []

    tunable_id = _s(d.get("tunable_id"), "(unknown tunable)")
    title = _s(d.get("title"))
    header = f"FlashRT tunable: {tunable_id}"
    if title:
        header += f" — {title}"
    parts.append(header)

    pre_rendered = _s(d.get("text"))
    if pre_rendered:
        parts.append(pre_rendered)
    else:
        summary = _s(d.get("summary"))
        if summary:
            parts.append(summary)

        signature = _s(d.get("signature"))
        if signature:
            parts.append("Entry point (this exact signature; FlashRT calls it directly):\n" + signature)
        entry_symbol = _s(d.get("entry_symbol"))
        if entry_symbol and not signature:
            parts.append(f"Entry symbol: {entry_symbol}")

        tensors = [t for t in _as_list(d.get("tensors")) if isinstance(t, Mapping)]
        if tensors:
            parts.append("Tensors (raw device pointers; FlashRT owns the allocations):\n"
                         + "\n".join(_render_tensor_row(t) for t in tensors))

        scalars = [t for t in _as_list(d.get("scalars")) if isinstance(t, Mapping)]
        if scalars:
            rows = []
            for sc in scalars:
                row = f"  - {_s(sc.get('name'), '?')}: {_s(sc.get('dtype'), '?')}"
                if sc.get("value") is not None:
                    row += f" = {sc.get('value')}"
                rows.append(row)
            parts.append("Scalar arguments:\n" + "\n".join(rows))

        launch = d.get("launch")
        if isinstance(launch, Mapping) and launch:
            rows = [f"  - {k}: {launch.get(k)}" for k in sorted(launch.keys())]
            parts.append("Launch configuration:\n" + "\n".join(rows))

        semantics = _s(d.get("semantics"))
        if semantics:
            parts.append("Semantics:\n" + semantics)

    fixture_mode = _s(d.get("fixture_mode"))
    if fixture_mode:
        parts.append(
            f"Fixture mode: {fixture_mode} — inputs come from FlashRT, not from your code."
        )

    gate = d.get("gate")
    if isinstance(gate, Mapping) and gate:
        metric = _s(gate.get("metric"), "cos")
        threshold = gate.get("threshold")
        gate_line = f"Correctness gate: {metric}"
        if threshold is not None:
            gate_line += f" >= {threshold}"
        gate_line += (
            " against the production kernel's live output on the same inputs "
            "(the reference is regenerated every round; there is no golden file)."
        )
        parts.append(gate_line)

    constraints = [c for c in (_s(x) for x in _as_list(d.get("constraints"))) if c]
    if constraints:
        parts.append("Constraints from the registry:\n" + "\n".join(f"  - {c}" for c in constraints))

    hw = _s(gpu_info)
    if hw:
        parts.append(hw)

    return "\n\n".join(p for p in parts if p.strip()).strip() + "\n"


def baseline_targets_text(describe: Mapping[str, Any] | None) -> str:
    """Render the incumbent's measured latency as the performance target."""
    d: Mapping[str, Any] = describe if isinstance(describe, Mapping) else {}
    base = d.get("baseline")
    if not isinstance(base, Mapping) or not base:
        return ""
    name = _s(base.get("name"), "incumbent")
    lines = [f"- incumbent: {name}"]
    lat = base.get("latency_us")
    if isinstance(lat, (int, float)):
        lines.append(f"- incumbent_latency_us: {float(lat):.3f}")
    notes = _s(base.get("notes"))
    if notes:
        lines.append(f"- notes: {notes}")
    lines.append(
        "- objective: speedup = incumbent_latency / candidate_latency, measured back to back "
        "in the same process. Beat 1.0."
    )
    return "\n".join(lines)


_INVARIANT_BLOCK = """Hard rules for this task (the referee enforces them):
- You are rewriting the body of exactly ONE kernel launch. Do NOT add, remove, split,
  merge or reorder launches. The launch boundary is fixed.
- The kernel must be CUDA-graph safe: no device allocation, no host synchronization,
  no default-stream launch, no printf, no cudaMalloc/cudaFree/cudaDeviceSynchronize.
  Dynamic shared memory must stay within the device's opt-in limit.
- Use the stream FlashRT passes in. Never create your own stream or event.
- Arguments are raw device pointers, not torch::Tensor. There is no PyTorch in this ABI.
- Keep the entry symbol and its exact signature; FlashRT links against it.
- The correctness reference is produced live by the production kernel on the same
  inputs, so there is no tolerance to game and no golden output to match textually."""


def _cuda_xml_and_guidelines_block(*, target_gpu: str) -> str:
    tg = _s(target_gpu, "the target GPU")
    return f"""IMPORTANT: Generate code in XML format with exactly 3 files with these strict names.
Only `kernel.h` and `kernel.cu` are shipped to the referee; `main.cpp` is required by the
container format but its contents are REPLACED with a fixed stub, so keep it trivial.

<header_file name="kernel.h">
- Declaration of the launch entry with the exact signature given in the specification
- Any struct/type definitions the entry needs
- Include guards and the headers you need
</header_file>

<cuda_file name="kernel.cu">
- The __global__ kernel(s) and __device__ helpers
- The host-side launch entry declared in kernel.h: it computes the grid/block and
  launches on the stream it is given. Nothing else.
- This file is the artifact. Everything that matters goes here.
</cuda_file>

<cpp_file name="main.cpp">
// unused for this task; emit exactly this one comment line
</cpp_file>

Code generation guidelines:
- Write for {tg} as described by the target profile above; do not assume a different
  architecture, and do not assume a numeric format the profile says is unavailable.
- Guard anything architecture-specific behind __CUDA_ARCH__ so the file still compiles.
- No torch/extension.h, no PYBIND11_MODULE, no at:: or torch:: types anywhere.
- No dynamic allocation, no host sync, no stream creation (see the hard rules).
- Prefer __restrict__ pointers and vectorized (16-byte) accesses where alignment allows."""


def code_format_text(*, language: str, target_gpu: str) -> str:
    """Format/output guidance for world-model prompts (`{code_format}`)."""
    if _s(language).lower() != "cuda":
        return ""
    return (_cuda_xml_and_guidelines_block(target_gpu=target_gpu) + "\n\n" + _INVARIANT_BLOCK).strip()


def _optimization_strategy_block(*, target_gpu: str) -> str:
    tg = _s(target_gpu, "the target GPU")
    return f"""Optimization strategy:
1. ENSURE CORRECTNESS FIRST. If the referee reported `got_to: compile`, fix the nvcc
   diagnostic verbatim in the log. If it reported `got_to: correctness`, the kernel
   compiled and ran but disagreed with the production kernel — look at the per-output
   cosine / rel_l2 before changing anything about performance.
2. If `graph_safe` is false, fix that before anything else: a graph-unsafe kernel can
   never be accepted no matter how fast it is.
3. Only once it passes, optimize. What this harness measures on {tg}:
   - The score is LATENCY at batch size 1 — a single launch, single stream. Not
     throughput and not bytes/second. An optimization that only pays off at large
     batch is worth nothing here.
   - M (the token/batch dimension) is small and fixed. Treat it as a constant.
   - Every reported latency includes a fixed harness overhead: an empty kernel
     measures a non-zero time. The referee reports `harness_overhead_us`. Subtract
     it before reasoning about kernel work, and never target a latency below it.
   - The referee reports `traffic_floor_us`: the measured cost of moving this
     tunable's operands with NO arithmetic, timed through the identical path.
     It is an unrolled reference streamer, not a proven bound — a very good
     kernel can pass it. Compare against that measured number, not intuition.
   - If your latency is far above `traffic_floor_us`, the gap is arithmetic and
     launch cost, not memory. Do not assume a low FLOP/byte ratio means the
     arithmetic is free — check the reported numbers instead of reasoning from
     the shape.
   - Weights are cold: both sides are measured under an L2 flush."""


def per_task_requirement_text(*, language: str, target_gpu: str, phase: str = "") -> str:
    """Task-specific requirements injected via `{per_task_requirement}`."""
    if _s(language).lower() != "cuda":
        return ""
    raw_ph = _s(phase).lower()
    if raw_ph in ("optimize", "optimization", "improve") or raw_ph.startswith("opt"):
        return (
            _optimization_strategy_block(target_gpu=target_gpu)
            + "\n\n"
            + _INVARIANT_BLOCK
            + "\n\n"
            + _cuda_xml_and_guidelines_block(target_gpu=target_gpu)
        ).strip()
    return (
        _INVARIANT_BLOCK + "\n\n" + _cuda_xml_and_guidelines_block(target_gpu=target_gpu)
    ).strip()


def _floor_lines(floor: Any, latency_us: Any) -> list[str]:
    """Surface the referee's measured floors as structured feedback.

    They also appear inside `log`, but that is truncated, and these two numbers
    are the only thing in the payload that says which resource is actually
    binding. Losing them to a long nvcc warning list would be silent.
    """
    if not isinstance(floor, Mapping):
        return []
    tf = floor.get("traffic_floor_us")
    ov = floor.get("harness_overhead_us")
    if not isinstance(tf, (int, float)) or not isinstance(ov, (int, float)):
        return []
    out = [f"floors: traffic_floor_us={float(tf):.3f} harness_overhead_us={float(ov):.3f}"
           + (f" floor_achieved_gbs={float(floor['floor_achieved_gbs']):.0f}"
              if isinstance(floor.get("floor_achieved_gbs"), (int, float)) else "")]
    if isinstance(latency_us, (int, float)):
        delta = float(latency_us) - float(tf)
        out.append(
            f"floors: your kernel is {delta:+.3f} us vs the traffic floor. The floor "
            "already pays for every byte this tunable moves, so that difference is "
            "arithmetic and launch cost, not memory.")
    return out


def format_evaluate_feedback(payload: Mapping[str, Any] | None, *, max_log_chars: int = 6000,
                             spread: Mapping[str, Any] | None = None) -> str:
    """Render one `evaluate` JSON payload as next-round feedback.

    `log` is load-bearing (design §1.1): it is how a compiler error or a numeric
    mismatch becomes the next prompt, so it is kept last and truncated from the front
    (the tail of an nvcc log carries the error summary).
    """
    p: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    lines: list[str] = []

    got_to = _s(p.get("got_to"), "unknown")
    ok = bool(p.get("ok", False))
    lines.append(f"referee: got_to={got_to} ok={str(ok).lower()}")

    if p.get("graph_safe") is not None:
        gs = bool(p.get("graph_safe"))
        lines.append(f"graph_safe={str(gs).lower()}" + ("" if gs else "  <-- BLOCKING: fix this first"))

    corr = p.get("correctness")
    if isinstance(corr, Mapping) and corr:
        for out_name in sorted(corr.keys()):
            entry = corr.get(out_name)
            if not isinstance(entry, Mapping):
                continue
            bits = [f"{k}={entry.get(k)}" for k in sorted(entry.keys())]
            lines.append(f"correctness[{out_name}]: " + " ".join(bits))

    if ok:
        lat = p.get("latency_us")
        base = p.get("baseline_latency_us")
        spd = p.get("speedup")
        perf = []
        if isinstance(lat, (int, float)):
            perf.append(f"latency_us={float(lat):.3f}")
        if isinstance(base, (int, float)):
            perf.append(f"incumbent_us={float(base):.3f}")
        if isinstance(spd, (int, float)):
            perf.append(f"speedup={float(spd):.3f}x")
        if perf:
            lines.append("perf: " + " ".join(perf))
        if isinstance(spread, Mapping) and spread.get("n"):
            lines.append(
                f"perf: scored as the MEDIAN of {int(spread['n'])} independent "
                f"evaluations (min {float(spread['min']):.3f}x, "
                f"max {float(spread['max']):.3f}x). A difference smaller than that "
                "min-max range is measurement noise, not a real improvement.")
        lines.extend(_floor_lines(p.get("floor"), lat))
    else:
        lines.append("perf: withheld — the candidate did not reach a trustworthy timing")

    dev = p.get("device")
    if isinstance(dev, Mapping) and dev:
        bits = [f"{k}={dev.get(k)}" for k in sorted(dev.keys())]
        lines.append("device: " + " ".join(bits))
        if dev.get("throttled"):
            lines.append("device: THROTTLED — treat this round's timing as unreliable")

    log = _s(p.get("log"))
    if log:
        if max_log_chars and len(log) > int(max_log_chars):
            log = "...<truncated>...\n" + log[-int(max_log_chars) :]
        lines.append("referee log:\n" + log)

    return "\n".join(lines).strip()


def unavailable_profile_text(target_gpu: str = "") -> str:
    """The block a prompt shows when no target profile exists. Never a guess."""
    from k_search.utils.cuda_gpu_info import get_gpu_info_or_placeholder

    return get_gpu_info_or_placeholder(None, str(target_gpu or ""))


__all__: Sequence[str] = (
    "baseline_targets_text",
    "code_format_text",
    "format_evaluate_feedback",
    "per_task_requirement_text",
    "render_describe",
    "unavailable_profile_text",
)
