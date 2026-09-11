# FlashRT integration — K-Search side

Implements §3 (K-Search adapter) and §4.2 (profile consumption) of
`FlashRT_dev/docs/kernel_search_integration.md`. Everything here runs with no GPU,
no CUDA toolkit and no FlashRT install; the referee is reached only through
`k_search/tasks/flashrt/transport.py`.

## Files

| File | Role |
|---|---|
| `k_search/utils/cuda_gpu_info.py` | target profile → prompt block; static table fallback |
| `k_search/tasks/flashrt/transport.py` | `local` \| `ssh` \| `service` referee invocation |
| `k_search/tasks/flashrt/prompts.py` | renders `describe` / `evaluate` into prompt blocks |
| `k_search/tasks/flashrt_tunable_task.py` | the `Task`: referee JSON → `EvalResult` |
| `tests/` | offline tests (`python -m unittest discover -s tests -t .`) |

Edited: `kernel_generator_prompts.py`, `world_model_prompts.py`, `world_model.py`
(prompt slots only), `generate_kernels_and_eval.py`, `k_search/tasks/__init__.py`.

## Decisions where the design was silent

**How the profile reaches the deep prompt builders.** §4.2 says the profile is injected
through a `{gpu_info}` slot "mirroring what the MLX path already does", but the MLX
path auto-detects with a zero-argument call from inside the prompt module, and the
CUDA path has a file path that arrives on the command line. Threading a new parameter
from the CLI down through `WorldModelManager` into every prompt builder would mean
editing the search loop, which was out of scope. So `get_gpu_info(profile_path,
target_gpu)` falls back to the `KSEARCH_TARGET_PROFILE` environment variable when
`profile_path` is `None`, and `generate_kernels_and_eval.py` sets that variable from
`--target-profile` immediately after parsing. Call sites stay identical in shape to
the MLX ones. `FlashRTTunableTask` passes its path explicitly and does not depend on
the variable.

**Profile artifact schema.** §4.1 describes the two layers in prose but names no
schema. Chosen shape (see `tests/test_cuda_gpu_info.py::A100_PROFILE` for a full
example):

```jsonc
{
  "schema_version": 1,
  "target": "local-a100",
  "hardware": { "device_name", "compute_capability", "arch", "sm_count",
                "shared_memory_per_sm_kb", "shared_memory_per_block_optin_kb",
                "registers_per_sm", "warp_size", "l2_cache_mb",
                "total_memory_gb", "free_memory_gb", "sm_clock_mhz", "memory_clock_mhz",
                "measured_copy_bandwidth_gb_s", "graph_node_replay_overhead_us",
                "tensor_cores", "supported_dtypes", "unsupported_dtypes", "features" },
  "build":    { "available", "reason", "flashrt_arch", "kernel_modules",
                "reachable_formats", "unreachable_formats", "nvcc_flags": {"<cmake target>": "..."} }
}
```

Every field is optional and renders as `unknown` when absent — never as a guess. A
*flat* document (no `hardware` key) is accepted as hardware-only, so a hand-written
stub artifact works during bring-up. `build.available: false` renders the build layer
as explicitly absent, which is the §4.1 "flash_rt not importable" case.

**`describe` output schema** (§7 open question 2 — "JSON with a rendered-text field,
or both?"). Answer: **both**. `render_describe` uses the structured fields
(`signature`, `tensors`, `scalars`, `launch`, `semantics`, `constraints`, `gate`,
`fixture_mode`, `baseline`) when present, but if the payload carries a `text` field it
is used verbatim as the body instead. That lets FlashRT improve the wording without a
K-Search release, while keeping the fields available for anything that wants to reason
over them. Missing fields are simply omitted, per §1.4 ("documentation, not a
specification").

**Tunable-id grammar** (§7 open question 1) is FlashRT's to decide. K-Search treats
the id as an opaque string and only sanitizes it for filesystem paths.

**Artifact namespacing** (§7 open question 6). The full fix touches `paths.py` and
`Solution.hash()` and was out of scope. Cheap mitigation applied instead: the task's
default `name` is `flashrt_<tunable>__<target>`, and artifacts live at
`<artifacts>/<task_name>/`, so an A100 run and a Thor run of the same tunable no
longer collide *by path*. They still collide by `Solution.hash()`, which excludes
`spec.target_hardware`. Unresolved.

**`--target-gpu` vs `--target-profile`** (§7 open question 7). Neither errors nor is
ignored: the profile wins, and a disagreement between the profile's device name and
`--target-gpu` emits two `WARNING:` lines in the block telling the model the profile
is authoritative. Erroring would break the common case of a slightly-differently
spelled SKU name.

**Status vocabulary.** `EvalResult.status` for a non-`ok` referee result is derived
from `got_to`: `compile → compile_error`, `lint`/`graph_safety → graph_unsafe`,
`correctness → wrong_answer`, `timing → timing_failed`, anything else → `failed`. A
transport-level failure (no JSON at all) is `failed`, or `timeout` when the transport
timed out. Only `"passed"` is load-bearing elsewhere in K-Search (`is_passed()`), so
the rest are free-form and chosen to be readable in logs.

**Perf nulling.** On any non-`ok` result, `latency_ms`, `reference_latency_ms`,
`speedup_factor`, `mean_vs_baseline_factor` and `metrics["score"]` are all `None`,
even when the referee reported numbers. A candidate that failed the correctness gate
can easily be "fast", and feeding that to the world model teaches it the wrong
lesson. `graph_safe`, `correctness` and `device` are still forwarded — they are
diagnostics, not performance evidence.

**Transport interface.** One method, `run(argv, *, sources=None, timeout_seconds=None)
-> TransportResult`, returning `returncode` / `stdout` / `stderr` / `timed_out`. The
literal token `{sources}` in `argv` is replaced by the directory the transport put the
sources in — that is what lets the same argv work when the directory has to be created
on a remote host. Transports never parse; `parse_referee_json` in the task does, and
tolerates chatter before the JSON object. `ssh` shells out to the system `ssh`/`scp`
so authentication and host config come from the user's setup; `service` uses `urllib`
so no HTTP dependency is added.

**`main.cpp`.** Replaced unconditionally with `MAIN_CPP_STUB` in
`make_solution_from_generated_code`. Per §7 open question 8 the generator's
required-file check runs *before* this hook, so the prompt still asks for three files;
`code_format_text` tells the model to emit a one-line comment there and that its
contents are discarded. `entry_point` is `kernel.cu::run` (there is no host entry
point in this ABI), so `Solution.get_entry_source()` returns the kernel.

**Static table provenance.** The §4.2 fallback table carries spec-sheet numbers, some
of them SKU-dependent (Thor's and Orin's especially). The rendered block is labelled
`static spec-sheet table — NOT measured`, quotes peak rather than achievable
bandwidth, and carries no launch-overhead number at all, so the model cannot mistake
it for a measured profile. When neither source exists the prompt says
`TARGET GPU PROFILE: unavailable` and tells the model not to assume an architecture.

## Not built (out of the requested scope)

- `scripts/flashrt_tunable_wm.sh` (listed in §3) — no wrapper script was requested.
- `--continue-from-world-model auto` refusing a target mismatch (§4.3) — that is
  world-model persistence logic, explicitly out of scope here.
- `Solution.hash()` / `paths.py` namespacing by target (§7 item 6).

## Nothing in the design turned out to be wrong

The one thing worth flagging: §3 lists `--target` as the new CLI flag, but §4.3
defines a *target* as bundling profile + transport + endpoint. Implementing a real
target registry would need a config file format nobody has specified, so the three
pieces are passed separately (`--flashrt-target` names the target for artifact
namespacing, `--target-profile` / `--transport` / `--flashrt-referee-host` /
`--flashrt-referee-endpoint` supply the parts). A `--target` that expands to those
four can be layered on later without touching the adapter.
