"""FlashRT tunable Task implementation.

K-Search keeps its search loop; FlashRT replaces the evaluator. This adapter is the
whole of the K-Search side: it shells out to the FlashRT referee through a transport,
maps the referee's JSON into `EvalResult`, and renders `describe` into prompts.

It contains no compiler invocation, no correctness comparison, no timing code and no
device probe. It must import and run on a host with no GPU and no FlashRT install;
only `run_benchmark` and `get_definition_text` need a reachable referee.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)
from k_search.tasks.flashrt import prompts as flashrt_prompts
from k_search.tasks.flashrt.transport import (
    SOURCES_DIR_TOKEN,
    LocalTransport,
    Transport,
    TransportResult,
)
from k_search.utils.cuda_gpu_info import get_gpu_info_or_placeholder


# The referee only consumes these; `main.cpp` exists solely because the CUDA container
# format demands three files.
REFEREE_SOURCE_FILES = ("kernel.h", "kernel.cu")

# Replaces whatever the model emitted as main.cpp. The generator's required-file check
# runs *before* solution construction, so the model must still emit a non-empty
# main.cpp — we can only replace its contents, not drop it from the contract.
MAIN_CPP_STUB = """// Unused for FlashRT tunable tasks.
//
// The referee compiles kernel.cu with FlashRT's own toolchain and links it into
// FlashRT's fixture. There is no PyTorch extension and no host entry point here.
// K-Search replaces this file's contents unconditionally; the kernel is the artifact.
"""

# `got_to` (how far the candidate got) -> EvalResult.status when ok is false.
_GOT_TO_STATUS: dict[str, str] = {
    "compile": "compile_error",
    "lint": "graph_unsafe",
    "graph_safety": "graph_unsafe",
    "correctness": "wrong_answer",
    "timing": "timing_failed",
    # A `timed` round with ok=false is the canary tripping, not a transport fault.
    "timed": "timing_failed",
}


@dataclass(frozen=True)
class FlashRTTunableTaskConfig:
    tunable_id: str
    target: str = ""
    target_profile: Optional[str] = None
    target_gpu: str = ""
    timeout_seconds: float = float(os.getenv("KSEARCH_FLASHRT_TIMEOUT_SECONDS", "900"))
    describe_timeout_seconds: float = float(os.getenv("KSEARCH_FLASHRT_DESCRIBE_TIMEOUT_SECONDS", "120"))
    max_failure_excerpt_chars: int = int(os.getenv("KSEARCH_FLASHRT_FAILURE_EXCERPT_CHARS", "4000"))
    # Repeating the whole evaluation was the wrong lever. Measured ICC is 0.128,
    # i.e. 87% of the variance is WITHIN a process, so a fresh process per repeat
    # pays ~12 s of startup to resample the same noise. The referee's estimator
    # was the real source (a median sitting in the gap of a bimodal distribution)
    # and fixing that took the ratio's spread from 1.10% to 0.14% for free. Raise
    # this only to cross-check a headline result, not as a matter of course.
    repeats: int = max(1, int(os.getenv("KSEARCH_FLASHRT_REPEATS", "1")))


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s or "")).strip("_")


def parse_referee_json(stdout: str) -> dict[str, Any] | None:
    """Extract the referee's JSON object from stdout, tolerating leading chatter."""
    text = str(stdout or "")
    if not text.strip():
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    return None


class FlashRTTunableTask:
    """K-Search Task: one FlashRT tunable, judged by the FlashRT referee."""

    def __init__(
        self,
        *,
        tunable_id: str,
        transport: Transport | None = None,
        target: str = "",
        target_profile: str | None = None,
        target_gpu: str = "",
        timeout_seconds: float | None = None,
        artifacts_dir: str | None = None,
        name: str | None = None,
    ) -> None:
        tid = str(tunable_id or "").strip()
        if not tid:
            raise ValueError("FlashRTTunableTask requires a tunable id (--flashrt-tunable)")

        # Namespace the task by (tunable, target): artifacts live under
        # <artifacts>/<task_name>/, so a Thor run and an A100 run of the same tunable
        # must not share a directory (design §4.3).
        tgt = str(target or "").strip()
        default_name = f"flashrt_{_sanitize(tid)}" + (f"__{_sanitize(tgt)}" if tgt else "")
        self._name = str(name or default_name)

        # Translate the profile before anything reads it. Shared prompt sites resolve
        # it through the KSEARCH_TARGET_PROFILE env var rather than through the task,
        # so re-pointing that var is what actually gets the numbers into the world
        # model. See k_search/tasks/flashrt/profile.py for why the two schemas differ.
        profile_path = str(target_profile) if target_profile else None
        if profile_path:
            profile_path = self._conform_profile(profile_path, artifacts_dir)

        cfg_kwargs: dict[str, Any] = {
            "tunable_id": tid,
            "target": tgt,
            "target_profile": profile_path,
            "target_gpu": str(target_gpu or ""),
        }
        if timeout_seconds is not None:
            cfg_kwargs["timeout_seconds"] = float(timeout_seconds)
        self._cfg = FlashRTTunableTaskConfig(**cfg_kwargs)

        self._transport: Transport = transport if transport is not None else LocalTransport()
        self._ksearch_artifacts_dir: str | None = (str(artifacts_dir) if artifacts_dir is not None else None)
        self._solutions: dict[str, Solution] = {}
        self._describe_cache: dict[str, Any] | None = None

        # Last-round feedback cache (read by generators via getattr).
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = 1
        self._last_round_summary_line: str = ""

    @staticmethod
    def _conform_profile(profile_path: str, artifacts_dir: str | None) -> str:
        """Rewrite a FlashRT profile into the schema the CUDA prompt blocks read.

        Best-effort: a profile that cannot be read or translated is passed through
        unchanged, because a degraded hardware block is better than no task.
        """
        from k_search.tasks.flashrt.profile import materialize_ksearch_profile
        from k_search.utils.cuda_gpu_info import TARGET_PROFILE_ENV_VAR, get_gpu_info

        try:
            dest = Path(artifacts_dir) if artifacts_dir else Path(profile_path).parent
            conformed = materialize_ksearch_profile(profile_path, dest)
        except Exception:
            return profile_path
        if conformed != profile_path:
            os.environ[TARGET_PROFILE_ENV_VAR] = conformed
            get_gpu_info.cache_clear()
        return conformed

    # ------------------------------------------------------------------ protocol

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        lang = str(language or "cuda").strip().lower()
        if lang != "cuda":
            raise ValueError(f"FlashRTTunableTask only supports language='cuda'; got {lang!r}")
        return flashrt_prompts.render_describe(self._describe(), gpu_info=self.get_gpu_info_block())

    def get_solution(self, solution_name: str) -> Solution | None:
        name = str(solution_name)
        if name in self._solutions:
            return self._solutions.get(name)
        try:
            d = load_ksearch_solution_json(
                solution_ref=name,
                definition_name=str(self.name or ""),
                artifacts_dir=self._ksearch_artifacts_dir,
            )
            sol = solution_from_json_dict(d)
            if str(sol.definition or "") != str(self.name or ""):
                return None
            self._solutions[str(sol.name)] = sol
            return sol
        except FileNotFoundError:
            return None
        except Exception:
            return None

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        sources = self._sources_for_referee(solution)
        missing = [f for f in REFEREE_SOURCE_FILES if not str(sources.get(f, "")).strip()]
        if missing:
            return self._finish(
                EvalResult(
                    status="failed",
                    log_excerpt=f"[flashrt] solution is missing required source files: {missing}",
                    metrics=self._base_metrics(got_to="sources", score=None),
                ),
                round_num=round_num,
                feedback=f"[flashrt] solution is missing required source files: {missing}",
            )

        res, payload, spread = self._evaluate_repeated(sources)

        if payload is None:
            reason = "timed out" if res.timed_out else f"exit={res.returncode}"
            log = f"[flashrt] referee produced no JSON ({reason})\n{res.combined_log(max_chars=8000)}"
            return self._finish(
                EvalResult(
                    status=("timeout" if res.timed_out else "failed"),
                    log_excerpt=log,
                    metrics=self._base_metrics(got_to="transport", score=None),
                ),
                round_num=round_num,
                feedback=log,
            )

        return self._finish(
            self._eval_result_from_payload(payload, transport_result=res),
            round_num=round_num,
            feedback=flashrt_prompts.format_evaluate_feedback(payload, spread=spread),
        )

    def _evaluate_repeated(self, sources: dict[str, str]) -> tuple[Any, dict[str, Any] | None, dict[str, float] | None]:
        """Evaluate `repeats` times and return the run holding the MEDIAN speedup.

        Returning a real run rather than a synthesised average keeps `latency_us`,
        `log` and `correctness` mutually consistent — the model is shown one
        coherent measurement, just a representative one instead of a lucky one.

        Only a run that reached a trustworthy timing is worth repeating: a compile
        or correctness failure is deterministic, so repeating it would triple the
        cost of the cheapest possible round for no information.
        """
        argv = ["evaluate", self._cfg.tunable_id, "--sources", SOURCES_DIR_TOKEN, "--json"]
        runs: list[tuple[float, Any, dict[str, Any]]] = []
        bad: list[tuple[Any, dict[str, Any]]] = []
        res = None
        payload = None
        for _ in range(max(1, int(self._cfg.repeats))):
            res = self._transport.run(argv, sources=sources,
                                      timeout_seconds=self._cfg.timeout_seconds)
            payload = parse_referee_json(res.stdout)
            if payload is None:
                return res, None, None
            speedup = payload.get("speedup")
            if not payload.get("ok") or not isinstance(speedup, (int, float)):
                # One transient canary trip should not throw away the samples that
                # did measure cleanly; only give up if no majority survives.
                bad.append((res, payload))
                if len(bad) > max(1, int(self._cfg.repeats)) // 2:
                    return bad[-1][0], bad[-1][1], None
                continue
            runs.append((float(speedup), res, payload))

        if not runs:
            return (bad[-1][0], bad[-1][1], None) if bad else (res, payload, None)
        runs.sort(key=lambda r: r[0])
        med_speedup, med_res, med_payload = runs[len(runs) // 2]
        spread = {
            "n": float(len(runs)),
            "median": med_speedup,
            "min": runs[0][0],
            "max": runs[-1][0],
        }
        med_payload = dict(med_payload)
        med_payload["speedup_samples"] = [r[0] for r in runs]
        med_payload["speedup_spread"] = spread
        return med_res, med_payload, spread

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        """Pass only kernel.cu to the world model; the rest is boilerplate."""
        try:
            sraw = str(raw or "")
            if not sraw.strip():
                return ""
            if str(language or "").lower() != "cuda":
                return sraw
            m = re.search(
                r'<cuda_file\s+name="kernel\.cu"\s*>([\s\S]*?)</cuda_file>',
                sraw,
                flags=re.IGNORECASE,
            )
            if m:
                cu = m.group(1)
                cu = re.sub(r"^\s*<!\[CDATA\[\s*", "", cu)
                cu = re.sub(r"\s*\]\]>\s*$", "", cu)
                cu = cu.strip()
                if cu:
                    return cu
            return sraw
        except Exception:
            return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, config=config, dump_traces=False, round_num=0)

    def get_config_for_logging(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "task_type": "flashrt",
            "task_name": self._name,
            "tunable_id": self._cfg.tunable_id,
            "target": self._cfg.target,
            "target_gpu": self._cfg.target_gpu,
            "target_profile": self._cfg.target_profile,
            "timeout_seconds": float(self._cfg.timeout_seconds),
        }
        try:
            cfg.update(self._transport.describe_for_logging())
        except Exception:
            cfg["transport"] = "unknown"
        return cfg

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for sol in solutions or []:
            if sol is None:
                continue
            er = self.run_benchmark(solution=sol, dump_traces=False, round_num=None)
            metrics = er.metrics if isinstance(er.metrics, dict) else {}
            out.append(
                {
                    "solution": str(getattr(sol, "name", "") or ""),
                    "status": str(er.status or ""),
                    "got_to": metrics.get("got_to"),
                    "graph_safe": metrics.get("graph_safe"),
                    "latency_ms": er.latency_ms,
                    "reference_latency_ms": er.reference_latency_ms,
                    "speedup": er.speedup_factor,
                    "score_name": metrics.get("score_name"),
                    "score": metrics.get("score"),
                    "device": metrics.get("device"),
                }
            )
        best = max(
            (s for s in out if s.get("status") == "passed"),
            key=lambda s: float(s.get("speedup") or 0.0),
            default=None,
        )
        return {
            "task": str(self._name),
            "tunable_id": str(self._cfg.tunable_id),
            "target": str(self._cfg.target),
            "solutions": out,
            "summary": {
                "best_solution": (best.get("solution") if best else None),
                "best_speedup": (best.get("speedup") if best else None),
            },
        }

    # -------------------------------------------------------------- optional hooks

    def get_last_round_trace_logs_for_prompt(self) -> str:
        return str(getattr(self, "_last_round_trace_logs_for_prompt", "") or "")

    def get_last_round_passed_count(self) -> int:
        try:
            return int(getattr(self, "_last_round_passed_count", 0) or 0)
        except Exception:
            return 0

    def get_last_round_total_workloads(self) -> int:
        try:
            return int(getattr(self, "_last_round_total_workloads", 1) or 1)
        except Exception:
            return 1

    def get_last_round_summary_line(self) -> str:
        return str(getattr(self, "_last_round_summary_line", "") or "")

    def has_last_round_feedback_trace(self) -> bool:
        return bool(str(getattr(self, "_last_round_trace_logs_for_prompt", "") or "").strip())

    def get_baseline_targets_text(self) -> str:
        try:
            return flashrt_prompts.baseline_targets_text(self._describe())
        except Exception:
            return ""

    def get_per_task_requirement_text(self, *, language: str, target_gpu: str, phase: str = "") -> str:
        try:
            return flashrt_prompts.per_task_requirement_text(
                language=str(language),
                target_gpu=(str(target_gpu) or self._cfg.target_gpu),
                phase=str(phase or ""),
            )
        except Exception:
            return ""

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        try:
            return flashrt_prompts.code_format_text(
                language=str(language), target_gpu=(str(target_gpu) or self._cfg.target_gpu)
            )
        except Exception:
            return ""

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        lang = str(language or "").strip().lower()
        if lang != "cuda":
            raise ValueError(f"FlashRTTunableTask only supports language='cuda'; got {lang!r}")

        files: dict[str, str] = {}
        if isinstance(cleaned_code, dict):
            files = {str(k): str(v) for k, v in cleaned_code.items()}
        else:
            files = self._parse_cuda_xml(str(raw_code or ""))

        # The kernel is the artifact: whatever the model wrote in main.cpp is discarded.
        sources = [
            SourceFile(path="kernel.h", content=str(files.get("kernel.h", "") or "")),
            SourceFile(path="kernel.cu", content=str(files.get("kernel.cu", "") or "")),
            SourceFile(path="main.cpp", content=MAIN_CPP_STUB),
        ]

        hw = str(target_gpu or "").strip() or str(self._cfg.target or self._cfg.target_gpu or "unknown")
        sol_name = f"{model_name}_{self._name}_cuda_r{int(round_num)}"
        sol = Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.CUDA,
                target_hardware=[hw],
                # kernel.cu is what the referee compiles; there is no main.cpp entry.
                entry_point="kernel.cu::run",
            ),
            sources=sources,
            description=f"FlashRT tunable {self._cfg.tunable_id} (round {round_num})",
        )
        self._solutions[sol_name] = sol
        return sol

    # ------------------------------------------------------------------- helpers

    def get_gpu_info_block(self) -> str:
        """The rendered target profile, or a plain statement that none is available."""
        return get_gpu_info_or_placeholder(self._cfg.target_profile, self._cfg.target_gpu)

    def register_solution(self, sol: Solution) -> None:
        if not isinstance(sol, Solution):
            raise TypeError("register_solution expects a k_search.tasks.task_base.Solution")
        self._solutions[str(sol.name)] = sol

    def list_tunables(self) -> TransportResult:
        """`referee list` — exposed for scripts/CLIs, not used by the search loop."""
        return self._transport.run(["list"], timeout_seconds=self._cfg.describe_timeout_seconds)

    def _describe(self) -> dict[str, Any]:
        if self._describe_cache is not None:
            return self._describe_cache
        res = self._transport.run(
            ["describe", self._cfg.tunable_id, "--json"],
            timeout_seconds=self._cfg.describe_timeout_seconds,
        )
        payload = parse_referee_json(res.stdout)
        if payload is None:
            raise RuntimeError(
                f"FlashRT referee `describe {self._cfg.tunable_id}` returned no JSON "
                f"(exit={res.returncode}, timed_out={res.timed_out}).\n"
                f"{res.combined_log(max_chars=4000)}"
            )
        self._describe_cache = payload
        return payload

    @staticmethod
    def _parse_cuda_xml(raw: str) -> dict[str, str]:
        patterns = {
            "kernel.h": r'<header_file name="kernel\.h">(.*?)</header_file>',
            "kernel.cu": r'<cuda_file name="kernel\.cu">(.*?)</cuda_file>',
            "main.cpp": r'<cpp_file name="main\.cpp">(.*?)</cpp_file>',
        }
        out: dict[str, str] = {}
        for fname, pat in patterns.items():
            m = re.search(pat, str(raw or ""), re.DOTALL)
            if m:
                out[fname] = m.group(1).strip()
        return out

    @staticmethod
    def _sources_for_referee(solution: Solution) -> dict[str, str]:
        by_path = {str(sf.path): str(sf.content or "") for sf in (solution.sources or [])}
        return {f: by_path.get(f, "") for f in REFEREE_SOURCE_FILES}

    def _base_metrics(self, *, got_to: str, score: float | None) -> dict[str, Any]:
        return {
            "score_name": "speedup_vs_incumbent",
            "score": score,
            "got_to": got_to,
            "tunable_id": str(self._cfg.tunable_id),
            "target": str(self._cfg.target),
        }

    def _eval_result_from_payload(
        self, payload: Mapping[str, Any], *, transport_result: TransportResult | None = None
    ) -> EvalResult:
        got_to = str(payload.get("got_to") or "unknown").strip().lower()
        ok = bool(payload.get("ok", False))
        log = str(payload.get("log") or "")
        if not log and transport_result is not None:
            log = transport_result.combined_log(max_chars=8000)

        metrics = self._base_metrics(got_to=got_to, score=None)
        metrics["graph_safe"] = payload.get("graph_safe")
        if isinstance(payload.get("correctness"), dict):
            metrics["correctness"] = payload.get("correctness")
        if isinstance(payload.get("device"), dict):
            metrics["device"] = payload.get("device")

        if not ok:
            # Never hand the world model partial performance evidence: a candidate that
            # failed correctness may well have a "fast" timing, and a throttled or
            # aborted run's numbers are noise. Perf fields stay null.
            return EvalResult(
                status=_GOT_TO_STATUS.get(got_to, "failed"),
                latency_ms=None,
                reference_latency_ms=None,
                mean_vs_baseline_factor=None,
                speedup_factor=None,
                log_excerpt=log,
                metrics=metrics,
            )

        latency_us = payload.get("latency_us")
        base_us = payload.get("baseline_latency_us")
        speedup = payload.get("speedup")

        latency_ms = float(latency_us) / 1000.0 if isinstance(latency_us, (int, float)) else None
        base_ms = float(base_us) / 1000.0 if isinstance(base_us, (int, float)) else None
        speedup_f = float(speedup) if isinstance(speedup, (int, float)) else None
        if speedup_f is None and latency_ms and base_ms and latency_ms > 0:
            speedup_f = base_ms / latency_ms

        metrics["score"] = speedup_f
        metrics["latency_us"] = latency_us
        metrics["baseline_latency_us"] = base_us

        return EvalResult(
            status="passed",
            latency_ms=latency_ms,
            reference_latency_ms=base_ms,
            # The objective is the ratio, not the microseconds (design §4.3).
            mean_vs_baseline_factor=speedup_f,
            speedup_factor=speedup_f,
            log_excerpt=log,
            metrics=metrics,
        )

    def _finish(self, er: EvalResult, *, round_num: int | None, feedback: str) -> EvalResult:
        """Populate the last-round caches and print one compact summary line."""
        passed = bool(er.is_passed())
        self._last_round_trace_logs_for_prompt = str(feedback or "")
        self._last_round_total_workloads = 1
        self._last_round_passed_count = 1 if passed else 0

        try:
            rn = str(int(round_num)) if round_num is not None else "?"
            metrics = er.metrics if isinstance(er.metrics, dict) else {}
            lat = er.latency_ms
            lat_text = f"{float(lat) * 1000.0:.3f} us" if isinstance(lat, (int, float)) else "-"
            spd = er.speedup_factor
            spd_text = f"{float(spd):.3f}x" if isinstance(spd, (int, float)) else "-"
            self._last_round_summary_line = (
                f"[{self._name}] Round {rn}: status={er.status} | got_to={metrics.get('got_to')} | "
                f"latency={lat_text} | speedup={spd_text} | tunable={self._cfg.tunable_id}"
            )
            print(self._last_round_summary_line, flush=True)
            if not passed:
                le = str(er.log_excerpt or "").strip()
                if le:
                    max_chars = int(self._cfg.max_failure_excerpt_chars or 4000)
                    if len(le) > max_chars:
                        le = le[:max_chars] + "...<truncated>..."
                    print(f"[{self._name}] Failure excerpt:\n{le}", flush=True)
        except Exception:
            pass
        return er
