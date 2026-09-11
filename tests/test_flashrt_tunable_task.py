"""FlashRT adapter tests. No GPU, no network, no FlashRT install.

Every referee call goes through a fake transport that returns canned JSON.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.tasks.flashrt.transport import SOURCES_DIR_TOKEN, TransportResult  # noqa: E402
from k_search.tasks.flashrt_tunable_task import (  # noqa: E402
    MAIN_CPP_STUB,
    FlashRTTunableTask,
    parse_referee_json,
)
from k_search.tasks.task_base import BuildSpec, Solution, SourceFile, SupportedLanguages  # noqa: E402


DESCRIBE_PAYLOAD = {
    "tunable_id": "pi05.decoder.gemm.m10_n4096_k4096.bf16",
    "title": "Decoder down-projection GEMM, M=10",
    "summary": "Weight-streaming GEMM at batch 1; cuBLASLt reaches 46% of the memory roofline here.",
    "signature": "void flashrt_gemm_m10(const __nv_bfloat16* __restrict__ A, ...)",
    "entry_symbol": "flashrt_gemm_m10",
    "tensors": [
        {"name": "A", "role": "in", "dtype": "bf16", "shape": [10, 4096], "layout": "row-major"},
        {"name": "W", "role": "in", "dtype": "bf16", "shape": [4096, 4096], "layout": "col-major"},
        {"name": "out", "role": "out", "dtype": "bf16", "shape": [10, 4096]},
    ],
    "semantics": "out = A @ W",
    "constraints": ["no allocation", "single launch"],
    "gate": {"metric": "cos", "threshold": 0.999},
    "fixture_mode": "synthetic",
    "baseline": {"name": "cuBLASLt", "latency_us": 44.9},
}

OK_PAYLOAD = {
    "got_to": "timed",
    "ok": True,
    "log": "ptxas info: 64 registers",
    "graph_safe": True,
    "correctness": {"out": {"cos": 0.999987, "rel_l2": 3.1e-4, "pass": True}},
    "latency_us": 9.1,
    "baseline_latency_us": 12.4,
    "speedup": 1.36,
    "device": {"target": "local-a100", "temp_c": 61, "clock_mhz": 1410, "throttled": False},
}


class FakeTransport:
    """Canned referee. Records what it was asked to run."""

    kind = "fake"

    def __init__(self, *, evaluate: Any = None, describe: Any = None, raw_stdout: str | None = None) -> None:
        self._evaluate = evaluate
        self._describe = describe if describe is not None else DESCRIBE_PAYLOAD
        self._raw_stdout = raw_stdout
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        sources: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResult:
        self.calls.append(
            {
                "argv": list(argv),
                "sources": dict(sources or {}),
                "timeout_seconds": timeout_seconds,
            }
        )
        verb = str(argv[0]) if argv else ""
        if verb == "describe":
            return TransportResult(returncode=0, stdout=json.dumps(self._describe))
        if self._raw_stdout is not None:
            return TransportResult(returncode=1, stdout=self._raw_stdout, stderr="boom")
        return TransportResult(returncode=0, stdout=json.dumps(self._evaluate))

    def describe_for_logging(self) -> dict[str, Any]:
        return {"transport": self.kind}


def _make_task(transport: Any, **kwargs: Any) -> FlashRTTunableTask:
    return FlashRTTunableTask(
        tunable_id="pi05.decoder.gemm.m10_n4096_k4096.bf16",
        transport=transport,
        target="local-a100",
        target_gpu="A100",
        **kwargs,
    )


def _solution(*, kernel_cu: str = "__global__ void k() {}", kernel_h: str = "#pragma once") -> Solution:
    return Solution(
        name="sol",
        definition="flashrt",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA, target_hardware=["A100"], entry_point="kernel.cu::run"
        ),
        sources=[
            SourceFile(path="kernel.h", content=kernel_h),
            SourceFile(path="kernel.cu", content=kernel_cu),
            SourceFile(path="main.cpp", content=MAIN_CPP_STUB),
        ],
    )


class ReferereJsonParsingTests(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(parse_referee_json('{"ok": true}'), {"ok": True})

    def test_json_after_chatter(self) -> None:
        out = parse_referee_json("loading kernels...\nwarning: x\n{\"ok\": false, \"got_to\": \"compile\"}\n")
        self.assertEqual(out, {"ok": False, "got_to": "compile"})

    def test_no_json_returns_none(self) -> None:
        self.assertIsNone(parse_referee_json(""))
        self.assertIsNone(parse_referee_json("segmentation fault"))


class EvalResultMappingTests(unittest.TestCase):
    def test_ok_maps_speedup_to_score(self) -> None:
        t = FakeTransport(evaluate=OK_PAYLOAD)
        task = _make_task(t)
        er = task.run_benchmark(solution=_solution(), round_num=1)

        self.assertEqual(er.status, "passed")
        self.assertTrue(er.is_passed())
        self.assertAlmostEqual(er.latency_ms, 0.0091)
        self.assertAlmostEqual(er.reference_latency_ms, 0.0124)
        self.assertAlmostEqual(er.speedup_factor, 1.36)
        self.assertAlmostEqual(er.mean_vs_baseline_factor, 1.36)
        self.assertAlmostEqual(er.score(), 1.36)
        self.assertEqual(er.metrics["score_name"], "speedup_vs_incumbent")
        self.assertEqual(er.metrics["got_to"], "timed")
        self.assertTrue(er.metrics["graph_safe"])
        self.assertEqual(er.metrics["device"]["target"], "local-a100")

    def test_evaluate_call_shape(self) -> None:
        t = FakeTransport(evaluate=OK_PAYLOAD)
        task = _make_task(t, timeout_seconds=42.0)
        task.run_benchmark(solution=_solution(), round_num=1)

        call = t.calls[-1]
        self.assertEqual(
            call["argv"],
            ["evaluate", "pi05.decoder.gemm.m10_n4096_k4096.bf16", "--sources", SOURCES_DIR_TOKEN, "--json"],
        )
        self.assertEqual(call["timeout_seconds"], 42.0)
        # Only the two files the referee consumes are shipped.
        self.assertEqual(sorted(call["sources"].keys()), ["kernel.cu", "kernel.h"])

    def test_compile_failure_nulls_perf_fields(self) -> None:
        payload = {
            "got_to": "compile",
            "ok": False,
            "log": "kernel.cu(12): error: identifier \"foo\" is undefined",
            "graph_safe": None,
            # A referee that leaks stale perf numbers must not reach the world model.
            "latency_us": 9.1,
            "baseline_latency_us": 12.4,
            "speedup": 1.36,
        }
        er = _make_task(FakeTransport(evaluate=payload)).run_benchmark(solution=_solution(), round_num=2)

        self.assertEqual(er.status, "compile_error")
        self.assertFalse(er.is_passed())
        self.assertIsNone(er.latency_ms)
        self.assertIsNone(er.reference_latency_ms)
        self.assertIsNone(er.speedup_factor)
        self.assertIsNone(er.mean_vs_baseline_factor)
        self.assertIsNone(er.metrics["score"])
        self.assertEqual(er.score(), -1.0)
        self.assertNotIn("latency_us", er.metrics)
        self.assertIn("identifier", er.log_excerpt)

    def test_correctness_failure_status_and_metrics(self) -> None:
        payload = {
            "got_to": "correctness",
            "ok": False,
            "log": "cosine below gate",
            "graph_safe": True,
            "correctness": {"out": {"cos": 0.81, "rel_l2": 0.4, "pass": False}},
        }
        er = _make_task(FakeTransport(evaluate=payload)).run_benchmark(solution=_solution(), round_num=3)

        self.assertEqual(er.status, "wrong_answer")
        self.assertIsNone(er.latency_ms)
        self.assertEqual(er.metrics["correctness"]["out"]["cos"], 0.81)

    def test_graph_unsafe_status(self) -> None:
        payload = {"got_to": "lint", "ok": False, "graph_safe": False, "log": "cudaMalloc in kernel launch path"}
        er = _make_task(FakeTransport(evaluate=payload)).run_benchmark(solution=_solution(), round_num=4)
        self.assertEqual(er.status, "graph_unsafe")
        self.assertFalse(er.metrics["graph_safe"])

    def test_unknown_got_to_falls_back_to_failed(self) -> None:
        er = _make_task(FakeTransport(evaluate={"got_to": "wat", "ok": False})).run_benchmark(
            solution=_solution(), round_num=5
        )
        self.assertEqual(er.status, "failed")

    def test_missing_speedup_is_derived_from_latencies(self) -> None:
        payload = dict(OK_PAYLOAD)
        payload.pop("speedup")
        er = _make_task(FakeTransport(evaluate=payload)).run_benchmark(solution=_solution(), round_num=6)
        self.assertAlmostEqual(er.speedup_factor, 12.4 / 9.1)

    def test_non_json_stdout_is_a_failure_not_a_crash(self) -> None:
        er = _make_task(FakeTransport(raw_stdout="Segmentation fault (core dumped)")).run_benchmark(
            solution=_solution(), round_num=7
        )
        self.assertEqual(er.status, "failed")
        self.assertIsNone(er.latency_ms)
        self.assertIn("no JSON", er.log_excerpt)
        self.assertIn("Segmentation fault", er.log_excerpt)

    def test_solution_without_kernel_cu_fails_before_the_referee(self) -> None:
        t = FakeTransport(evaluate=OK_PAYLOAD)
        sol = _solution(kernel_cu="")
        er = _make_task(t).run_benchmark(solution=sol, round_num=8)
        self.assertEqual(er.status, "failed")
        self.assertIn("kernel.cu", er.log_excerpt)
        self.assertEqual(t.calls, [])

    def test_feedback_hooks_carry_the_round_forward(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        task.run_benchmark(solution=_solution(), round_num=1)
        self.assertEqual(task.get_last_round_passed_count(), 1)
        self.assertEqual(task.get_last_round_total_workloads(), 1)
        self.assertTrue(task.has_last_round_feedback_trace())
        logs = task.get_last_round_trace_logs_for_prompt()
        self.assertIn("got_to=timed", logs)
        self.assertIn("speedup=1.360x", logs)

        payload = {"got_to": "compile", "ok": False, "log": "error: expected a \";\""}
        task2 = _make_task(FakeTransport(evaluate=payload))
        task2.run_benchmark(solution=_solution(), round_num=1)
        self.assertEqual(task2.get_last_round_passed_count(), 0)
        self.assertIn("withheld", task2.get_last_round_trace_logs_for_prompt())


class SolutionConstructionTests(unittest.TestCase):
    def test_main_cpp_is_replaced_by_the_stub(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        sol = task.make_solution_from_generated_code(
            cleaned_code={
                "kernel.h": "#pragma once\nvoid launch();",
                "kernel.cu": "__global__ void k() {}",
                "main.cpp": "#include <torch/extension.h>\nPYBIND11_MODULE(m, x) {}",
            },
            raw_code="<ignored/>",
            round_num=3,
            model_name="gpt-5",
            target_gpu="A100",
            language="cuda",
        )
        by_path = {s.path: s.content for s in sol.sources}
        self.assertEqual(sorted(by_path), ["kernel.cu", "kernel.h", "main.cpp"])
        self.assertEqual(by_path["main.cpp"], MAIN_CPP_STUB)
        self.assertNotIn("PYBIND11_MODULE", by_path["main.cpp"])
        self.assertTrue(by_path["main.cpp"].strip())
        # The kernel itself is untouched.
        self.assertEqual(by_path["kernel.cu"], "__global__ void k() {}")
        self.assertEqual(sol.spec.entry_point, "kernel.cu::run")
        self.assertEqual(sol.spec.language, SupportedLanguages.CUDA)

    def test_solution_can_be_built_from_raw_xml(self) -> None:
        raw = (
            '<header_file name="kernel.h">\n#pragma once\n</header_file>\n'
            '<cuda_file name="kernel.cu">\n__global__ void k() {}\n</cuda_file>\n'
            '<cpp_file name="main.cpp">\nint main(){}\n</cpp_file>'
        )
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        sol = task.make_solution_from_generated_code(
            cleaned_code=None,
            raw_code=raw,
            round_num=1,
            model_name="m",
            target_gpu="A100",
            language="cuda",
        )
        by_path = {s.path: s.content for s in sol.sources}
        self.assertEqual(by_path["kernel.cu"], "__global__ void k() {}")
        self.assertEqual(by_path["main.cpp"], MAIN_CPP_STUB)

    def test_generated_solution_round_trips_through_run_benchmark(self) -> None:
        t = FakeTransport(evaluate=OK_PAYLOAD)
        task = _make_task(t)
        sol = task.make_solution_from_generated_code(
            cleaned_code={"kernel.h": "h", "kernel.cu": "cu", "main.cpp": "junk"},
            raw_code="",
            round_num=1,
            model_name="m",
            target_gpu="A100",
            language="cuda",
        )
        task.run_benchmark(solution=sol, round_num=1)
        # main.cpp never reaches the referee.
        self.assertEqual(t.calls[-1]["sources"], {"kernel.h": "h", "kernel.cu": "cu"})

    def test_non_cuda_language_is_rejected(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        with self.assertRaises(ValueError):
            task.make_solution_from_generated_code(
                cleaned_code={}, raw_code="", round_num=1, model_name="m", target_gpu="A100", language="triton"
            )

    def test_registered_solution_is_retrievable(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        sol = _solution()
        task.register_solution(sol)
        self.assertIs(task.get_solution("sol"), sol)
        self.assertIsNone(task.get_solution("missing"))


class DefinitionAndPromptTests(unittest.TestCase):
    def test_definition_text_renders_describe_and_profile(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        text = task.get_definition_text(language="cuda")
        self.assertIn("pi05.decoder.gemm.m10_n4096_k4096.bf16", text)
        self.assertIn("flashrt_gemm_m10", text)
        self.assertIn("out = A @ W", text)
        self.assertIn("cos >= 0.999", text)
        self.assertIn("synthetic", text)
        # Static-table hardware block, since no profile artifact was supplied.
        self.assertIn("sm_80", text)

    def test_describe_is_fetched_once(self) -> None:
        t = FakeTransport(evaluate=OK_PAYLOAD)
        task = _make_task(t)
        task.get_definition_text(language="cuda")
        task.get_definition_text(language="cuda")
        task.get_baseline_targets_text()
        self.assertEqual(sum(1 for c in t.calls if c["argv"][0] == "describe"), 1)

    def test_describe_failure_raises_with_the_referee_log(self) -> None:
        class BrokenTransport(FakeTransport):
            def run(self, argv, *, sources=None, timeout_seconds=None):  # type: ignore[override]
                return TransportResult(returncode=2, stdout="", stderr="no such tunable")

        task = _make_task(BrokenTransport())
        with self.assertRaises(RuntimeError) as ctx:
            task.get_definition_text(language="cuda")
        self.assertIn("no such tunable", str(ctx.exception))

    def test_baseline_targets_text(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        text = task.get_baseline_targets_text()
        self.assertIn("cuBLASLt", text)
        self.assertIn("44.900", text)
        self.assertIn("Beat 1.0", text)

    def test_prompt_hooks_state_the_invariants(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        req = task.get_per_task_requirement_text(language="cuda", target_gpu="A100", phase="generate")
        fmt = task.get_code_format_text(language="cuda", target_gpu="A100")
        for text in (req, fmt):
            self.assertIn("ONE kernel launch", text)
            self.assertIn("graph", text.lower())
        self.assertIn("REPLACED with a fixed stub", fmt)
        self.assertEqual(task.get_code_format_text(language="triton", target_gpu="A100"), "")

    def test_config_for_logging_is_json_friendly(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        cfg = task.get_config_for_logging()
        json.dumps(cfg)
        self.assertEqual(cfg["task_type"], "flashrt")
        self.assertEqual(cfg["target"], "local-a100")
        self.assertEqual(cfg["transport"], "fake")

    def test_task_name_is_namespaced_by_tunable_and_target(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        self.assertEqual(task.name, "flashrt_pi05.decoder.gemm.m10_n4096_k4096.bf16__local-a100")

    def test_code_for_world_model_extracts_kernel_cu(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        raw = '<cuda_file name="kernel.cu">\n__global__ void k() {}\n</cuda_file>'
        self.assertEqual(task.code_for_world_model_from_raw(raw=raw, language="cuda"), "__global__ void k() {}")

    def test_final_evaluation_report(self) -> None:
        task = _make_task(FakeTransport(evaluate=OK_PAYLOAD))
        report = task.run_final_evaluation(solutions=[_solution()])
        json.dumps(report)
        self.assertEqual(report["summary"]["best_speedup"], 1.36)
        self.assertEqual(report["solutions"][0]["status"], "passed")

    def test_missing_tunable_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FlashRTTunableTask(tunable_id="  ", transport=FakeTransport())


if __name__ == "__main__":
    unittest.main()
