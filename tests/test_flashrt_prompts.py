"""Rendering of the referee's `describe` and `evaluate` payloads into prompt blocks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.tasks.flashrt import prompts  # noqa: E402


class RenderDescribeTests(unittest.TestCase):
    def test_empty_describe_still_renders_something(self) -> None:
        text = prompts.render_describe(None)
        self.assertIn("unknown tunable", text)
        self.assertTrue(text.endswith("\n"))

    def test_pre_rendered_text_is_used_verbatim(self) -> None:
        text = prompts.render_describe(
            {"tunable_id": "t1", "text": "HAND WRITTEN BLOCK", "summary": "ignored", "signature": "ignored"},
            gpu_info="HW BLOCK",
        )
        self.assertIn("HAND WRITTEN BLOCK", text)
        self.assertNotIn("ignored", text)
        self.assertIn("HW BLOCK", text)

    def test_tensor_rows_include_role_dtype_shape_layout_and_aliasing(self) -> None:
        text = prompts.render_describe(
            {
                "tunable_id": "t1",
                "tensors": [
                    {"name": "x", "role": "inout", "dtype": "bf16", "shape": [10, 4096], "layout": "row-major",
                     "alias_of": "y", "note": "written in place"}
                ],
            }
        )
        self.assertIn("x [inout] bf16 shape=10x4096", text)
        self.assertIn("layout=row-major", text)
        self.assertIn("ALIASES y", text)
        self.assertIn("written in place", text)

    def test_gate_explains_the_live_reference(self) -> None:
        text = prompts.render_describe({"tunable_id": "t1", "gate": {"metric": "cos", "threshold": 0.999}})
        self.assertIn("cos >= 0.999", text)
        self.assertIn("regenerated every round", text)


class FeedbackRenderingTests(unittest.TestCase):
    def test_passing_payload_reports_the_ratio(self) -> None:
        text = prompts.format_evaluate_feedback(
            {"got_to": "timed", "ok": True, "latency_us": 9.1, "baseline_latency_us": 12.4, "speedup": 1.36}
        )
        self.assertIn("speedup=1.360x", text)
        self.assertIn("incumbent_us=12.400", text)

    def test_failing_payload_withholds_perf(self) -> None:
        text = prompts.format_evaluate_feedback(
            {"got_to": "correctness", "ok": False, "latency_us": 1.0, "speedup": 99.0,
             "correctness": {"out": {"cos": 0.4, "pass": False}}}
        )
        self.assertIn("withheld", text)
        self.assertNotIn("99.0", text)
        self.assertIn("cos=0.4", text)

    def test_graph_unsafe_is_flagged_as_blocking(self) -> None:
        text = prompts.format_evaluate_feedback({"got_to": "lint", "ok": False, "graph_safe": False})
        self.assertIn("BLOCKING", text)

    def test_throttled_device_is_flagged(self) -> None:
        text = prompts.format_evaluate_feedback(
            {"got_to": "timed", "ok": True, "device": {"throttled": True, "temp_c": 91}}
        )
        self.assertIn("THROTTLED", text)

    def test_long_log_keeps_the_tail(self) -> None:
        log = ("noise\n" * 5000) + "error: the thing that matters"
        text = prompts.format_evaluate_feedback({"got_to": "compile", "ok": False, "log": log}, max_log_chars=200)
        self.assertIn("the thing that matters", text)
        self.assertIn("truncated", text)


if __name__ == "__main__":
    unittest.main()
