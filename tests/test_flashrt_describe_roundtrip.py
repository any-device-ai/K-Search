"""A real referee payload must survive `render_describe`.

`tests/test_flashrt_prompts.py` builds payloads by hand using this renderer's own
key names, so it could not see that the referee emits different ones. The renderer
silently fell through to its degraded branch and delivered 1,430 characters of an
8,360-character payload: no shapes, no entry signature, no correctness gate, no
shared-memory cap, no cost model. This test pins the contract by feeding the
renderer exactly what the referee produces.

Skips when the referee is not runnable. No GPU needed -- `describe` is offline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.tasks.flashrt import prompts as flashrt_prompts  # noqa: E402

REFEREE = Path(os.environ.get("FLASHRT_ROOT", "")) / "scripts" / "kopt-referee"
TUNABLE = "pi05.decoder.gemm_bf16.dtype_bf16-k_1024-m_10-n_4096"
PROFILE = Path(os.environ.get("FLASHRT_ROOT", "")) / "targets" / "local-a100.json"


def _describe() -> dict:
    argv = [str(REFEREE), "describe", TUNABLE]
    if PROFILE.is_file():
        argv += ["--profile", str(PROFILE)]
    raw = subprocess.run(argv, capture_output=True, text=True, timeout=180).stdout
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


@unittest.skipUnless(REFEREE.is_file(), "FlashRT referee not present")
class RealPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.payload = _describe()
        except Exception as e:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"referee describe unavailable: {e}")
        cls.rendered = flashrt_prompts.render_describe(cls.payload, gpu_info="")

    def test_the_tunable_is_named(self) -> None:
        self.assertNotIn("(unknown tunable)", self.rendered)
        self.assertIn(TUNABLE, self.rendered)

    def test_tensor_shapes_reach_the_prompt(self) -> None:
        # The world model spent 150 rounds asking "what are the actual N and K".
        self.assertIn("4096x1024", self.rendered)
        self.assertIn("10x1024", self.rendered)

    def test_entry_signature_reaches_the_prompt(self) -> None:
        self.assertIn("kopt_entry(", self.rendered)

    def test_correctness_gate_reaches_the_prompt(self) -> None:
        self.assertIn("0.9999", self.rendered)

    def test_shared_memory_cap_reaches_the_prompt(self) -> None:
        self.assertIn(str(self.payload["max_dynamic_smem_bytes"]), self.rendered)

    def test_cost_model_reaches_the_prompt(self) -> None:
        self.assertIn(str(self.payload["cost_model"]["bytes_moved"]), self.rendered)

    def test_most_of_the_payload_survives(self) -> None:
        # The degraded branch delivered ~17% of the payload; anything near that
        # means the key names have drifted apart again.
        self.assertGreater(len(self.rendered), 6000)


class AliasTests(unittest.TestCase):
    """Neither spelling may regress; K-Search's own names keep priority."""

    def test_referee_spelling_is_accepted(self) -> None:
        out = flashrt_prompts.render_describe(
            {"id": "t1", "rendered": "BODY", "c_signature": "void f()"}, gpu_info="")
        self.assertIn("t1", out)
        self.assertIn("BODY", out)

    def test_ksearch_spelling_still_wins(self) -> None:
        out = flashrt_prompts.render_describe(
            {"id": "ignored", "tunable_id": "t2", "rendered": "R", "text": "T"},
            gpu_info="")
        self.assertIn("t2", out)
        self.assertIn("T", out)
        self.assertNotIn("ignored", out)

    def test_empty_payload_is_still_safe(self) -> None:
        self.assertIsInstance(flashrt_prompts.render_describe(None), str)


if __name__ == "__main__":
    unittest.main()
