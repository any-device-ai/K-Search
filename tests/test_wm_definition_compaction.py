"""`compact_definition_for_wm_prompt` must not return an empty spec.

Section selection matches five exact headers that only flashinfer-bench emits.
For kernelbench, gpu_mode, mlx_mamba and flashrt nothing matched, so the world
model's per-round prompt carried two title lines and no shapes, dtypes or
constraints — which is why a 150-round flashrt search kept listing the tensor
dimensions among its open questions.

No GPU, no network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.kernel_generators.world_model import (  # noqa: E402
    compact_definition_for_wm_prompt as compact,
)

FLASHINFER_STYLE = """Name: gemm
Type: matmul

Axes:
  M: 10
  N: 4096
Inputs:
  x: bf16 [M, K]
Outputs:
  y: bf16 [M, N]
Constraints:
  one launch
Reference Implementation:
  def f(x, w): return x @ w.T
"""

MARKDOWN_STYLE = """# Tunable pi05.decoder.gemm_bf16.dtype_bf16-k_1024-m_10-n_4096
# Small-M BF16 GEMM, FFN gate, Pi0.5 decoder on A100.

## Parameters
  x          in      bf16  [10x1024] row_major
  W          weight  bf16  [4096x1024] row_major

## Constraints
  - M is 10 and fixed: this is decode, not prefill.
"""


class HeaderSelectionTests(unittest.TestCase):
    def test_flashinfer_style_still_uses_section_selection(self) -> None:
        out = compact(FLASHINFER_STYLE)
        self.assertIn("Axes:", out)
        self.assertIn("Reference Implementation: (excerpt)", out)
        # Selection, not truncation: the blank lines between sections are dropped.
        self.assertNotIn("\n\n", out)


class FallbackTests(unittest.TestCase):
    def test_markdown_style_no_longer_collapses_to_the_title(self) -> None:
        out = compact(MARKDOWN_STYLE)
        self.assertIn("4096x1024", out)
        self.assertIn("M is 10 and fixed", out)
        self.assertGreater(len(out.splitlines()), 2)

    def test_fallback_is_bounded(self) -> None:
        out = compact("# Title\n# Summary\n" + ("x" * 50_000))
        self.assertLessEqual(len(out), 6100)

    def test_empty_in_empty_out(self) -> None:
        self.assertEqual(compact(""), "")
        self.assertEqual(compact("   \n  "), "")

    def test_a_two_line_definition_is_returned_whole(self) -> None:
        self.assertIn("Summary", compact("# Title\n# Summary\n"))


if __name__ == "__main__":
    unittest.main()
