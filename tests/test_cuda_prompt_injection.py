"""The CUDA prompt templates must carry the target profile, and must not hardcode
one architecture's advice. No GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_search.kernel_generators import kernel_generator_prompts as kgp  # noqa: E402
from k_search.kernel_generators import world_model_prompts as wmp  # noqa: E402
from k_search.kernel_generators.world_model import (  # noqa: E402
    build_decision_tree_edit_prompt,
    build_world_model_prompts,
)
from k_search.utils.cuda_gpu_info import TARGET_PROFILE_ENV_VAR, get_gpu_info  # noqa: E402


PROFILE = {
    "target": "thor-01",
    "hardware": {
        "device_name": "NVIDIA Jetson AGX Thor",
        "compute_capability": "11.0",
        "arch": "sm_110",
        "sm_count": 20,
        "measured_copy_bandwidth_gb_s": 221.0,
        "graph_node_replay_overhead_us": 3.0,
        "supported_dtypes": ["bf16", "fp16", "fp8_e4m3"],
    },
    "build": {"available": False, "reason": "flash_rt not importable"},
}


class GpuInfoSlotTests(unittest.TestCase):
    def setUp(self) -> None:
        get_gpu_info.cache_clear()
        self._saved = os.environ.pop(TARGET_PROFILE_ENV_VAR, None)

    def tearDown(self) -> None:
        get_gpu_info.cache_clear()
        os.environ.pop(TARGET_PROFILE_ENV_VAR, None)
        if self._saved is not None:
            os.environ[TARGET_PROFILE_ENV_VAR] = self._saved

    def _use_profile(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(PROFILE, fh)
        self.addCleanup(lambda: os.unlink(path))
        os.environ[TARGET_PROFILE_ENV_VAR] = path
        get_gpu_info.cache_clear()
        return path

    # -- templates declare the slot ---------------------------------------------------

    def test_all_cuda_templates_have_a_gpu_info_slot(self) -> None:
        for name, template in (
            ("CUDA_PROMPT", kgp.CUDA_PROMPT),
            ("CUDA_OPTIMIZATION_PROMPT", kgp.CUDA_OPTIMIZATION_PROMPT),
            ("CUDA_ACTION_PROMPT", wmp.CUDA_ACTION_PROMPT),
            ("CUDA_DEBUG_PROMPT", wmp.CUDA_DEBUG_PROMPT),
            ("CUDA_IMPROVE_PROMPT", wmp.CUDA_IMPROVE_PROMPT),
        ):
            self.assertIn("{gpu_info}", template, f"{name} is missing the gpu_info slot")

    # -- the bug fix ------------------------------------------------------------------

    def test_hints_no_longer_hardcode_h100_mma(self) -> None:
        hints = kgp.CUDA_OPTIMIZATION_HINTS
        self.assertNotIn("H100", hints)
        self.assertNotIn("MMA", hints)
        self.assertIn("target GPU profile", hints)

    def test_static_table_target_reaches_the_generation_prompt(self) -> None:
        prompt = kgp.get_prompt_from_definition_text("cuda", "spec text", "A100")
        self.assertIn("sm_80", prompt)
        self.assertIn("spec text", prompt)
        # A100 has no FP8: the prompt must say so rather than let the model assume.
        self.assertIn("UNAVAILABLE dtypes", prompt)

    def test_unknown_target_says_so_instead_of_guessing(self) -> None:
        prompt = kgp.get_prompt_from_definition_text("cuda", "spec", "SomeFutureGPU")
        self.assertIn("TARGET GPU PROFILE: unavailable", prompt)
        self.assertNotIn("sm_90a", prompt)

    # -- profile artifact flows through every CUDA prompt builder ---------------------

    def test_profile_reaches_every_cuda_prompt_builder(self) -> None:
        self._use_profile()
        builders = {
            "generation": lambda: kgp.get_prompt_from_definition_text("cuda", "spec", "A100"),
            "optimization": lambda: kgp.get_optimization_prompt_from_definition_text(
                "cuda", definition_text="spec", trace_logs="log", current_code="code", target_gpu="A100"
            ),
            "action": lambda: wmp.get_generate_code_from_action_prompt_from_text(
                "cuda", definition_text="spec", base_code="base", action_text="act", target_gpu="A100"
            ),
            "spec_action": lambda: wmp.get_generate_code_from_spec_with_action_prompt_from_text(
                "cuda", definition_text="spec", action_text="act", target_gpu="A100"
            ),
            "debug": lambda: wmp.get_debug_generated_code_prompt_from_text(
                "cuda",
                definition_text="spec",
                trace_logs="log",
                base_code="base",
                buggy_code="buggy",
                action_text="act",
                debug_round=1,
                target_gpu="A100",
            ),
            "improve": lambda: wmp.get_improve_generated_code_prompt_from_text(
                "cuda",
                definition_text="spec",
                trace_logs="log",
                base_code="base",
                current_code="cur",
                debug_round=1,
                target_gpu="A100",
            ),
        }
        for label, build in builders.items():
            text = build()
            self.assertIn("NVIDIA Jetson AGX Thor", text, f"{label} prompt lost the profile")
            self.assertIn("sm_110", text, f"{label} prompt lost the arch")
            # The profile is authoritative; the mismatched --target-gpu must be flagged.
            self.assertIn("WARNING", text, f"{label} prompt lost the mismatch warning")

    def test_non_cuda_builders_are_untouched(self) -> None:
        self._use_profile()
        triton = kgp.get_prompt_from_definition_text("triton", "spec", "A100")
        self.assertNotIn("Jetson", triton)
        self.assertNotIn("{gpu_info}", triton)


class WorldModelPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        get_gpu_info.cache_clear()
        self._saved = os.environ.pop(TARGET_PROFILE_ENV_VAR, None)

    def tearDown(self) -> None:
        get_gpu_info.cache_clear()
        os.environ.pop(TARGET_PROFILE_ENV_VAR, None)
        if self._saved is not None:
            os.environ[TARGET_PROFILE_ENV_VAR] = self._saved

    def _init_prompt(self, *, language: str, target_gpu: str) -> str:
        return build_world_model_prompts(
            definition_text="spec",
            target_gpu=target_gpu,
            language=language,
            previous_world_model_json=None,
            current_code_excerpt=None,
            eval_result=None,
            chosen_action_text=None,
            prediction=None,
        ).init_prompt

    def _edit_prompt(self, *, language: str, target_gpu: str) -> str:
        return build_decision_tree_edit_prompt(
            world_model_json="{}",
            definition_text="spec",
            target_gpu=target_gpu,
            language=language,
            current_code_excerpt="code",
            current_tree_path="root",
            chosen_action_text=None,
            prediction=None,
            eval_result=None,
        )

    def test_cuda_world_model_init_prompt_carries_the_profile(self) -> None:
        text = self._init_prompt(language="cuda", target_gpu="A100")
        self.assertIn("sm_80", text)
        self.assertIn("108", text)
        self.assertNotIn("Target GPU: A100", text)
        self.assertIn("Language: cuda", text)

    def test_cuda_decision_tree_edit_prompt_carries_the_profile(self) -> None:
        text = self._edit_prompt(language="cuda", target_gpu="A100")
        self.assertIn("sm_80", text)
        self.assertNotIn("Target GPU: A100", text)
        self.assertIn("Language: cuda", text)

    def test_unknown_cuda_target_states_the_gap(self) -> None:
        text = self._init_prompt(language="cuda", target_gpu="SomeFutureGPU")
        self.assertIn("TARGET GPU PROFILE: unavailable", text)

    def test_non_cuda_world_model_prompts_keep_the_one_line_hint(self) -> None:
        for build in (self._init_prompt, self._edit_prompt):
            text = build(language="triton", target_gpu="A100")
            self.assertIn("Target GPU: A100", text)
            self.assertNotIn("sm_80", text)


if __name__ == "__main__":
    unittest.main()
