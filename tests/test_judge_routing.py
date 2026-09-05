"""Tests for RAGAS judge provider routing.

The bug worth guarding is silent misrouting: a judge that answers, scores and
writes a manifest while running on a provider nobody asked for.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from newsqa_rag.evaluation.metrics import (
    FIREWORKS_BASE_URL,
    JUDGE_MIN_MAX_TOKENS,
    _resolve_judge_provider,
)


class ResolveJudgeProviderTests(unittest.TestCase):
    """Provider comes from the model name, never from the environment."""

    def test_glm_models_route_to_fireworks(self):
        for model in (
            "accounts/fireworks/models/glm-5p3-flash",
            "accounts/fireworks/models/glm-5p3",
            "GLM-5p3-Flash",
        ):
            with self.subTest(model=model):
                self.assertEqual(_resolve_judge_provider(model), "fireworks")

    def test_known_families_keep_their_providers(self):
        cases = {
            "gemini-3.1-flash-lite": "gemini",
            "gemini-3.7-flash": "gemini",
            "deepseek-chat": "deepseek",
            "gpt-4o-mini": "openai",
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                self.assertEqual(_resolve_judge_provider(model), expected)

    def test_a_stray_deepseek_key_cannot_hijack_a_glm_judge(self):
        """The original bug: provider was inferred from whichever key existed.

        A DEEPSEEK_API_KEY left in .env silently ran the judge on DeepSeek while
        the manifest still recorded GLM.
        """

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-leftover"}):
            self.assertEqual(
                _resolve_judge_provider("accounts/fireworks/models/glm-5p3-flash"),
                "fireworks",
            )
            self.assertEqual(_resolve_judge_provider("gemini-3.1-flash-lite"), "gemini")
            self.assertEqual(_resolve_judge_provider("gpt-4o-mini"), "openai")

    def test_an_explicit_provider_always_wins(self):
        self.assertEqual(
            _resolve_judge_provider("accounts/fireworks/models/glm-5p3-flash", "openai"),
            "openai",
        )
        self.assertEqual(_resolve_judge_provider("gpt-4o-mini", "fireworks"), "fireworks")


class JudgeOutputBudgetTests(unittest.TestCase):
    """GLM 5.3 Flash is a reasoning model; a low cap returns empty content."""

    def test_minimum_output_budget_clears_the_measured_reasoning_cost(self):
        # Measured against the live endpoint: 64 tokens produced 274 characters
        # of reasoning_content and an EMPTY message; 512 produced the answer
        # after spending 279 completion tokens.
        self.assertGreaterEqual(JUDGE_MIN_MAX_TOKENS, 512)

    def test_fireworks_base_url_is_the_openai_compatible_path(self):
        self.assertTrue(FIREWORKS_BASE_URL.endswith("/inference/v1"))
        self.assertTrue(FIREWORKS_BASE_URL.startswith("https://"))


class JudgeCliTests(unittest.TestCase):
    """The runner has to be able to ask for the provider that now exists."""

    def test_fireworks_is_selectable_on_the_command_line(self):
        import ast
        from pathlib import Path

        from newsqa_rag.model_gateway import PROJECT_ROOT

        source = (PROJECT_ROOT / "scripts/judge_benchmark_predictions.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        choices = {
            element.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "choices" and isinstance(keyword.value, ast.List)
            for element in keyword.value.elts
            if isinstance(element, ast.Constant)
        }
        self.assertIn("fireworks", choices)

    def test_reasoning_ablation_arguments_are_exposed(self):
        from unittest.mock import patch

        from scripts.judge_benchmark_predictions import parse_args

        argv = [
            "judge_benchmark_predictions.py",
            "--run-dir", "run",
            "--judge-provider", "fireworks",
            "--judge-model", "accounts/fireworks/models/glm-5p3-flash",
            "--reasoning-effort", "none",
            "--judge-max-tokens", "512",
            "--results-file", "judge_results_none.jsonl",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.reasoning_effort, "none")
        self.assertEqual(args.judge_max_tokens, 512)
        self.assertEqual(args.results_file, "judge_results_none.jsonl")


if __name__ == "__main__":
    unittest.main()
