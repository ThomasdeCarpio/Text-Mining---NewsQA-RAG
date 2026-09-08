"""Guard the artifact-to-slide boundary without requiring a TeX installation."""

import importlib.util
import json
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("latex_numbers", ROOT / "scripts/build_latex_numbers.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class LatexNumbersTest(TestCase):
    def test_committed_macros_match_current_artifacts(self):
        actual = [line for line in BUILDER.OUT.read_text(encoding="utf-8").splitlines()
                  if line.startswith(r"\newcommand")]
        expected = [rf"\newcommand{{\Num{key}}}{{{value}}}"
                    for key, value in BUILDER.collect().items()]
        self.assertEqual(actual, expected, "Run python3 scripts/build_latex_numbers.py")

    def test_preview_baseline_and_denominators(self):
        numbers = BUILDER.collect()
        report = json.loads((ROOT / "docs/reports/phase2/report_baseline_p0_d5.json").read_text())
        self.assertEqual(int(numbers["BaselineExpected"]), int(numbers["NDev"]))
        self.assertEqual(int(numbers["BaselineSuccessful"]), report["coverage"]["successful"])
        self.assertEqual(int(numbers["BaselineJudged"]), report["ragas"]["n_samples"])
        self.assertEqual(int(numbers["NResolved"].replace(".", "")),
                         sum(int(numbers[k]) for k in ("NDev", "NHeldout", "NReserve")))
        for macro, section, metric in [
            ("PzeroAC", "ragas", "answer_correctness"),
            ("PzeroFaith", "ragas", "faithfulness"),
            ("PzeroEM", "qa", "exact_match"), ("PzeroFone", "qa", "f1"),
            ("PzeroCitFone", "citations", "citation_f1"),
            ("PzeroCitVal", "citations", "citation_validity"),
        ]:
            self.assertEqual(numbers[macro], BUILDER.vn(report[section][metric]))

    def test_retriever_plot_lengths_match_displayed_scores(self):
        numbers = BUILDER.collect()
        plots = {key: value for key, value in numbers.items() if key.endswith("Plot")}
        self.assertTrue(plots)
        for key, value in plots.items():
            self.assertGreaterEqual(float(value), 0)
            self.assertLessEqual(float(value), 1)
            self.assertEqual(BUILDER.vn(float(value)), numbers[key.removesuffix("Plot")])

    def test_guardrail_display_preserves_boundary_failure(self):
        numbers = BUILDER.collect()
        significance = BUILDER.load_json("docs/reports/phase2/paired_significance.json")
        guard = next(g for g in significance["runs"]["p2_d3"]["guardrails"]
                     if g["metric"] == "Faithfulness")
        shown = numbers["PtwodThreeGuardFaithExact"].replace("$-$", "-").replace("{,}", ".")
        self.assertLess(float(shown), guard["tolerance"])
        self.assertEqual(float(shown), guard["delta"])
