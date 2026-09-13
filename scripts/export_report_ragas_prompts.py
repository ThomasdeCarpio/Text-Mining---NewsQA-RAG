#!/usr/bin/env python3
"""Export the exact RAGAS 0.4.3 judge prompt templates used in Phase 2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from newsqa_rag.evaluation.metrics import _ragas_shim  # noqa: E402

_ragas_shim()

import ragas  # noqa: E402
from ragas.metrics._answer_correctness import (  # noqa: E402
    CorrectnessClassifier,
    QuestionAnswerGroundTruth,
)
from ragas.metrics._answer_relevance import (  # noqa: E402
    ResponseRelevanceInput,
    ResponseRelevancePrompt,
)
from ragas.metrics._context_precision import ContextPrecisionPrompt, QAC  # noqa: E402
from ragas.metrics._context_recall import (  # noqa: E402
    ContextRecallClassificationPrompt,
    QCA,
)
from ragas.metrics._faithfulness import (  # noqa: E402
    NLIStatementInput,
    NLIStatementPrompt,
    StatementGeneratorInput,
    StatementGeneratorPrompt,
)


OUTPUT = ROOT / "docs/latex/report/appendix/ragas_prompts"
TEMPLATES = {
    "statement_generation": (
        StatementGeneratorPrompt(),
        StatementGeneratorInput(question="<question>", answer="<answer or reference>"),
        ["Faithfulness", "Answer Correctness"],
    ),
    "faithfulness_verdict": (
        NLIStatementPrompt(),
        NLIStatementInput(context="<retrieved contexts>", statements=["<statement>"]),
        ["Faithfulness"],
    ),
    "answer_correctness_classification": (
        CorrectnessClassifier(),
        QuestionAnswerGroundTruth(
            question="<question>",
            answer=["<answer statement>"],
            ground_truth=["<reference statement>"],
        ),
        ["Answer Correctness"],
    ),
    "answer_relevancy": (
        ResponseRelevancePrompt(),
        ResponseRelevanceInput(response="<generated answer>"),
        ["Answer Relevancy"],
    ),
    "context_precision": (
        ContextPrecisionPrompt(),
        QAC(question="<question>", context="<one retrieved context>", answer="<reference>"),
        ["Context Precision"],
    ),
    "context_recall": (
        ContextRecallClassificationPrompt(),
        QCA(question="<question>", context="<retrieved contexts>", answer="<reference>"),
        ["Context Recall"],
    ),
}


def main() -> None:
    if ragas.__version__ != "0.4.3":
        raise RuntimeError(f"Expected RAGAS 0.4.3, got {ragas.__version__}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (prompt, placeholder_input, metrics) in TEMPLATES.items():
        content = "\n".join(
            line.rstrip() for line in prompt.to_string(placeholder_input).splitlines()
        ).rstrip() + "\n"
        path = OUTPUT / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
        records.append(
            {
                "file": path.name,
                "metrics": metrics,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    manifest = {"ragas_version": ragas.__version__, "templates": records}
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Exported {len(records)} RAGAS prompts to {OUTPUT}")


if __name__ == "__main__":
    main()
