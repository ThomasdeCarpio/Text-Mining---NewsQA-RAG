"""Contracts used by the Phase 2B prompt and context-depth tournament."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from newsqa_rag.evaluation.benchmark_io import stable_hash
from newsqa_rag.evaluation.testset import sha256_file
from newsqa_rag.llm import OpenAILLM
from newsqa_rag.model_gateway import PROJECT_ROOT
from scripts.collect_benchmark_predictions import main
from scripts.generate_phase2b_split_notebooks import VARIANTS


def test_prompt_registry_contains_preregistered_variants():
    path = PROJECT_ROOT / "configs/experiments/phase2_generation_prompts.yaml"
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert registry["schema_version"] == 1
    assert set(registry["prompts"]) == {"p0", "p1", "p2", "p3"}
    assert all(record["system_prompt"].strip() for record in registry["prompts"].values())


def test_baseline_prompt_is_exactly_preserved():
    path = Path(PROJECT_ROOT) / "configs/experiments/phase2_generation_prompts.yaml"
    registry = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert registry["prompts"]["p0"]["system_prompt"] == OpenAILLM.DEFAULT_SYSTEM_PROMPT


def test_split_notebooks_have_fixed_assignments_and_shared_subset_checks():
    notebook_root = Path(PROJECT_ROOT) / "notebooks/Tests"
    assert len(VARIANTS) == 9
    for filename, stage, platform, options in VARIANTS:
        notebook = json.loads((notebook_root / filename).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert f"PLATFORM='{platform}'" in source
        assert "SCREENING_QUESTIONS=80" in source
        assert "JUDGE_CALIBRATION_QUESTIONS=20" in source
        assert "FINAL_HELDOUT_ARTICLES=50" in source
        assert "HF_ARTIFACT_REPO_ID='ThomasAnderson2009/newsqa-rag-phase2-locked-v2'" in source
        assert "HF_ARTIFACT_REVISION='locked-bge-m3-512-64-deduplicated-v2'" in source
        assert "HF_ARTIFACT_SHA256='fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24'" in source
        if stage != "prepare":
            assert "Preparation subset hashes verified" in source
            assert "PREPARATION_BUNDLE_PATH" in source
        else:
            assert "HF_BASELINE_REPO_ID='ThomasAnderson2009/newsqa-rag-phase2-experiments'" in source
            assert "HF_BASELINE_REVISION='353c517decd13f6d13c1615e2980a3e8d340d434'" in source
            assert "HF_BASELINE_SHA256='45ec892225d92688d9ce12b262223543e8b5c84cc5d273e698a9040f4ec0ba31'" in source
            assert "Configure the read-only HF_TOKEN Kaggle secret" in source
            assert "'baseline_artifact'" in source
            assert "import newsqa_rag" in source
            assert "preparation_bundle_manifest.json" in source
        if stage == "prompt":
            assert f"PROMPT_ID='{options['prompt_id']}'" in source
        if stage == "context":
            assert f"CONTEXT_DEPTH={options['depth']}" in source
            assert "CONTEXT_DEPTH in {1,3}" in source
        if stage == "heldout":
            assert "len(heldout_ids)==284" in source
            assert "len(heldout_reserve_ids)==587" in source


def test_collector_generates_from_frozen_trace_with_registered_prompt(tmp_path):
    testset = tmp_path / "testset.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    sparse = tmp_path / "sparse.pkl"
    source = tmp_path / "source_retrievals.jsonl"
    prompt = tmp_path / "prompt.txt"
    config_path = tmp_path / "config.yaml"
    manifest_path = tmp_path / "variant.json"
    run_dir = tmp_path / "run"
    entry = {
        "question_id": "q1",
        "article_key": "a1",
        "question": "Who won?",
        "ground_truth": "Ada",
        "relevant_chunk_ids": ["c1"],
    }
    trace = {
        "question": "Who won?",
        "retrieved_chunks": [
            {"id": "c1", "text": "Ada won.", "score": 1.0},
            {"id": "c2", "text": "Noise.", "score": 0.5},
        ],
        "reranked_chunks": [
            {"id": "c1", "text": "Ada won.", "score": 1.0},
            {"id": "c2", "text": "Noise.", "score": 0.5},
        ],
        "retrieved_ids": ["c1", "c2"],
        "contexts": ["Ada won.", "Noise."],
        "timing_ms": {"retrieve_ms": 1.0, "rerank_ms": 1.0, "retrieval_total_ms": 2.0},
    }
    testset.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    chunks.write_text("{}\n", encoding="utf-8")
    sparse.write_bytes(b"index")
    source.write_text(json.dumps({
        **entry,
        "status": "success",
        "run_fingerprint": "source-run",
        "trace": trace,
    }) + "\n", encoding="utf-8")
    prompt.write_text("Registered prompt", encoding="utf-8")
    config = {
        "llm": {"model": "generator-test", "temperature": 0, "max_tokens": 32},
        "retrieval": {
            "top_k": 2,
            "sparse": {"method": "bge-m3"},
            "reranker": {"type": "cross-encoder", "top_n": 2},
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "pipeline": {"config_sha256": stable_hash(config)},
        "database": {"indexed": False, "path": "unused", "collection": "unused"},
        "artifacts": {
            "testset_resolved": {"path": str(testset), "sha256": sha256_file(testset)},
            "chunks": {"path": str(chunks), "sha256": sha256_file(chunks)},
            "bm25": {"path": str(sparse), "sha256": sha256_file(sparse)},
        },
    }), encoding="utf-8")

    class FakeLLM:
        _effective_model = "generator-test"
        last_usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

        def generate_rag_answer(self, question, contexts, *, system_prompt=None):
            assert question == "Who won?"
            assert contexts == ["Ada won."]
            assert system_prompt == "Registered prompt"
            return "Ada [1]"

    argv = [
        "collect_benchmark_predictions.py",
        "--retriever", "sparse",
        "--reranker", "cross-encoder",
        "--testset", str(testset),
        "--variant-manifest", str(manifest_path),
        "--config", str(config_path),
        "--run-dir", str(run_dir),
        "--top-k", "2",
        "--rerank-top-n", "2",
        "--generator-model", "generator-test",
        "--prompt-id", "p-test",
        "--system-prompt-file", str(prompt),
        "--context-depth", "1",
        "--source-retrievals", str(source),
        "--warmup-queries", "0",
    ]
    with (
        patch("sys.argv", argv),
        patch(
            "scripts.collect_benchmark_predictions.load_generation_client_settings",
            return_value=SimpleNamespace(model="generator-test", provider="test"),
        ),
        patch("scripts.collect_benchmark_predictions.get_llm", return_value=FakeLLM()),
        patch(
            "scripts.collect_benchmark_predictions.get_retriever",
            side_effect=AssertionError("frozen generation must not load a retriever"),
        ),
    ):
        main()

    prediction = json.loads((run_dir / "predictions.jsonl").read_text())
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert prediction["result"]["contexts"] == ["Ada won."]
    assert prediction["result"]["retrieved_ids"] == ["c1", "c2"]
    assert manifest["inputs"]["prompt_id"] == "p-test"
    assert manifest["inputs"]["rag_prompt"] == "Registered prompt"
    assert manifest["inputs"]["context_depth"] == 1
