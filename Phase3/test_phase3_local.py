"""Local-only tests using synthetic fixtures and fake HTTP; no real API calls."""
import ast
import copy
import io
import json
import sqlite3
import runpy
import types
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import phase3_run as runner
import phase3_api as local_runtime
import phase3_metrics as metrics
import review_phase3 as review


class FakeClock:
    def __init__(self, now=1788778800.0):
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def response(text="answer"):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}


class QuotaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "quota.sqlite3"
        self.clock = FakeClock()

    def ledger(self, **kwargs):
        return local_runtime.QuotaLedger(self.path, clock=self.clock.time, sleep=self.clock.sleep, **kwargs)

    def test_rate_spacing_across_policies_and_restarts(self):
        for n in range(17):
            ledger = self.ledger()
            self.assertIsNotNone(ledger.acquire("run", str(n), "B0" if n < 8 else "B1"))
        with sqlite3.connect(self.path) as db:
            times = [r[0] for r in db.execute("SELECT started FROM requests ORDER BY id")]
        self.assertTrue(all(b - a >= 4.19 for a, b in zip(times, times[1:])))
        self.assertTrue(all(sum(t - 60 < other <= t for other in times) <= 15 for t in times))
        self.assertTrue(all(seconds <= 30 for seconds in self.clock.sleeps))

    def test_daily_limit_is_shared_and_persistent(self):
        self.ledger(daily=3, reserve=1).acquire("smoke", "a", "B0")
        self.ledger(daily=3, reserve=1).acquire("development", "a", "B1")
        with self.assertRaisesRegex(local_runtime.RunPaused, "daily request budget"):
            self.ledger(daily=3, reserve=1).acquire("final", "b", "B0")

    def test_pacific_day_not_utc_day(self):
        ledger = self.ledger()
        before = datetime(2026, 9, 7, 6, 59, tzinfo=timezone.utc).timestamp()
        after = datetime(2026, 9, 7, 7, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(ledger.day(before), "2026-09-06")
        self.assertEqual(ledger.day(after), "2026-09-07")

    def test_retries_capped_across_resume(self):
        ledger = self.ledger()
        parse = lambda text: (_ for _ in ()).throw(ValueError("invalid schema"))
        calls = []
        transport = lambda payload, key: calls.append(1) or response()
        result = local_runtime.generate_one(ledger, "run", "a", "B1", {}, parse, "not-a-real-key", transport=transport)
        self.assertEqual(result["status"], "exhausted")
        local_runtime.generate_one(self.ledger(), "run", "a", "B1", {}, parse, "not-a-real-key", transport=transport)
        self.assertEqual(len(calls), 3)
        self.assertEqual(ledger.usage("run")["reported_tokens_all_attempts"]["total_tokens"], 36)

    def test_success_recovered_without_repeat_after_export_gap(self):
        ledger = self.ledger()
        parse = lambda text: {"answer": text, "answerability": "answerable", "citations": [1]}
        first = local_runtime.generate_one(ledger, "run", "a", "B0", {}, parse, "private-fixture", transport=lambda payload, key: response())
        def forbidden(*args):
            self.fail("Completed API response must be recovered, not sent again")
        second = local_runtime.generate_one(self.ledger(), "run", "a", "B0", {}, parse, "private-fixture", transport=forbidden)
        self.assertEqual(first, second)
        self.assertEqual(ledger.attempts("run", "a", "B0"), 1)
        self.assertNotIn(b"private-fixture", self.path.read_bytes())

    def test_429_pauses_and_cooldown_survives_restart(self):
        ledger = self.ledger()
        def limited(*args):
            raise urllib.error.HTTPError("https://example.invalid", 429, "secret-fixture", {"Retry-After": "90"}, io.BytesIO(b"secret-fixture"))
        with self.assertRaises(local_runtime.RunPaused) as raised:
            local_runtime.generate_one(ledger, "run", "a", "B0", {}, lambda text: {}, "secret-fixture", transport=limited)
        self.assertNotIn("secret-fixture", str(raised.exception))
        with self.assertRaisesRegex(local_runtime.RunPaused, "cooldown"):
            self.ledger().acquire("run", "b", "B1")
        self.assertEqual(ledger.usage("run")["http_attempts"], 1)

    def test_auth_failure_stops_without_retry_storm(self):
        ledger = self.ledger()
        def rejected(*args):
            raise urllib.error.HTTPError("https://example.invalid", 403, "private", {}, None)
        with self.assertRaisesRegex(local_runtime.RunPaused, "HTTP 403"):
            local_runtime.generate_one(ledger, "run", "a", "B0", {}, lambda text: {}, "fixture", transport=rejected)
        self.assertEqual(ledger.usage("run")["http_attempts"], 1)

    def test_truncated_generation_is_not_success(self):
        data = response()
        data["choices"][0]["finish_reason"] = "length"
        result = local_runtime.generate_one(self.ledger(), "run", "a", "B0", {}, lambda text: {}, "fixture", transport=lambda *args: data)
        self.assertEqual(result["status"], "exhausted")

    def test_provider_block_is_terminal_and_recovered_after_export_gap(self):
        # Shape observed in diagnostic requests 410/411: no message field.
        for finish_reason in ("content_filter: PROHIBITED_CONTENT", "content_filter"):
            with self.subTest(finish_reason=finish_reason):
                ledger = self.ledger()
                data = {"choices": [{"finish_reason": finish_reason, "index": 0}],
                        "usage": {"prompt_tokens": 2289, "completion_tokens": 0, "total_tokens": 2289}}
                def forbidden(*args):
                    self.fail("A blocked response must not be parsed or requested again")
                result = local_runtime.generate_one(ledger, "run", finish_reason, "B0", {}, forbidden,
                                                    "private-fixture", transport=lambda *args: data)
                self.assertEqual(result["status"], "provider_blocked")
                self.assertEqual(result["error"], "provider_content_filter")
                self.assertEqual(result["finish_reason"], finish_reason)
                self.assertIsNone(result["answerability"])
                self.assertIsNone(result["answer"])
                self.assertEqual(result["citations"], [])
                self.assertEqual(result["usage"]["input_tokens"], 2289)
                self.assertEqual(result["usage"]["output_tokens"], 0)
                with ledger.connect() as db:
                    outcome, payload = db.execute("SELECT outcome,payload FROM requests WHERE case_id=?", (finish_reason,)).fetchone()
                self.assertEqual(outcome, "provider_blocked")
                self.assertEqual(json.loads(payload), result)
                recovered = local_runtime.generate_one(self.ledger(), "run", finish_reason, "B0", {}, forbidden,
                                                       "private-fixture", transport=forbidden)
                self.assertEqual(recovered, result)
                self.assertEqual(ledger.attempts("run", finish_reason, "B0"), 1)
                self.assertNotIn(b"private-fixture", self.path.read_bytes())

    def test_missing_message_preserves_nonblocking_finish_reason(self):
        ledger = self.ledger()
        data = {"choices": [{"finish_reason": "length"}], "usage": {"completion_tokens": 512}}
        result = local_runtime.generate_one(ledger, "run", "a", "B0", {}, lambda text: {}, "fixture", transport=lambda *args: data)
        self.assertEqual(result["status"], "exhausted")
        with ledger.connect() as db:
            attempts = db.execute("SELECT outcome,payload FROM requests ORDER BY id").fetchall()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(all(outcome == "invalid_response" for outcome, _ in attempts))
        self.assertTrue(all(json.loads(payload)["finish_reason"] == "length" for _, payload in attempts))

    def test_export_rebuilt_from_durable_results(self):
        ledger = self.ledger()
        ledger.save_prediction("run", "a", "B0", {"case_id": "a", "status": "success"})
        target = Path(self.directory.name) / "predictions.jsonl"
        rows = local_runtime.export_predictions(ledger, "run", [{"case_id": "a"}], "B0", target)
        self.assertEqual(json.loads(target.read_text()), rows[0])
        with self.assertRaises(sqlite3.IntegrityError):
            ledger.save_prediction("run", "a", "B0", {})

    def test_payload_preserves_selected_generator_without_credentials(self):
        result = local_runtime.request_payload(runner.GENERATOR, "system", "user")
        self.assertEqual({k: result[k] for k in runner.GENERATOR}, runner.GENERATOR)
        self.assertNotIn("api_key", result)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = review.read_jsonl(review.OUTPUT / "cases.jsonl")
        cls.by_origin = {(c["origin"]["file"], c["origin"]["row"]): c for c in cls.cases}

    def original(self, row, file="review_queue.jsonl"):
        return self.by_origin[file, row]

    def test_all_200_slots_and_partitions(self):
        review.validate_cases(self.cases)
        self.assertEqual(len([c for c in self.cases if c["partition"] == "development"]), 140)
        self.assertEqual(len([c for c in self.cases if c["origin"]["file"] == "authored_cases.jsonl"]), 44)

    def test_every_row_has_source_and_individual_review(self):
        for c in self.cases:
            self.assertTrue(c["review"]["notes"], c["case_id"])
            self.assertTrue(c["source_evidence"] or c["construction"].get("source_evidence_spans"), c["case_id"])
            self.assertNotEqual(c.get("human_review", {}).get("decision"), "approved")

    def test_source_backed_repairs(self):
        self.assertEqual(self.original(20)["ground_truth"], "Alina Cho")
        self.assertIn("25", self.original(21)["question"])
        self.assertIn("cancellation", self.original(37)["question"])
        self.assertIn("before", self.original(30)["question"])
        self.assertIn("could", self.original(117)["question"])
        self.assertEqual(self.original(112)["ground_truth"], "CALO")
        self.assertEqual(self.original(146)["ground_truth"], "CALO")

    def test_ablation_removes_overlapping_answers(self):
        for row, forbidden in ((60, {"3bf88ec8cf4b_chunk_1"}), (61, {"9c0085af9fe6_chunk_0", "9c0085af9fe6_chunk_2"}),
                               (72, {"599a60d41b7d_chunk_0", "68721fdeab81_chunk_1", "68721fdeab81_chunk_2"}),
                               (135, {"427ee3c05963_chunk_0", "e3238110fae0_chunk_0"}), (137, {"5904f89522e7_chunk_0"})):
            c = self.original(row)
            self.assertFalse(forbidden & {x["id"] for x in c["contexts"]})
            self.assertTrue(c["contexts"])

    def test_weak_replacements_and_narrower_questions(self):
        for row in (82, 86, 94, 147, 148):
            c = self.original(row)
            self.assertEqual(c["case_type"], "partial_weak_evidence")
            self.assertNotEqual(c["case_id"], c["origin"]["case_id"])
        self.assertIn("warning", self.original(88)["question"])
        self.assertEqual(self.original(91)["ground_truth"], "Sobibor")
        self.assertIn("degree", self.original(93)["question"])

    def test_natural_misses_are_not_silently_reused_or_gated(self):
        for c in self.cases:
            if c["case_type"] == "natural_retrieval_miss":
                self.assertEqual(c["scope"], "provided_context")
                self.assertEqual(c["contexts"], [])
                if c["origin"]["case_id"].startswith("abs_natural_retrieval_miss_"):
                    self.assertTrue(c["historical_contexts"])
                self.assertEqual(c["review"]["runtime_status"], "pending_retrieval")

    def test_counterfactual_questions_have_evidence_and_changes(self):
        cases = [self.original(n, "authored_cases.jsonl") for n in range(1, 23)]
        self.assertEqual(len({c["question"] for c in cases}), 22)
        self.assertEqual(sum(bool(c["review"]["changes"]) for c in cases), 12)
        self.assertIn("radar outages", cases[3]["question"])
        self.assertIn("train", cases[14]["question"])
        self.assertIn("prison sentence", cases[16]["question"])
        self.assertIn("donate", cases[17]["question"])

    def test_four_external_replacements_are_globally_withheld(self):
        settings = runner.load_settings()
        corpus = runner.active_corpus(settings)
        ids = {c["id"] for c in corpus}
        self.assertEqual(len(corpus), 19259)
        for row in (23, 29, 41, 42):
            c = self.original(row, "authored_cases.jsonl")
            self.assertEqual(c["construction"]["source"], "locally_withheld_Phase3_article")
            self.assertTrue(c["source_evidence"])
            self.assertFalse(ids & set(c["source_gold_chunk_ids"]))
        external = [c for c in self.cases if c["case_type"] == "external_unanswerable"]
        self.assertEqual(len({c["source_article_id"] for c in external}), 22)

    def test_removed_replacements_still_have_control_and_exclude_source(self):
        corpus = runner.active_corpus(runner.load_settings())
        for row in (96, 99, 108, 151, 152, 153, 154):
            c = self.original(row)
            self.assertTrue(any(v["case_type"] == "answerable_control" and v["base_question_id"] == c["base_question_id"] and v["partition"] == c["partition"] for v in self.cases))
            matching = [v for v in corpus if v["metadata"]["canonical_article_id"] == c["source_article_id"]]
            self.assertTrue(matching)
            self.assertTrue(all(runner.excluded(c, v) for v in matching))

    def test_provided_text_identity_not_replaced_with_uploaded_text(self):
        c = self.original(146)
        self.assertTrue(any(v["id"] == "80024e136b72_chunk_2" for v in c["contexts"]))
        self.assertNotIn("80024e136b72_chunk_2", {v["id"] for v in review.read_jsonl(review.HERE / "chunks.jsonl")})

    def test_rebuild_deterministic_and_originals_unchanged(self):
        before = {p: review.file_hash(p) for p in review.BUNDLE.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            report = review.prepare(output=Path(directory))
            self.assertEqual(report["changed_cases"], 79)
            self.assertEqual(review.file_hash(Path(directory) / "cases.jsonl"), review.file_hash(review.OUTPUT / "cases.jsonl"))
        self.assertEqual(before, {p: review.file_hash(p) for p in before})

    def test_recovered_external_articles_and_attribution(self):
        sources = review.read_jsonl(review.HERE / "external_source_evidence.jsonl")
        self.assertEqual(len(sources), 18)
        for c in self.cases:
            self.assertEqual(c["review"]["content_status"], "reviewed")
            self.assertTrue(c["source_evidence"])
        self.assertIn("Michael Ware", self.original(37, "authored_cases.jsonl")["question"])
        self.assertIn("11th anniversary", self.original(32, "authored_cases.jsonl")["question"])
        self.assertEqual(self.original(24, "authored_cases.jsonl")["construction"]["source_expected_answer"], "Brrr, the Cowardly Lion")

    def test_observed_natural_misses_not_forced_to_old_quota(self):
        self.assertEqual(sum(c["case_type"] == "natural_retrieval_miss" for c in self.cases), 3)
        self.assertEqual(sum(c["case_type"] == "answerable_control" for c in self.cases), 87)
        for n in (73, 74, 75, 76, 77, 78, 141, 142):
            c = self.original(n)
            self.assertEqual(c["answerability_label"], "answerable")
            self.assertEqual(c["scope"], "full_corpus")
        self.assertEqual(self.original(101)["source_article_id"], self.original(42)["source_article_id"])

    def test_barnett_name_is_not_the_requested_statement(self):
        c = self.original(33)
        self.assertEqual(c["case_id"], c["origin"]["case_id"])
        self.assertEqual(c["case_type"], "natural_retrieval_miss")
        self.assertEqual(c["source_gold_chunk_ids"], ["5d574141dec9_chunk_1"])
        self.assertEqual(c["contexts"], [])
        self.assertEqual(c["review"]["runtime_status"], "pending_retrieval")
        corpus = {r["id"]: r["text"] for r in review.read_jsonl(review.HERE / "chunks.jsonl")}
        self.assertNotIn("aggressively", corpus["5d574141dec9_chunk_0"])
        self.assertIn("aggressively", c["source_evidence"][0]["text"])


class MetricsTests(unittest.TestCase):
    def case(self, negative=False, scope="full_corpus"):
        return {"case_id": "a", "case_type": "counterfactual" if negative else "answerable_control", "question": "Which city?",
                "answerability_label": "insufficient_evidence" if negative else "answerable", "accepted_answers": ["Paris"],
                "partition": "development", "scope": scope}

    def prediction(self, **kwargs):
        return {"status": "success", "answerability": "answerable", "answer": "Paris [1]", "citations": [1], **kwargs}

    def test_prompt_is_exactly_newly_supplied_p2(self):
        self.assertEqual(runner.baseline_prompt(review.HERE / "phase2_generation_prompts.yaml"), metrics.B0_PROMPT)
        self.assertIn("one short answer sentence", metrics.B0_PROMPT)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "prompts.yaml"
            p.write_text("prompts:\n  p2:\n    system_prompt: >-\n      changed\n")
            with self.assertRaises(ValueError):
                runner.baseline_prompt(p)

    def test_b0_abstention_is_exact_not_substring(self):
        self.assertEqual(metrics.parse_response(metrics.ABSTENTION, "B0", 1)["answerability"], "insufficient_evidence")
        self.assertEqual(metrics.parse_response(metrics.ABSTENTION + " But Paris is likely.", "B0", 1)["answerability"], "answerable")

    def test_b1_schema_and_citation_limits(self):
        for value in ([1, True], [0], [2], "[1]"):
            with self.assertRaises(ValueError):
                metrics.parse_response(json.dumps({"answerability": "answerable", "answer": "Paris", "citations": value}), "B1", 1)
        with self.assertRaises(ValueError):
            metrics.parse_response('{"answerability":"insufficient_evidence","answer":"Paris","citations":[]}', "B1", 1)
        self.assertEqual(metrics.parse_response('```json\n{"answerability":"answerable","answer":"Paris","citations":[1]}\n```', "B1", 1)["answer"], "Paris")

    def test_qa_matches_project_normalization(self):
        self.assertEqual(metrics.qa_scores("Answer: Paris [1; 2]\nSources: [1]", ["Paris"]), (1.0, 1.0))
        self.assertEqual(metrics.qa_scores("The Paris", ["Paris"]), (0.0, 2 / 3))

    def test_failure_not_rewarded_as_correct_abstention(self):
        for status in ("exhausted", "provider_blocked"):
            with self.subTest(status=status):
                result = metrics.summarize([self.case(True)], {"a": self.prediction(status=status, answerability=None, answer=None)})
                self.assertEqual(result["false_answer_rate"], 1)
                self.assertEqual(result["generation_success_rate"], 0)
                self.assertEqual(result["abstention_f1"], 0)

    def test_correction_flag_does_not_claim_hallucination(self):
        r = metrics.per_case(self.case(True), self.prediction(answer="The premise is wrong; the city was Paris."))
        self.assertIn("counterfactual_non_abstention_review_correction", r["flags"])
        self.assertFalse(any("hallucination" in f for f in r["flags"]))

    def test_citation_validity_ratio(self):
        r = metrics.per_case(self.case(), self.prediction(invalid_citations=[7]))
        self.assertEqual(r["citation_valid"], .5)

    def test_gate_does_not_apply_to_provided_or_failed_results(self):
        for c, p in ((self.case(scope="provided_context"), self.prediction()), (self.case(), self.prediction(status="exhausted")),
                     (self.case(), self.prediction(status="provider_blocked", answerability=None, answer=None))):
            r = metrics.apply_gate([c], {"a": p}, {"a": {"top1_reranker_score": -5.0}}, 1.0)
            self.assertFalse(r["a"]["gate"]["rejected"])
            self.assertEqual(r["a"]["status"], p["status"])

    def test_calibration_final_refused_and_development_only(self):
        c = self.case()
        with self.assertRaises(ValueError):
            metrics.calibrate([{**c, "partition": "final_test"}], {}, {})
        r = metrics.calibrate([c], {"a": self.prediction()}, {"a": {"top1_reranker_score": 5.0}})
        self.assertEqual(r["status"], "selected")
        self.assertLessEqual(r["threshold"], 5.0)


class RunnerTests(unittest.TestCase):
    def trace_fixture(self, root):
        settings = runner.load_settings()
        settings["work"] = root
        row = {"id": "fixture_chunk_0", "text": "Paris is the city.", "metadata": {"article_id": "fixture", "canonical_article_id": "article"}, "score": 3.0, "reranker_score": 3.0}
        case = {"case_id": "fixture", "scope": "full_corpus", "case_type": "answerable_control", "revision": "revision", "source_article_id": "article", "source_gold_chunk_ids": [row["id"]],
                "review": {"runtime_status": "pending_retrieval", "content_status": "reviewed"}}
        trace = {"mode": "selected_retrieval", "case_revision": "revision", "contexts": [row], "retrieved_chunks": [row], "top1_reranker_score": 3.0}
        stored = {"identity": {"fixture": True}, "traces": {"fixture": trace}}
        local_runtime.atomic_json(root / "retrievals.json", stored)
        return settings, case, row, stored

    def test_trace_rejects_stale_text_invalid_score_and_missing_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, case, row, stored = self.trace_fixture(root)
            with patch.object(runner, "retrieval_identity", return_value=stored["identity"]), patch.object(runner, "active_corpus", return_value=[row]):
                self.assertIn("fixture", runner.checked_traces(settings, [case]))
                for key, value in (("top1_reranker_score", float("nan")), ("retrieved_chunks", []), ("case_revision", "old")):
                    bad = copy.deepcopy(stored)
                    bad["traces"]["fixture"][key] = value
                    local_runtime.atomic_json(root / "retrievals.json", bad)
                    with self.assertRaises(ValueError):
                        runner.checked_traces(settings, [case])
                bad = copy.deepcopy(stored)
                bad["traces"]["fixture"]["contexts"][0]["text"] = "substituted text"
                local_runtime.atomic_json(root / "retrievals.json", bad)
                with self.assertRaisesRegex(ValueError, "substituted"):
                    runner.checked_traces(settings, [case])

    def test_freeze_requires_actual_evidence_notes_not_pending_template(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, case, row, stored = self.trace_fixture(root)
            decision = {"case_id": "fixture", "decision": "pending", "trace_sha256": local_runtime.stable_hash(stored["traces"]["fixture"]),
                        "evidence_notes": "", "evidence_basis_checked": False, "scope_verified": False}
            review.write_jsonl(root / "retrieval_review.jsonl", [decision])
            with patch.object(runner, "draft", return_value=[case]), patch.object(runner, "retrieval_identity", return_value=stored["identity"]), patch.object(runner, "active_corpus", return_value=[row]):
                with self.assertRaises(ValueError):
                    runner.freeze(settings)
                decision.update(decision="accept", evidence_notes="Synthetic TEST ONLY: this fixture passage says Paris is the city.", evidence_basis_checked=True, scope_verified=True)
                review.write_jsonl(root / "retrieval_review.jsonl", [decision])
                frozen = runner.freeze(settings)
                self.assertEqual(frozen["case_count"], 1)
                runner.frozen_inputs(settings)
                review.write_jsonl(root / "retrieval_review.jsonl", [{**decision, "evidence_notes": "changed"}])
                with self.assertRaisesRegex(ValueError, "changed"):
                    runner.frozen_inputs(settings)

    def test_natural_miss_requires_a_source_in_active_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, case, row, stored = self.trace_fixture(root)
            case.update(case_type="natural_retrieval_miss", scope="provided_context", source_article_id="missing", source_gold_chunk_ids=["missing_chunk_0"])
            with patch.object(runner, "retrieval_identity", return_value=stored["identity"]), patch.object(runner, "active_corpus", return_value=[row]):
                with self.assertRaisesRegex(ValueError, "source to exist"):
                    runner.checked_traces(settings, [case])

    def test_smoke_one_each_and_never_final(self):
        selected = runner.stage_cases(review.read_jsonl(review.OUTPUT / "cases.jsonl"), "smoke")
        self.assertEqual(len(selected), 7)
        self.assertEqual(len({c["case_type"] for c in selected}), 7)
        self.assertTrue(all(c["partition"] == "development" for c in selected))

    def test_current_preflight_no_network(self):
        with patch.object(local_runtime.urllib.request, "build_opener", side_effect=AssertionError("Network forbidden")), patch.object(runner, "model_cached", return_value=False):
            result = runner.preflight(runner.load_settings(), "retrieve")
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(any("bge-reranker-large" in error for error in result["errors"]))
        self.assertFalse(any("testset" in error or "locked corpus" in error for error in result["errors"]))

    def test_dense_only_bge_m3_cache_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "models--BAAI--bge-m3/snapshots/fixture"
            snapshot.mkdir(parents=True)
            for name in ("config.json", "tokenizer_config.json", "pytorch_model.bin"):
                (snapshot / name).write_text("fixture")
            with patch.dict(runner.os.environ, {"HF_HUB_CACHE": directory}):
                self.assertFalse(runner.model_cached("BAAI/bge-m3"))
                for name in ("sparse_linear.pt", "colbert_linear.pt"):
                    (snapshot / name).write_text("fixture")
                self.assertTrue(runner.model_cached("BAAI/bge-m3"))

    def test_generation_cannot_bypass_missing_context_validation(self):
        with patch.object(runner, "generate_one", side_effect=AssertionError("No API")):
            with tempfile.TemporaryDirectory() as directory:
                settings = runner.load_settings()
                settings["work"] = Path(directory)
                with self.assertRaises(OSError):
                    runner.run(settings, "smoke", "fixture")

    def test_exclusion_checks_canonical_and_physical_ids(self):
        case = {"case_type": "removed_article", "source_article_id": "canonical", "source_gold_chunk_ids": ["physical_chunk_0"]}
        for meta in ({"article_id": "physical"}, {"article_id": "alternate", "canonical_article_id": "canonical"}):
            self.assertTrue(runner.excluded(case, {"id": "another", "metadata": meta}))

    def test_notebooks_execute_default_cells_offline(self):
        for name in ("14a_phase_3_abstention_preparation_colab.ipynb", "14b_phase_3_abstention_evaluation_colab.ipynb"):
            path = review.HERE / name
            book = json.loads(path.read_text())
            namespace = {}
            with patch.object(local_runtime.urllib.request, "build_opener", side_effect=AssertionError("No network")), patch("builtins.print"), patch("getpass.getpass", side_effect=AssertionError("No key prompt")):
                for cell in book["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [])
                        exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
            self.assertFalse(namespace.get("EXECUTE", namespace.get("COLLECT_RETRIEVALS")))

    def test_fake_full_run_smoke_reuse_final_lock_and_resume(self):
        cases = []
        for partition in ("development", "final_test"):
            for neg in (False, True):
                cases.append({"case_id": partition + str(neg), "partition": partition, "case_type": "removed_article" if neg else "answerable_control",
                              "answerability_label": "insufficient_evidence" if neg else "answerable", "scope": "full_corpus",
                              "accepted_answers": [] if neg else ["Paris"], "question": "Unknown?" if neg else "Which city?"})
        traces = {c["case_id"]: {"contexts": [{"id": "x", "text": "The city is Paris."}], "top1_reranker_score": 0.0 if c["accepted_answers"] == [] else 5.0} for c in cases}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = runner.load_settings()
            settings["work"] = root
            clock = FakeClock()
            ledger = local_runtime.QuotaLedger(root / "usage.sqlite", clock=clock.time, sleep=clock.sleep)
            calls = []
            def transport(payload, key):
                calls.append(payload)
                self.assertTrue(payload["messages"][1]["content"].startswith("Context:\n[1]"))
                structured = "Return exactly one JSON object" in payload["messages"][0]["content"]
                return response('{"answerability":"answerable","answer":"Paris","citations":[1]}' if structured else "Paris [1]")
            with patch.object(runner, "frozen_inputs", return_value=(cases, traces, {"fixture": True})), patch.object(runner, "QuotaLedger", return_value=ledger), patch("builtins.print"):
                with self.assertRaises(OSError):
                    runner.run(settings, "final", "fixture", transport=transport)
                self.assertEqual(calls, [])
                runner.run(settings, "smoke", "fixture", transport=transport)
                self.assertEqual(len(calls), 4)
                runner.run(settings, "development", "fixture", transport=transport)
                self.assertEqual(len(calls), 4)
                result = runner.run(settings, "final", "fixture", transport=transport)
                self.assertEqual(len(calls), 8)
                runner.run(settings, "development", "fixture", transport=transport)
                runner.run(settings, "final", "fixture", transport=transport)
                self.assertEqual(len(calls), 8)
                self.assertIn("B2", result["summaries"])
                self.assertEqual(result["summaries"]["B2"]["false_answer_rate"], 0.0)
                # A post-run runtime fix must not overwrite a completed experiment.
                original_hash = runner.file_hash
                with patch.object(runner, "file_hash", side_effect=lambda path: "changed-runtime" if path.name == "phase3_api.py" else original_hash(path)):
                    with self.assertRaisesRegex(local_runtime.RunPaused, "different experiment"):
                        runner.run(settings, "final", "fixture", transport=transport)
                self.assertEqual(len(calls), 8)


if __name__ == "__main__":
    unittest.main()
