"""Construction and validation helpers for the NewsQA abstention benchmark."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from newsqa_rag.evaluation.benchmark_io import stable_hash, utc_now
from newsqa_rag.evaluation.testset import DatasetBuildError, sha256_file

SCHEMA_VERSION = 1
CASE_TYPES = {
    "answerable_control",
    "natural_retrieval_miss",
    "controlled_context_ablation",
    "removed_article",
    "external_unanswerable",
    "counterfactual",
    "partial_weak_evidence",
}
CONTEXT_SCOPED_TYPES = {
    "natural_retrieval_miss",
    "controlled_context_ablation",
    "partial_weak_evidence",
}
CORPUS_SCOPED_TYPES = {"removed_article", "external_unanswerable"}
TARGETS = {
    "pilot": {
        "answerable_control": 20,
        "natural_retrieval_miss": 5,
        "controlled_context_ablation": 5,
        "removed_article": 5,
        "external_unanswerable": 5,
        "counterfactual": 5,
        "partial_weak_evidence": 5,
    },
    "full": {
        "answerable_control": 200,
        "natural_retrieval_miss": 50,
        "controlled_context_ablation": 50,
        "removed_article": 50,
        "external_unanswerable": 50,
        "counterfactual": 50,
        "partial_weak_evidence": 50,
    },
    "compact_200": {
        "answerable_control": 80,
        "natural_retrieval_miss": 10,
        "controlled_context_ablation": 22,
        "removed_article": 22,
        "external_unanswerable": 22,
        "counterfactual": 22,
        "partial_weak_evidence": 22,
    },
}

PARTITION_TARGETS = {
    "compact_200": {
        "development": {
            "answerable_control": 56,
            "natural_retrieval_miss": 7,
            "controlled_context_ablation": 16,
            "removed_article": 16,
            "external_unanswerable": 15,
            "counterfactual": 15,
            "partial_weak_evidence": 15,
        },
        "final_test": {
            "answerable_control": 24,
            "natural_retrieval_miss": 3,
            "controlled_context_ablation": 6,
            "removed_article": 6,
            "external_unanswerable": 7,
            "counterfactual": 7,
            "partial_weak_evidence": 7,
        },
    }
}


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(f"Malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise DatasetBuildError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def save_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(target)


def article_partitions(questions: list[dict], seed: int, development_articles: int) -> dict[str, str]:
    articles = sorted({str(row["article_key"]) for row in questions})
    if development_articles >= len(articles):
        raise DatasetBuildError("development_articles must be smaller than article count")
    random.Random(seed).shuffle(articles)
    development = set(articles[:development_articles])
    return {article: ("development" if article in development else "final_test") for article in articles}


def compact_article_partitions(
    questions: list[dict],
    natural_miss_rows: list[dict],
    seed: int,
    development_articles: int,
) -> dict[str, str]:
    """Create an article split with enough observed misses for both partitions."""
    articles = sorted({str(row["article_key"]) for row in questions})
    if development_articles >= len(articles):
        raise DatasetBuildError("development_articles must be smaller than article count")
    miss_counts = Counter(str(row["article_key"]) for row in natural_miss_rows)
    dev_misses = PARTITION_TARGETS["compact_200"]["development"]["natural_retrieval_miss"]
    final_misses = PARTITION_TARGETS["compact_200"]["final_test"]["natural_retrieval_miss"]
    if sum(miss_counts.values()) < dev_misses + final_misses:
        raise DatasetBuildError(
            "compact_200 requires at least "
            f"{dev_misses + final_misses} natural retrieval misses; "
            f"observed {sum(miss_counts.values())}"
        )
    rng = random.Random(seed)
    for _ in range(10_000):
        candidate = list(articles)
        rng.shuffle(candidate)
        development = set(candidate[:development_articles])
        observed_development = sum(
            count for article, count in miss_counts.items() if article in development
        )
        observed_final = sum(miss_counts.values()) - observed_development
        if observed_development >= dev_misses and observed_final >= final_misses:
            return {
                article: ("development" if article in development else "final_test")
                for article in articles
            }
    raise DatasetBuildError(
        "Could not construct the compact_200 article split with the required "
        "natural-miss capacity"
    )


def chunk_indexes(chunks: list[dict]) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, list[str]]]:
    by_id: dict[str, dict] = {}
    by_canonical: dict[str, list[str]] = defaultdict(list)
    physical_by_canonical: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        metadata = chunk.get("metadata") or {}
        canonical = str(metadata.get("canonical_article_id") or "")
        physical = str(metadata.get("article_id") or "")
        if not chunk_id or chunk_id in by_id or not canonical:
            raise DatasetBuildError(f"Invalid or duplicate chunk record: {chunk_id!r}")
        by_id[chunk_id] = chunk
        by_canonical[canonical].append(chunk_id)
        if physical and physical not in physical_by_canonical[canonical]:
            physical_by_canonical[canonical].append(physical)
    return by_id, dict(by_canonical), dict(physical_by_canonical)


def _case_id(case_type: str, base_id: str, question: str) -> str:
    digest = stable_hash({"type": case_type, "base": base_id, "question": question})[:20]
    return f"abs_{case_type}_{digest}"


def _base_case(row: dict, case_type: str, partition: str, *, question: str | None = None) -> dict:
    expected = "answerable" if case_type == "answerable_control" else "insufficient_evidence"
    scope = "provided_context" if case_type in CONTEXT_SCOPED_TYPES else "full_corpus"
    base_id = str(row["question_id"])
    text = str(question if question is not None else row["question"])
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": _case_id(case_type, base_id, text),
        "base_question_id": base_id,
        "source_question_id": row.get("source_question_id", base_id),
        "case_type": case_type,
        "partition": partition,
        "question": text,
        "answerability_label": expected,
        "scope": scope,
        "source_article_id": row.get("article_key"),
        "accepted_answers": deepcopy(row.get("accepted_answers") or [row.get("ground_truth", "")]),
        "ground_truth": row.get("ground_truth"),
        "source_gold_chunk_ids": list(row.get("relevant_chunk_ids") or []),
        "gold_relevant_chunk_ids": (
            list(row.get("relevant_chunk_ids") or []) if expected == "answerable" else []
        ),
        "provided_context_chunk_ids": [],
        "excluded_chunk_ids": [],
        "excluded_article_ids": [],
        "construction": {"method": case_type, "status": "generated"},
        "human_review": {
            "decision": "pending",
            "reviewer_id": "",
            "scope_verified": False,
            "notes": "",
        },
        "secondary_review": {"decision": "not_selected", "reviewer_id": "", "notes": ""},
    }


def _trace_chunks(record: dict) -> list[str]:
    trace = record.get("trace") or record.get("result") or {}
    ranked = trace.get("reranked_chunks") or trace.get("retrieved_chunks") or []
    return [str(item.get("id")) for item in ranked if item.get("id")]


def retrieval_map(records: list[dict]) -> dict[str, list[str]]:
    mapped = {}
    for record in records:
        if record.get("status") != "success":
            continue
        question_id = str(record.get("question_id") or record.get("base_question_id") or "")
        ids = _trace_chunks(record)
        if question_id and ids:
            mapped[question_id] = ids
    return mapped


def _sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if count <= 0:
        return []
    if len(rows) < count:
        return list(rows)
    return rng.sample(rows, count)


def _partition_of(candidate: dict | tuple) -> str:
    row = candidate[0] if isinstance(candidate, tuple) else candidate
    return str(row.get("partition") or "")


def _sample_for_type(
    candidates: list,
    case_type: str,
    mode: str,
    rng: random.Random,
) -> list:
    partition_targets = PARTITION_TARGETS.get(mode)
    if not partition_targets:
        return _sample(candidates, TARGETS[mode][case_type], rng)
    selected = []
    for partition in ("development", "final_test"):
        pool = [item for item in candidates if _partition_of(item) == partition]
        target = partition_targets[partition][case_type]
        if case_type == "answerable_control":
            by_article: dict[str, list] = defaultdict(list)
            for item in pool:
                row = item[0] if isinstance(item, tuple) else item
                by_article[str(row.get("article_key") or row.get("source_article_id"))].append(item)
            article_ids = sorted(by_article)
            rng.shuffle(article_ids)
            chosen = [rng.choice(by_article[article]) for article in article_ids[:target]]
            if len(chosen) < target:
                chosen_ids = {id(item) for item in chosen}
                remaining = [item for item in pool if id(item) not in chosen_ids]
                chosen.extend(_sample(remaining, target - len(chosen), rng))
            selected.extend(chosen)
        else:
            selected.extend(_sample(pool, target, rng))
    return selected


def _answer_bearing_chunk_ids(row: dict, candidate_ids: Iterable[str], by_chunk: dict[str, dict]) -> set[str]:
    answers = {
        " ".join(str(answer).lower().split()).strip(".,!?;:\"'")
        for answer in (row.get("accepted_answers") or [row.get("ground_truth", "")])
    }
    answers -= {"", "yes", "no"}
    leaking = set()
    for chunk_id in candidate_ids:
        chunk = by_chunk.get(chunk_id)
        if chunk is None:
            continue
        text = " ".join(str(chunk.get("text") or "").lower().split())
        if any(len(answer) >= 4 and answer in text for answer in answers):
            leaking.add(chunk_id)
    return leaking


def _normalize_authored_case(raw: dict, questions: dict[str, dict], partitions: dict[str, str]) -> dict:
    case_type = str(raw.get("case_type") or "")
    if case_type not in {"external_unanswerable", "counterfactual"}:
        raise DatasetBuildError(f"Authored case has unsupported type: {case_type!r}")
    base_id = str(raw.get("base_question_id") or "")
    question = str(raw.get("question") or "").strip()
    if not question:
        raise DatasetBuildError("Authored abstention case requires a question")
    if case_type == "counterfactual":
        if base_id not in questions:
            raise DatasetBuildError(f"Unknown counterfactual base_question_id: {base_id}")
        case = _base_case(questions[base_id], case_type, partitions[questions[base_id]["article_key"]], question=question)
    else:
        source_article = str(raw.get("source_article_id") or "")
        partition = str(raw.get("partition") or "development")
        if partition not in {"development", "final_test"}:
            raise DatasetBuildError(f"Invalid external case partition: {partition!r}")
        synthetic = {
            "question_id": base_id or stable_hash(question)[:32],
            "question": question,
            "article_key": source_article or None,
            "accepted_answers": [],
            "ground_truth": None,
            "relevant_chunk_ids": [],
        }
        case = _base_case(synthetic, case_type, partition, question=question)
    case["construction"].update(deepcopy(raw.get("construction") or {}))
    case["construction"]["status"] = "authored_proposal"
    for key in ("excluded_article_ids", "excluded_chunk_ids", "provided_context_chunk_ids"):
        case[key] = list(raw.get(key) or [])
    return case


def build_review_queue(
    questions: list[dict],
    chunks: list[dict],
    *,
    mode: str,
    seed: int,
    development_articles: int = 50,
    retrieval_records: list[dict] | None = None,
    authored_cases: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    if mode not in TARGETS:
        raise DatasetBuildError(f"Unknown abstention build mode: {mode}")
    targets = TARGETS[mode]
    by_question = {str(row["question_id"]): row for row in questions}
    if len(by_question) != len(questions):
        raise DatasetBuildError("Question IDs must be unique")
    by_chunk, chunks_by_article, physical_by_article = chunk_indexes(chunks)
    rng = random.Random(seed)
    trace_map = retrieval_map(retrieval_records or [])
    observed_misses = []
    for row in questions:
        ranked = trace_map.get(str(row["question_id"]), [])
        gold = set(row.get("relevant_chunk_ids") or [])
        if ranked and gold.isdisjoint(ranked) and not _answer_bearing_chunk_ids(row, ranked, by_chunk):
            observed_misses.append(row)
    partitions = (
        compact_article_partitions(questions, observed_misses, seed, development_articles)
        if mode == "compact_200"
        else article_partitions(questions, seed, development_articles)
    )
    pool = [
        {**row, "partition": partitions[row["article_key"]]}
        for row in questions
        if mode in {"full", "compact_200"}
        or partitions[row["article_key"]] == "development"
    ]
    cases: list[dict] = []

    control_pool = pool
    if mode == "compact_200":
        control_pool = []
        for row in pool:
            ranked = trace_map.get(str(row["question_id"]), [])
            if set(row.get("relevant_chunk_ids") or []) & set(ranked[:3]):
                control_pool.append(row)
    selected_control_rows = _sample_for_type(
        control_pool, "answerable_control", mode, rng
    )
    for row in selected_control_rows:
        case = _base_case(row, "answerable_control", partitions[row["article_key"]])
        case["provided_context_chunk_ids"] = (
            trace_map.get(str(row["question_id"]), [])[:5]
            or list(row.get("relevant_chunk_ids") or [])
        )
        cases.append(case)

    observed_miss_ids = {str(row["question_id"]) for row in observed_misses}
    misses = [
        (row, trace_map[str(row["question_id"])])
        for row in pool
        if str(row["question_id"]) in observed_miss_ids
    ]
    for row, ranked in _sample_for_type(misses, "natural_retrieval_miss", mode, rng):
        case = _base_case(row, "natural_retrieval_miss", partitions[row["article_key"]])
        case["provided_context_chunk_ids"] = ranked
        case["construction"]["retrieval_observed"] = True
        cases.append(case)

    derived_pool = selected_control_rows if mode == "compact_200" else pool
    ablation_pool = []
    for row in derived_pool:
        ranked = trace_map.get(str(row["question_id"]), [])
        gold = set(row.get("relevant_chunk_ids") or [])
        leaking = _answer_bearing_chunk_ids(row, ranked, by_chunk)
        excluded = gold | leaking
        remaining = [item for item in ranked if item not in excluded]
        if remaining:
            ablation_pool.append((row, remaining, excluded))
    for row, remaining, excluded in _sample_for_type(
        ablation_pool, "controlled_context_ablation", mode, rng
    ):
        case = _base_case(row, "controlled_context_ablation", partitions[row["article_key"]])
        case["excluded_chunk_ids"] = sorted(excluded)
        case["provided_context_chunk_ids"] = remaining
        cases.append(case)

    removal_pool = [row for row in derived_pool if row.get("article_key") in chunks_by_article]
    for row in _sample_for_type(removal_pool, "removed_article", mode, rng):
        article = str(row["article_key"])
        case = _base_case(row, "removed_article", partitions[article])
        case["excluded_chunk_ids"] = sorted(chunks_by_article[article])
        case["excluded_article_ids"] = sorted(physical_by_article.get(article, []))
        cases.append(case)

    weak_pool = []
    for row in derived_pool:
        article_chunks = chunks_by_article.get(str(row.get("article_key")), [])
        gold = set(row.get("relevant_chunk_ids") or [])
        leaking = _answer_bearing_chunk_ids(row, article_chunks, by_chunk)
        excluded = gold | leaking
        remaining = [chunk_id for chunk_id in article_chunks if chunk_id not in excluded]
        if remaining:
            weak_pool.append((row, remaining, excluded))
    for row, remaining, excluded in _sample_for_type(weak_pool, "partial_weak_evidence", mode, rng):
        case = _base_case(row, "partial_weak_evidence", partitions[row["article_key"]])
        case["excluded_chunk_ids"] = sorted(excluded)
        case["provided_context_chunk_ids"] = remaining[:5]
        cases.append(case)

    normalized_authored = [
        _normalize_authored_case(raw, by_question, partitions) for raw in (authored_cases or [])
    ]
    for case_type in ("external_unanswerable", "counterfactual"):
        candidates = [row for row in normalized_authored if row["case_type"] == case_type]
        if mode == "compact_200" and case_type == "counterfactual":
            control_ids = {str(row["question_id"]) for row in selected_control_rows}
            candidates = [row for row in candidates if row["base_question_id"] in control_ids]
        cases.extend(_sample_for_type(candidates, case_type, mode, rng))

    cases.sort(key=lambda row: (row["partition"], row["case_type"], row["case_id"]))
    observed = Counter(row["case_type"] for row in cases)
    observed_by_partition = {
        partition: dict(sorted(Counter(
            row["case_type"] for row in cases if row["partition"] == partition
        ).items()))
        for partition in ("development", "final_test")
    }
    deficits = {name: count - observed.get(name, 0) for name, count in targets.items() if observed.get(name, 0) < count}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "awaiting_human_review",
        "mode": mode,
        "seed": seed,
        "development_articles": development_articles,
        "partition_method": (
            "seeded_article_split_stratified_by_observed_natural_miss"
            if mode == "compact_200"
            else "seeded_article_split"
        ),
        "partition_articles": {
            partition: sorted(
                article for article, assigned in partitions.items() if assigned == partition
            )
            for partition in ("development", "final_test")
        },
        "targets": targets,
        "observed": dict(sorted(observed.items())),
        "partition_targets": PARTITION_TARGETS.get(mode),
        "observed_by_partition": observed_by_partition,
        "deficits": deficits,
        "instructions": {
            "authored_case_types": ["external_unanswerable", "counterfactual"],
            "natural_miss_requires_retrieval_trace": True,
            "finalization_requires_exact_targets": True,
        },
    }
    return cases, manifest


def validate_cases(cases: list[dict], chunks: list[dict], *, require_approved: bool = False) -> dict:
    by_chunk, chunks_by_article, _ = chunk_indexes(chunks)
    ids: set[str] = set()
    errors: list[str] = []
    partitions_by_base: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    secondary_counts = Counter()
    for case in cases:
        case_id = str(case.get("case_id") or "")
        case_type = str(case.get("case_type") or "")
        counts[case_type] += 1
        if not case_id or case_id in ids:
            errors.append(f"duplicate_or_missing_case_id:{case_id}")
        ids.add(case_id)
        if case_type not in CASE_TYPES:
            errors.append(f"{case_id}:invalid_case_type")
        expected = "answerable" if case_type == "answerable_control" else "insufficient_evidence"
        if case.get("answerability_label") != expected:
            errors.append(f"{case_id}:wrong_answerability_label")
        scope = "provided_context" if case_type in CONTEXT_SCOPED_TYPES else "full_corpus"
        if case.get("scope") != scope:
            errors.append(f"{case_id}:wrong_scope")
        unknown = (set(case.get("provided_context_chunk_ids") or []) | set(case.get("excluded_chunk_ids") or [])) - set(by_chunk)
        if unknown:
            errors.append(f"{case_id}:unknown_chunks:{sorted(unknown)[:3]}")
        gold = set(case.get("gold_relevant_chunk_ids") or [])
        source_gold = set(case.get("source_gold_chunk_ids") or [])
        provided = set(case.get("provided_context_chunk_ids") or [])
        if expected == "answerable" and not gold:
            errors.append(f"{case_id}:answerable_without_gold")
        if case_type in CONTEXT_SCOPED_TYPES and source_gold & provided:
            errors.append(f"{case_id}:gold_context_leakage")
        if case_type in CONTEXT_SCOPED_TYPES and not provided:
            errors.append(f"{case_id}:empty_controlled_context")
        if case_type in CONTEXT_SCOPED_TYPES:
            normalized_answers = {
                " ".join(str(answer).lower().split()).strip(".,!?;:\"'")
                for answer in (case.get("accepted_answers") or [])
            }
            normalized_answers -= {"", "yes", "no"}
            for chunk_id in provided:
                text = " ".join(str(by_chunk[chunk_id].get("text") or "").lower().split())
                if any(len(answer) >= 4 and answer in text for answer in normalized_answers):
                    errors.append(f"{case_id}:answer_text_leakage:{chunk_id}")
                    break
        if case_type == "removed_article":
            source = case.get("source_article_id")
            if source not in chunks_by_article or set(chunks_by_article[source]) != set(case.get("excluded_chunk_ids") or []):
                errors.append(f"{case_id}:incomplete_article_removal")
            if not case.get("excluded_article_ids"):
                errors.append(f"{case_id}:missing_physical_article_exclusion")
        if case_type == "external_unanswerable":
            source = str(case.get("source_article_id") or "")
            if not source:
                errors.append(f"{case_id}:missing_external_article_id")
            elif source in chunks_by_article:
                errors.append(f"{case_id}:external_article_exists_in_corpus")
        if case_type == "counterfactual":
            changed_field = str((case.get("construction") or {}).get("changed_field") or "")
            if changed_field not in {"subject", "event", "location", "date", "quantity"}:
                errors.append(f"{case_id}:invalid_counterfactual_changed_field")
        review = case.get("human_review") or {}
        if require_approved:
            if review.get("decision") != "approved":
                errors.append(f"{case_id}:not_approved")
            if not str(review.get("reviewer_id") or "").strip():
                errors.append(f"{case_id}:missing_reviewer_id")
            if expected == "insufficient_evidence":
                if not str(review.get("notes") or "").strip():
                    errors.append(f"{case_id}:missing_negative_review_notes")
                if review.get("scope_verified") is not True:
                    errors.append(f"{case_id}:scope_not_verified")
            if case_type in CORPUS_SCOPED_TYPES | {"counterfactual"}:
                secondary = case.get("secondary_review") or {}
                if secondary.get("decision") != "approved":
                    errors.append(f"{case_id}:missing_secondary_approval")
                secondary_id = str(secondary.get("reviewer_id") or "").strip()
                if not secondary_id or secondary_id == str(review.get("reviewer_id") or "").strip():
                    errors.append(f"{case_id}:invalid_secondary_reviewer")
            else:
                secondary = case.get("secondary_review") or {}
                if secondary.get("decision") == "approved":
                    secondary_id = str(secondary.get("reviewer_id") or "").strip()
                    if not secondary_id or secondary_id == str(review.get("reviewer_id") or "").strip():
                        errors.append(f"{case_id}:invalid_secondary_reviewer")
                    else:
                        secondary_counts[case_type] += 1
        base_id = str(case.get("base_question_id") or case_id)
        partitions_by_base[base_id].add(str(case.get("partition") or ""))
    leakage = sorted(base for base, values in partitions_by_base.items() if len(values) > 1)
    errors.extend(f"partition_leakage:{base}" for base in leakage)
    if require_approved:
        for case_type, count in counts.items():
            if case_type in CORPUS_SCOPED_TYPES | {"counterfactual"}:
                continue
            required = math.ceil(count * 0.2)
            if secondary_counts[case_type] < required:
                errors.append(
                    f"secondary_review_coverage:{case_type}:"
                    f"required={required}:observed={secondary_counts[case_type]}"
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors else "failed",
        "cases": len(cases),
        "counts_by_type": dict(sorted(counts.items())),
        "errors": errors,
    }


def artifact_record(path: str | Path) -> dict:
    target = Path(path)
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
