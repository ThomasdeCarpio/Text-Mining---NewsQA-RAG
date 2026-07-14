"""Reproducible NewsQA evaluation-dataset construction utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


DATASET_SCHEMA_VERSION = "2.1"
DEFAULT_DATASET_NAME = "lucadiliello/newsqa"
DEFAULT_DATASET_REVISION = "728e52920b8e4ffcfaad93fa47556f26a1d82546"


class DatasetBuildError(RuntimeError):
    """Raised when source data or generated artifacts violate the data contract."""


@dataclass(frozen=True)
class SampleSpec:
    """Parameters defining one reproducible article sample."""

    split: str
    n_articles: int
    seed: int
    role: str


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for hashing and manifests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def article_id_for_context(context: str) -> str:
    return f"newsqa_{sha256_text(context)[:16]}"


def normalized_context_hash(context: str) -> str:
    normalized = re.sub(r"\s+", " ", context).strip().lower()
    return sha256_text(normalized)


def _derived_title(context: str) -> str:
    for line in context.splitlines():
        value = line.strip()
        if value:
            return value[:160]
    return context.strip()[:160]


def distribution_summary(values: Sequence[float | int]) -> dict:
    """Compact deterministic distribution summary for dataset manifests."""

    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        position = round((len(ordered) - 1) * fraction)
        return round(ordered[position], 4)

    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def extract_evidence_spans(sample: dict, context: str) -> list[dict]:
    """Convert NewsQA inclusive offsets to validated [start, end) spans."""

    spans: list[dict] = []
    for label in sample.get("labels") or []:
        starts = label.get("start") or []
        ends = label.get("end") or []
        if len(starts) != len(ends):
            raise DatasetBuildError(
                f"Question {sample.get('key', '<unknown>')} has mismatched evidence offsets"
            )
        for start_value, end_value in zip(starts, ends):
            start = int(start_value)
            end = int(end_value) + 1
            if not 0 <= start < end <= len(context):
                raise DatasetBuildError(
                    f"Question {sample.get('key', '<unknown>')} has invalid span "
                    f"[{start}, {end}) for context length {len(context)}"
                )
            spans.append({"start": start, "end": end, "text": context[start:end]})

    if spans:
        return spans

    answers = sample.get("answers") or []
    answer = str(answers[0]).strip() if answers else ""
    start = context.find(answer) if answer else -1
    if start < 0:
        raise DatasetBuildError(
            f"Question {sample.get('key', '<unknown>')} has no usable evidence span"
        )
    return [{"start": start, "end": start + len(answer), "text": answer}]


def question_record(sample: dict, context: str, article_id: str) -> dict:
    answers = sample.get("answers") or []
    if not answers or not str(answers[0]).strip():
        raise DatasetBuildError(f"Question {sample.get('key', '<unknown>')} has no answer")
    question_id = str(sample.get("key") or "").strip()
    if not question_id:
        raise DatasetBuildError("NewsQA row is missing its question key")
    return {
        "question_id": question_id,
        "article_id": article_id,
        "question": str(sample.get("question") or "").strip(),
        "ground_truth": str(answers[0]).strip(),
        "evidence_spans": extract_evidence_spans(sample, context),
    }


def _default_dataset_factory(
    dataset_name: str, revision: str
) -> Callable[[str], Iterable[dict]]:
    def load(split: str) -> Iterable[dict]:
        from datasets import load_dataset

        return load_dataset(
            dataset_name,
            split=split,
            revision=revision,
            streaming=True,
        )

    return load


def sample_articles(
    dataset_factory: Callable[[str], Iterable[dict]],
    spec: SampleSpec,
    excluded_normalized_hashes: set[str] | None = None,
    include_questions: bool = True,
) -> tuple[list[dict], dict]:
    """Uniformly sample articles, then rescan the entire split to collect all rows."""

    excluded = excluded_normalized_hashes or set()
    source_counts: Counter[str] = Counter()
    normalized_by_id: dict[str, str] = {}
    raw_hash_by_id: dict[str, str] = {}
    context_chars_by_id: dict[str, int] = {}

    for sample in dataset_factory(spec.split):
        context = str(sample.get("context") or "")
        if not context:
            raise DatasetBuildError(f"Encountered an empty context in split {spec.split}")
        article_id = article_id_for_context(context)
        normalized_hash = normalized_context_hash(context)
        if normalized_hash in excluded:
            continue
        raw_hash = sha256_text(context)
        previous = raw_hash_by_id.setdefault(article_id, raw_hash)
        if previous != raw_hash:
            raise DatasetBuildError(f"Article ID collision for {article_id}")
        normalized_by_id[article_id] = normalized_hash
        context_chars_by_id[article_id] = len(context)
        source_counts[article_id] += 1

    candidates = sorted(source_counts)
    if len(candidates) < spec.n_articles:
        raise DatasetBuildError(
            f"Requested {spec.n_articles} {spec.split} articles, but only "
            f"{len(candidates)} are available after exclusions"
        )
    selected_ids = set(random.Random(spec.seed).sample(candidates, spec.n_articles))

    collected: dict[str, dict] = {}
    seen_questions: set[str] = set()
    second_pass_counts: Counter[str] = Counter()
    for sample in dataset_factory(spec.split):
        context = str(sample.get("context") or "")
        article_id = article_id_for_context(context)
        if article_id not in selected_ids:
            continue
        second_pass_counts[article_id] += 1
        article = collected.setdefault(
            article_id,
            {
                "article_id": article_id,
                "context_sha256": sha256_text(context),
                "normalized_context_sha256": normalized_context_hash(context),
                "split": spec.split,
                "role": spec.role,
                "context": context,
                "metadata": {"title": _derived_title(context), "publisher": "CNN"},
                "questions": [],
            },
        )
        if article["context"] != context:
            raise DatasetBuildError(f"Conflicting contexts for {article_id}")
        if include_questions:
            question = question_record(sample, context, article_id)
            if question["question_id"] in seen_questions:
                raise DatasetBuildError(f"Duplicate question ID {question['question_id']}")
            seen_questions.add(question["question_id"])
            article["questions"].append(question)

    if set(collected) != selected_ids:
        missing = sorted(selected_ids - set(collected))[:5]
        raise DatasetBuildError(f"Second pass did not recover selected articles: {missing}")
    for article_id in selected_ids:
        if source_counts[article_id] != second_pass_counts[article_id]:
            raise DatasetBuildError(
                f"Incomplete question collection for {article_id}: expected "
                f"{source_counts[article_id]}, got {second_pass_counts[article_id]}"
            )

    articles = []
    for article_id in sorted(collected):
        article = collected[article_id]
        article["source_question_count"] = source_counts[article_id]
        article["questions"].sort(key=lambda item: item["question_id"])
        articles.append(article)

    stats = {
        "split": spec.split,
        "role": spec.role,
        "seed": spec.seed,
        "candidate_articles": len(candidates),
        "selected_articles": len(articles),
        "selected_questions": sum(source_counts[item] for item in selected_ids),
        "selected_article_ids": sorted(selected_ids),
        "coverage": {
            "candidate_article_chars": distribution_summary(
                [context_chars_by_id[item] for item in candidates]
            ),
            "candidate_questions_per_article": distribution_summary(
                [source_counts[item] for item in candidates]
            ),
            "selected_article_chars": distribution_summary(
                [len(item["context"]) for item in articles]
            ),
            "selected_questions_per_article": distribution_summary(
                [source_counts[item] for item in selected_ids]
            ),
            "selected_question_chars": distribution_summary(
                [len(question["question"]) for item in articles for question in item["questions"]]
            ),
            "selected_answer_chars": distribution_summary(
                [len(question["ground_truth"]) for item in articles for question in item["questions"]]
            ),
            "selected_normalized_answer_position": distribution_summary(
                [
                    question["evidence_spans"][0]["start"] / max(len(item["context"]), 1)
                    for item in articles
                    for question in item["questions"]
                ]
            ),
        },
    }
    return articles, stats


def build_selection_bundle(
    dataset_name: str = DEFAULT_DATASET_NAME,
    revision: str = DEFAULT_DATASET_REVISION,
    evaluation_count: int = 200,
    distractor_count: int = 800,
    seed: int = 42,
    dataset_factory: Callable[[str], Iterable[dict]] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Build the locked validation sample and train distractor corpus."""

    factory = dataset_factory or _default_dataset_factory(dataset_name, revision)
    evaluation_articles, evaluation_stats = sample_articles(
        factory,
        SampleSpec("validation", evaluation_count, seed, "evaluation"),
        include_questions=True,
    )
    excluded = {item["normalized_context_sha256"] for item in evaluation_articles}
    distractor_articles, distractor_stats = sample_articles(
        factory,
        SampleSpec("train", distractor_count, seed, "distractor"),
        excluded_normalized_hashes=excluded,
        include_questions=False,
    )
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset": {"name": dataset_name, "revision": revision},
        "sampling": {
            "method": "uniform_without_replacement_over_sorted_context_hashes",
            "seed": seed,
            "evaluation": evaluation_stats,
            "distractors": distractor_stats,
        },
    }
    return evaluation_articles, distractor_articles, manifest


def chunk_char_ranges(context: str, chunks: Sequence[dict]) -> list[tuple[int, int]]:
    """Locate the production chunk texts in the unmodified source context."""

    ranges: list[tuple[int, int]] = []
    cursor = 0
    for chunk in chunks:
        text = chunk["text"]
        position = context.find(text, cursor)
        if position < 0:
            position = context.find(text)
        if position < 0:
            raise DatasetBuildError(f"Could not align chunk {chunk['id']} to its article")
        ranges.append((position, position + len(text)))
        cursor = position + 1
    return ranges


def map_spans_to_chunks(
    chunks: Sequence[dict], ranges: Sequence[tuple[int, int]], spans: Sequence[dict]
) -> list[str]:
    relevant: list[str] = []
    for chunk, (chunk_start, chunk_end) in zip(chunks, ranges):
        if any(chunk_start < span["end"] and span["start"] < chunk_end for span in spans):
            relevant.append(chunk["id"])
    if not relevant:
        raise DatasetBuildError("Evidence spans did not overlap any production chunk")
    return relevant


def derive_chunked_testsets(
    evaluation_articles: Sequence[dict],
    distractor_articles: Sequence[dict],
    chunker,
    annotations: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Chunk the complete corpus and construct original and clarified testsets."""

    annotations = annotations or {}
    chunks: list[dict] = []
    original_rows: list[dict] = []
    clarified_rows: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for article in [*evaluation_articles, *distractor_articles]:
        article_chunks = chunker.chunk_article(
            {
                "text": article["context"],
                "metadata": {
                    "url": "",
                    "title": article["metadata"].get("title", ""),
                    "publisher": article["metadata"].get("publisher", "CNN"),
                    "publish_date": "",
                    "author": "",
                },
            },
            filename=article["article_id"],
        )
        for chunk in article_chunks:
            if chunk["id"] in seen_chunk_ids:
                raise DatasetBuildError(f"Duplicate chunk ID {chunk['id']}")
            seen_chunk_ids.add(chunk["id"])
            chunk["metadata"].update(
                {
                    "canonical_article_id": article["article_id"],
                    "dataset_split": article["split"],
                    "corpus_role": article["role"],
                }
            )
        chunks.extend(article_chunks)

        if article["role"] != "evaluation":
            continue
        ranges = chunk_char_ranges(article["context"], article_chunks)
        all_chunk_ids = [item["id"] for item in article_chunks]
        for question in article["questions"]:
            relevant = map_spans_to_chunks(
                article_chunks, ranges, question["evidence_spans"]
            )
            annotation = annotations.get(question["question_id"], {})
            base = {
                "question_id": question["question_id"],
                "question_variant": "original",
                "question": question["question"],
                "ground_truth": question["ground_truth"],
                "article_key": article["article_id"],
                "evidence_spans": question["evidence_spans"],
                "evidence": question["evidence_spans"][0]["text"],
                "relevant_chunk_ids": relevant,
                "article_chunk_ids": all_chunk_ids,
                "standalone_label": annotation.get("final_label", "unreviewed"),
                "ambiguity_reasons": annotation.get("reason_codes", []),
            }
            original_rows.append(base)
            clarified = annotation.get("final_clarified_question")
            if clarified:
                clarified_rows.append(
                    {
                        **base,
                        "question_id": f"{question['question_id']}::clarified",
                        "source_question_id": question["question_id"],
                        "question": clarified,
                        "question_variant": "clarified",
                    }
                )

    original_rows.sort(key=lambda item: item["question_id"])
    clarified_rows.sort(key=lambda item: item["question_id"])
    chunks.sort(key=lambda item: item["id"])
    return original_rows, clarified_rows, chunks


def derive_reviewed_artifacts(
    original_rows: Sequence[dict],
    annotations: dict[str, dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Create reviewed, resolved, clarified, and excluded artifacts without rechunking."""

    reviewed_original_rows: list[dict] = []
    clarified_rows: list[dict] = []
    resolved_rows: list[dict] = []
    excluded_rows: list[dict] = []
    original_ids = {row["question_id"] for row in original_rows}
    unknown_annotations = sorted(set(annotations) - original_ids)
    if unknown_annotations:
        raise DatasetBuildError(
            f"Review annotations contain unknown question IDs: {unknown_annotations[:5]}"
        )

    for row in original_rows:
        question_id = row["question_id"]
        annotation = annotations.get(question_id)
        if annotation is None:
            raise DatasetBuildError(f"Missing review annotation for {question_id}")

        clarified = annotation.get("final_clarified_question")
        reviewed_fields = {
            "standalone_label": annotation.get("final_label", "unreviewed"),
            "ambiguity_reasons": annotation.get("reason_codes", []),
            "ground_truth": annotation.get("ground_truth", row["ground_truth"]),
            "accepted_answers": annotation.get(
                "accepted_answers", [annotation.get("ground_truth", row["ground_truth"])]
            ),
            "evidence_spans": annotation.get("evidence_spans", row["evidence_spans"]),
            "evidence": annotation.get("evidence", row.get("evidence", "")),
            "relevant_chunk_ids": annotation.get(
                "relevant_chunk_ids", row["relevant_chunk_ids"]
            ),
            "answer_modified": annotation.get("answer_modified", False),
        }
        if annotation.get("answer_modified"):
            reviewed_fields.update(
                {
                    "source_ground_truth": annotation.get(
                        "source_ground_truth", row["ground_truth"]
                    ),
                    "source_evidence_spans": annotation.get(
                        "source_evidence_spans", row["evidence_spans"]
                    ),
                    "answer_review_notes": annotation.get("answer_review_notes", ""),
                }
            )
        if annotation.get("excluded"):
            excluded_rows.append(
                {
                    **row,
                    **reviewed_fields,
                    "source_question_id": question_id,
                    "excluded": True,
                    "exclusion_reasons": annotation.get("exclusion_reasons", []),
                    "review_notes": annotation.get("review_notes", ""),
                }
            )
            continue
        reviewed_original_rows.append(
            {
                **row,
                **reviewed_fields,
                "source_question_id": question_id,
                "question": row["question"],
                "question_variant": "original",
            }
        )
        resolved_rows.append(
            {
                **row,
                **reviewed_fields,
                "source_question_id": question_id,
                "question": clarified or row["question"],
                "question_variant": "clarified" if clarified else "original",
            }
        )
        if clarified:
            clarified_rows.append(
                {
                    **row,
                    **reviewed_fields,
                    "question_id": f"{question_id}::clarified",
                    "source_question_id": question_id,
                    "question": clarified,
                    "question_variant": "clarified",
                }
            )

    reviewed_original_rows.sort(key=lambda item: item["question_id"])
    clarified_rows.sort(key=lambda item: item["question_id"])
    resolved_rows.sort(key=lambda item: item["question_id"])
    excluded_rows.sort(key=lambda item: item["question_id"])
    return reviewed_original_rows, clarified_rows, resolved_rows, excluded_rows


def derive_reviewed_testsets(
    original_rows: Sequence[dict],
    annotations: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Compatibility wrapper returning the paired clarified and resolved sets."""

    _, clarified_rows, resolved_rows, _ = derive_reviewed_artifacts(
        original_rows, annotations
    )
    return clarified_rows, resolved_rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_testset(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def save_jsonl(entries: Iterable[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(canonical_json(entry) + "\n")
    os.replace(temporary, output)


def save_testset(entries: list[dict], path: str | Path) -> None:
    save_jsonl(entries, path)


def artifact_record(path: str | Path, root: str | Path | None = None) -> dict:
    artifact = Path(path)
    relative = os.path.relpath(artifact, root) if root else str(artifact)
    return {"path": relative, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)}


# Backward-compatible helper used by notebooks and older scripts.
def build_article_testset(
    chunker,
    n_articles: int = 15,
    max_scan: int | None = None,
    split: str = "validation",
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> tuple[list[dict], list[dict]]:
    if max_scan is not None:
        print("WARNING: max_scan is deprecated and ignored; the complete split is scanned.")
    factory = _default_dataset_factory(dataset_name, DEFAULT_DATASET_REVISION)
    articles, _ = sample_articles(
        factory,
        SampleSpec(split, n_articles, 42, "evaluation"),
        include_questions=True,
    )
    rows, _, chunks = derive_chunked_testsets(articles, [], chunker)
    return rows, chunks


class NewsQATestSetBuilder:
    """Compatibility adapter around the complete-scan builder."""

    def __init__(self, chunker, overlap_threshold: float = 0.6, seed: int = 42):
        self.chunker = chunker
        self.seed = seed

    def build(
        self,
        n_articles: int,
        output_path: str,
        split: str = "validation",
        dataset_name: str = DEFAULT_DATASET_NAME,
    ) -> list[dict]:
        factory = _default_dataset_factory(dataset_name, DEFAULT_DATASET_REVISION)
        articles, _ = sample_articles(
            factory,
            SampleSpec(split, n_articles, self.seed, "evaluation"),
            include_questions=True,
        )
        rows, _, _ = derive_chunked_testsets(articles, [], self.chunker)
        save_testset(rows, output_path)
        return rows
