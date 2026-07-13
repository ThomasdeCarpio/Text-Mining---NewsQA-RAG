"""
NewsQA test set builder.

Loads the NewsQA dataset (lucadiliello/newsqa on HuggingFace), samples N articles,
chunks them using the configured chunker, maps each answer span to chunk IDs, and
saves a JSONL file for use by run_benchmark.py.

Dataset fields used:
  context  — full article text
  question — question string
  answers  — list of answer strings (we take the first)
  labels   — list of {start: [int], end: [int]} character offsets (inclusive) in context
  key      — unique sample ID

JSONL output schema (one JSON line per question):
  {
    "question":            str,
    "ground_truth":        str,        # first answer string
    "article_key":         str,        # NewsQA key for the article group
    "relevant_chunk_ids":  list[str],  # chunk IDs whose text contains the answer span
    "evidence":            str,        # raw answer text from the label span
    "article_chunk_ids":   list[str],  # all chunk IDs from this article (for context)
  }
"""

import hashlib
import json
import os
import random
from itertools import islice


class NewsQATestSetBuilder:
    """Build and serialize an evaluation test set from NewsQA."""

    def __init__(self, chunker, overlap_threshold: float = 0.6, seed: int = 42):
        """
        Args:
            chunker: TextChunker (or any chunker with a chunk_article() method).
            overlap_threshold: Minimum word overlap ratio to consider a chunk relevant.
            seed: Random seed for reproducible article sampling.
        """
        self.chunker = chunker
        self.overlap_threshold = overlap_threshold
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        n_articles: int,
        output_path: str,
        split: str = "train",
        dataset_name: str = "lucadiliello/newsqa",
    ) -> list[dict]:
        """
        Build the test set and save it to output_path.

        Args:
            n_articles: Number of unique articles (contexts) to sample.
            output_path: Path for the output JSONL file.
            split: HuggingFace dataset split ("train" / "validation" / "test").
            dataset_name: HuggingFace dataset identifier.

        Returns:
            List of test set entries (same as written to JSONL).
        """
        from datasets import load_dataset

        print(f"Loading NewsQA ({split}) ...")
        raw = load_dataset(dataset_name, split=split, streaming=True)

        # Group samples by unique article key (context hash)
        print("Grouping by article ...")
        articles = self._group_by_article(raw)

        # Sample n_articles
        all_keys = list(articles.keys())
        if n_articles >= len(all_keys):
            print(f"Requested {n_articles} articles but only {len(all_keys)} available. Using all.")
            sampled_keys = all_keys
        else:
            rng = random.Random(self.seed)
            sampled_keys = rng.sample(all_keys, n_articles)

        print(f"Sampled {len(sampled_keys)} articles.")

        entries = []
        for idx, article_key in enumerate(sampled_keys, 1):
            samples = articles[article_key]
            context = samples[0]["context"]

            # Chunk the article text
            article_data = {
                "text": context,
                "metadata": {
                    "url": article_key,
                    "title": "",
                    "publish_date": "",
                    "publisher": "CNN",
                    "author": "",
                },
            }
            chunks = self.chunker.chunk_article(article_data, filename=article_key)
            all_chunk_ids = [c["id"] for c in chunks]

            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(sampled_keys)} articles ...")

            for sample in samples:
                answer_text, evidence_span = self._extract_answer(sample, context)
                if not answer_text:
                    continue  # skip unanswerable questions

                relevant_ids = self._map_to_chunks(answer_text, evidence_span, chunks)

                entries.append({
                    "question": sample["question"],
                    "ground_truth": answer_text,
                    "article_key": article_key,
                    "relevant_chunk_ids": relevant_ids,
                    "evidence": evidence_span,
                    "article_chunk_ids": all_chunk_ids,
                })

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        n_with_relevant = sum(1 for e in entries if e["relevant_chunk_ids"])
        print(
            f"\nTest set saved to {output_path}"
            f"\n  Total questions : {len(entries)}"
            f"\n  With relevant chunks : {n_with_relevant} ({100*n_with_relevant//max(len(entries),1)}%)"
            f"\n  No relevant chunks  : {len(entries) - n_with_relevant}"
        )
        return entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _group_by_article(self, dataset_iterable) -> dict[str, list[dict]]:
        """Stream the dataset and group samples by article key."""
        groups: dict[str, list[dict]] = {}
        for sample in dataset_iterable:
            key = sample.get("key", "")
            if not key:
                # Derive a key from a prefix of the context if key is missing
                key = sample["context"][:80]
            groups.setdefault(key, []).append(sample)
        return groups

    def _extract_answer(self, sample: dict, context: str) -> tuple[str, str]:
        """
        Return (answer_text, evidence_span) from a sample.

        Uses the first `labels` entry to extract the exact span from context;
        falls back to the first element of `answers`.
        """
        answers = sample.get("answers", [])
        labels = sample.get("labels", [])

        if not answers or answers[0] in ("", "None", None):
            return "", ""

        answer_text = str(answers[0]).strip()

        # Try to extract the span using character offsets from labels
        evidence_span = answer_text
        if labels:
            first_label = labels[0]
            starts = first_label.get("start", [])
            ends = first_label.get("end", [])
            if starts and ends:
                start_idx = int(starts[0])
                end_idx = int(ends[0]) + 1  # end is inclusive in this dataset
                if 0 <= start_idx < end_idx <= len(context):
                    evidence_span = context[start_idx:end_idx].strip()

        return answer_text, evidence_span

    def _map_to_chunks(
        self, answer_text: str, evidence_span: str, chunks: list[dict]
    ) -> list[str]:
        """
        Find which chunks contain the answer evidence.

        Strategy (in order of preference):
        1. Exact substring match of evidence_span in chunk text.
        2. Exact substring match of answer_text in chunk text.
        3. Fuzzy word overlap: overlap(evidence_words, chunk_words) >= threshold.
        """
        relevant = []
        evidence_lower = evidence_span.lower()
        answer_lower = answer_text.lower()
        evidence_words = set(evidence_lower.split())

        for chunk in chunks:
            text_lower = chunk["text"].lower()

            # Strategy 1 & 2: substring
            if evidence_lower in text_lower or answer_lower in text_lower:
                relevant.append(chunk["id"])
                continue

            # Strategy 3: fuzzy word overlap
            if evidence_words:
                chunk_words = set(text_lower.split())
                overlap = len(evidence_words & chunk_words) / len(evidence_words)
                if overlap >= self.overlap_threshold:
                    relevant.append(chunk["id"])

        return relevant


# ---------------------------------------------------------------------------
# Article-grouped builder + offset-based evidence->chunk mapping
#
# NewsQA `key` is per-question, so we group by `context` to get real articles
# with multiple questions. Evidence spans are exact char offsets, so we map them
# to chunks by range overlap (more precise than word overlap). Reference impl for
# notebooks/03_newsqa_mini_dataset.ipynb and scripts/build_mini_testset.py.
# ---------------------------------------------------------------------------

def evidence_to_span(sample: dict, context: str) -> tuple:
    """(start, end, text) of the answer evidence in context; fallback to answer substring."""
    labels = sample.get("labels") or []
    if labels and labels[0].get("start") and labels[0].get("end"):
        a, b = int(labels[0]["start"][0]), int(labels[0]["end"][0]) + 1  # end inclusive in NewsQA
        if 0 <= a < b <= len(context):
            return a, b, context[a:b]
    ans = (sample.get("answers") or [""])[0]
    i = context.find(ans)
    return (i, i + len(ans), ans) if i >= 0 else (None, None, ans)


def chunk_char_ranges(context: str, chunks: list[dict]) -> list[tuple]:
    """True [start, end) of each chunk within context, walking forward (handles overlap)."""
    ranges, cur = [], 0
    for c in chunks:
        pos = context.find(c["text"], cur)
        if pos < 0:                       # whitespace drift: retry on a short prefix
            pos = context.find(c["text"][:40])
        if pos < 0:
            ranges.append((None, None))
            continue
        ranges.append((pos, pos + len(c["text"])))
        cur = pos + 1                     # next chunk may overlap, so +1 not +len
    return ranges


def map_evidence_to_chunks(chunks: list[dict], ranges: list[tuple],
                           start, end, evidence: str) -> list[str]:
    """Chunk IDs whose char-range overlaps the evidence span; substring fallback."""
    if start is not None:
        hit = [c["id"] for c, (s, e) in zip(chunks, ranges)
               if s is not None and s < end and start < e]
        if hit:
            return hit
    evl = evidence.lower()
    return [c["id"] for c in chunks if evl and evl in c["text"].lower()]


def build_article_testset(
    chunker,
    n_articles: int = 15,
    max_scan: int = 800,
    split: str = "train",
    dataset_name: str = "lucadiliello/newsqa",
) -> tuple[list[dict], list[dict]]:
    """
    Build an article-grouped NewsQA test set with offset-based evidence->chunk mapping.

    Returns:
        (entries, all_chunks) — entries follow the schema in docs/evaluation.md §6.1;
        all_chunks are the chunk dicts (for optionally building a matching collection).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    groups: dict[str, list[dict]] = {}
    for i, s in enumerate(ds):
        if i >= max_scan:
            break
        groups.setdefault(s["context"], []).append(s)
    picked = list(groups.items())[:n_articles]

    entries, all_chunks = [], []
    for context, rows in picked:
        akey = "newsqa_" + hashlib.md5(context.encode("utf-8")).hexdigest()[:12]
        chunks = chunker.chunk_article(
            {"text": context, "metadata": {"url": "", "title": context.splitlines()[0][:80], "publisher": "CNN"}},
            filename=akey,
        )
        ranges = chunk_char_ranges(context, chunks)
        all_chunks.extend(chunks)
        for s in rows:
            a, b, ev = evidence_to_span(s, context)
            entries.append({
                "question": s["question"],
                "ground_truth": (s.get("answers") or [""])[0],
                "article_key": akey,
                "relevant_chunk_ids": map_evidence_to_chunks(chunks, ranges, a, b, ev),
                "evidence": ev,
                "article_chunk_ids": [c["id"] for c in chunks],
            })
    return entries, all_chunks


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def load_testset(path: str) -> list[dict]:
    """Load a JSONL test set file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_testset(entries: list[dict], path: str) -> None:
    """Save a list of test set entries to JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # ponytail self-check: the offset->chunk mapping is the part that silently corrupts a test set
    ctx = "AAA. BBB. CCC."
    chunks = [{"id": "a", "text": "AAA. BBB."}, {"id": "b", "text": "BBB. CCC."}]
    ranges = chunk_char_ranges(ctx, chunks)
    assert ranges == [(0, 9), (5, 14)], ranges
    assert map_evidence_to_chunks(chunks, ranges, 0, 3, "AAA") == ["a"]
    assert map_evidence_to_chunks(chunks, ranges, 10, 13, "CCC") == ["b"]
    assert map_evidence_to_chunks(chunks, ranges, 5, 8, "BBB") == ["a", "b"]  # spans the overlap
    assert evidence_to_span({"labels": [{"start": [0], "end": [2]}]}, ctx) == (0, 3, "AAA")
    assert evidence_to_span({"answers": ["CCC"]}, ctx) == (10, 13, "CCC")     # fallback
    print("testset self-check OK")
