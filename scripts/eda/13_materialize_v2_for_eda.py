"""Materialize v2.0.0 chunks + gold mapping locally so the EDA can be re-run on it.

Restoration is append-only, so questions and evidence spans are untouched; only
the chunk boundaries and chunk IDs move. derive_chunked_testsets does exactly
that remapping, so we reuse it rather than reasoning about boundaries.
"""
import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
from newsqa_rag.ingestion.chunker import TextChunker
from newsqa_rag.evaluation.testset import derive_chunked_testsets

OUT = ROOT / "outputs" / "eda_v2" / "final"
OUT.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "data/evaluation/newsqa_200_11064_restored/staging/corpus"
V1 = ROOT / "data/evaluation/newsqa_200_11064/final"


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


ev = rd(SRC / "evaluation_articles.jsonl")
di = rd(SRC / "distractor_articles.jsonl")
print(f"articles: {len(ev)} evaluation + {len(di)} distractor", flush=True)

original_rows, clarified_rows, chunks = derive_chunked_testsets(
    ev, di, TextChunker(chunk_size=512, chunk_overlap=64)
)
print(f"chunks: {len(chunks):,}", flush=True)

with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps(c) + "\n")

# Carry the new gold mapping onto every v1 variant file, keyed by question id.
# Questions themselves are unchanged by restoration.
gold = {r["question_id"]: r["relevant_chunk_ids"] for r in original_rows}
article_chunks = {}
for c in chunks:
    article_chunks.setdefault(c["metadata"]["canonical_article_id"], []).append(c["id"])

for name in ("testset_original", "testset_reviewed_original", "testset_resolved",
             "testset_clarified"):
    rows = rd(V1 / f"{name}.jsonl")
    missed = 0
    for r in rows:
        key = r.get("source_question_id") or r["question_id"]
        if key in gold:
            r["relevant_chunk_ids"] = gold[key]
        else:
            missed += 1
        if "article_chunk_ids" in r:
            r["article_chunk_ids"] = article_chunks.get(r["article_key"], [])
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {name}: {len(rows):,} rows, {missed} unmapped", flush=True)

for name in ("review_annotations", "excluded_questions"):
    src = V1 / f"{name}.jsonl"
    if src.exists():
        (OUT / f"{name}.jsonl").write_bytes(src.read_bytes())
print("wrote", OUT)
