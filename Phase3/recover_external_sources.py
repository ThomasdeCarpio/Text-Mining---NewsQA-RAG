"""Recover only the original external proposals' source passages.

Explicit, user-authorized download of the 1.6 MB validation file, not an old
testset comparison or a corpus replacement. Never add these texts to retrieval.
"""
import hashlib
import json
from pathlib import Path

from review_phase3 import BUNDLE, HERE, read_jsonl, write_jsonl

REVISION = "728e52920b8e4ffcfaad93fa47556f26a1d82546"
FILE = "data/validation-00000-of-00001-3cf888b12fff1dd6.parquet"


def recover():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("lucadiliello/newsqa", FILE, repo_type="dataset", revision=REVISION)
    texts = {hashlib.sha256(r["context"].encode()).hexdigest(): r["context"]
             for r in pq.read_table(path, columns=["context"]).to_pylist()}
    retired = {r["row"] for r in json.loads((HERE / "review_decisions.json").read_text())["external_replacements"]}
    result = []
    for row, case in enumerate(read_jsonl(BUNDLE / "authored_cases.jsonl"), 1):
        if case["case_type"] != "external_unanswerable" or row in retired:
            continue
        construction = case["construction"]
        digest = construction["source_context_sha256"]
        text = texts[digest]
        assert construction["source_revision"] == REVISION
        assert all(text[s["start"]:s["end"]] == s["text"] for s in construction["source_evidence_spans"])
        result.append({"authored_row": row, "source_article_id": case["source_article_id"],
                       "text": text, "sha256": digest, "source_revision": REVISION,
                       "source_url": f"https://huggingface.co/datasets/lucadiliello/newsqa/blob/{REVISION}/{FILE}"})
    assert len(result) == 18
    write_jsonl(HERE / "external_source_evidence.jsonl", result)
    print("Recovered 18 exact source passages; all original span offsets match. AI content review is separate.")


if __name__ == "__main__":
    recover()
