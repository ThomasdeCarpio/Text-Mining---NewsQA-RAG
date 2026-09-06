import json, sys, collections, statistics
sys.path.insert(0, "common")
from newsqa_rag.ingestion.chunker import TextChunker

def count(root, label):
    ch = TextChunker(chunk_size=512, chunk_overlap=64)
    dist = collections.Counter(); total = 0; n = 0; toks = []
    import tiktoken; enc = tiktoken.get_encoding("cl100k_base")
    for name in ("evaluation_articles.jsonl", "distractor_articles.jsonl"):
        for line in open(f"{root}/{name}", encoding="utf-8"):
            a = json.loads(line)
            text = a["context"]
            k = len(ch.text_splitter.split_text(text))
            dist[k] += 1; total += k; n += 1
            toks.append(len(enc.encode(text)))
    print(f"--- {label}")
    print(f"  articles          {n:,}")
    print(f"  chunks            {total:,}")
    print(f"  mean chunks/bai   {total/n:.2f}")
    print(f"  median tokens/bai {statistics.median(toks):,.0f}   p90 {statistics.quantiles(toks, n=10)[8]:,.0f}   max {max(toks):,}")
    print(f"  distribution      {dict(sorted(dist.items()))}")
    return total, n

count("data/evaluation/newsqa_200_11064/staging/corpus", "PUBLISHED v1.0.0 (truoc restore)")
count("data/evaluation/newsqa_200_11064_restored/staging/corpus", "RESTORED v2.0.0 (sau restore)")
