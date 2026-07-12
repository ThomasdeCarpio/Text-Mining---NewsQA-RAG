# News Article Crawler

This crawler discovers recent English news articles from trusted sources and stores
normalized HTML, rendered raw HTML, metadata, and crawl manifests in a Hugging Face
Bucket. Local filesystem storage remains available for offline development and the
existing ingestion pipeline.

## Install

```bash
pip install -r crawler/requirements-crawler.txt
python -m playwright install chromium
```

Copy the crawler variables from `.env.example` to `.env` and set a Hugging Face
write token and bucket ID:

```dotenv
HF_TOKEN="hf-your-write-token-here"
HF_BUCKET_ID="your-username/newsqa-crawler"
HF_BUCKET_PRIVATE="true"
HF_BUCKET_PREFIX=""
CRAWLER_STORAGE="hf-bucket"
```

The crawler creates the configured bucket when it does not exist. Keep the bucket
private unless the stored publisher content is cleared for redistribution.

## Run

```bash
python crawler/crawl_articles.py --max-articles 50
```

Useful options:

```bash
python crawler/crawl_articles.py \
  --sources reuters,bbc,npr,guardian,ap \
  --categories World,Business,Technology \
  --since-hours 72 \
  --max-articles 20
```

Storage modes:

```bash
# Hugging Face Bucket only (default)
python crawler/crawl_articles.py --storage hf-bucket

# Local HTML only
python crawler/crawl_articles.py --storage filesystem

# Upload to the bucket and retain local ingestion-ready HTML
python crawler/crawl_articles.py --storage both
```

Hugging Face Bucket objects use deterministic URL hashes:

```text
<prefix>/articles/<source>/<hash-prefix>/<url-hash>/clean.html.gz
<prefix>/articles/<source>/<hash-prefix>/<url-hash>/raw.html.gz
<prefix>/articles/<source>/<hash-prefix>/<url-hash>/metadata.json
<prefix>/manifests/YYYY/MM/DD/<crawl-run-id>.jsonl.gz
```

Local outputs used by `filesystem` and `both`:

- `data/articles/`: normalized HTML for ingestion.
- `data/raw_articles/`: rendered raw HTML for debugging.
- `data/crawl_manifest.jsonl`: crawl status, metadata, output paths, and errors.

Vietnam-related articles are skipped by default. Use `--include-vietnam` only when
that content is intentionally needed.

Reuters remains configured as a source, but its public website may return HTTP 401
to automated browsers. The crawler does not bypass publisher access controls; a
licensed Reuters delivery feed is required when public pages reject the request.
