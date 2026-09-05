# Web UI

The React/Vite client calls FastAPI at `VITE_API_BASE_URL`, defaulting to
`http://localhost:8000`.

```bash
# Terminal 1: repository root
python -m uvicorn newsqa_app.api.main:app --reload --port 8000

# Terminal 2
cd app/frontend
npm run dev
```

Demo admin login: `admin` / `admin123`.

| Screen | Purpose |
| --- | --- |
| `/chat` | SSE chat, current-session history, and citations |
| `/retrieval` | Inspect the configured collection and one retrieval query |
| `/dashboard` | Preview/run experiments and load saved per-run analysis |

## Evaluation Desk

No metric is loaded without a selected experiment report. After choosing
**Load saved results**, select a run to update:

- MRR, NDCG, Recall, latency, coverage, and run-time cards;
- the comparison chart across runs in that experiment; and
- Failure Analysis from that run's `report.json`.

The dashboard reads YAML only from `configs/experiments/` and writes results
only under the configured project-local output directory. Edit YAML in Git,
not in the browser.

Login and roles are coursework-demo guards, not production authorization.
Chat history remains in API memory; experiment state is persisted on disk.
