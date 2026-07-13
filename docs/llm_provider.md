# LLM provider switching

## Status: ⛔ Proxy server NOT implemented yet — planned

There is **no** central proxy/gateway for switching LLM providers today. A small
proxy server is **planned** so the app can point at one endpoint and swap providers
(OpenAI / DeepSeek / local / …) behind it, without touching call sites. This doc
records the current state so that work is easy to pick up.

## How switching works right now (ad-hoc, env-driven)

Provider choice is decided **inline at each call site**, driven by environment
variables. There are three independent spots:

| What | File | Rule |
| --- | --- | --- |
| Generation LLM | `src/llm.py` (`OpenAILLM._get_client` / `_model`) | `DEEPSEEK_API_KEY` set → DeepSeek (`deepseek-chat`, `base_url=https://api.deepseek.com`); else `OPENAI_API_KEY` / `OPENAI_BASE_URL` |
| RAGAS judge LLM | `src/evaluation/metrics.py` (`_ragas_judge`) | Same DeepSeek-first rule; embeddings always local (`all-MiniLM`, free) |
| Embeddings | `src/indexing/embeddings.py` | `config["embedding"]["provider"]`: `sentence-transformers` (local) or `openai` |

Keys live in a gitignored `.env` (loaded via `python-dotenv`):

```
DEEPSEEK_API_KEY=sk-...      # generation + RAGAS judge both prefer DeepSeek when set
# OPENAI_API_KEY=sk-...      # fallback for generation + judge
# OPENAI_BASE_URL=...        # optional: any OpenAI-compatible endpoint
```

**Limitation this causes:** the DeepSeek-first rule is duplicated in `llm.py` and
`metrics.py`, and provider choice is implicit (whatever key is in env wins). Adding
a third provider means editing every call site. That is exactly what the proxy is
meant to remove.

## What the planned proxy should centralize (when implemented)

- One `base_url` for the whole app; the proxy routes to the chosen provider.
- Provider/model selection in **one** place (config or a proxy route), not per call site.
- Call sites go back to a plain OpenAI-compatible client pointed at the proxy —
  delete the `DEEPSEEK_API_KEY` branches in `src/llm.py` and `src/evaluation/metrics.py`.

Until then, the env-driven rule above is the source of truth. Nothing in the code
assumes a proxy yet, so introducing one is additive.
