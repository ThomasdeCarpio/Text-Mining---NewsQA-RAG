# Model gateway

Generation and remote embeddings use an OpenAI-compatible client configured in
`.env`:

```dotenv
OPENAI_API_KEY="..."
OPENAI_BASE_URL="https://api.xah.io/v1"
CHAT_MODE="auto"
CHAT_MODEL="gpt-5.4"
```

`CHAT_MODE=direct` skips retrieval. `auto` uses local RAG when available and
otherwise falls back to direct chat. `rag` requests RAG but still falls back if
the local index cannot initialize.

Benchmark generation selects its provider from the requested model name:

- `gemini-*` uses `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) through Google's
  OpenAI-compatible endpoint;
- `deepseek-*` uses `DEEPSEEK_API_KEY` through the direct DeepSeek endpoint;
- other model names use `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.

An unrelated provider key never overrides the requested model. Credentials are
read only from the environment and are not serialized into configs, Chroma
metadata, or benchmark reports.

Phase 2 verifies the locked Gemini models with:

```bash
python scripts/verify_gemini_models.py
```

To use remote embeddings, update `embedding.provider`, `model_name`, and exact
`dimensions` in `configs/config.yaml`, then rebuild the collection.

Validate configured endpoints without printing prompts, responses, or secrets:

```bash
python scripts/check_model_gateway.py --chat-model MODEL
```

Use `--help` for optional Responses, Messages, embedding, streaming, Gemini,
and Ollama checks.
