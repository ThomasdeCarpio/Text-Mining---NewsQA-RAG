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

`DEEPSEEK_API_KEY` switches benchmark generation to the direct DeepSeek client.
Credentials are read only from the environment and are not serialized into
configs, Chroma metadata, or benchmark reports.

To use remote embeddings, update `embedding.provider`, `model_name`, and exact
`dimensions` in `configs/config.yaml`, then rebuild the collection.

Validate configured endpoints without printing prompts, responses, or secrets:

```bash
python scripts/check_model_gateway.py --chat-model MODEL
```

Use `--help` for optional Responses, Messages, embedding, streaming, Gemini,
and Ollama checks.
