# LLM provider switching

## Current status

Generation clients are centralized in `src/model_gateway.py`. The project does
not host another proxy process; it connects to XAH, OpenAI, or another
OpenAI-compatible gateway through `OPENAI_BASE_URL`.

`DEEPSEEK_API_KEY` remains a convenience override introduced by the evaluation
pipeline. When it is set, generation uses DeepSeek directly. Otherwise,
generation uses the shared OpenAI-compatible gateway settings.

| Capability | Configuration |
| --- | --- |
| Chat and RAG generation | `DEEPSEEK_API_KEY`, or `OPENAI_API_KEY` + `OPENAI_BASE_URL` + `CHAT_MODEL` |
| RAGAS judge | DeepSeek when `DEEPSEEK_API_KEY` is set; otherwise OpenAI via `langchain-openai` |
| Remote embeddings | `OPENAI_API_KEY` + `OPENAI_BASE_URL` and the embedding block in `configs/config.yaml` |
| Local embeddings | `sentence-transformers` provider in `configs/config.yaml` |

Example XAH configuration:

```dotenv
OPENAI_API_KEY="sk-..."
OPENAI_BASE_URL="https://api.xah.io/v1"
CHAT_MODEL="deepseek-v4-flash-free"
DEEPSEEK_API_KEY=""
```

Example direct DeepSeek configuration:

```dotenv
DEEPSEEK_API_KEY="sk-..."
```

Keep only the intended provider credential populated. Embeddings continue to
use the embedding provider selected in `configs/config.yaml`; a DeepSeek key
does not redirect embedding requests. See `docs/model_gateway.md` for endpoint
and coding-agent configuration details.
