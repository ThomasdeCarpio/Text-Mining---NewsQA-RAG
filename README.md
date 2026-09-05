# NewsQA RAG

Hệ thống hỏi đáp trên NewsQA/CNN: tìm kiếm dense/BM25/hybrid, rerank,
sinh câu trả lời có trích dẫn và so sánh các cấu hình bằng experiment.

## Bắt đầu từ đâu?

| Bạn muốn làm gì? | Đọc/chạy ở đâu? |
| --- | --- |
| Chạy web app | Phần **Chạy ứng dụng** bên dưới |
| Test một retriever, reranker hoặc model mới | [Hướng dẫn experiment](docs/archive/experiments.md) |
| Hiểu MRR, NDCG, Recall và failure | [Metrics và kết quả](docs/archive/evaluation.md) |
| Tạo bộ câu hỏi/ground truth mới | [Tạo evaluation dataset](docs/archive/evaluation_dataset.md) |
| Chia sẻ/tải evaluation dataset private | [Hugging Face evaluation dataset](docs/dev-docs/huggingface_evaluation_dataset.md) |
| Chạy thủ công một cấu hình | [Benchmark CLI](docs/archive/benchmarking.md) |

Điểm dễ nhầm nhất:

- **evaluation dataset** là bộ câu hỏi + đáp án/chunk đúng;
- **experiment** là một phép so sánh các cấu hình trên cùng dataset;
- **run** là một cấu hình cụ thể trong experiment;
- **report** là kết quả đã sinh ra sau khi run.

Test tính năng mới thường chỉ cần tạo **experiment YAML mới**, không cần tạo
dataset mới.

## Cài đặt

Khuyến nghị Python 3.11+ và Node.js 20+.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy `.env.example` thành `.env`. Dense retrieval tải Sentence Transformer ở
lần dùng đầu tiên; BM25 không cần API key.

## Chạy ứng dụng

```bash
# Terminal 1: API tại http://localhost:8000
python -m uvicorn newsqa_app.api.main:app --reload --port 8000

# Terminal 2: UI tại http://localhost:5173
cd app/frontend
npm install
npm run dev
```

Tài khoản demo: `admin` / `admin123`.

## Test nhanh một tính năng mới

```powershell
Copy-Item configs/experiments/newsqa_retrieval_smoke.yaml `
  configs/experiments/my_feature.yaml
```

Trong file mới, đổi `experiment.id`, giữ baseline trong `fixed`, chỉ đặt yếu tố
cần so sánh trong `matrix`. Sau đó:

```bash
python scripts/run_experiment.py configs/experiments/my_feature.yaml --dry-run
python scripts/run_experiment.py configs/experiments/my_feature.yaml
```

Kết quả nằm ở `outputs/experiments/<experiment.id>/` và xuất hiện trong trang
**Evaluation Desk**. Xem ví dụ đầy đủ tại [docs/archive/experiments.md](docs/archive/experiments.md).

## Bản đồ repository

| Đường dẫn | Loại | Công dụng |
| --- | --- | --- |
| `app/backend/newsqa_app/` | Code | FastAPI routers và services (chỉ app dùng) |
| `app/frontend/` | Code | React/Vite frontend |
| `app/assets/` | Asset | File tĩnh của frontend (Vite `publicDir`) |
| `common/newsqa_rag/` | Code | Thư viện RAG dùng chung cho app, scripts và notebook |
| `configs/config.yaml` | Cấu hình | Chunking, embedding, retrieval và LLM mặc định |
| `configs/experiments/` | Cấu hình | Mỗi YAML định nghĩa một experiment có thể chạy |
| `evaluation/` | Metadata | Manifest dataset/index và quyết định review; không phải code |
| `data/` | Dữ liệu local | Raw/processed data, Chroma và BM25; Git bỏ qua |
| `scripts/` | CLI | Các lệnh ingest, build index, benchmark và experiment |
| `outputs/` | Kết quả | Mọi artifact chạy ra: benchmark, experiment, EDA, build frontend, presentation |
| `notebooks/` | Tham khảo | Notebook nghiên cứu cũ; không thuộc luồng chạy chính |
| `docs/` | Tài liệu | Hướng dẫn theo từng công việc |
| `tests/` | Kiểm tra | Test offline cho backend/pipeline |

Nếu thấy `database/`, `models/` hoặc `dev-docs/` ở máy local: chúng không được
Git theo dõi. Database đang dùng được đặt bởi `RAG_DB_PATH` (mặc định
`data/chroma_db`); `database/` chỉ là dữ liệu thử/legacy.

## Luồng evaluation hiện tại

```text
YAML experiment
  -> testset + manifest/index tương ứng
  -> collect retrieval/generation traces (resume được)
  -> tính metric deterministic
  -> RAGAS judge (tùy chọn)
  -> comparison trong outputs/ + dashboard
```

Đọc thêm: [kiến trúc](docs/dev-docs/architecture.md), [database](docs/dev-docs/database.md),
[model gateway](docs/dev-docs/model_gateway.md), [UI](docs/dev-docs/ui.md), [crawler](docs/dev-docs/crawler.md).
