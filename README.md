# NewsQA RAG

Hệ thống hỏi đáp trên NewsQA/CNN: tìm kiếm dense/BM25/hybrid, rerank,
sinh câu trả lời có trích dẫn và so sánh các cấu hình bằng experiment.

## Bắt đầu từ đâu?

| Bạn muốn làm gì? | Đọc/chạy ở đâu? |
| --- | --- |
| Chạy web app | Phần **Chạy ứng dụng** bên dưới |
| Test một retriever, reranker hoặc model mới | [Hướng dẫn experiment](docs/experiments.md) |
| Hiểu MRR, NDCG, Recall và failure | [Metrics và kết quả](docs/evaluation.md) |
| Tạo bộ câu hỏi/ground truth mới | [Tạo evaluation dataset](docs/evaluation_dataset.md) |
| Chạy thủ công một cấu hình | [Benchmark CLI](docs/benchmarking.md) |

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
python -m uvicorn newsqa_rag.api.main:app --reload --port 8000

# Terminal 2: UI tại http://localhost:5173
cd frontend
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
**Evaluation Desk**. Xem ví dụ đầy đủ tại [docs/experiments.md](docs/experiments.md).

## Bản đồ repository

| Đường dẫn | Loại | Công dụng |
| --- | --- | --- |
| `backend/newsqa_rag/` | Code | Thư viện Python và FastAPI backend |
| `frontend/` | Code | React/Vite frontend |
| `configs/config.yaml` | Cấu hình | Chunking, embedding, retrieval và LLM mặc định |
| `configs/experiments/` | Cấu hình | Mỗi YAML định nghĩa một experiment có thể chạy |
| `evaluation/` | Metadata | Manifest dataset/index và quyết định review; không phải code |
| `data/` | Dữ liệu local | Raw/processed data, Chroma và BM25; Git bỏ qua |
| `scripts/` | CLI | Các lệnh ingest, build index, benchmark và experiment |
| `outputs/` | Kết quả | Experiment, benchmark, trace và presentation đã sinh |
| `notebooks/` | Tham khảo | Notebook nghiên cứu cũ; không thuộc luồng chạy chính |
| `docs/` | Tài liệu | Hướng dẫn theo từng công việc |
| `tests/` | Kiểm tra | Test offline cho backend/pipeline |
| `outputs/` | Artifact | Slide và tài liệu đã xuất |

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

Đọc thêm: [kiến trúc](docs/architecture.md), [database](docs/database.md),
[model gateway](docs/model_gateway.md), [UI](docs/ui.md), [crawler](docs/crawler.md).
