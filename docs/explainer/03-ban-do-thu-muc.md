# 3. Bản đồ thư mục — file nào ở đâu, để làm gì?

Tra cứu nhanh. Xem quy ước ✅ / 🎭 / ⛔ ở [README.md](README.md) của bộ tài
liệu này.

## `configs/` — Cấu hình

- `config.yaml` — nơi **lẽ ra** khai báo mọi tham số hệ thống (model
  embedding dùng model nào, cắt đoạn dài bao nhiêu, top-k tìm bao nhiêu kết
  quả...). File này **có nội dung** nhưng ⛔ **chưa có code nào đọc nó cả** —
  các phần khác của hệ thống hiện đang tự khai báo config riêng, cứng trong
  code (xem mục "vì sao" ở file 04).
- `setting.py` — ⛔ trống. Lẽ ra là nơi code Python đọc `config.yaml` +
  file `.env` (chứa API key) thành 1 chỗ dùng chung.

## `data/` — Dữ liệu (không đưa lên Git, vì quá nặng/nhạy cảm)

- `raw/` — bài báo gốc dạng HTML, đầu vào của bước ingest.
- `processed/` — bài báo đã dọn sạch (JSON), kết quả trung gian.
- `chroma_db/` — ✅ **database vector thật**, chứa các đoạn văn bản đã được
  đánh index, đây là "kho" mà tính năng tìm kiếm tra cứu vào.

## `scripts/` — Các lệnh chạy tay (CLI)

- `ingest.py` — ✅ **THẬT, dùng lệnh này để nạp dữ liệu**. Chạy toàn bộ Giai
  đoạn A (xem file 02): dọn → cắt → vector hoá → lưu vào ChromaDB.
- `build_chroma_collection.py` — ⛔ Định làm y hệt `ingest.py` nhưng cho phép
  chỉnh tham số qua `configs/config.yaml` thay vì cứng trong code. Hiện tại
  file này mới chỉ có phần mô tả (docstring), chưa có code thật.
- `run_benchmark.py` — ⛔ Định dùng để tự động chấm điểm hệ thống (câu hỏi
  mẫu → so sánh câu trả lời AI với đáp án đúng → tính điểm Ragas). Chưa có
  code thật.
- `query.py` — ⛔ Định làm 1 CLI để hỏi nhanh 1 câu ngay trên terminal (không
  cần mở web). Chưa có code thật.
- `inspect_collection.py` — công cụ phụ để xem/debug ChromaDB đang chứa gì.

## `src/` — Toàn bộ logic xử lý (không phải web)

### `src/ingestion/` ✅ — Dọn & cắt bài báo (đã xong)
- `loader.py`, `cleaner.py`, `chunker.py` — 3 bước dọn/cắt mô tả ở file 02.

### `src/indexing/` — Lưu trữ có thể tìm được
- `embeddings.py` ✅ — đổi văn bản thành vector.
- `chroma_store.py` ✅ — đọc/ghi vào ChromaDB.
- `bm25_index.py` ⛔ — định làm 1 cách tìm kiếm khác, theo *từ khoá khớp
  chính xác* (khác với tìm theo nghĩa). Xem file 04 để hiểu vì sao cần cả 2.

### `src/retrieval/` — Cách tìm đoạn văn bản liên quan tới câu hỏi
- `dense.py` ✅ — cách tìm theo "nghĩa" (dùng vector), đang chạy thật, là
  thứ đứng sau trang Retrieval Playground.
- `hybrid.py` ⛔ — định kết hợp tìm theo nghĩa + tìm theo từ khoá để chính
  xác hơn.
- `reranker.py` ⛔ — định thêm 1 bước "chấm điểm lại" các kết quả tìm được
  cho chính xác hơn nữa trước khi đưa cho AI.

### `src/agents/` ⛔ — "Bộ não" quyết định các bước (chưa có gì)
- `rag_agent.py` — định là nơi: nhận câu hỏi → tự quyết định gọi tìm kiếm →
  đưa kết quả cho AI → trả lời.
- `orchestrator.py` — định là nơi cho phép AI **lặp lại nhiều bước** (tìm
  nhiều lần, so sánh nhiều nguồn) thay vì chỉ 1 bước, dùng framework
  LangGraph.

### `src/tools/` ⛔ — Các "công cụ" mà bộ não ở trên có thể gọi
- `retrieval_tools.py` — bọc `src/retrieval/` thành dạng mà 1 AI agent có
  thể tự quyết định "gọi hay không gọi".
- `ingestion_tools.py` — định làm công cụ để agent tự đi thu thập tin tức
  mới (crawl) khi cần, thay vì chỉ dùng dữ liệu có sẵn.

### `src/evaluation/` ⛔ — Đo điểm chất lượng
- `metrics.py` — định dùng thư viện **Ragas** để tính các điểm số như "câu
  trả lời có trung thực với tài liệu gốc không", "tìm đúng tài liệu không".

### `src/services/` — Lớp "quyết định logic" đứng sau các trang web
- `types.py` — định nghĩa hình dạng dữ liệu dùng chung (VD: 1 tin nhắn chat
  trông như thế nào, 1 "nguồn trích dẫn" gồm những trường gì).
- `session_store.py` — nơi tạm giữ lịch sử chat của từng người dùng (hiện
  giữ trong RAM, mất khi tắt server — đủ dùng cho demo).
- `auth_service.py` 🎭 — kiểm tra đăng nhập, hiện dùng 2 tài khoản viết cứng
  trong code (`admin`/`admin123`, `analyst`/`pass123`), không có database
  người dùng thật.
- `chat_service.py` 🎭 — xử lý khi người dùng gửi câu hỏi trong "News Chat".
  Hiện trả lời giả (xem file 02).
- `eval_service.py` 🎭 — cung cấp số liệu cho trang "Evaluation Desk". Hiện
  toàn bộ là số bịa sẵn.
- `retrieval_service.py` ✅ — xử lý khi người dùng tìm kiếm ở "Retrieval
  Playground". Đây là service **thật duy nhất** trong `src/services/` tính
  đến hiện tại.

### `src/llm.py` ⛔ — Kết nối tới AI (OpenAI)
File trống. Lẽ ra là nơi khởi tạo kết nối tới model AI (GPT) để dùng ở
`rag_agent.py`.

## `api/` — "Tổng đài" HTTP (FastAPI), nối `ui/` với `src/services/`

- `main.py` — khởi động server, đăng ký các nhóm đường dẫn (routers) bên
  dưới, bật CORS (cho phép trình duyệt gọi từ 1 địa chỉ mạng khác).
- `schemas.py` — định nghĩa hình dạng dữ liệu gửi/nhận qua mạng (giống hợp
  đồng: "gọi endpoint này thì phải gửi gì, sẽ nhận lại gì").
- `routers/auth.py` — xử lý đăng nhập.
- `routers/chat.py` — xử lý gửi câu hỏi trong chat (kèm cơ chế gửi từng
  phần câu trả lời dần dần thay vì đợi xong hết mới hiện, gọi là streaming).
- `routers/admin.py` — cấp số liệu cho Evaluation Desk.
- `routers/retrieval.py` ✅ — cấp kết quả tìm kiếm cho Retrieval Playground.

## `ui/` — Giao diện web (React + Vite + TypeScript)

- `src/api/client.ts` — nơi *duy nhất* trong `ui/` biết cách gọi `api/` qua
  mạng. Các trang khác không tự gọi mạng, chỉ gọi hàm trong file này.
- `src/context/AuthContext.tsx` — nhớ ai đang đăng nhập, vai trò gì (lưu
  tạm trong trình duyệt, mất khi bấm "Logout").
- `src/pages/` — mỗi file là 1 trang: `LoginPage`, `ChatPage`,
  `DashboardPage` (Evaluation Desk), `RetrievalPage` (Retrieval Playground).
- `src/components/` — các mảnh giao diện dùng lại nhiều nơi (bong bóng chat,
  thẻ hiển thị 1 kết quả tìm kiếm, thẻ số liệu...).

## `docs/` — Tài liệu

- `database.md`, `ingestion_guide.md`, `indexing_guide.md` — hướng dẫn kỹ
  thuật chi tiết để *code* các phần tương ứng (dành cho người sẽ code, không
  phải để hiểu tổng quan).
- `ui.md` — mô tả đầy đủ yêu cầu giao diện/tính năng mong muốn.
- `explainer/` — chính là 4 file bạn đang đọc.
