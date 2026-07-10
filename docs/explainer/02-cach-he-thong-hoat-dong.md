# 2. Hệ thống hoạt động ra sao?

Hãy tưởng tượng cả hệ thống như một **thư viện + bàn tư vấn tra cứu**. Có 2
giai đoạn lớn, tách rời nhau:

- **Giai đoạn A — Chuẩn bị thư viện** (chạy 1 lần, hoặc mỗi khi có bài báo
  mới): đưa bài báo thô vào, dọn sạch, cắt nhỏ, đánh số, xếp lên kệ sao cho
  dễ tìm.
- **Giai đoạn B — Phục vụ người hỏi** (chạy mỗi lần có người dùng gõ câu
  hỏi trên web): tra kệ tìm đoạn liên quan, đưa cho AI đọc, AI viết câu trả
  lời.

Bên dưới là sơ đồ toàn cảnh, đánh dấu rõ phần nào **✅ chạy thật** và phần
nào **🎭 còn giả lập**.

```
GIAI ĐOẠN A — CHUẨN BỊ THƯ VIỆN  (scripts/ingest.py)          ✅ THẬT
═══════════════════════════════════════════════════════════════════
  data/raw/*.html                     (bài báo thô, dạng HTML)
        │
        ▼
  src/ingestion/cleaner.py            Dọn HTML → lấy text sạch +
                                       metadata (tiêu đề, ngày, tác giả...)
        │
        ▼
  data/processed/*.json               (bài báo đã sạch, dạng JSON)
        │
        ▼
  src/ingestion/chunker.py            Cắt mỗi bài báo thành nhiều
                                       "đoạn nhỏ" (chunk), ~500 token/đoạn
        │
        ▼
  src/indexing/embeddings.py          Biến mỗi đoạn thành 1 dãy số
                                       (vector) đại diện cho "ý nghĩa"
        │
        ▼
  src/indexing/chroma_store.py        Lưu (đoạn văn bản + vector + metadata)
        │                             vào ChromaDB — như xếp thẻ mục lục
        ▼                             lên kệ, nhưng xếp theo "ý nghĩa"
  data/chroma_db/  (collection "newsqa_cnn")
```

```
GIAI ĐOẠN B — PHỤC VỤ NGƯỜI HỎI
═══════════════════════════════════════════════════════════════════
  Người dùng mở trình duyệt → gõ câu hỏi
        │
        ▼
  ui/  (React)  ── gọi HTTP/SSE ──▶  api/  (FastAPI)
        │                                  │
        │                                  ▼
        │                     src/services/*_service.py
        │                     (nơi quyết định logic thật sự)
        │                                  │
        ├─── Trang "Retrieval Playground" ─┤
        │    (chỉ tìm, không cần AI)       ▼
        │                     src/retrieval/dense.py            ✅ THẬT
        │                     → biến câu hỏi thành vector,
        │                       tìm trong ChromaDB, trả về
        │                       các đoạn văn bản gần nghĩa nhất
        │
        └─── Trang "News Chat"  ───────────▶
                             src/services/chat_service.py        🎭 MOCK
                             → LẼ RA phải: tìm đoạn liên quan (như trên)
                               rồi nhờ AI (LLM) đọc + viết câu trả lời
                             → HIỆN TẠI: chỉ trả về câu trả lời bịa sẵn
                               + 2 nguồn trích dẫn giả, không hề tìm kiếm
                               hay gọi AI thật

                             src/services/eval_service.py         🎭 MOCK
                             → LẼ RA phải: đọc kết quả benchmark thật
                               (điểm Ragas) để hiển thị ở Evaluation Desk
                             → HIỆN TẠI: trả về vài con số bịa sẵn
```

## Giải thích từng bước Giai đoạn A (chuẩn bị dữ liệu)

Chạy bằng lệnh `python scripts/ingest.py` (xem README mục "Run Ingestion").

1. **Dọn dẹp (`cleaner.py`)**: bài báo thô là file HTML (có thẻ `<div>`,
   quảng cáo, menu web...). Bước này dùng thư viện `newspaper3k` để lọc ra
   *chỉ* phần nội dung bài báo + các thông tin đi kèm (tiêu đề, ngày đăng,
   tác giả, tên báo).
2. **Cắt nhỏ (`chunker.py`)**: một bài báo dài không thể nhét nguyên vào AI
   đọc mỗi lần (tốn kém, chậm). Nên chia mỗi bài thành nhiều đoạn ~500 "token"
   (đơn vị đo độ dài văn bản mà AI dùng, gần bằng 375–400 từ tiếng Anh) — mỗi
   đoạn có thể được tìm/lấy ra riêng lẻ.
3. **Vector hoá (`embeddings.py`)**: máy tính không "hiểu" chữ như người —
   nó cần đổi mỗi đoạn văn bản thành 1 dãy số (vector) sao cho 2 đoạn có
   *nghĩa* gần nhau thì có vector gần nhau trong không gian số. Việc này do
   1 model AI nhỏ tên `all-MiniLM-L6-v2` làm (chạy ngay trên máy, không cần
   gọi API trả phí).
4. **Lưu trữ (`chroma_store.py`)**: tất cả (đoạn văn bản + vector + metadata)
   được lưu vào **ChromaDB** — một loại database chuyên để "tìm theo nghĩa"
   (vector database), khác database thường chỉ tìm khớp chữ.

Sau bước này, ta có 1 "kho" gồm hàng trăm đoạn văn bản đã sẵn sàng để tìm
kiếm theo ý nghĩa câu hỏi.

## Giải thích Giai đoạn B (khi người dùng hỏi)

### Trang "Retrieval Playground" (đã chạy thật)

Đây là tính năng để **test riêng phần tìm kiếm**, không liên quan AI viết
câu trả lời:

1. Bạn gõ 1 câu hỏi trên web.
2. Câu hỏi đó cũng được vector hoá y hệt bước 3 ở trên.
3. ChromaDB so sánh vector câu hỏi với vector của tất cả các đoạn đã lưu,
   trả về những đoạn có vector **gần nhất** (nghĩa là liên quan nhất).
4. Web hiển thị các đoạn đó kèm điểm "distance" (khoảng cách — càng nhỏ
   càng liên quan) và thời gian xử lý (để biết bước nào chậm).

Đây chính là phần **"Retrieval" (R)** trong "RAG" — đã hoạt động thật.

### Trang "News Chat" (còn đang giả lập)

Đây được thiết kế để làm luôn cả 2 chữ còn lại "Augmented Generation" (AG):
sau khi tìm được đoạn liên quan, phải **đưa đoạn đó cho 1 AI (LLM, ví dụ
GPT)** đọc và yêu cầu viết câu trả lời dựa trên đó. Phần này **chưa được xây
dựng** — hiện tại khi bạn gõ câu hỏi, hệ thống **không hề tìm kiếm gì cả**,
chỉ trả về 1 câu trả lời và 2 nguồn trích dẫn đã viết sẵn trong code, luôn
giống nhau bất kể bạn hỏi gì. Xem vì sao và cần làm gì trong
[04-con-thieu-gi-va-tai-sao.md](04-con-thieu-gi-va-tai-sao.md).

### Trang "Evaluation Desk" (còn đang giả lập)

Tương tự — các con số ở đây (VD tỷ lệ trả lời đúng) hiện là số cố định viết
sẵn trong code, **không phải** kết quả đo thật từ hệ thống.

## Vai trò của `api/` và `ui/`

- **`ui/` (React)**: đây là những gì bạn *nhìn thấy và bấm* trên trình duyệt.
  Nó không tự làm gì cả — mọi thao tác (đăng nhập, gửi câu hỏi, tải trang
  đánh giá) đều gửi yêu cầu qua mạng (HTTP) tới `api/`.
- **`api/` (FastAPI)**: đây là "tổng đài" nhận yêu cầu từ `ui/`, rồi gọi đúng
  hàm xử lý tương ứng trong `src/services/`, rồi trả kết quả về cho `ui/`
  hiển thị. Tách `ui/` và `api/` thành 2 chương trình riêng (chạy ở 2 cổng
  mạng khác nhau: `ui` ở cổng 5173, `api` ở cổng 8000) giúp sau này có thể
  thay giao diện web khác (VD app di động) mà không phải viết lại phần xử lý
  logic.
- **`src/services/`**: đây là nơi **quyết định thật sự** phải làm gì khi có
  yêu cầu — gọi tìm kiếm, gọi AI, hay (hiện tại) chỉ trả dữ liệu giả. `api/`
  chỉ là lớp "phiên dịch" giữa web và các hàm này.
