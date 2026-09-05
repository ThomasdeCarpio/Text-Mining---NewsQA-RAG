# Kế Hoạch Thực Nghiệm Chi Tiết - Giai Đoạn 1: Tối Ưu Truy Xuất (Phase 1: Retrieval & Reranking Tournament)

---

## 1. Mục Tiêu của Giai Đoạn 1 (Phase 1 Goal)

### 🎯 Mục tiêu Cốt lõi
Tìm ra **"Golden Retriever"** — Cấu hình tối ưu nhất về **Kích thước đoạn cắt (Chunking)**, **Phương pháp Truy xuất (Retrieval)**, và **Mô hình Tái xếp hạng (Reranker)** để làm nền tảng cho hệ thống RAG.

### ❓ Tại sao Giai đoạn 1 là quan trọng nhất?
* **Nguyên lý "Rác vào thì Rác ra" (Garbage In, Garbage Out)**: Trong hệ thống RAG, mô hình sinh LLM chỉ có thể trả lời đúng nếu được cung cấp chính xác các đoạn văn bản chứa thông tin bằng chứng (*Ground-truth Evidence*).
* Nếu bộ truy xuất lấy sai hoặc đặt bằng chứng ở vị trí quá xa (bị loãng ngữ cảnh), LLM sẽ sinh ra **ảo giác (Hallucination)** hoặc từ chối trả lời.
* **Chi phí 0 VNĐ & Tốc độ cao**: Giai đoạn này là **100% Information Retrieval Offline**, không gọi LLM API sinh văn bản, giúp nhóm thử nghiệm sâu toàn bộ các phương pháp mà không tốn chi phí.

---

## 2. Hệ Thống Metric Đánh Giá (Evaluation Metrics)

Dưới đây là định nghĩa chi tiết tất cả các chỉ số đo lường định lượng trong bài test:

| Tên viết tắt | Tên đầy đủ (Full Name) | Ý nghĩa khái niệm | Tại sao cần dùng trong RAG? |
| :--- | :--- | :--- | :--- |
| **MRR@K** | **Mean Reciprocal Rank at K** | Điểm trung bình của thứ hạng nghịch đảo ($\frac{1}{\text{rank}}$) của **đoạn văn đúng đầu tiên** tìm được trong top $K$. *(Nếu đoạn đúng đứng top 1: được 1.0; đứng top 2: được 0.5; không tìm thấy: được 0)*. | **Metric quan trọng nhất**. Đo khả năng đẩy đoạn chứa câu trả lời lên vị trí đầu tiên để LLM đọc thấy ngay. |
| **Hit@K** (Hit Rate@K) | **Hit Rate at K** | Tỷ lệ phần trăm các câu hỏi tìm được **ít nhất 1 đoạn văn đúng** nằm trong top $K$ kết quả trả về. | Đo lường độ tin cậy của hệ thống — Đảm bảo LLM có tài liệu đúng trong prompt để không trả lời sai. |
| **Recall@K** | **Recall at K** | Tỷ lệ số đoạn văn đúng tìm được trong top $K$ trên **tổng số đoạn văn bằng chứng chuẩn** của câu hỏi. | Đánh giá độ bao phủ — Rất quan trọng với các câu hỏi phức tạp cần tổng hợp từ nhiều đoạn văn khác nhau. |
| **NDCG@K** | **Normalized Discounted Cumulative Gain at K** | Đo lường chất lượng toàn diện của toàn bộ danh sách xếp hạng, phạt nặng nếu các đoạn đúng bị đẩy lùi xuống cuối danh sách. | Đánh giá độ mượt mà và thứ tự ưu tiên của toàn bộ danh sách $K$ tài liệu trả về. |
| **P50 Latency** | **Median Latency (50th Percentile)** | Thời gian phản hồi trung vị (50% số câu hỏi xử lý nhanh hơn mức này), tính bằng mili-giây (ms). | Đo tốc độ phản hồi thông thường của hệ thống trong điều kiện thực tế. |
| **P95 Latency** | **95th Percentile Latency** | Thời gian phản hồi ở nhóm chậm nhất (95% câu hỏi nhanh hơn mức này), tính bằng ms. | Đảm bảo trải nghiệm người dùng, phát hiện các trường hợp truy vấn bị nghẽn hoặc quá tải. |
| **Time Breakdown** | **Per-stage Latency** | Đo thời gian chi tiết từng mắt xích: `Retrieve Time` (Dense/BM25) vs `Rerank Time` (Cross-Encoder). | Phục vụ phân tích điểm nghẽn hiệu năng và vẽ biểu đồ Pareto Frontier. |

---

## 3. Kiến Trúc Giải Đấu 3 Vòng (3-Stage Tournament Design)

Để kiểm tra toàn diện tất cả các mô hình mà không làm bùng nổ số lượng thực nghiệm, bài test được thiết kế thành **3 Vòng phân cấp logic (Tổng cộng đúng 23 bài test)**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ VÒNG 1: SÀNG LỌC MÔ HÌNH ĐƠN LẺ (8 runs - Trên Chunk Chuẩn 512/64)                     │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ Nhánh Dense (4 Embedding Models)       │ Nhánh Sparse (4 Lexical Methods)              │
│ 1. all-MiniLM-L6-v2 (384-d, 22M)       │ 1. BM25Okapi                                  │
│ 2. BAAI/bge-small-en-v1.5 (384-d, 33M) │ 2. BM25+ (BM25Plus)                           │
│ 3. intfloat/e5-base-v2 (768-d, 110M)   │ 3. BM25 + Snowball Stemmer & Stopwords ⭐      │
│ 4. BAAI/bge-large-en-v1.5 (1024-d, 335M│ 4. SPLADE / BGE-M3 Sparse                     │
│ ──► Chọn ra "Best Dense Model"         │ ──► Chọn ra "Best Sparse Model"               │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ VÒNG 2: MA TRẬN ĐỐI XỨNG 3 RETRIEVERS x 3 RERANKERS (9 runs)                           │
├──────────────────────┬────────────────────────┬───────────────────┬────────────────────┤
│ Phương pháp Truy xuất│ No-op (Không Rerank)   │ MiniLM-L-6 (22M)  │ BGE-Large (560M) 🚀│
├──────────────────────┼────────────────────────┼───────────────────┼────────────────────┤
│ 1. Best Dense        │ Run 1                  │ Run 2             │ Run 3              │
│ 2. Best Sparse       │ Run 4                  │ Run 5             │ Run 6              │
│ 3. Hybrid (RRF)      │ Run 7                  │ Run 8             │ Run 9              │
├──────────────────────┴────────────────────────┴───────────────────┴────────────────────┤
│ ──► Đo lường mức cải thiện của Reranker lớn (560M) vs Reranker nhẹ (22M)               │
│ ──► Chọn ra QUÁN QUÂN TUYỆT ĐỐI (GOLDEN RETRIEVAL PIPELINE)                            │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ VÒNG 3: KIỂM CHỨNG TÍNH BỀN VỮNG TRÊN 3 KÍCH THƯỚC CHUNK (6 runs)                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Thử nghiệm Golden Pipeline trên 3 Kích thước (với No-op vs Cross-Encoder):             │
│ • Chunk 256 / Overlap 32 (Đoạn ngắn - 36.973 chunks)                                  │
│ • Chunk 512 / Overlap 64 (Đoạn chuẩn - 19.263 chunks)                                  │
│ • Chunk 1024 / Overlap 128 (Đoạn dài)                                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Chi Tiết Các Vòng Thực Nghiệm

### 4.1. Vòng 1: Sàng Lọc Mô Hình Đơn Lẻ (8 Runs)
* **Môi trường**: Cố định Chunk `512/64 recursive`, Top $K=10$, Top $N=5$, `No-op Reranker`.
* **Mục tiêu**: Tìm ra đại diện mạnh nhất của trường phái Semantic (Dense) và trường phái Từ vựng (Sparse).
  1. **Đấu trường Dense**:
     - `all-MiniLM-L6-v2` (Mốc cơ sở 2021)
     - `BAAI/bge-small-en-v1.5` (MTEB Leader)
     - `intfloat/e5-base-v2` (Phi đối xứng có prefix `query:`/`passage:`)
     - `BAAI/bge-large-en-v1.5` (Biểu diễn ngữ nghĩa sâu)
  2. **Đấu trường Sparse**:
     - `BM25Okapi` (Khớp từ khóa gốc)
     - `BM25+` / `BM25L` (Điều chỉnh phạt độ dài)
     - `BM25 + Snowball Stemmer & Stopwords` (Chuẩn hóa hình thái từ vựng)
     - `SPLADE` / `BGE-M3 Sparse` (Mở rộng từ đồng nghĩa bằng Neural)

---

### 4.2. Vòng 2: Ma Trận Đối Xứng 3 Retrievers $\times$ 3 Rerankers (9 Runs)
* **Môi trường**: Cố định Chunk `512/64`, Initial Top $K=20$, Post-Rerank Top $N=5$.
* **Ma trận 9 tổ hợp**:
  - `Dense` $\times$ `[No-op, MiniLM-L6 (22M), BGE-Large (560M)]`
  - `Sparse` $\times$ `[No-op, MiniLM-L6 (22M), BGE-Large (560M)]`
  - `Hybrid (RRF)` $\times$ `[No-op, MiniLM-L6 (22M), BGE-Large (560M)]`
* **Câu hỏi khoa học**:
  - Reranker tác động mạnh nhất lên phương pháp nào?
  - Tăng tham số Reranker gấp 25 lần (từ 22M lên 560M) giúp tăng thêm bao nhiêu % MRR@5 và phải trả giá bằng bao nhiêu ms độ trễ?
  - $\rightarrow$ **Chọn ra cấu hình có điểm cân bằng Pareto tốt nhất làm Golden Pipeline**.

---

### 4.3. Vòng 3: Kiểm Chứng Kích Thước Đoạn Cắt (6 Runs)
* **Mục tiêu**: Kiểm tra xem Golden Pipeline có giữ vững vị trí số 1 trên các tỷ lệ cắt văn bản khác nhau hay không.
* **Tỷ lệ Overlap chuẩn $12.5\%$**:
  - `Chunk 256` (Overlap 32): Đoạn ngắn, vector tập trung, kiểm tra nguy cơ đứt gãy bằng chứng.
  - `Chunk 512` (Overlap 64): Điểm cân bằng chuẩn công nghiệp.
  - `Chunk 1024` (Overlap 128): Đoạn dài, kiểm tra hiện tượng pha loãng vector.
* Chạy mỗi kích thước với cả `No-op` và `Cross-Encoder` để đo mức độ phục hồi bằng chứng của Reranker.

---

## 5. Tập Dữ Liệu Thực Nghiệm (Evaluation Dataset)

* **Kho bài báo (Corpus)**: `11.064` bài báo CNN (200 bài validation gốc + 10.864 bài train distractor để tạo môi trường tìm kiếm thực tế).
* **Tổng câu hỏi**: `1.340` câu, trong đó `1.336` câu có bản resolved.
* **Tập câu hỏi thực chạy ở Giai đoạn 1**: `281` câu resolved thuộc `50` bài
  development (seed `42`). Giải đấu **không** chạy trên toàn bộ tập câu hỏi;
  partition chia theo bài báo nên 150 bài final-test được giữ nguyên chưa đụng tới.
* **Nguồn lưu trữ canonical**: Hugging Face **public** dataset [`MatchaMacchiato/newsqa_200_11064_v2.0.0`](https://huggingface.co/datasets/MatchaMacchiato/newsqa_200_11064_v2.0.0),
  ghim tại commit `b81c8db6847a23272665946c0c43c72e9a212fd9` (chưa có tag `v2.0.0`).
* **Khác biệt so với `v1.0.0`**: văn bản bài báo đã được phục hồi từ trang CNN
  lưu trữ theo cách *append-only* — `4.603` bài được nối thêm `5.351.580` ký tự,
  nên mọi offset ký tự của evidence giữ nguyên. Điểm số của `v1.0.0` và `v2.0.0`
  **không so sánh trực tiếp được** với nhau.

---

## 6. Kết Quả Kỳ Vọng Sau Giai Đoạn 1

1. **Bảng Xếp Hạng Toàn Diện 23 Cấu Hình**: Bảng số liệu chuẩn khoa học gồm đầy đủ `MRR@5`, `NDCG@5`, `Hit@1`, `Hit@5`, `Recall@5`, `P50 Latency (ms)`.
2. **Bộ Biểu Đồ Khoa Học 300 DPI**:
   - Biểu đồ so sánh các Embedding Models & Sparse Variants.
   - Biểu đồ Dumbbell thể hiện mức tăng trưởng $\Delta\text{MRR@5}$ của Reranker.
   - Biểu đồ Đường biên Pareto (*Pareto Frontier: Accuracy vs Latency*).
   - Biểu đồ phân rã thời gian (*Latency Breakdown: Retrieve vs Rerank*).
3. **Cấu hình Golden Retriever chính thức** để khóa lại và chuyển giao sang Giai đoạn 2 (Generation & SOTA Frameworks).
