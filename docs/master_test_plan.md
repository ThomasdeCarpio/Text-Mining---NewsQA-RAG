# Kế hoạch Thực nghiệm & Đánh giá RAG (Master Test Plan)

Tài liệu này là bản kế hoạch tổng thể cho toàn bộ các thực nghiệm đánh giá, nghiên cứu bóc tách (ablation study), và định hướng mở rộng kiến trúc sinh nâng cao (Novel/SOTA Generation Frameworks) cho hệ thống **NewsQA RAG**.

---

## 1. Cấu hình Baseline (Chuẩn cơ sở)

Cấu hình Baseline là điểm xuất phát trước Giai đoạn 1, giữ lại làm mốc so
sánh cho mọi cải tiến. Đây **không** còn là cấu hình đang chạy: Giai đoạn 1
đã khóa lại một cấu hình khác (xem mục 1.1).

| Thành phần | Cấu hình Baseline |
| :--- | :--- |
| **Kho ngữ liệu** | 11.064 bài báo CNN (200 bài validation chứa câu hỏi + 10.864 bài train distractor) |
| **Chunking** | Recursive chunking (`chunk_size: 512`, `chunk_overlap: 64`) |
| **Embedding Model** | `all-MiniLM-L6-v2` (384 chiều, khoảng cách cosine) |
| **Vector DB** | ChromaDB (HNSW index: $M=16, ef=50$) |
| **Sparse Search** | Okapi BM25 |
| **Retrieval Stage** | Hybrid Search (Dense 70% + BM25 30%, lấy top $K=10$) |
| **Reranker** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`, lọc ra top $N=5$) |
| **LLM Generator** | `gemini-2.0-flash` (hoặc `gpt-4o-mini`, nhiệt độ $0.0$, max tokens $1024$) |
| **Tập đánh giá** | `newsqa_200_11064` v2.0.0 — 1.340 câu hỏi, 1.336 câu resolved |

### 1.1. Cấu hình đã khóa sau Giai đoạn 1

Kết quả giải đấu 3 vòng (`notebooks/13_phase_1_tournament_report.ipynb`) đã
thay thế phần truy xuất của baseline. Đây là cấu hình `configs/config.yaml`
đang dùng:

| Thành phần | Cấu hình đã khóa | Thay đổi so với Baseline |
| :--- | :--- | :--- |
| **Retriever** | BGE-M3 learned-sparse (`BAAI/bge-m3`) | thay Hybrid 70/30 |
| **Chunking** | Recursive `512/64` | giữ nguyên |
| **Top K** | 20 ứng viên | tăng từ 10 |
| **Reranker** | `BAAI/bge-reranker-large`, top $N=5$ | thay `ms-marco-MiniLM-L-6-v2` |
| **Hybrid** | tắt | Hybrid thua Sparse thuần ở Vòng 2 |

Đo trên 281 câu resolved của 50 bài development: `Hit@5 0,957` · `MRR@5
0,880` · `NDCG@5 0,898` · `P50 513 ms`.

---

## 2. Hệ thống Đánh giá & Chấm điểm (Evaluators & Judges)

Hệ thống đánh giá được chia làm 2 tầng: **Deterministic (Định lượng bằng mã, chi phí 0đ)** và **LLM-as-a-Judge (Chấm ngữ nghĩa bằng RAGAS)**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        HỆ THỐNG METRIC ĐÁNH GIÁ                        │
├──────────────────────────────────┬─────────────────────────────────────┤
│   Metric Deterministic (0đ)      │        LLM Judge (RAGAS)            │
│  • Hit Rate@K, Recall@K          │  • Faithfulness (Độ trung thực/án ngữ)│
│  • MRR@K, NDCG@K                 │  • Answer Relevancy (Độ liên quan)  │
│  • Exact Match (EM), Token F1    │  • Context Precision (Độ chuẩn ngữ cảnh)│
│  • Citation Precision, Recall, F1│  • Context Recall (Độ phủ ngữ cảnh) │
│  • Độ trễ P95 & Chi phí API      │  • Answer Correctness (Độ chính xác)│
└──────────────────────────────────┴─────────────────────────────────────┘
```

### 2.1. Metric Định lượng (Deterministic - Chạy hoàn toàn offline)
1. **Truy xuất (Retrieval Metrics)** (Đo tại mức ban đầu $K$ và sau rerank $N$):
   - **Hit Rate@K**: Tỷ lệ câu hỏi tìm được ít nhất 1 chunk đúng trong top-$K$.
   - **Recall@K**: Tỷ lệ số chunk đúng tìm được trên tổng số chunk ground truth.
   - **MRR@K (Mean Reciprocal Rank)**: Thứ hạng nghịch đảo của chunk đúng đầu tiên (thưởng rất lớn nếu chunk đúng đứng ở vị trí 1).
   - **NDCG@K**: Đánh giá vị trí xuất hiện của tất cả các chunk đúng.
2. **Chất lượng câu trả lời**:
   - **Exact Match (EM)**: Tỷ lệ câu trả lời khớp chính xác tuyệt đối với đáp án chuẩn.
   - **Token F1**: Điểm F1 trùng khớp từ vựng giữa câu trả lời sinh ra và ground truth.
3. **Trích dẫn nguồn (Citation Metrics)**:
   - **Citation Validity**: Tỷ lệ trích dẫn `[1]`, `[2]` tham chiếu đúng chunk có thực.
   - **Citation Precision & Recall**: Đo xem chunk được trích dẫn có đúng là chunk chứa evidence hay không.
4. **Hiệu năng hệ thống**:
   - **Độ trễ (P50, P90, P95 tính bằng ms)**: Thời gian truy xuất, rerank, và sinh câu trả lời.
   - **Ước tính chi phí**: Số lượng token tiêu thụ và chi phí USD/1.000 lượt hỏi.

### 2.2. Metric Chấm bằng LLM (RAGAS Judge)
Sử dụng script [`scripts/judge_benchmark_predictions.py`](file:///Users/thomas200905/Documents/Thomas/HCMUS/Third%20Year/Semester%209/Text%20Mining/project/Text-Mining---NewsQA-RAG/scripts/judge_benchmark_predictions.py):
- **Faithfulness**: Đo mức độ trung thực của câu trả lời (phát hiện ảo giác - hallucination).
- **Answer Relevancy**: Đánh giá câu trả lời có đi đúng trọng tâm câu hỏi hay không.
- **Context Precision**: Đo xem các chunk xếp hạng đầu có thực sự chứa thông tin cần thiết không.
- **Context Recall**: Đo xem ngữ cảnh truy xuất có đủ thông tin để trả lời toàn diện câu hỏi không.

> [!TIP]
> **Quy tắc Judge độc lập**: LLM đóng vai trò Judge phải mạnh hơn hoặc khác với LLM Generator (Ví dụ dùng `gemini-1.5-pro` làm Judge để chấm điểm cho `gemini-2.0-flash` hoặc `gpt-4o-mini`).

---

---

## 3. Chiến lược Đánh giá 2 Giai đoạn & Tối ưu Chi phí API

### 3.1. Quy trình Thực nghiệm 2 Giai đoạn (Two-Phase Protocol)
Để đảm bảo tính khoa học cao nhất và tiết kiệm tài nguyên:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Giai đoạn 1: Sàng lọc & Tối ưu Truy xuất (Giải Đấu 3 Vòng - 23 runs)    │
│ • 100% Offline IR (Chi phí 0đ) trên 281 câu development                │
│ • Vòng 1: Sàng lọc 4 Dense Models + 4 Sparse Models (8 runs)           │
│ • Vòng 2: Ma trận Đối xứng 3 Retrievers x 3 Rerankers (9 runs)         │
│   (No-op vs MiniLM-L6 vs BGE-Reranker-Large)                           │
│ • Vòng 3: Kiểm chứng Golden Pipeline trên 3 Kích thước Chunk (6 runs)  │
│ ──► Khóa "Golden Retriever" Tuyệt Đối (MRR@5 cao nhất)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼ (Cố định Golden Retriever)
┌────────────────────────────────────────────────────────────────────────┐
│ Giai đoạn 2: Đánh giá End-to-End & Thử nghiệm Sinh nâng cao            │
│ • Cố định Golden Retriever từ Giai đoạn 1                              │
│ • So sánh Direct LLM vs Baseline RAG vs Novel Frameworks (Self-RAG)    │
│ • Đánh giá RAGAS Judge & Biểu đồ Pareto Frontier                       │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Tận dụng Gemini Free Tier để làm End-to-End hoàn toàn miễn phí (0đ)
- **Google AI Studio (Gemini 3.1 Flash-Lite / 2.0 Flash / 1.5 Flash)**:
  - Cung cấp gói **Free** cực kỳ mạnh: **15 RPM**, **1.000.000 TPM**, và **500 – 1.500 requests/ngày**.
  - Tốc độ cực nhanh (~0.8s) và **hoàn toàn miễn phí 0đ**.
- Hệ thống `model_gateway.py` trong project đã tích hợp sẵn và chỉ cần điền `GEMINI_API_KEY` hoặc `OPENAI_API_KEY` trong file `.env`.

---

## 4. Chi tiết Kế hoạch Thực nghiệm Giai đoạn 1 (Master Retrieval Tournament)

### Vòng 1: Sàng lọc Mô hình Đơn lẻ (Model Screening - 8 runs)
Chạy trên cấu hình Chunk chuẩn `512/64 recursive`:
1. **Nhánh Dense (4 Embedding Models)**:
   - `all-MiniLM-L6-v2` (384-d, 22M params - Baseline)
   - `BAAI/bge-small-en-v1.5` (384-d, 33M params - MTEB Leader)
   - `intfloat/e5-base-v2` (768-d, 110M params - Asymmetric)
   - `BAAI/bge-large-en-v1.5` (1024-d, 335M params - Deep representation)
   - $\rightarrow$ **Chọn ra `Best Dense Model`**.
2. **Nhánh Sparse (4 Lexical/Sparse Methods)**:
   - `BM25Okapi` (Standard whitespace matching)
   - `BM25+` / `BM25L` (Document length normalization adjustment)
   - `BM25 + Snowball Stemmer & Stopwords` (Morphological normalization)
   - `SPLADE` / `BGE-M3 Sparse` (Learned sparse neural term expansion)
   - $\rightarrow$ **Chọn ra `Best Sparse Model`**.

---

### Vòng 2: Ma Trận Đối Xứng 3 Retrievers $\times$ 3 Rerankers (9 runs)
So sánh tương tác giữa 3 phương pháp truy xuất và 3 cấp độ Reranker trên Chunk `512/64`:

$$\begin{array}{|l|c|c|c|}
\hline
\textbf{Phương pháp Truy xuất} & \textbf{No-op (Không Rerank)} & \textbf{MiniLM-L-6 (22M)} & \textbf{BGE-Large (560M)} \\
\hline
\textbf{Best Dense} & \text{Run 1} & \text{Run 2} & \text{Run 3} \\
\textbf{Best Sparse} & \text{Run 4} & \text{Run 5} & \text{Run 6} \\
\textbf{Hybrid (RRF Fusion)} & \text{Run 7} & \text{Run 8} & \text{Run 9} \\
\hline
\end{array}$$

* **Câu hỏi nghiên cứu**:
  1. *Reranker mô hình lớn (BGE-Large 560M) cải thiện bao nhiêu % so với model nhẹ (MiniLM 22M)?*
  2. *Sự đánh đổi về độ trễ (Latency trade-off) có xứng đáng trong thực tế không?*
* $\rightarrow$ **Xác định QUÁN QUÂN TUYỆT ĐỐI (Golden Retrieval Pipeline)**.

---

### Vòng 3: Kiểm Chứng Tính Bền Vững Trên Kích Thước Chunk (6 runs)
Thử nghiệm Quán quân trên 3 kích thước đoạn cắt với tỷ lệ overlap chuẩn $12.5\%$:
1. `chunk_256` (overlap 32) với `No-op` và `Cross-Encoder`
2. `chunk_512` (overlap 64) với `No-op` và `Cross-Encoder`
3. `chunk_1024` (overlap 128) với `No-op` và `Cross-Encoder`

---

## 5. Chi tiết Kế hoạch Thực nghiệm Giai đoạn 2 (Generation & Novel SOTA)

---

### Phase 2A: Khóa baseline end-to-end

Trước các ablation bên dưới, chạy đúng một baseline RAG trên tập `resolved`
development: retrieval BGE-M3, BGE-large reranker, top 5 context,
`gemini-3.1-flash-lite` generator với reasoning `minimal` và GLM-5.3-Flash
RAGAS judge với reasoning `low`. Bước này
không so sánh direct LLM; direct LLM và các model khác là thí nghiệm tiếp theo
sau khi baseline đã được ghi nhận. Xem
[`phase_2_baseline_test_plan.md`](Detailed%20Test%20Plans/phase_2_baseline_test_plan.md)
cho protocol đầy đủ và
[`phase_2_execution_guide.md`](Detailed%20Test%20Plans/phase_2_execution_guide.md)
cho hướng dẫn chạy ngắn.

---

### Thí nghiệm 4: Độ sâu Ngữ cảnh cung cấp cho LLM (Context Depth Ablation)
* **Câu hỏi nghiên cứu**: Đưa bao nhiêu chunk vào prompt ($N=1, 3, 5, 8, 10$) sẽ cho kết quả trả lời tốt nhất mà không gây nhiễu (distraction) cho LLM?
* **Biến kiểm soát**: Cấu hình Retrieval + Reranker tốt nhất, Generator `gemini-3.1-flash-lite` với reasoning `minimal`.
* **Biến độc lập**: Số lượng chunk cung cấp: $N \in \{1, 3, 5, 8, 10\}$.
* **Metric đo lường**: Answer Exact Match (EM), Token F1, Citation Precision/Recall, Số token tiêu thụ, Độ trễ sinh câu trả lời.
* **Lệnh chạy**:
  ```bash
  python scripts/run_experiment.py configs/experiments/ablation_4_context_depth.yaml
  ```

---

### Thí nghiệm 5: So sánh Mô hình Sinh LLM (Generator Model Comparison)
* **Câu hỏi nghiên cứu**: Mô hình LLM nào tuân thủ trích dẫn tốt nhất và ít sinh ảo giác nhất khi được cung cấp cùng ngữ cảnh?
* **Biến kiểm soát**: Cùng tập ngữ cảnh top 5, Nhiệt độ $= 0.0$.
* **Biến độc lập**:
  1. `gemini-2.0-flash` (Tốc độ cao, miễn phí qua Google AI Studio)
  2. `gpt-4o-mini` (OpenAI)
  3. `deepseek-chat` (DeepSeek V3)
  4. Direct LLM Fallback (Hỏi trực tiếp LLM không có RAG - làm mốc chứng minh giá trị của RAG)
* **Metric đo lường**: Token F1, RAGAS Faithfulness, RAGAS Answer Relevancy, Citation Validity, Chi phí/1.000 câu.
* **Lệnh chạy**:
  ```bash
  python scripts/run_experiment.py configs/experiments/ablation_5_generator.yaml
  ```

---

### Thí nghiệm 6: Tác động của Độ mơ hồ của Câu hỏi (Query Ambiguity Study)
* **Câu hỏi nghiên cứu**: Câu hỏi mơ hồ trong thực tế (thiếu tiêu đề bài báo) làm suy giảm chất lượng RAG ra sao, và việc làm rõ câu hỏi (Disambiguation) giúp ích thế nào?
* **Biến kiểm soát**: Cùng hệ thống RAG baseline.
* **Biến độc lập**:
  1. `testset_reviewed_original.jsonl`: Câu hỏi gốc NewsQA (ngắn gọn, có thể mơ hồ).
  2. `testset_clarified.jsonl`: Câu hỏi đã được làm rõ thông tin thực thể độc lập.
* **Metric đo lường**: Chênh lệch $\Delta$ Hit Rate@5, MRR@5, Answer Exact Match, Phân loại nguyên nhân lỗi (Failure Analysis).
* **Lệnh chạy**:
  ```bash
  python scripts/run_experiment.py configs/experiments/ablation_6_ambiguity.yaml
  ```

---

### Thí nghiệm 7: Đánh giá Tổng thể & Đường biên Pareto (Pareto Frontier)
* **Mục tiêu**: Lập biểu đồ **Pareto Frontier** cho bài báo cáo cuối kỳ để minh họa sự đánh đổi giữa **Chất lượng (MRR / Answer F1)**, **Độ trễ (ms)**, và **Chi phí ($/1.000 câu)**.

```text
▲ Chất lượng (MRR@5 / Answer F1)
│               ● [Cấu hình C: Hybrid + BGE Reranker + Gemini 2.0 Flash] (Tối ưu nhất)
│         ● [Cấu hình B: Dense + MiniLM Reranker]
│   ● [Cấu hình A: BM25 No-op]
│
└────────────────────────────────────────────────────────► Độ trễ (ms) / Chi phí ($)
```

---

## 5. Mở rộng Kiến trúc Sinh Nâng cao (Novel / SOTA Frameworks for Phase 2)

Sau khi Giai đoạn 1 xác định được **"Golden Retriever"** (Bộ truy xuất tối ưu nhất), nhóm có thể triển khai thêm các kiến trúc sinh tiên tiến từ các bài báo khoa học hàng đầu:

| Framework / Paper | Nguyên lý hoạt động | Cách triển khai | Giá trị học thuật |
| :--- | :--- | :--- | :--- |
| **Self-RAG** *(Asai et al., ICLR 2024)* | LLM tự tạo các **Reflection Tokens** (tự đánh giá xem context có liên quan không, câu trả lời có được context hỗ trợ hay không). | Prompting / Multi-step evaluation | Giải quyết triệt để ảo giác (Hallucination), tăng Faithfulness. |
| **Corrective RAG (CRAG)** *(Yan et al., 2024)* | Thêm bộ đánh giá ngữ cảnh (**Context Evaluator**). Nếu chunk truy xuất bị nhiễu, hệ thống lọc bớt trước khi đưa vào LLM. | Python filter rule / Lightweight LLM check | Tăng Context Precision, giảm sai lệch thông tin. |
| **Attributed CoT (Attributed Chain-of-Thought)** | Yêu cầu LLM suy luận từng bước (Step-by-step reasoning), mỗi bước suy luận bắt buộc phải gán trích dẫn `[1]`, `[2]` trước khi kết luận. | Prompt Engineering có cấu trúc | Tăng Citation Precision và Exact Match đáng kể. |
| **Adaptive / Self-Consistency RAG** | Lấy mẫu nhiều câu trả lời với nhiệt độ khác nhau và tổng hợp kết quả (Majority Voting). | Sampling + Voting mechanism | Tăng độ ổn định cho các câu hỏi phức tạp. |

---

## 6. Cấu trúc Báo cáo & Thuyết trình Dự án Cuối kỳ

Bố cục 3 phần chuẩn nghiên cứu khoa học cho bài báo cáo cuối môn:

```text
Phần 1: Nghiên cứu Truy xuất & Chỉ mục (Retrieval & Indexing Ablation)
  └── Trình bày so sánh BM25 vs Dense vs Hybrid, Kích thước Chunk, và Reranker (Chọn ra Golden Retriever).

Phần 2: Đánh giá Kiến trúc Sinh & So sánh SOTA (Generation & Frameworks)
  └── Cố định Golden Retriever:
        • Direct LLM (Mốc chứng minh giá trị của RAG)
        • Baseline RAG (Standard Prompting)
        • Novel Framework (Self-RAG / CRAG / Attributed CoT do nhóm tự triển khai)
        • Đánh giá RAGAS (Faithfulness, Relevancy, Precision, Recall).

Phần 3: Phân tích Lỗi & Đánh giá Pareto (Failure Analysis & Trade-offs)
  └── Phân tích các ca câu hỏi bị sai (do retrieval hay do LLM), vẽ biểu đồ Pareto Frontier (Chất lượng vs Độ trễ vs Chi phí).
```

---

## 7. Tổng kết Danh sách File YAML Thực nghiệm

| File Cấu hình | Nội dung Thí nghiệm | Chi phí LLM |
| :--- | :--- | :---: |
| `configs/experiments/ablation_1_chunking.yaml` | Chunk size (256, 512, 1024), Strategy | 0đ (Offline) |
| `configs/experiments/ablation_2_retrieval.yaml` | Sparse vs Dense vs Hybrid RRF | 0đ (Offline) |
| `configs/experiments/ablation_3_reranking.yaml` | No-op vs MiniLM vs BGE Reranker | 0đ (Offline) |
| `configs/experiments/ablation_4_context_depth.yaml` | Độ sâu ngữ cảnh $N \in \{1, 3, 5, 8, 10\}$ | Thấp |
| `configs/experiments/ablation_5_generator.yaml` | Gemini 2.0 vs GPT-4o-mini vs No-RAG | 0đ (Free Tier) |
| `configs/experiments/ablation_6_ambiguity.yaml` | Câu hỏi gốc vs Câu hỏi làm rõ | 0đ (Offline) |
| `configs/experiments/ablation_final_benchmark.yaml` | Tổng hợp End-to-End Pareto Benchmark | 0đ (Free Tier) |
