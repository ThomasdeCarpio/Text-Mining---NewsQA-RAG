# Báo cáo Chi tiết Phase 1 + Phase 2 — Hệ thống NewsQA RAG

> **Dự án:** NewsQA RAG — Text Mining (HK3/Năm 3)
> **Kho ngữ liệu:** `MatchaMacchiato/newsqa_200_11064_v2.0.0` — 11.064 bài báo, **22.766 chunks** (bản đã phục hồi phần đuôi bị cắt)
> **Tập đánh giá:** 1.152 câu hỏi `resolved` đã khử trùng lặp ngữ nghĩa và có người duyệt, chia theo bài báo
> **Bản tóm tắt (đọc trước):** [report.md](report.md)  
> **Báo cáo chi tiết từng phase:** [Phase 1 — Retrieval Tournament](phase1/report.md) · [Phase 1 (bản tiếng Anh, gọn)](../phase1_results.md) · [EDA](../eda/eda_report.md)
> **Nguồn số liệu:** `docs/reports/phase1/{round1,round2,round3}.csv`, `winner_lock.jsonl`, `paired_significance.json` và artifact Phase 2 trên Kaggle/Colab

---

## 0. Trạng thái một trang

| Giai đoạn | Nội dung | Trạng thái | Kết quả chốt |
| :--- | :--- | :--- | :--- |
| **EDA** | Khảo sát dữ liệu, phát hiện truncation & nhiễu nhãn | ✅ Xong | Ngưỡng nhiễu **7,0%–24,5%**; phục hồi 4.603 bài |
| **Phase 1** | Giải đấu 23 cấu hình truy xuất, 3 vòng | ✅ **Đã khóa** | BGE-M3 sparse → bge-reranker-large → top 5, chunk 512/64 |
| **Phase 2A** | Baseline end-to-end trên 281 câu development | ✅ Xong | Answer Correctness **0,6308** |
| **Phase 2B.1** | Sàng lọc 4 prompt (P0–P3) | ✅ Xong | **P2** là prompt duy nhất qua guardrail |
| **Phase 2B.2** | Sàng lọc độ sâu context, 2 prompt × 3 depth | ✅ Xong | Hai finalist: P2-depth3, P2-depth5 |
| **Phase 2B.3** | Xác nhận finalist trên đủ 281 câu | ✅ Xong | **Winner: P2-depth5** (§5.5) |
| **Phase 2B.4** | Held-out 284 câu, chạy đúng một lần | ⛔ Chưa chạy | *chưa được phép chạm vào* |
| **Phase 3** | Abstention — 200 case, 3 cấu hình B0/B1/B2 | ⛔ Bị chặn | chờ duyệt tay dataset |

> [!IMPORTANT]
> **Winner Phase 2B là P2-depth5.** Nó là **cấu hình hợp lệ duy nhất**: qua đủ 4 guardrail đã đăng ký, Answer Correctness **0,7350** (+0,1043 so với P0, CI95 [+0,0844; +0,1240]).
>
> **P2-depth3 đạt điểm cao hơn (0,7711) nhưng bị loại** vì trượt 2 guardrail — và kiểm định theo cặp cho thấy cả hai mức tụt đều **có ý nghĩa thống kê**, không phải nhiễu đo. Đây là trường hợp bộ quy tắc đăng ký trước làm đúng việc của nó: chặn một cấu hình đổi tính trung thực lấy điểm số.
>
> Còn thiếu để đóng Phase 2: chạy held-out, audit mù 30 cặp, và kiểm độ ổn định API (§8).

---

## 1. Pipeline end-to-end hiện tại

```mermaid
flowchart LR
    Q["Câu hỏi resolved"] --> R["BGE-M3 learned sparse<br/>top_k = 20"]
    R --> RR["bge-reranker-large<br/>top_n = 5"]
    RR --> D{"Cắt context<br/>1 / 3 / 5"}
    D --> G["Gemini 3.1 Flash-Lite<br/>temperature 0"]
    G --> A["Câu trả lời + citation"]
    A --> J["Deterministic metrics<br/>+ RAGAS / GLM judge"]
```

| Thành phần | Cấu hình | Nguồn quyết định | Còn được đổi không? |
| :--- | :--- | :--- | :--- |
| Chunking | Recursive, 512 / overlap 64 | Phase 1 vòng 3 | **Không** — đã khóa |
| Retriever | BGE-M3 learned sparse, top 20 | Phase 1 vòng 1 | **Không** — đã khóa |
| Reranker | `BAAI/bge-reranker-large`, top 5 | Phase 1 vòng 2 | **Không** — đã khóa |
| Prompt | P0 (control) / **P2** (ứng viên) | Phase 2B.1 | Đang tinh chỉnh |
| Context depth | 1 / 3 / **5** | Phase 2B.2–2B.3 | Đang tinh chỉnh |
| Generator | `gemini-3.1-flash-lite`, temp 0, minimal reasoning | — | Không tinh chỉnh trọng số |
| Judge | `glm-5p3-flash` (Fireworks), low reasoning | EDA §9 | **Bắt buộc tách khỏi generator** |
| Seed | 42 | — | — |

> [!NOTE]
> **Phase 2 không huấn luyện lại gì cả.** Nó chỉ đổi hai biến: nội dung prompt và số context đưa vào. Retrieval + rerank chạy **đúng một lần**, mọi thí nghiệm depth chỉ cắt ngắn danh sách đã rerank sẵn. Nhờ vậy mọi khác biệt đo được quy hết cho generation, không lẫn dao động truy xuất.

---

## 2. Hai bộ quy tắc phán quyết, và vì sao chúng phải khác nhau

Đây là chỗ dễ đọc nhầm nhất khi ghép hai phase, nên nói rõ ngay.

| | **Phase 1 (truy xuất)** | **Phase 2 (sinh)** |
| :--- | :--- | :--- |
| Metric chính | nDCG@5 | RAGAS Answer Correctness |
| Cách phán quyết | CI95 của **hiệu theo từng câu** không chứa 0 | Qua đủ **4 guardrail**, rồi mới so metric chính |
| Ràng buộc thực tiễn | Chênh < 0,070 = nằm trong nhiễu nhãn | Coverage ≥ 95%; Faithfulness không giảm quá 0,02; Citation F1 và Citation Validity không giảm quá 0,01 |
| Chi phí một lần chạy | GPU, không tốn tiền API | Tốn tiền API, có rủi ro sinh sai |
| Vì sao khác | Sai một chút chỉ là xếp hạng kém | Sai một chút là **hệ thống nói điều không có trong bằng chứng** |

Phase 1 hỏi *"cái nào tốt hơn?"*. Phase 2 hỏi *"cái nào tốt hơn **mà không đánh đổi tính trung thực**?"* — nên nó cần guardrail chứ không chỉ cần một điểm số cao hơn.

**Điểm chung của cả hai:** so sánh luôn **theo cặp trên cùng bộ câu hỏi**, bootstrap gom cụm theo bài báo. Lý do đã trình bày kỹ ở [Phase 1 §1.2](phase1/report.md): độ khó của từng câu hỏi là nguồn nhiễu **dùng chung**, và nó tự triệt tiêu khi lấy hiệu trước rồi mới tổng hợp. Khoảng CI biên (tính riêng từng cấu hình) quá rộng, không bao giờ phát hiện được chênh lệch thật cỡ 0,02.

> [!WARNING]
> Ngưỡng nhiễu 7,0% và guardrail Phase 2 là **hai loại ràng buộc khác nhau**, đừng gộp. 7,0% nói về *ý nghĩa thực tiễn* (chênh nhỏ hơn thế thì không đáng tin, vì bộ chấm có sai sót hệ thống). Guardrail nói về *điều không được phép đánh đổi*. Một cấu hình có thể vượt ngưỡng nhiễu mà vẫn bị loại vì guardrail — và đó chính là chuyện đang xảy ra với P2-depth3.

---

## 3. Dữ liệu: một tập, ba lần chia

| Phần | Số bài | Số câu | Dùng để làm gì | Đã chạm chưa |
| :--- | ---: | ---: | :--- | :--- |
| Development | 50 | **281** | Baseline, sàng lọc, chọn winner | Đã chạm nhiều lần |
| Held-out final | 50 | 284 | Đánh giá winner **đúng một lần** | Chưa |
| Held-out reserve | 100 | 587 | Nghiên cứu mở rộng | Chưa |
| **Tổng** | **200** | **1.152** | | |

- Chia theo `article_key`, **không** chia ngẫu nhiên từng câu — nên không có câu hỏi nào của cùng một bài báo xuất hiện ở cả hai bên.
- Tập `resolved` là tập báo cáo chính; `original` dùng làm **cận dưới** (nhiều câu cụt, mơ hồ).
- 1.336 → 1.152 câu sau khi khử trùng lặp ngữ nghĩa (155 cụm gồm 339 câu, giữ 1 đại diện mỗi cụm, có người duyệt).
- Full set 1.336 câu chỉ dành cho sensitivity analysis, **không** dùng chọn winner.

> [!IMPORTANT]
> Mọi con số trong báo cáo này, cả Phase 1 lẫn Phase 2, đều đo trên **cùng 281 câu development**. Đó là điều khiến các so sánh xuyên phase ở §6 hợp lệ — nhưng cũng là lý do phải nhớ: **281 câu này là tập tinh chỉnh**, số trên nó lệch về phía lạc quan. Con số dùng để công bố là con số held-out, và nó chưa tồn tại.

---

## 4. Phase 1 — kết luận đã khóa

Chi tiết đầy đủ: [phase1/report.md](phase1/report.md). Phần này chỉ giữ những gì Phase 2 cần.

### 4.1. Đã thử gì

23 cấu hình, 3 vòng phân tầng thay vì quét đủ 72 tổ hợp:

| Vòng | Không gian | Người thắng |
| :--- | :--- | :--- |
| 1 | 4 dense × 4 sparse (không rerank) | **BGE-M3 sparse** (nDCG@5 = 0,8317) |
| 2 | 3 retriever × 3 reranker | **BGE-M3 + bge-reranker-large** (0,8976) |
| 3 | 3 chunk size × 2 reranker | **512/64** — nhưng là *kết quả null*, ba kích thước không tách được nhau |

### 4.2. Cấu hình đã khóa và hiệu năng của nó

```yaml
retrieval:
  retriever: sparse
  sparse: { method: bge-m3, model: BAAI/bge-m3 }
  top_k: 20
  reranker: { enabled: true, model: BAAI/bge-reranker-large, top_n: 5 }
chunking: { strategy: recursive, chunk_size: 512, chunk_overlap: 64 }
```

| Metric | `resolved` (thực tế) | `original` (cận dưới) |
| :--- | ---: | ---: |
| Hit@1 | **0,8221** | 0,4093 |
| Hit@3 | **0,9324** | — |
| Hit@5 | **0,9573** | 0,5445 |
| nDCG@5 | **0,8976** | 0,4824 |
| MRR@5 | **0,8797** | 0,4651 |
| Recall@5 | **0,9555** | 0,5409 |
| Latency tổng (P50) | 510,1 ms | 505,7 ms |

### 4.3. Ba điều Phase 1 chứng minh được, và một điều chưa

**Chứng minh được** (CI95 của hiệu theo cặp không chứa 0):

1. **Sparse áp đảo dense**, +0,1634 nDCG@5 trên `resolved`, +0,1981 trên `original`. Không phải chênh lệch nhỏ, không phải nhiễu.
2. **Reranker đáng giá thật**: +0,0659 nDCG@5, +0,0961 nDCG@1. Đổi lấy ~445 ms.
3. **BGE-M3 hơn BM25-stemmed trên tập `original`** (+0,0680, CI [+0,0311; +0,1056]) nhưng **không** tách được trên `resolved` (+0,0194). Sự bất đối xứng này khớp đúng cơ chế EDA đã dự đoán — xem §6, luồng 1.

**Chưa chứng minh được, và đã ghi nhận trung thực:**

- **Dense chưa tái lập được.** Cả 4 mô hình sparse chạy lại cho sai lệch 0,0000; cả 4 mô hình dense đều trôi, cao nhất 0,0143 — **lớn hơn khoảng cách giữa các mô hình dense** (e5-base hơn bge-small chỉ 0,0094). Trên `original`, ba vị trí đầu hoán đổi hết. Nguyên nhân: `chroma_store.py:33` dựng HNSW không seed, `embeddings.py:148` nhúng theo batch không cố định, và repo không đặt `torch.manual_seed` ở đâu cả.
- Hệ quả: **"best dense" là một lựa chọn không phân định được**, và bản thân điều đó là một phát hiện đáng báo cáo. Nó không ảnh hưởng cấu hình khóa, vì sparse thắng dense ở khoảng cách gấp hơn 10 lần biên độ trôi.
- **Cách sửa đã có:** chỉ mục Phase 1 nay đã được đóng gói kèm SHA-256 (`scripts/package_index_artifacts.py`). Nạp lại đúng chỉ mục đó thay vì dựng lại sẽ loại bỏ nguồn trôi phía kho ngữ liệu — xem §8, việc số 5.

---

## 5. Phase 2 — trạng thái hiện tại

### 5.1. Phase 2A — Baseline (P0, depth 5, 281/281 câu)

| Nhóm | Metric | Giá trị |
| :--- | :--- | ---: |
| Độ phủ | Coverage | 281/281 (100%) |
| Truy xuất | Hit@5 / Recall@5 / nDCG@5 / MRR@5 | 0,9573 / 0,9555 / 0,8976 / 0,8797 |
| Đáp án | **Answer Correctness** | **0,6308** |
| | Exact Match / Token F1 | 0,0000 / 0,2648 |
| | Answer Relevancy | 0,7950 |
| Tính trung thực | Faithfulness | 0,9761 |
| Context | Precision / Recall | 0,8980 / 0,9644 |
| Trích dẫn | Citation F1 / Validity | 0,8025 / 0,9893 |

Latency end-to-end: trung vị 2,77 s nhưng trung bình 13,04 s và P95 53,63 s — đuôi dài đến từ API của generator, không phải từ pipeline truy xuất (vốn chỉ 510 ms). Judge tiêu 3.372 request, ~4,68 triệu token, ~0,79 USD.

**Đọc baseline này cho đúng:** retrieval và grounding đã mạnh. Vấn đề còn lại không phải hallucination, mà là **chọn đúng thông tin để trả lời**. Model thường chứa đáp án đúng nhưng trả lời quá dài, trộn chi tiết từ các bài cùng chủ đề, hoặc trả sai loại thông tin được hỏi. Faithfulness 0,9761 không cứu được điều đó: model có thể diễn đạt trung thực một distractor mà vẫn không trả lời đúng câu hỏi.

### 5.2. Kiểm chứng nối tiếp: cấu hình khóa đã được mang nguyên vẹn sang Phase 2

| Metric truy xuất | Phase 1 (`winner_lock.jsonl`) | Phase 2A (baseline) | Lệch |
| :--- | ---: | ---: | ---: |
| Hit@5 | 0,9573 | 0,9573 | 0,0000 |
| Recall@5 | 0,9555 | 0,9555 | 0,0000 |
| nDCG@5 | 0,8976 | 0,8976 | 0,0000 |
| MRR@5 | 0,8797 | 0,8797 | 0,0000 |

Khớp tuyệt đối đến 4 chữ số thập phân. Đây không phải trang trí: nó là **bằng chứng kiểm chứng được** rằng Phase 2 chạy trên đúng cấu hình Phase 1 đã khóa, đúng tập câu hỏi, đúng chỉ mục — không có biến số nào lọt vào giữa hai phase. Mọi khác biệt Phase 2 đo được vì thế thuộc về generation.

*(Nhắc lại đặc điểm của sparse: nó xác định. Chính vì thế bảng này khớp đúng 0,0000. Nếu cấu hình khóa là dense, bảng này đã không thể khớp như vậy — xem §4.3.)*

### 5.3. Phase 2B.1 — Sàng lọc prompt (depth 5)

QA F1 và token đo trên 80 câu; các metric RAGAS đo trên cùng một tập con cố định 20 câu.

| Prompt | QA F1 | Answer Correctness | Faithfulness | Answer Relevancy | Citation F1 | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 (control) | 0,2417 | 0,7118 | 1,000 | 0,8131 | 0,7808 | 2.997 |
| P1 | 0,2570 | 0,6885 | 0,975 | 0,7426 | 0,7863 | 2.732 |
| **P2** | **0,3999** | **0,8363** | **1,000** | 0,7388 | 0,8133 | **1.470** |
| P3 | 0,2790 | 0,7190 | 0,975 | **0,8153** | **0,8146** | 1.912 |

- **P2 là prompt mới duy nhất qua hết guardrail.** Hơn P0 +0,1245 Answer Correctness, CI95 theo cặp gom cụm bài báo [+0,0325; +0,2296] — không chứa 0.
- P2 tạo **9 exact match** (P0: 0) và giảm ~51% output token.
- P1 không cải thiện correctness. P3 cải thiện độ súc tích và citation nhưng mức tăng correctness quá nhỏ.
- Answer Relevancy của P2 giảm. Nguyên nhân có cơ sở dữ liệu, không phải lỗi — xem §6, luồng 2.

> [!NOTE]
> Tập screening 80 câu được chốt **một lần** với seed 42 và không đổi giữa các run: phân tầng theo loại câu hỏi và theo việc gold evidence có nằm trong top 5 hay không, đồng thời cân bằng theo bài báo. RAGAS chỉ chấm 20 câu cố định trong số đó để giảm chi phí — nên **không được đọc số RAGAS screening như số final**. P0 ở depth 5 tái sử dụng từ baseline, không gọi API lại.

### 5.4. Phase 2B.2 — Sàng lọc độ sâu context (2 prompt × 3 depth)

Theo đúng plan §4.2, depth được quét trên **hai** prompt tốt nhất của vòng P — P0 làm control và P2 làm ứng viên — nên hiệu ứng của depth tách được khỏi hiệu ứng của prompt.

| Prompt | Depth | QA F1 | Answer Correctness | Faithfulness | Answer Relevancy | Citation F1 | Citation Validity | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 1 | 0,2543 | 0,7063 | 0,9750 | 0,7365 | 0,7250 | 0,8167 | 2.246 |
| P0 | 3 | 0,2527 | 0,7046 | **0,9875** | 0,7667 | 0,7792 | 0,9500 | 2.698 |
| P0 | 5 | 0,2417 | 0,7118 | **1,0000** | **0,8131** | 0,7808 | 0,9750 | 2.997 |
| P2 | 1 | **0,5809** | 0,8671 | 0,9500 | 0,4825 | 0,5500 | 0,6125 | **779** |
| P2 | 3 | 0,5311 | **0,8795** | 0,9250 | 0,6806 | 0,8125 | 0,9500 | 1.047 |
| P2 | 5 | 0,3999 | 0,8363 | **1,0000** | 0,7388 | **0,8133** | **0,9750** | 1.470 |

Ba điều đọc được từ bảng này:

1. **Depth gần như không tác động lên P0** — Answer Correctness dao động 0,7046–0,7118 qua cả ba mức. Prompt dài, trả lời dài thì thêm hay bớt context không đổi được nhiều.
2. **Depth tác động rất mạnh lên P2** (0,8363–0,8795). Khi câu trả lời bị ép ngắn, việc chọn đúng bằng chứng nào để đọc trở thành yếu tố quyết định.
3. **Citation Validity xuống theo depth ở cả hai prompt** (P0: 0,9750 → 0,8167; P2: 0,9750 → 0,6125). Đây không phải đặc tính của prompt mà của **context bị cắt** — đúng cơ chế được định lượng ở §6, luồng 3.

Depth 1 bị loại vì mất bằng chứng và trích dẫn quá nhiều. Depth 3 và depth 5 của P2 được đưa lên chạy đủ 281 câu. **Bảng screening không tự quyết định winner** — nó chỉ chọn ứng viên.

### 5.5. Phase 2B.3 — Finalist và quyết định winner

Cả hai finalist đã chạy đủ **281/281** trên tập development, coverage generation và RAGAS đều 100%, không có record failed, missing hay trùng.

| Metric | P0-depth5 | P2-depth3 | P2-depth5 |
| :--- | ---: | ---: | ---: |
| Exact Match | 0,0000 | **0,2633** | 0,0925 |
| Token F1 | 0,2648 | **0,5549** | 0,4191 |
| **Answer Correctness** | 0,6308 | **0,7711** | 0,7350 |
| Faithfulness | 0,9761 | 0,9561 | **0,9801** |
| Citation F1 | 0,8025 | 0,8381 | **0,8391** |
| Citation Validity | **0,9893** | 0,9751 | 0,9858 |
| Answer Relevancy | **0,7950** | 0,5694 | 0,7092 |
| Input / output tokens | 634.008 / 10.754 | **400.241 / 3.883** | 640.752 / 5.035 |

#### Áp bộ guardrail đã đăng ký

Quy tắc lấy nguyên từ `phase_2_generation_tuning_plan.md` §7, khóa **trước khi** xem kết quả. Một cấu hình phải qua **cả bốn** mới được coi là hợp lệ; chỉ trong nhóm hợp lệ mới so Answer Correctness.

| Guardrail (ngưỡng) | P2-depth3 | P2-depth5 |
| :--- | :--- | :--- |
| Coverage ≥ 95% | 100% ✅ | 100% ✅ |
| Faithfulness giảm ≤ 0,02 | **−0,020029** ❌ | +0,003992 ✅ |
| Citation F1 giảm ≤ 0,01 | +0,035567 ✅ | +0,036635 ✅ |
| Citation Validity giảm ≤ 0,01 | **−0,014235** ❌ | −0,003559 ✅ |
| **Kết luận** | **BỊ LOẠI** (2/4 trượt) | **HỢP LỆ** (4/4 qua) |

> [!IMPORTANT]
> **Winner: P2-depth5.** Nó là cấu hình hợp lệ duy nhất, nên quy tắc "chọn Answer Correctness cao nhất trong nhóm hợp lệ" không cần đến bước tie-break nào.
>
> P2-depth3 đạt Answer Correctness **cao hơn 0,0360** nhưng không được chọn. Đây chính là tình huống bộ guardrail được thiết kế để xử lý: nó chặn việc đổi tính trung thực lấy điểm số.

#### Kiểm định theo cặp — bootstrap gom cụm bài báo

Mọi so sánh dùng hiệu theo từng câu, bootstrap 2.000 lần lấy mẫu **theo bài báo** (50 cụm), seed 42. Gom cụm theo bài là bắt buộc: nhiều câu hỏi cùng một bài không phải quan sát độc lập. Tái lập: `scripts/phase2_paired_comparison.py`; kết quả đầy đủ ở `docs/reports/phase2/paired_significance.json`.

**P2-depth5 so với P0** — cải thiện chất lượng, không mất grounding:

| Metric | Hiệu | CI95 của hiệu | Phán quyết |
| :--- | ---: | :--- | :--- |
| Answer Correctness | **+0,1043** | [+0,0844; +0,1240] | **có ý nghĩa** |
| Token F1 | +0,1542 | [+0,1317; +0,1786] | **có ý nghĩa** |
| Exact Match | +0,0925 | [+0,0596; +0,1296] | **có ý nghĩa** |
| Citation F1 | +0,0366 | [+0,0127; +0,0616] | **có ý nghĩa** |
| Faithfulness | +0,0040 | [−0,0127; +0,0189] | không tách được |
| Citation Validity | −0,0036 | [−0,0112; +0,0000] | không tách được |
| Answer Relevancy | −0,0858 | [−0,1213; −0,0508] | **có ý nghĩa** (giảm) |

Hai dòng "không tách được" ở đây là **kết quả tốt**, không phải kết quả yếu: chúng nói rằng P2-depth5 **không hề làm giảm** grounding và citation validity — mạnh hơn hẳn so với việc chỉ nằm trong ngưỡng dung sai.

**P2-depth3 so với P0** — mức tụt guardrail là thật:

| Metric | Hiệu | CI95 của hiệu | Phán quyết |
| :--- | ---: | :--- | :--- |
| Answer Correctness | +0,1403 | [+0,1131; +0,1674] | **có ý nghĩa** |
| Faithfulness | **−0,0200** | **[−0,0372; −0,0048]** | **có ý nghĩa** (giảm) |
| Citation Validity | **−0,0142** | **[−0,0282; −0,0034]** | **có ý nghĩa** (giảm) |
| Citation F1 | +0,0356 | [−0,0011; +0,0730] | không tách được |
| Answer Relevancy | −0,2256 | [−0,2679; −0,1857] | **có ý nghĩa** (giảm) |

> [!CAUTION]
> **Mức tụt của P2-depth3 không phải nhiễu đo.** Khoảng CI95 của cả Faithfulness lẫn Citation Validity đều **không chứa 0**. Việc con số Faithfulness chỉ vượt ngưỡng 0,02 đúng 0,00003 dễ khiến người ta tưởng đây là chuyện làm tròn — không phải. Cái sát ngưỡng là *vị trí* của mức tụt so với vạch quy ước; *sự tồn tại* của mức tụt thì đã được chứng minh chắc chắn.

**P2-depth5 so với P2-depth3** — đây là bản chất của đánh đổi:

| Metric | Hiệu (d5 − d3) | CI95 của hiệu | Phán quyết |
| :--- | ---: | :--- | :--- |
| Answer Correctness | −0,0360 | [−0,0554; −0,0190] | **có ý nghĩa** |
| Faithfulness | **+0,0240** | [+0,0056; +0,0440] | **có ý nghĩa** |
| Answer Relevancy | +0,1398 | [+0,1081; +0,1733] | **có ý nghĩa** |
| Exact Match | −0,1708 | [−0,2159; −0,1246] | **có ý nghĩa** |

Không có cấu hình nào tốt hơn tuyệt đối. Depth 3 mua correctness bằng grounding; depth 5 làm ngược lại. **Bộ quy tắc đăng ký trước là thứ phân xử**, chứ không phải sở thích của nhóm sau khi đã nhìn thấy cả hai cột số.

Cái giá của depth 5: input token cao hơn depth 3 **60%** (640.752 so với 400.241). Đây là chi phí có thật của việc giữ nguyên tính trung thực, và phải được nêu ra chứ không giấu đi.

Riêng P2-depth3 có **7 câu trả về canonical abstention** không kèm citation: 5 câu vốn không có gold evidence trong top 5, 2 câu có gold ở rank 4 nên bị depth 3 cắt mất. Cơ chế được định lượng đầy đủ ở §6, luồng 3.

### 5.6. Audit 30 câu Answer Correctness thấp

Mẫu có chủ đích, phân tầng, seed 42 (`selection: purposive_stratified_low_correctness_audit`): 82 câu ứng viên có điểm < 0,5, lấy 30 câu từ 27 bài theo bốn tầng — nặng/vừa × gold có/không trong top 5. **Không** dùng tỷ lệ trong mẫu này để ước lượng tỷ lệ trên toàn bộ 281 câu.

| Nhóm lỗi | Số câu |
| :--- | ---: |
| Đúng nhưng quá dài | 10 |
| Vấn đề gold / evidence / cách chấm | 9 |
| Retrieval hỏng thật | 4 |
| Đúng một phần, trộn mốc thời gian | 3 |
| Judge không đồng thuận với người duyệt | 3 |
| Sai answer type | 1 |

Người duyệt cho rằng **23/30 câu đúng về ngữ nghĩa** và điểm tự động thấp là không hợp lý; chỉ 7/30 là lỗi end-to-end thật. Kết quả audit này chính là thứ dẫn tới hướng tối ưu của P2: **trả lời trực tiếp, đúng answer type, ít chi tiết, chỉ cite bằng chứng cần thiết.**

> [!WARNING]
> **Đây không phải audit mà plan yêu cầu để đóng Phase 2.** `phase_2_generation_tuning_plan.md` §8 đăng ký một audit khác: **mù cấu hình**, trên **30 cặp output P0 ↔ winner**, do **hai reviewer** độc lập chấm, có báo cáo tỷ lệ đồng thuận. Audit ở trên là mẫu một chiều trên câu điểm thấp của riêng baseline, do một người cùng một mô hình soát (`reviewer_id: "Thomas + Codex"`) — hữu ích để chẩn đoán, nhưng không thay thế được. Xem §8, việc số 2.

---

## 6. Sợi chỉ xuyên suốt: EDA → Phase 1 → Phase 2

Đây là phần trả lời câu hỏi *"vì sao chọn cấu hình này?"* bằng cơ chế dữ liệu, chứ không chỉ bằng bảng điểm.

### Luồng 1 — Nhiễu nhãn không biến mất khi sang tầng generation

| Tầng | Biểu hiện |
| :--- | :--- |
| **EDA §7** | 7,0%–24,5% câu hỏi có một chunk từ **bài báo khác** trả lời thỏa đáng, nhưng bộ chấm chỉ chấp nhận đúng một gold chunk ⇒ tính là sai |
| **Phase 1** | Sinh ra quy tắc: chênh lệch < 0,070 không được đọc thành tiến bộ kỹ thuật |
| **Phase 2** | Audit 30 câu điểm thấp: **9 câu** thuộc nhóm "vấn đề gold/evidence/cách chấm"; người duyệt thấy **23/30** thực ra đúng |

Cùng một hiện tượng, đo ở hai tầng khác nhau. Giả định *closed-world* phạt cả retriever lẫn generator theo đúng một cách: hệ thống tìm ra và diễn đạt trung thực một bài báo tương đương, rồi bị chấm sai.

> **Hệ quả phải ghi vào báo cáo:** Answer Correctness 0,6308 của baseline là **cận dưới**, không phải năng lực thật của hệ thống. Điều đó không làm số ấy vô dụng — vì mọi cấu hình đều chịu cùng một khoản phạt, nên **so sánh giữa các cấu hình vẫn hợp lệ**; chỉ có giá trị tuyệt đối là bị nén xuống.

Đây cũng là lý do Answer Correctness được chọn làm metric chính thay vì Exact Match: gold NewsQA thường là span ngắn, trong khi model có thể paraphrase đúng.

### Luồng 2 — Gold là span ngắn, và điều đó quyết định prompt nào thắng

Đáp án NewsQA thường là một span rất ngắn: một thực thể, một mốc thời gian, một con số.

- **Bằng chứng rõ nhất:** baseline có Answer Correctness **0,6308** nhưng Exact Match **0,0000**. Model biết câu trả lời, nhưng gói nó trong một đoạn văn.
- **P2 làm đúng một việc:** ép trả lời trực tiếp, đúng answer type, bỏ chi tiết thừa. Kết quả: EM 0,0000 → **0,2633**, Token F1 0,2648 → **0,5549**, output token giảm 63,9%.
- **Cùng nguyên nhân đó giải thích Answer Relevancy tụt 0,2256.** Metric ấy thưởng câu trả lời đầy đủ ngữ cảnh — trong khi dữ liệu này lại thưởng câu trả lời cụt. Hai thứ mâu thuẫn nhau **về bản chất dữ liệu**, không phải P2 sinh câu trả lời kém.

Vì vậy P2 thắng không phải nhờ may mắn trong prompt engineering, mà vì nó khớp với hình dạng của nhãn. Và cũng vì vậy Answer Relevancy được theo dõi nhưng **không** được đặt làm guardrail — quyết định đó được đưa ra *trước* khi thấy con số −0,2256.

### Luồng 3 — Đường cong Hit@k của Phase 1 định giá chính xác việc cắt context

Đây là chỗ Phase 1 trả lời trực tiếp cho một câu hỏi của Phase 2, **không tốn thêm một lệnh gọi API nào**.

| Depth | Hit@k của cấu hình khóa (`resolved`) | Số câu **mất sạch bằng chứng** so với depth 5 | Recall@k |
| ---: | ---: | ---: | ---: |
| 1 | 0,8221 | **38** / 281 (13,5%) | 0,8149 |
| 3 | 0,9324 | **7** / 281 (2,5%) | 0,9288 |
| 5 | 0,9573 | 0 (mốc) | 0,9555 |

Đối chiếu với những gì Phase 2 quan sát được:

- **Depth 3 lấy mất bằng chứng của đúng 7 câu** — và Phase 2B.3 ghi nhận **đúng 7 câu abstention không kèm citation**, trong đó 2 câu có gold ở rank 4, tức bị chính depth 3 cắt. Đây là cơ chế đứng sau việc Citation Validity tụt 0,0142 và Faithfulness tụt 0,0200: không phải parser hỏng, không phải model bịa chỉ số citation, mà là **context bị cắt cụt đúng chỗ có bằng chứng**.
- **Depth 1 lấy mất bằng chứng của 38 câu**, và Citation Validity của nó sụp xuống 0,6125. Độ lớn khớp với dự đoán.

> **Bài học phương pháp:** độ sâu context không phải một cái núm miễn phí. Phase 1 đã đo sẵn cái giá của nó dưới dạng đường cong Hit@k, nên đáng lẽ có thể **dự đoán** kết quả sàng lọc depth trước khi chi tiền API. Với hệ thống sau, nên đọc Hit@k trước rồi mới chọn depth để thử.

### Luồng 4 — Phục hồi truncation, và vì sao đáp án vẫn nằm sớm

- **EDA §3–§4:** 41,6% bài báo bị cắt ở ngưỡng 640–680 từ (lỗi từ tập NewsQA gốc). Toàn bộ 4.603 bài đã được phục hồi bằng thao tác **chỉ nối thêm vào cuối** — mọi offset ký tự của evidence span giữ nguyên, không cần ánh xạ lại ground truth.
- **Đo lại vị trí bằng chứng:** trung vị 16,5% → **16,0%** độ dài bài. Đáp án NewsQA nằm ở phần đầu bài báo, đúng chỗ truncation không chạm tới.
- **Phase 1:** giải thích vì sao Hit@1 đạt tới 0,8221 — chunk chứa đáp án thường là chunk đầu tiên của bài.
- **Phase 2:** giải thích vì sao depth 1 vẫn giữ được **QA F1 cao nhất** (0,5809) dù mất bằng chứng của 38 câu. Depth 1 thất bại vì grounding và citation, **không phải** vì mất đáp án. Hai chuyện khác nhau, và số liệu tách được chúng ra.

### Luồng 5 — Hai hiện tượng trùng lặp câu hỏi, đừng gộp

- **Tác dụng phụ của việc làm rõ câu hỏi (EDA §6):** 47 nhóm câu hỏi vốn khác nhau trở thành giống hệt nhau từng chữ sau khi bổ sung ngữ cảnh — 96 câu (7,2%), tương đương 49 câu bị hỏi trùng. Không câu nào trở nên bất khả thi (cùng bài, cùng gold chunk), nhưng 47 bài báo đó bị **tính trọng số gấp đôi** trong điểm số.
- **Khử trùng lặp ngữ nghĩa (bước riêng, chạy sau):** LLM đề xuất + người duyệt gom 155 cụm gồm 339 câu, giữ 1 đại diện mỗi cụm ⇒ loại 184 câu, 1.336 → **1.152**.
- **Cả Phase 1 lẫn Phase 2 đều chạy trên tập đã khử trùng lặp này.** Đó là điều kiện để bảng §5.2 khớp được 0,0000.

### Luồng 6 — Tách judge khỏi generator

EDA §9 phát hiện notebook cũ đặt `JUDGE_MODEL = GENERATOR_MODEL`, tức LLM tự chấm bài của chính nó. Phase 2 tách hẳn: generator `gemini-3.1-flash-lite`, judge `glm-5p3-flash` trên endpoint Fireworks riêng. Đây là lý do các con số Phase 2 dùng được, và là chi tiết bắt buộc phải nêu khi trình bày.

---

## 7. Những gì báo cáo này **chưa** được phép kết luận

1. **Chưa có con số nào để công bố.** Mọi số Phase 2 đo trên 281 câu development — tập đã dùng để tinh chỉnh, nên lệch lạc quan theo cấu trúc. Winner đã được khóa, nhưng con số dùng để công bố là con số held-out, và held-out chưa được chạm vào.
2. **Winner mới chỉ thắng theo quy tắc, chưa được nghiệm thu.** Ba việc bắt buộc trong plan vẫn còn thiếu: chạy held-out, audit mù 30 cặp hai reviewer, và kiểm độ ổn định API. Chưa xong ba việc đó thì Phase 2 chưa đóng được.
3. **Không kết luận được P2-depth5 "tốt hơn" P2-depth3 về chất lượng.** Depth 3 có Answer Correctness cao hơn có ý nghĩa. Depth 5 thắng vì nó **hợp lệ**, còn depth 3 thì không. Hai chuyện khác nhau, và báo cáo phải nói đúng chuyện thứ hai.
4. **Không phân định được best dense** (e5-base vs bge-small), và dense nói chung chưa tái lập được (§4.3).
5. **Không kết luận được gì về chunk size** — vòng 3 là kết quả null, ba kích thước chồng lấn nhau. Chọn 512 vì lý do vận hành, không vì nó thắng.
6. **Không kết luận được về hybrid** — thua sparse thuần nhưng khoảng CI chồng lấn, nên phải nói là "không chứng minh được có lợi", không phải "đã chứng minh có hại".
7. **Phase 1 chưa chạy held-out.** `phase_1_execution_guide.md` bước 5 yêu cầu chạy cấu hình đã khóa **đúng một lần trên 150 bài final-test**. Chưa có artifact nào. Phase 1 đã khóa xong *lựa chọn*, nhưng chưa hoàn tất *nghiệm thu*.
8. **Chưa có phân tầng `gold_in_top5` / `gold_not_in_top5`** và chưa có article-level macro song song với micro. Cả hai plan đều yêu cầu; số hiện có mới là micro tổng.
9. **Contextual chunking đang chạy** (notebook 16). Chưa có kết quả, và chưa được đưa vào bất kỳ lập luận nào ở trên.

---

## 8. Còn thiếu gì để đóng từng phase

Bảng này lấy thẳng **điều kiện chấp nhận trong test plan** làm checklist, không phải danh sách tự nghĩ ra. Cột cuối trỏ tới artifact chứng minh.

### 8.1. Phase 1

| # | Điều kiện (nguồn) | Trạng thái | Bằng chứng |
| ---: | :--- | :--- | :--- |
| 1 | 23 cấu hình, 3 vòng, chạy trên 50 bài development | ✅ | `round{1,2,3}.csv` |
| 2 | Chọn Best Dense/Sparse theo MRR@5, tie-break nDCG@5 → Hit@5 → P50 | ✅ | `round1_winners.jsonl` |
| 3 | Khóa winner | ✅ | `winner_lock.jsonl` |
| 4 | **Chạy đúng một lần trên 150 bài final-test** (exec guide bước 5) | ❌ **chưa làm** | — |
| 5 | Biểu đồ đọc thẳng từ `comparison.json`, không nhập tay | ⚠️ | `scripts/generate_retrieval_figures.py` chưa nối được vào CSV các vòng |

### 8.2. Phase 2

| # | Điều kiện (nguồn) | Trạng thái | Bằng chứng |
| ---: | :--- | :--- | :--- |
| 1 | Baseline 281 câu, coverage ≥ 95% (2A §6) | ✅ 100% | `report_baseline_p0_d5.json` |
| 2 | Hai finalist đủ coverage deterministic + RAGAS (2B §10) | ✅ 100% | `report_finalist_p2_d{3,5}.json` |
| 3 | Winner chọn đúng quy tắc đăng ký, không dùng held-out (2B §10) | ✅ | `paired_significance.json` |
| 4 | **`phase2b_winner_decision.json`**: winner, hash hai finalist, người duyệt (2B §5.2 bước 7) | ⏳ **cần tạo** | — |
| 5 | **Chạy winner một lần trên 284 câu held-out** (2B §10) | ❌ **chưa làm** | thư mục `held out test` rỗng |
| 6 | **Audit mù 30 cặp P0 ↔ winner, hai reviewer, báo cáo đồng thuận** (2B §8) | ❌ **chưa làm** | audit hiện có là loại khác (§5.6) |
| 7 | **Lặp 25 câu cố định cho P0 và winner để đo độ ổn định API** (2B §8) | ❌ **chưa làm** | — |
| 8 | Báo cáo phân tầng `gold_in_top5` / `gold_not_in_top5` (2A §3.3, 2B §6) | ❌ **chưa làm** | — |
| 9 | Báo cáo article-level macro bên cạnh micro (2A §3.3) | ❌ **chưa làm** | — |
| 10 | Không lưu API key trong output/artifact (2A §6) | ✅ | đã kiểm |
| 11 | Đóng gói winner để app và benchmark nạp cùng một artifact (2B §10) | ⏳ | sau khi có held-out |

### 8.3. Phase 3 — bị chặn

Phase 3 chạy **compact_200**: 200 case (140 development / 60 final) thuộc 7 loại, đánh giá 3 cấu hình B0 (baseline Phase 2), B1 (prompt abstention có schema JSON), B2 (B1 + gate theo điểm reranker).

| # | Điều kiện | Trạng thái |
| ---: | :--- | :--- |
| 1 | Khóa winner Phase 2 | ⏳ chờ mục 8.2/5 |
| 2 | Duyệt tay toàn bộ case; `removed_article`, `external_unanswerable`, `counterfactual` cần **reviewer thứ hai** | ❌ chưa xong |
| 3 | Finalize: đúng 200 case, đúng quota 140/60, không leakage giữa partition | ❌ chưa |
| 4 | Chạy B0/B1 trên 140 dev, hiệu chỉnh B2, khóa threshold | ❌ chưa |
| 5 | Chạy đúng một lần trên 60 case final | ❌ chưa |

### 8.4. Thứ tự làm

| # | Việc | Chặn cái gì |
| ---: | :--- | :--- |
| 1 | Tạo `phase2b_winner_decision.json` (P2-depth5, hash hai finalist, người duyệt) | Held-out |
| 2 | Audit mù 30 cặp P0 ↔ P2-depth5, hai người chấm độc lập | Đóng Phase 2 — **cần chia việc giữa hai người** |
| 3 | Lặp 25 câu cố định × 2 cấu hình để đo độ ổn định API | Đóng Phase 2 |
| 4 | Bổ sung phân tầng gold-in-top5 và article macro từ dữ liệu **đã có** | Đóng Phase 2 — không tốn API |
| 5 | Chạy notebook `13i` trên 284 câu held-out, **đúng một lần** | Số công bố |
| 6 | Chạy vòng 1 lại trên chỉ mục đã đóng gói để cố định số dense | §4.3 |
| 7 | Chạy Phase 1 held-out trên 150 bài final-test | Nghiệm thu Phase 1 |
| 8 | Duyệt dataset abstention 200 case | Toàn bộ Phase 3 |

**Việc số 4 nên làm trước** vì nó không tốn một lệnh gọi API nào: mọi thứ cần thiết đã nằm trong `docs/reports/phase2/scores/*.jsonl` (có sẵn `article_key` và điểm retrieval theo từng câu).

**Việc số 6 cũng rẻ.** Vòng 1 là retrieval-only, và `scripts/run_phase1_kaggle.py` bỏ qua toàn bộ bước dựng chỉ mục nếu tìm thấy `indexes/round1/index_manifest.json`:

```python
round1_manifest_path = indexes / "round1/index_manifest.json"
if round1_manifest_path.exists():
    round1_manifest = json.loads(round1_manifest_path.read_text(encoding="utf-8"))
else:
    round1_manifest = parallel_build(...)   # phần tốn GPU
```

Attach dataset chỉ mục, chép vào `WORK_ROOT/indexes/round1`, chạy `--stop-after round1`. Kiểm trước: manifest phải liệt kê đủ **4 dense + 4 sparse** — bản đóng gói từ lần chạy `SMOKE_MODE`/`FAST_MODE` chỉ chứa `all-MiniLM-L6-v2`.

---

## 9. Ba chỗ báo cáo lệch với test plan, đã ghi nhận

Các test plan trong `docs/Detailed Test Plans/` là văn bản **đăng ký trước**. Khi thực tế chạy khác với plan, cách xử lý đúng là ghi nhận độ lệch — không sửa lùi plan.

**1. P3 đã được sửa sau khi plan viết xong.**

| Nguồn | P3 là gì |
| :--- | :--- |
| `phase_2_generation_tuning_plan.md` §4.1 | "Kết hợp P1/P2; citation `[i]` bắt buộc cho mỗi khẳng định chính" |
| `configs/experiments/phase2_generation_prompts.yaml` | `event_disambiguated_concise` — chọn đoạn khớp sự kiện trước khi trả lời |

Bản đã chạy là bản trong YAML. Việc sửa này **có cơ sở** — nó nhắm vào phát hiện distractor collision của EDA §7 — nhưng plan cần một ghi chú sửa đổi có ngày tháng. P3 không được chọn nên độ lệch này không ảnh hưởng winner.

**2. Metric chọn winner vòng 1.** Exec guide Phase 1 quy định chọn theo **MRR@5**, tie-break lần lượt nDCG@5 → Hit@5 → P50. Các bảng trong báo cáo dẫn nDCG@5 làm chỉ số đầu. Đã kiểm: **không đổi người thắng** — BGE-M3 (MRR@5 0,8059) và e5-base (0,6316) đứng đầu ở cả hai metric.

**3. Latency của cấu hình khóa có hai giá trị.** 510,1 ms đo ở vòng 2 (`winner_lock.jsonl`), 512,7 ms đo lại ở vòng 3 (`round3.csv`) — cùng cấu hình, hai lần đo, lệch 2,6 ms. Báo cáo này dẫn 510,1 ms; `phase_2_baseline_test_plan.md` dẫn 512,7 ms. Cả hai đều đúng; chênh lệch là dao động đo giữa hai lần chạy.

---

## Phụ lục A — Provenance

Mọi thứ cần để tái lập, lấy từ các test plan và manifest của run.

### Dữ liệu

| Thuộc tính | Giá trị |
| :--- | :--- |
| Raw dataset | `MatchaMacchiato/newsqa_200_11064_v2.0.0` |
| Raw revision | `b81c8db6847a23272665946c0c43c72e9a212fd9` |
| Locked artifact repo | `ThomasAnderson2009/newsqa-rag-phase2-locked-v2` |
| Artifact tag | `locked-bge-m3-512-64-deduplicated-v2` |
| Artifact commit | `bb73e682f472933c212f2c6a3f9575c652b280fd` |
| Artifact ZIP SHA-256 | `fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24` |
| Corpus | 11.064 bài báo, 22.766 chunks |
| Primary set | 200 bài, 1.152 câu resolved đã khử trùng lặp ngữ nghĩa |

### Chia tập

Chia theo `article_key`, shuffle seed `42`, lấy 50 bài đầu làm development. 150 bài còn lại (871 câu) là held-out pool; từ đó lấy 50 bài bằng thứ tự hash seed `46` làm final subset (284 câu), 100 bài còn lại (587 câu) là reserve.

### Fingerprint của run Phase 2

| Run | `run_fingerprint` |
| :--- | :--- |
| Baseline P0-depth5 | `ef7c1a5b12f2fb91c476fe97dbb91131646de39bc44dc81d9e56cd24129b3a70` |
| `config_sha256` | `ddbb3637110ed8e431ca312449741d562f8b32dc5d6674e7d6c2761d9696b770` |
| `testset_sha256` | `80f1b12acd6090aa353a89578c58c98e7d70d02dc4a271f2c724b370694c3e80` |
| `variant_manifest_sha256` | `8f7f6a77a8754b00d0f43442d4fb44593635a479a06ee072e71bd01c2519e975` |
| Judge fingerprint | `a4c91b2a3436adfa0c2ae4665cbc18f8bcc958b18210e99bd2f022481afa6532` |

### Model và tham số

| | |
| :--- | :--- |
| Generator | `gemini-3.1-flash-lite`, `temperature=0`, `max_tokens=512`, `reasoning_effort=minimal` |
| Giãn request generation | 4,2 giây (giới hạn 15 RPM của free key) |
| Judge | `accounts/fireworks/models/glm-5p3-flash`, `reasoning_effort=low`, tối đa 2.048 output token |
| Judge runtime | timeout 300 s, batch 1, 1 worker, 3 SDK retry |
| Seed | 42 |

Generator và judge dùng hai model, hai provider và hai credential khác nhau (EDA §9). Khóa API chỉ được tiêm vào đúng subprocess, không nằm trong checkpoint, notebook output hay manifest.

---

## Phụ lục B — Nguồn của từng con số

| Nhóm số liệu | Nguồn |
| :--- | :--- |
| Vòng 1 / 2 / 3 | `docs/reports/phase1/{round1,round2,round3}.csv` |
| Cấu hình khóa & đường cong Hit@k | `docs/reports/phase1/winner_lock.jsonl` |
| Kiểm định theo cặp Phase 1 | `docs/reports/phase1/paired_significance.json`, notebook `15` |
| Số EDA (bản v2.0.0) | `scripts/eda/12_chunk_counts.py`, `13_materialize_v2_for_eda.py`, `14_rerun_on_v2.py`; xem Phụ lục A của [phase1/report.md](phase1/report.md) |
| Phase 2A baseline | `docs/reports/phase2/report_baseline_p0_d5.json` |
| Phase 2B.1 / 2B.2 screening | Notebook `13b`–`13f`; report.json của từng run |
| Phase 2B.3 finalist | `docs/reports/phase2/report_finalist_p2_d{3,5}.json` |
| Điểm theo từng câu (dùng cho kiểm định) | `docs/reports/phase2/scores/*.jsonl` |
| Quyết định winner + CI theo cặp | `docs/reports/phase2/paired_significance.json`, `scripts/phase2_paired_comparison.py` |
| Phase 2B.4 held-out | Notebook `13i` — chưa chạy |
| Nội dung prompt P0–P3 | `configs/experiments/phase2_generation_prompts.yaml` |
| Quy tắc phán quyết | `docs/Detailed Test Plans/phase_2_generation_tuning_plan.md` §7 |
| Chỉ mục Phase 1 đã đóng gói | `scripts/package_index_artifacts.py`, notebook `17` |
