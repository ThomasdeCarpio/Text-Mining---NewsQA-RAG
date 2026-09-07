# Báo cáo Thực nghiệm Phase 2: Baseline End-to-End & Tối ưu Generation

> **Kế hoạch đăng ký trước:** [`phase_2_baseline_test_plan.md`](../../Detailed%20Test%20Plans/phase_2_baseline_test_plan.md) (2A) · [`phase_2_generation_tuning_plan.md`](../../Detailed%20Test%20Plans/phase_2_generation_tuning_plan.md) (2B)
> **Nền tảng:** [Phase 1 — Retrieval Tournament](../phase1/report.md) · [EDA](../../eda/eda_report.md)
> **Tóm tắt toàn dự án:** [../report.md](../report.md)
> **Artifact số liệu:** `report_baseline_p0_d5.json`, `report_finalist_p2_d{3,5}.json`, `scores/*.jsonl`, `paired_significance.json`

---

## 1. Phase 2 làm gì, và không làm gì

Phase 1 trả lời *"lấy đoạn văn nào?"*. Phase 2 trả lời *"đưa cho LLM thế nào để nó trả lời đúng?"*

Phase 2 tối ưu **prompt và cấu hình sinh**. Tên học thuật đúng là *prompt/configuration tuning* — **không phải supervised fine-tuning**, vì không có trọng số nào của Gemini được cập nhật.

| Trong phạm vi | Ngoài phạm vi |
| :--- | :--- |
| Nội dung prompt (P0–P3) | Đổi embedding, retriever, reranker, chunking |
| Số context đưa vào generator (1/3/5) | Query rewriting, Agentic RAG |
| | Huấn luyện trọng số model |
| | Dùng held-out để sửa prompt hay tham số |
| | Trộn bài toán abstention vào benchmark answerable |

### Thiết kế then chốt: retrieval chạy đúng một lần

Retrieval và rerank chạy **một lần duy nhất**, lưu lại danh sách 5 chunk đã xếp hạng. Mọi thí nghiệm depth chỉ **cắt ngắn danh sách sẵn có** — không truy xuất lại, không gọi lại Gemini cho các record đã trùng.

Hệ quả: mọi khác biệt đo được **chắc chắn** thuộc về generation. Cùng question ID, cùng ranked trace, chỉ khác prompt và số context. Đây là điều làm cho các so sánh trong báo cáo này quy kết được nguyên nhân.

---

## 2. Hợp đồng cố định

### 2.1. Dữ liệu và artifact

| Thuộc tính | Giá trị |
| :--- | :--- |
| Raw dataset | `MatchaMacchiato/newsqa_200_11064_v2.0.0` |
| Raw revision | `b81c8db6847a23272665946c0c43c72e9a212fd9` |
| Locked artifact repo | `ThomasAnderson2009/newsqa-rag-phase2-locked-v2` |
| Artifact tag | `locked-bge-m3-512-64-deduplicated-v2` |
| Artifact commit | `bb73e682f472933c212f2c6a3f9575c652b280fd` |
| ZIP SHA-256 | `fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24` |
| Corpus | 11.064 bài báo, 22.766 chunks |
| Primary set | 200 bài, 1.152 câu `resolved` đã khử trùng lặp ngữ nghĩa |

Artifact gồm `chunks.jsonl`, testset đã khử trùng lặp, chỉ mục BGE-M3 sparse, Chroma dự phòng và `bundle_manifest.json`. Phase 2 **tải chỉ mục đã khóa**, không dựng lại trong mỗi run. Notebook kiểm checksum trước khi chạy; artifact sai hash làm run dừng, không tự chuyển sang dựng lại từ raw NewsQA.

Việc khử trùng lặp đã có người duyệt: 184 câu trùng ngữ nghĩa trong cùng bài được gộp thành 155 cụm; câu đại diện giữ hợp nhất `accepted_answers`, evidence span và `relevant_chunk_ids`.

### 2.2. Chia tập

Chia theo `article_key`, shuffle seed `42`, lấy 50 bài đầu làm development.

| Partition | Số bài | Số câu | Mục đích | Đã chạm |
| :--- | ---: | ---: | :--- | :--- |
| Development | 50 | **281** | Baseline và toàn bộ tinh chỉnh | Nhiều lần |
| Held-out final | 50 | 284 | Đánh giá winner **đúng một lần**, sample seed `46` | Chưa |
| Held-out reserve | 100 | 587 | Mở rộng sau; cũng là nguồn của Phase 3 | Chưa |
| **Tổng** | **200** | **1.152** | | |

Đây là **cùng article-level split đã dùng ở Phase 1**. Con số 281 không phải giới hạn `n_eval` mà là toàn bộ semantic target thuộc 50 bài development sau khi khử trùng lặp.

Tập bổ sung 1.336 câu (bản đầy đủ, chưa khử trùng lặp) **chỉ** dùng cho sensitivity analysis sau khi đã khóa cấu hình, và không được đưa trở lại mục tiêu chọn prompt.

### 2.3. Cấu hình pipeline

| Thành phần | Giá trị |
| :--- | :--- |
| Question variant | `resolved` (242 câu `clarified` + 39 câu `original` không cần sửa) |
| Chunking | recursive, 512 / overlap 64 |
| Retriever | BGE-M3 learned sparse, `top_k=20` |
| Reranker | `BAAI/bge-reranker-large`, batch 8, top 5 |
| Generator | `gemini-3.1-flash-lite`, `temperature=0`, `max_tokens=512`, `reasoning_effort=minimal` |
| Giãn request generation | 4,2 giây (giới hạn 15 RPM của free key) |
| Judge | `accounts/fireworks/models/glm-5p3-flash`, `reasoning_effort=low`, ≤ 2.048 output token |
| Judge runtime | timeout 300 s, batch 1, 1 worker, 3 SDK retry |
| Seed | 42 |

> [!IMPORTANT]
> **Generator và judge dùng hai model, hai provider và hai credential khác nhau.** EDA §9 phát hiện notebook cũ đặt `JUDGE_MODEL = GENERATOR_MODEL`, tức LLM tự chấm bài của chính nó. Việc tách này là điều kiện để các con số RAGAS trong báo cáo dùng được. Khóa API chỉ được tiêm vào đúng subprocess, không nằm trong checkpoint, notebook output hay manifest.

### 2.4. Fingerprint

| | |
| :--- | :--- |
| `run_fingerprint` (baseline) | `ef7c1a5b12f2fb91c476fe97dbb91131646de39bc44dc81d9e56cd24129b3a70` |
| `config_sha256` | `ddbb3637110ed8e431ca312449741d562f8b32dc5d6674e7d6c2761d9696b770` |
| `testset_sha256` | `80f1b12acd6090aa353a89578c58c98e7d70d02dc4a271f2c724b370694c3e80` |
| `variant_manifest_sha256` | `8f7f6a77a8754b00d0f43442d4fb44593635a479a06ee072e71bd01c2519e975` |
| Judge fingerprint | `a4c91b2a3436adfa0c2ae4665cbc18f8bcc958b18210e99bd2f022481afa6532` |

---

## 3. Metric và quy tắc phán quyết

| Vai trò | Metric |
| :--- | :--- |
| **Chính** | RAGAS Answer Correctness |
| **Guardrail grounding** | Faithfulness |
| **Guardrail citation** | Citation F1, Citation Validity |
| Hỗ trợ | Exact Match, token F1, Answer Relevancy |
| Chẩn đoán context | Context Precision, Context Recall |
| Vận hành | Coverage, latency, token, chi phí |

### Quy tắc chọn winner — khóa trước khi xem kết quả

1. Một cấu hình **hợp lệ** khi so với P0: Faithfulness không giảm quá `0,02`; Citation F1 không giảm quá `0,01`; Citation Validity không giảm quá `0,01`; coverage generation và RAGAS ≥ 95%.
2. Trong nhóm hợp lệ, chọn **Answer Correctness cao nhất**.
3. Chênh AC dưới `0,01` ⇒ chọn token cost thấp hơn.
4. Chi phí chênh dưới 5% ⇒ chọn generation P95 thấp hơn.
5. Vẫn hòa ⇒ chọn cấu hình đơn giản hơn.

> [!NOTE]
> **Vì sao Phase 2 dùng guardrail còn Phase 1 thì không.** Phase 1 hỏi *"cái nào tốt hơn?"* — sai một chút chỉ là xếp hạng kém. Phase 2 hỏi *"cái nào tốt hơn **mà không đánh đổi tính trung thực**?"* — sai một chút nghĩa là hệ thống nói điều không có trong bằng chứng. Ngưỡng là **ràng buộc thực tiễn**, không phải khẳng định ý nghĩa thống kê; CI và effect size vẫn phải báo cáo đầy đủ.

### Cách so sánh

Mọi so sánh dùng **hiệu theo từng câu**, bootstrap 2.000 lần lấy mẫu **theo bài báo** (50 cụm), seed 42. Gom cụm theo bài là bắt buộc: nhiều câu hỏi cùng một bài không phải quan sát độc lập, coi chúng là độc lập sẽ làm khoảng tin cậy hẹp giả tạo.

Tái lập: `python scripts/phase2_paired_comparison.py --scores-dir docs/reports/phase2/scores --output docs/reports/phase2/paired_significance.json`

Exact Match không phải metric chính vì gold NewsQA thường là span ngắn trong khi model có thể paraphrase đúng. **CI chồng lấn không được đọc là "hai cấu hình tương đương"** — muốn tuyên bố tương đương thì phải đăng ký equivalence margin và dùng equivalence test riêng.

---

## 4. Bốn câu hỏi nghiên cứu, và câu trả lời

Plan 2B §3 đặt bốn câu hỏi. Đây là câu trả lời sau khi chạy xong.

| # | Câu hỏi | Trả lời |
| ---: | :--- | :--- |
| 1 | Prompt grounding rõ ràng có tăng faithfulness mà không giảm correctness? | **Không.** P1 làm giảm cả hai. Baseline đã đạt Faithfulness 0,9761 — không còn dư địa để cải thiện, nên siết grounding chỉ làm mất correctness. |
| 2 | Định dạng answer ngắn có hợp gold span NewsQA hơn không? | **Có, rất rõ.** P2 đưa Exact Match từ 0,0000 lên 0,0925 và Token F1 từ 0,2648 lên 0,4191 trên đủ 281 câu; cả hai đều có ý nghĩa thống kê. |
| 3 | Citation contract chặt hơn có tăng Citation F1? | **Có, mức vừa phải.** +0,0366, CI95 [+0,0127; +0,0616]. |
| 4 | 1, 3 hay 5 context cho trade-off tốt nhất? | **5** — sau khi áp guardrail. Depth 3 tốt hơn về correctness *và* chi phí, nhưng làm giảm grounding một cách có ý nghĩa. Xem §8. |

---

## 5. Phase 2A — Baseline (P0, depth 5, 281 câu)

### 5.1. Kết quả

| Nhóm | Metric | Giá trị |
| :--- | :--- | ---: |
| Độ phủ | Coverage generation / RAGAS | 281/281 (100%) / 100% |
| Truy xuất | Hit@1 / Hit@3 / Hit@5 | 0,8221 / 0,9324 / 0,9573 |
| | Recall@5 / MRR@5 / nDCG@5 | 0,9555 / 0,8797 / 0,8976 |
| Đáp án | **Answer Correctness** | **0,6308** |
| | Exact Match / Token F1 | 0,0000 / 0,2648 |
| | Answer Relevancy | 0,7950 |
| Grounding | Faithfulness | 0,9761 |
| | Context Precision / Recall | 0,8980 / 0,9644 |
| Trích dẫn | Citation F1 / Precision / Recall | 0,8025 / 0,7566 / 0,9199 |
| | Citation Validity / Coverage | 0,9893 / 0,9893 |
| Chi phí | Input / output token | 634.008 / 10.754 |

**Latency:** trung vị 2,77 s, trung bình 13,04 s, P95 53,63 s. Đuôi dài đến hoàn toàn từ API generator (`llm_ms` P95 = 53,13 s); phần truy xuất chỉ 76,5 ms và rerank 492,6 ms. Judge tiêu 3.372 request, ~4,68 triệu token, ~0,79 USD.

### 5.2. Kiểm chứng nối tiếp với Phase 1

| Metric | Phase 1 `winner_lock.jsonl` | Phase 2A baseline | Lệch |
| :--- | ---: | ---: | ---: |
| Hit@5 | 0,9573 | 0,9573 | 0,0000 |
| Recall@5 | 0,9555 | 0,9555 | 0,0000 |
| nDCG@5 | 0,8976 | 0,8976 | 0,0000 |
| MRR@5 | 0,8797 | 0,8797 | 0,0000 |

Khớp tuyệt đối đến 4 chữ số. Đây là **bằng chứng kiểm chứng được** rằng Phase 2 chạy đúng cấu hình Phase 1 đã khóa, đúng tập câu hỏi, đúng chỉ mục — không biến số nào lọt vào giữa hai phase.

Khớp được 0,0000 là nhờ chỉ mục sparse có tính xác định. Nếu cấu hình khóa là dense thì bảng này không thể khớp như vậy — xem [Phase 1 §3.2C](../phase1/report.md) về việc dense chưa tái lập được.

### 5.3. Chẩn đoán: vấn đề không phải hallucination

Ba con số đọc cùng nhau: **AC 0,6308 · EM 0,0000 · Faithfulness 0,9761.**

Hệ thống **không bịa** — nó nói dài. Model biết đáp án nhưng gói trong một đoạn văn, trộn chi tiết từ các bài cùng chủ đề, hoặc trả sai loại thông tin được hỏi.

Faithfulness cao **không** bảo đảm correctness cao: model có thể diễn đạt trung thực một distractor mà vẫn không trả lời đúng câu hỏi. Citation gần như luôn hợp lệ (0,9893) nhưng precision thấp hơn recall (0,7566 so với 0,9199) — model cite nhiều chunk hơn mức cần.

**Chẩn đoán này quyết định toàn bộ hướng của Phase 2B:** tối ưu phải nhắm vào *chọn đúng thông tin và dừng lại đúng lúc*, không phải vào chống bịa.

---

## 6. Audit 30 câu Answer Correctness thấp

Mẫu có chủ đích, phân tầng, seed 42 (`purposive_stratified_low_correctness_audit`): 82 câu ứng viên có điểm < 0,5, lấy 30 câu từ 27 bài theo bốn tầng (nặng/vừa × gold có/không trong top 5). **Không** dùng tỷ lệ trong mẫu này để ước lượng tỷ lệ trên toàn bộ 281 câu.

| Nhóm lỗi chính | Số câu |
| :--- | ---: |
| Đúng nhưng quá dài | 10 |
| Vấn đề gold / evidence / cách chấm | 9 |
| Retrieval hỏng thật | 4 |
| Đúng một phần, trộn mốc thời gian | 3 |
| Judge không đồng thuận với người duyệt | 3 |
| Sai answer type | 1 |

Người duyệt đánh giá **23/30 câu đúng về ngữ nghĩa**, điểm tự động thấp là không hợp lý; chỉ **7/30** là lỗi end-to-end thật.

**Nối với EDA:** nhóm "vấn đề gold/evidence/cách chấm" (9 câu) chính là hiện tượng EDA §7 đo được ở tầng truy xuất — 7,0%–24,5% câu có distractor trả lời thỏa đáng nhưng bị giả định *closed-world* chấm là sai. Cùng một hiện tượng, đo ở hai tầng.

> **Hệ quả:** Answer Correctness 0,6308 là **cận dưới**, không phải năng lực thật. Điều đó không làm số ấy vô dụng — mọi cấu hình chịu cùng khoản phạt, nên **so sánh giữa các cấu hình vẫn hợp lệ**; chỉ giá trị tuyệt đối bị nén xuống.

> [!WARNING]
> **Đây không phải audit mà plan yêu cầu để đóng Phase 2.** Plan 2B §8 đăng ký một audit khác: **mù cấu hình**, trên **30 cặp output P0 ↔ winner**, do **hai reviewer** độc lập chấm, có báo cáo tỷ lệ đồng thuận. Audit ở trên là mẫu một chiều trên câu điểm thấp của riêng baseline, do một người cùng một mô hình soát (`reviewer_id: "Thomas + Codex"`). Hữu ích để chẩn đoán, nhưng không thay thế được. Xem §10.

---

## 7. Phase 2B.1 — Sàng lọc prompt

### 7.1. Bốn prompt là gì

**Nội dung nguyên văn cả bốn prompt: [`docs/prompts.md`](../../prompts.md)** — kèm cách một request được ghép, giải thích `context_depth`, và bảng SHA-256 chứng minh prompt trong repo đúng là prompt đã chạy. Nguồn duy nhất: `configs/experiments/phase2_generation_prompts.yaml`.

Mỗi prompt là một giả thuyết nhắm vào một nhóm lỗi có số đo từ §6.

| ID | Tên | Nội dung cốt lõi | Nhắm nhóm lỗi | Cỡ nhóm |
| :--- | :--- | :--- | :--- | ---: |
| `P0` | `baseline` | Prompt RAG có citation thông thường | — (control) | — |
| `P1` | `strict_grounding` | *"Do not add background knowledge, assumptions, or plausible details."* | Hallucination | ~0 |
| `P2` | `concise_answer_type` | *"Give one short answer sentence containing only the information type requested… Do not repeat the question, list alternative answers, or add background details."* | Đúng nhưng quá dài + sai answer type | **11/30** |
| `P3` | `event_disambiguated_concise` | *"First identify the passage that best matches all identifying details… Do not combine facts, figures, or dates from different events, articles, or reporting periods."* | Trộn bài / trộn mốc thời gian | 3/30 |

Cả bốn dùng chung một câu từ chối chuẩn hóa: `I cannot find this information in the provided context.` Đây chính là chuỗi xuất hiện trong các abstention ở §8, và là hạt giống của Phase 3.

### 7.2. Kết quả (depth 5)

QA F1 và token đo trên 80 câu screening; RAGAS đo trên cùng một tập con cố định 20 câu.

| Prompt | QA F1 | Answer Correctness | Faithfulness | Answer Relevancy | Citation F1 | Citation Validity | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 0,2417 | 0,7118 | **1,0000** | **0,8131** | 0,7808 | 0,9750 | 2.997 |
| P1 | 0,2570 | 0,6885 | 0,9750 | 0,7426 | 0,7863 | 0,9625 | 2.732 |
| **P2** | **0,3999** | **0,8363** | **1,0000** | 0,7388 | 0,8133 | 0,9750 | **1.470** |
| P3 | 0,2790 | 0,7190 | 0,9750 | 0,8153 | **0,8146** | 0,9625 | 1.912 |

**Kết luận:**

- **P2 là prompt mới duy nhất qua hết guardrail.** Hơn P0 +0,1245 Answer Correctness, CI95 theo cặp gom cụm bài báo [+0,0325; +0,2296]. Tạo 9 exact match (P0: 0) và giảm ~51% output token.
- **P1 thất bại một cách có ích.** Faithfulness của P0 đã là 1,000 trên tập screening và 0,9761 trên full dev — P1 nhắm vào một vấn đề **không tồn tại**, và mất correctness khi làm vậy. Kết quả âm này xác nhận chẩn đoán §5.3.
- **P3 đúng hướng nhưng chưa đủ mạnh.** Citation F1 và Answer Relevancy tốt nhất, nhưng correctness chỉ 0,7190 so với 0,8363 của P2. Đáng chú ý: P3 **chứa cả chỉ thị của P2** cộng thêm phần chọn đoạn, mà correctness lại thấp hơn hẳn — có thể vì chỉ thị dài làm loãng mệnh lệnh súc tích. RAGAS chỉ chấm 20 câu nên đây là quan sát, chưa phải kết luận.

**Prompt thắng là prompt nhắm vào nhóm lỗi lớn nhất.** Không phải may.

> [!NOTE]
> Tập screening 80 câu và tập calibration 20 câu được chốt **một lần** với seed 42, phân tầng theo loại câu hỏi và theo `gold_in_top5`, cân bằng theo bài báo, và **không đổi** giữa các run. Vì RAGAS chỉ chấm 20 câu, **không được đọc số RAGAS screening như số final**. P0 ở depth 5 tái sử dụng từ baseline, không gọi API lại — nên ba thư mục screening có ba bản P0 giống hệt nhau.

---

## 8. Phase 2B.2 — Sàng lọc độ sâu context

Theo đúng plan §4.2, depth được quét trên **hai** prompt tốt nhất của vòng P — P0 làm control và P2 làm ứng viên — nên hiệu ứng của depth tách được khỏi hiệu ứng của prompt.

| Prompt | Depth | QA F1 | Answer Correctness | Faithfulness | Answer Relevancy | Citation F1 | Citation Validity | Output tokens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P0 | 1 | 0,2543 | 0,7063 | 0,9750 | 0,7365 | 0,7250 | 0,8167 | 2.246 |
| P0 | 3 | 0,2527 | 0,7046 | 0,9875 | 0,7667 | 0,7792 | 0,9500 | 2.698 |
| P0 | 5 | 0,2417 | 0,7118 | **1,0000** | **0,8131** | 0,7808 | 0,9750 | 2.997 |
| P2 | 1 | **0,5809** | 0,8671 | 0,9500 | 0,4825 | 0,5500 | 0,6125 | **779** |
| P2 | 3 | 0,5311 | **0,8795** | 0,9250 | 0,6806 | 0,8125 | 0,9500 | 1.047 |
| P2 | 5 | 0,3999 | 0,8363 | **1,0000** | 0,7388 | **0,8133** | **0,9750** | 1.470 |

Ba điều đọc được:

1. **Depth gần như không tác động lên P0** — AC dao động 0,7046–0,7118 qua cả ba mức. Prompt dài, trả lời dài thì thêm hay bớt context không đổi được nhiều.
2. **Depth tác động rất mạnh lên P2** (0,8363–0,8795). Khi câu trả lời bị ép ngắn, chọn đúng bằng chứng nào để đọc trở thành yếu tố quyết định.
3. **Citation Validity xuống theo depth ở cả hai prompt** (P0: 0,9750 → 0,8167; P2: 0,9750 → 0,6125). Đây là đặc tính của **context bị cắt**, không phải của prompt.

### Phase 1 đã định giá sẵn việc cắt context

Đường cong Hit@k của cấu hình khóa cho biết chính xác mỗi mức depth tốn bao nhiêu bằng chứng — **không tốn thêm một lệnh gọi API nào**:

| Depth | Hit@k (`resolved`) | Số câu **mất sạch bằng chứng** so với depth 5 | Recall@k |
| ---: | ---: | ---: | ---: |
| 1 | 0,8221 | **38** / 281 (13,5%) | 0,8149 |
| 3 | 0,9324 | **7** / 281 (2,5%) | 0,9288 |
| 5 | 0,9573 | 0 (mốc) | 0,9555 |

Đối chiếu với quan sát thực tế: **depth 3 mất bằng chứng của đúng 7 câu**, và run finalist P2-depth3 ghi nhận **đúng 7 câu abstention không kèm citation** — 5 câu vốn không có gold trong top 5, 2 câu có gold ở rank 4 nên bị depth 3 cắt. Con số khớp chính xác.

**Bài học phương pháp:** độ sâu context không phải cái núm miễn phí. Kết quả sàng lọc depth lẽ ra **dự đoán được từ Phase 1** trước khi chi tiền API. Với hệ thống sau, đọc Hit@k trước rồi mới chọn depth để thử.

Depth 1 bị loại. Depth 3 và depth 5 của P2 được đưa lên chạy đủ 281 câu. **Bảng screening không tự quyết định winner** — nó chỉ chọn ứng viên.

---

## 9. Phase 2B.3 — Finalist và quyết định winner

Cả hai finalist chạy đủ **281/281**, coverage generation và RAGAS đều 100%, không có record failed, missing hay trùng.

### 9.1. Số liệu

| Metric | P0-depth5 | P2-depth3 | P2-depth5 |
| :--- | ---: | ---: | ---: |
| Exact Match | 0,0000 | **0,2633** | 0,0925 |
| Token F1 | 0,2648 | **0,5549** | 0,4191 |
| **Answer Correctness** | 0,6308 | **0,7711** | 0,7350 |
| Faithfulness | 0,9761 | 0,9561 | **0,9801** |
| Citation F1 | 0,8025 | 0,8381 | **0,8391** |
| Citation Validity | **0,9893** | 0,9751 | 0,9858 |
| Answer Relevancy | **0,7950** | 0,5694 | 0,7092 |
| Input / output token | 634.008 / 10.754 | **400.241 / 3.883** | 640.752 / 5.035 |

### 9.2. Áp guardrail

| Guardrail (ngưỡng) | P2-depth3 | P2-depth5 |
| :--- | :--- | :--- |
| Coverage ≥ 95% | 100% ✅ | 100% ✅ |
| Faithfulness giảm ≤ 0,02 | **−0,020029** ❌ | +0,003992 ✅ |
| Citation F1 giảm ≤ 0,01 | +0,035567 ✅ | +0,036635 ✅ |
| Citation Validity giảm ≤ 0,01 | **−0,014235** ❌ | −0,003559 ✅ |
| **Kết luận** | **BỊ LOẠI** (2/4 trượt) | **HỢP LỆ** (4/4 qua) |

> [!IMPORTANT]
> **Winner: P2-depth5.** Là cấu hình hợp lệ duy nhất, nên quy tắc "chọn Answer Correctness cao nhất trong nhóm hợp lệ" không cần tới bước tie-break nào.
>
> P2-depth3 đạt Answer Correctness **cao hơn 0,0360** nhưng không được chọn. Đây chính là tình huống bộ guardrail được thiết kế để xử lý: nó chặn việc đổi tính trung thực lấy điểm số.

### 9.3. Kiểm định theo cặp

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

Hai dòng "không tách được" ở đây là **kết quả tốt**: chúng nói rằng P2-depth5 **không hề làm giảm** grounding và citation validity — mạnh hơn hẳn so với việc chỉ nằm trong ngưỡng dung sai.

**P2-depth3 so với P0** — mức tụt guardrail là thật:

| Metric | Hiệu | CI95 của hiệu | Phán quyết |
| :--- | ---: | :--- | :--- |
| Answer Correctness | +0,1403 | [+0,1131; +0,1674] | **có ý nghĩa** |
| Faithfulness | **−0,0200** | **[−0,0372; −0,0048]** | **có ý nghĩa** (giảm) |
| Citation Validity | **−0,0142** | **[−0,0282; −0,0034]** | **có ý nghĩa** (giảm) |
| Citation F1 | +0,0356 | [−0,0011; +0,0730] | không tách được |
| Answer Relevancy | −0,2256 | [−0,2679; −0,1857] | **có ý nghĩa** (giảm) |

> [!CAUTION]
> **Mức tụt của P2-depth3 không phải nhiễu đo.** CI95 của cả Faithfulness lẫn Citation Validity đều **không chứa 0**. Con số Faithfulness chỉ vượt ngưỡng 0,02 đúng 0,00003, dễ khiến người ta tưởng đây là chuyện làm tròn — không phải. Cái sát ngưỡng là *vị trí* của mức tụt so với vạch quy ước; *sự tồn tại* của mức tụt đã được chứng minh chắc chắn.

**P2-depth5 so với P2-depth3** — bản chất của đánh đổi:

| Metric | Hiệu (d5 − d3) | CI95 của hiệu | Phán quyết |
| :--- | ---: | :--- | :--- |
| Answer Correctness | −0,0360 | [−0,0554; −0,0190] | **có ý nghĩa** |
| Faithfulness | **+0,0240** | [+0,0056; +0,0440] | **có ý nghĩa** |
| Answer Relevancy | +0,1398 | [+0,1081; +0,1733] | **có ý nghĩa** |
| Exact Match | −0,1708 | [−0,2159; −0,1246] | **có ý nghĩa** |
| Citation F1 | +0,0011 | [−0,0235; +0,0242] | không tách được |
| Citation Validity | +0,0107 | [−0,0037; +0,0269] | không tách được |

Không cấu hình nào tốt hơn tuyệt đối. Depth 3 mua correctness bằng grounding; depth 5 làm ngược lại. **Bộ quy tắc đăng ký trước là thứ phân xử** — không phải sở thích của nhóm sau khi đã nhìn thấy cả hai cột số.

### 9.4. Cái giá phải trả

Depth 5 tốn input token cao hơn depth 3 **60%** (640.752 so với 400.241) và output token cao hơn 30%. Đây là chi phí có thật của việc giữ nguyên tính trung thực, và phải được nêu ra chứ không giấu đi.

Quy tắc 3 và 4 (ưu tiên chi phí, rồi latency) **không được kích hoạt** vì chỉ có một cấu hình hợp lệ. Nếu về sau có hai cấu hình cùng hợp lệ mà chênh AC dưới 0,01, depth 3 sẽ thắng nhờ chi phí — nhưng đó không phải tình huống hiện tại.

### 9.5. Phân tầng theo retrieval, và macro theo bài báo

Cả hai test plan yêu cầu hai lát cắt bổ sung. Không lát nào tốn thêm một lệnh gọi API — retrieval bị đóng băng nên `hit_rate@5` từng câu đã nằm sẵn trong `scores/*.jsonl`.

**Phân tầng `gold_in_top5`.** Retrieval đưa được bằng chứng ra trước mặt model ở **269/281** câu; **12 câu còn lại thì không**.

| | n | P0 (gốc) | P2-depth3 | **P2-depth5 (winner)** |
| :--- | ---: | ---: | ---: | ---: |
| `gold_in_top5` | 269 | 0,6522 | 0,7971 | **0,7597** |
| `gold_not_in_top5` | **12** | 0,1504 | 0,1885 | **0,1829** |

Đây là kết quả có ý nghĩa phương pháp rõ nhất của §9: **prompt không cứu được một lần truy xuất trượt.** Trên 12 câu không có gold trong context, đổi prompt chỉ nhích Answer Correctness từ 0,1504 lên 0,1829 — và mức 0,18 đó gần như hoàn toàn là điểm ngữ nghĩa cho một câu trả lời từ chối đúng cách, không phải câu trả lời đúng. Ngược lại, trên 269 câu có gold, cùng thay đổi prompt đó tăng **+0,1075**.

Hệ quả cho việc chọn ưu tiên: **trần của tầng sinh bị chặn bởi Hit@5 = 0,9573 của Phase 1.** Muốn vượt qua trần đó thì phải quay lại sửa retrieval, không phải viết lại prompt.

**Macro theo bài báo.** Micro (trung bình theo câu) để một bài đóng góp 12 câu lấn át một bài đóng góp 2 câu. Macro trung bình theo bài trước, rồi trung bình theo 50 bài:

| Cấu hình | Answer Correctness (micro) | (article macro) | Chênh |
| :--- | ---: | ---: | ---: |
| P0 | 0,6308 | 0,6344 | +0,0036 |
| P2-depth3 | 0,7711 | 0,7709 | −0,0002 |
| **P2-depth5** | **0,7350** | **0,7319** | −0,0031 |

Ba mức chênh đều dưới 0,004 — **không có bài báo nào chi phối kết quả**. Winner không đổi dù đọc theo cách nào. Đây là kiểm tra vững chắc, không phải một chỉ số bổ sung.

Số sinh ra từ `scripts/phase2_paired_comparison.py`, ghi vào `paired_significance.json` (`by_stratum`, `answer_correctness_article_macro`).

---

## 10. Đối chiếu với kế hoạch đăng ký trước

### 10.1. Điều kiện chấp nhận

| # | Điều kiện | Nguồn | Trạng thái |
| ---: | :--- | :--- | :--- |
| 1 | Artifact hash, source revision, config, prompt ghi đầy đủ | 2A §6 | ✅ §2 |
| 2 | Development đúng 50 bài / 281 câu; held-out 871 câu chưa bị truy cập | 2A §6 | ✅ |
| 3 | Generation coverage ≥ 95%, mục tiêu 100% | 2A §6 | ✅ 100% |
| 4 | RAGAS coverage ≥ 95% | 2A §6 | ✅ 100% |
| 5 | Mỗi score truy ngược được bằng `question_id`, `article_key`, run fingerprint | 2A §6 | ✅ `scores/*.jsonl` |
| 6 | Không lưu API key trong output/artifact | 2A §6 | ✅ |
| 7 | Baseline không chọn lại retriever/reranker/chunking | 2A §6 | ✅ |
| 8 | Hai finalist đủ coverage deterministic + RAGAS | 2B §10 | ✅ |
| 9 | Winner chọn đúng quy tắc đăng ký, không dùng held-out | 2B §10 | ✅ §9.2 |
| 10 | Dùng thuật ngữ "tối ưu prompt/cấu hình sinh", không tuyên bố fine-tune | 2B §10 | ✅ §1 |
| 11 | `phase2b_winner_decision.json` — winner, hash hai finalist, người duyệt | 2B §5.2 | ✅ `phase2b_winner_decision.json` |
| 12 | Chạy winner một lần trên 284 câu held-out | 2B §10 | ✅ **§11** |
| 13 | **Audit mù 30 cặp P0 ↔ winner, hai reviewer, báo cáo đồng thuận** | 2B §8 | ❌ **chưa làm** |
| 14 | **Lặp 25 câu cố định cho P0 và winner để đo độ ổn định API** | 2B §8 | ❌ **chưa làm** |
| 15 | Báo cáo phân tầng `gold_in_top5` / `gold_not_in_top5` | 2A §3.3, 2B §6 | ✅ §9.5, §11.3 |
| 16 | Báo cáo article-level macro bên cạnh micro | 2A §3.3 | ✅ §9.5, §11.2 |
| 17 | Đóng gói winner để app và benchmark nạp cùng một artifact | 2B §10 | ⏳ còn lại |
| 18 | **Hiệu chuẩn judge trên 20 câu `judge_calibration`** | 2A §4 | ❌ **tập đã có, chưa dùng** |

Còn lại 3 việc: 13, 14, 18. Cả ba đều không chặn con số công bố, nhưng 18 là việc quyết định xem AC 0,7157 đáng tin đến đâu.

### 10.2. Chỗ thực tế lệch với kế hoạch

Test plan là văn bản **đăng ký trước**. Khi thực tế chạy khác plan, cách xử lý đúng là ghi nhận độ lệch — không sửa lùi plan.

**P3 đã được sửa sau khi plan viết xong.**

| Nguồn | P3 là gì |
| :--- | :--- |
| Plan §4.1 | "Kết hợp P1/P2; citation `[i]` bắt buộc cho mỗi khẳng định chính" |
| `phase2_generation_prompts.yaml` | `event_disambiguated_concise` — chọn đoạn khớp sự kiện trước khi trả lời |

Bản đã chạy là bản trong YAML. Việc sửa **có cơ sở** — nó nhắm vào phát hiện distractor collision của EDA §7 — nhưng plan cần một ghi chú sửa đổi có ngày tháng. P3 không được chọn nên độ lệch này không ảnh hưởng winner.

---

## 11. Held-out — con số công bố

Chạy ngày 2026-09-06, cấu hình **P2-depth5**, **đúng một lần**, trên 284 câu / 50 bài chưa từng bị chạm tới.

### 11.1. Bằng chứng đăng ký trước

Điều khiến con số này dùng được không phải là bản thân nó, mà là dấu vết chứng minh nó không bị chọn lại sau khi nhìn thấy kết quả:

| Mốc | Thời điểm | Bằng chứng |
| :--- | :--- | :--- |
| Chốt danh sách 284 id (seed 46) | trước cả Phase 2B | `heldout_ids_sha256 = 09a04e57…` |
| Ký duyệt winner | **2026-09-06 13:31:30Z** | `decision_sha256 = e2fa5915…`, reviewer `thomas200905` |
| Bắt đầu run held-out | **2026-09-06 14:23:24Z** | `heldout_access.json` |
| Hoàn tất | 2026-09-06 18:08:53Z | coverage 284/284, 0 lỗi |

Quyết định đóng băng **trước run 52 phút**. Kiểm lại 5 hash trong `heldout_access.json`: `decision`, `heldout_ids`, `predictions`, `judge_results` đều khớp. `retrievals_sha256` khớp với bản trong `heldout_trace/` — bản sao thứ hai dưới `runs/` là một lần tuần tự hóa lại, lệch 26 KB; không ảnh hưởng kết quả nhưng ghi lại để khỏi hiểu nhầm.

Bản ghi quyết định cũng tự khai `"heldout_outputs_accessed_for_selection": false`.

### 11.2. Kết quả

| | Dev (281) | **Held-out (284)** | article macro |
| :--- | ---: | ---: | ---: |
| **Answer Correctness** | 0,7350 | **0,7157** | 0,7230 |
| Exact Match | 0,0925 | 0,1092 | — |
| Token F1 | 0,4191 | 0,4313 | 0,4370 |
| Faithfulness | 0,9801 | 0,9249 | 0,9318 |
| Citation F1 | 0,8391 | 0,7289 | 0,7315 |
| Citation Validity | 0,9858 | 0,9437 | 0,9483 |
| Answer Relevancy | 0,7092 | 0,6485 | 0,6477 |
| **Hit@5** (truy xuất) | 0,9573 | **0,8768** | — |

Chi phí: 0,169 USD sinh + 0,768 USD chấm = **0,937 USD**. Latency tổng P50 1.957,5 ms (so với 2.770,8 ms của baseline P0 — P2 sinh câu ngắn hơn nên nhanh hơn).

Micro và macro lệch dưới 0,008 ở mọi metric → **không bài báo nào chi phối số công bố**.

Điểm cần chú ý ngay: Answer Correctness chỉ tụt **0,0193**, nhưng Hit@5 tụt **0,0805**. Tập held-out khó hơn hẳn ở tầng truy xuất — không phải ở tầng sinh.

### 11.3. Phân tầng — và phát hiện quan trọng nhất

| Held-out | n | Answer Corr. | Token F1 | Citation F1 | Citation Validity | Faithfulness |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_in_top5` | 249 | **0,7689** | 0,4694 | 0,8313 | 0,9839 | 0,9552 |
| `gold_not_in_top5` | 35 | 0,3375 | 0,1602 | **0,0000** | 0,6571 | 0,7095 |

Ba điều đọc ra được:

1. **Trên các câu truy xuất làm đúng việc, held-out (0,7689) *cao hơn* development (0,7597).** Prompt P2 tổng quát hóa được sang dữ liệu chưa từng thấy. Không có dấu hiệu overfit.
2. **Tỉ lệ truy xuất trượt tăng gần ba lần:** 12/281 (4,3%) → 35/284 (12,3%). Đây là toàn bộ nguyên nhân của khoảng cách dev → held-out.
3. **Citation F1 trên nhóm trượt bằng đúng 0,0000** — theo định nghĩa, vì không có đoạn đúng nào trong context để mà trích dẫn nên recall bằng 0. Faithfulness cũng tụt xuống 0,7095: không có bằng chứng thì model bám vào đoạn sai.

> **Toàn bộ khoảng cách development → held-out là câu chuyện của truy xuất, không phải của prompt.**

Kết luận này chỉ rút ra được vì hai phase dùng chung một vết truy xuất đóng băng và mọi điểm số đều truy ngược được về từng câu hỏi (§1).

### 11.4. Hệ quả cho việc chọn ưu tiên

Với Hit@5 = 0,8768, khoảng **12% số câu held-out không có đường nào để trả lời đúng**, bất kể prompt viết thế nào. Đầu tư tiếp vào prompt engineering sẽ cho lợi ích giảm dần.

Đòn bẩy còn lại, theo thứ tự:

1. **Query rewriting** — RAG path hiện chưa có bước này, mà EDA cho thấy 11,3% câu hỏi vẫn phụ thuộc ngữ cảnh ngay cả sau khi resolve.
2. **Truy xuất** — contextual chunking đã bị loại có bằng chứng (§5.3 Phase 1), nhưng hướng dense + reranker mạnh hơn chưa được thử lại sau khi sửa lỗi tái lập.
3. **Abstention (Phase 3)** — 35 câu ở nhóm trượt là ca dùng trực tiếp: hệ thống vẫn trả lời chúng thay vì nói không biết.

**Artifact:** `docs/reports/phase2/heldout/` (access record, summary micro+macro, subgroups, per-question và per-article scores) · `docs/reports/phase2/scores/p2_d5_heldout.jsonl`.

---

## 12. Kết luận

1. **Winner Phase 2B là P2-depth5**, Answer Correctness 0,7350 (+0,1043 so với baseline, CI95 [+0,0844; +0,1240]), không làm giảm grounding hay citation validity.
2. **Prompt đúng dạng là đòn bẩy lớn nhất ở tầng sinh.** Không phải vì prompt engineering khéo, mà vì gold NewsQA là span ngắn — P2 khớp với hình dạng của nhãn.
3. **Ít context giúp giảm token và distractor, nhưng depth quá thấp làm mất bằng chứng thật** — và Phase 1 đã định giá chính xác mức mất đó trước khi thí nghiệm chạy.
4. **Bộ guardrail đăng ký trước đã làm đúng việc của nó**, loại một cấu hình có điểm cao hơn nhưng đánh đổi tính trung thực.
5. **Chưa có con số nào để công bố.** Mọi số ở trên đo trên tập tinh chỉnh 281 câu, lệch lạc quan theo cấu trúc. Số công bố là số held-out 284 câu, và held-out chưa được chạm vào.
6. **Phase 2 chưa đóng được** cho tới khi xong bốn việc 12–16 ở §10.1.
