# Kế Hoạch Thực Nghiệm Chi Tiết - Giai Đoạn 3: Abstention và Bằng Chứng Yếu

## 1. Mục Tiêu

Đánh giá khả năng hệ thống **không trả lời khi bằng chứng không đủ**, thay vì
dùng các chunk gần nghĩa nhất để tạo một câu trả lời không được hỗ trợ.

Đây là một benchmark riêng, chạy sau Phase 2. Không trộn các câu không có bằng
chứng vào benchmark retrieval Phase 1 vì:

- MRR, Recall, Hit Rate và NDCG cần ít nhất một gold chunk;
- retriever luôn trả về top-k, kể cả khi tất cả kết quả đều không liên quan;
- gán điểm retrieval bằng 0 cho một câu đúng ra phải bị từ chối sẽ trộn lẫn
  **xếp hạng bằng chứng** với **phát hiện không đủ bằng chứng**.

Ba câu hỏi nghiên cứu:

1. Khi gold evidence không nằm trong context, generator có từ chối hay đoán?
2. Khi không có gold evidence trong toàn corpus, pipeline có nhận biết được
   rằng top-k chỉ là các kết quả “ít sai nhất” không?
3. Có thể dùng retrieval/reranker score để từ chối sớm mà vẫn giữ chất lượng
   trên câu trả lời được hay không?

---

## 2. Phạm Vi và Giả Thuyết

### 2.1. Giữ nguyên benchmark hiện tại

Tập NewsQA resolved hiện tại vẫn là benchmark **answerable QA**. Kết quả Phase
1 và Phase 2 tiếp tục được báo cáo độc lập. Abstention suite có manifest,
partition và bảng kết quả riêng.

### 2.2. Hai loại “không có bằng chứng” khác nhau

| Trạng thái | Gold evidence trong corpus | Gold evidence trong top-k | Ý nghĩa |
|---|---:|---:|---|
| Answerable | Có | Có | Pipeline phải trả lời và trích dẫn |
| Retrieval miss | Có | Không | Đánh giá phản ứng của generator khi retrieval thất bại |
| Corpus-unanswerable | Không | Không | Đánh giá rejection end-to-end |

Không được gọi `retrieval miss` là `corpus-unanswerable`: lỗi đầu thuộc retrieval,
lỗi sau là điều kiện dữ liệu.

### 2.3. Giả thuyết chính

- Prompt yêu cầu “không đủ bằng chứng” sẽ giảm hallucination nhưng có thể tăng
  từ chối sai.
- Score threshold giúp từ chối sớm, nhưng threshold phải được hiệu chỉnh trên
  development và khóa trước final-test.
- Kết hợp retrieval score, reranker score và đánh giá sufficiency từ LLM có thể
  tốt hơn từng tín hiệu riêng lẻ.

---

## 3. Hợp Đồng Đầu Ra

Generator phải trả về cấu trúc có thể chấm xác định:

```json
{
  "answerability": "answerable",
  "answer": "Natalie Cole",
  "citations": [1]
}
```

Hoặc:

```json
{
  "answerability": "insufficient_evidence",
  "answer": null,
  "citations": []
}
```

Quy tắc:

- không suy ra fact vượt quá context;
- không citation khi abstain;
- không coi một câu trả lời mơ hồ như “có thể là ...” là abstention;
- output sai schema được tính là lỗi, không tự diễn giải bằng heuristic câu chữ.

---

## 4. Xây Dựng Abstention Suite

### 4.1. Quy mô đề xuất

Pilot trước với `50` mẫu. Sau khi duyệt quy trình, khóa benchmark chính khoảng
`500` mẫu:

| Nhóm | Số mẫu mục tiêu | Nhãn |
|---|---:|---|
| Answerable controls | 200 | `answerable` |
| Natural retrieval miss | 50 | Gold có trong corpus nhưng locked retriever không lấy được |
| Controlled context ablation | 50 | Loại toàn bộ gold/overlap chunk khỏi context |
| Removed article | 50 | Loại toàn bộ source article bằng corpus overlay |
| External unanswerable | 50 | Câu hỏi từ NewsQA article ngoài corpus |
| Counterfactual | 50 | `insufficient_evidence` |
| Partial/weak evidence | 50 | `insufficient_evidence` |

Tỷ lệ này phục vụ so sánh mô hình, không đại diện cho tỷ lệ câu không trả lời
trong thực tế. Báo cáo phải nêu rõ prevalence nhân tạo và không diễn giải
accuracy tổng như performance ngoài production.

### 4.2. Cách tạo từng nhóm

**Answerable controls**

- lấy từ câu resolved đã human-review;
- gold article và gold chunk tồn tại;
- giữ nguyên answer, evidence spans và provenance.

**Natural retrieval miss**

- chỉ chọn miss quan sát được từ retrieval trace của pipeline đã khóa;
- gold evidence vẫn tồn tại trong corpus nhưng không xuất hiện trong top-k;
- không dùng case này để giả lập corpus-unanswerable.

**Controlled context ablation**

- bắt đầu từ câu answerable;
- loại toàn bộ `relevant_chunk_ids` khỏi context đưa cho generator;
- giữ các top-ranked non-gold chunks để mô phỏng retrieval thất bại thực tế;
- đây chỉ là generator test, không đưa vào điểm retrieval end-to-end.

**Removed article**

- không xóa vật lý corpus/index Phase 2;
- lưu `excluded_article_ids` và toàn bộ `excluded_chunk_ids` dưới dạng overlay;
- lúc chạy phải áp dụng filter trước khi chọn top-k cuối cùng.

**External unanswerable**

- lấy câu từ bài báo không được index trong corpus thực nghiệm;
- xác minh target article và các bản duplicate/near-duplicate không có trong
  corpus;
- tìm kiếm toàn corpus và human-review top-k để loại trường hợp vẫn có câu trả
  lời hợp lệ ở một bài khác.

**Counterfactual**

- thay tối thiểu một subject, event, location, date hoặc quantity;
- câu mới phải tự nhiên nhưng fact được hỏi không được bài báo/corpus hỗ trợ;
- không tạo câu vô lý hoặc lộ rõ là câu giả;
- lưu câu gốc, phép biến đổi và lý do không trả lời được.

**Partial/weak evidence**

- giữ các chunk đúng chủ đề/sự kiện nhưng thiếu fact cần để trả lời;
- không được chứa answer span hoặc một paraphrase đủ để suy ra đáp án;
- human-review phải xác nhận rằng abstention là lựa chọn duy nhất có thể bảo vệ.

Vì chunk overlap là `64`, “xóa một chunk” không đủ đảm bảo mất bằng chứng. Mọi
chunk chứa/giao evidence span và chunk lân cận chứa answer hoặc paraphrase đủ
để suy ra answer đều phải được loại và ghi trong overlay.

### 4.3. Chống leakage và duplicate

- partition theo **base article/base question**, không theo biến thể;
- câu gốc và mọi negative dẫn xuất luôn nằm cùng partition;
- không dùng 150 bài final-test hiện tại để tạo hoặc hiệu chỉnh negative;
- chuẩn hóa URL, title và context để phát hiện article duplicate;
- dùng lexical/semantic search để tìm fact tương đương trong distractor corpus;
- Codex/LLM chỉ đề xuất; con người quyết định nhãn cuối.

### 4.4. Schema tối thiểu

```json
{
  "case_id": "...",
  "base_question_id": "...",
  "case_type": "corpus_unanswerable",
  "question": "...",
  "answerability_label": "insufficient_evidence",
  "scope": "full_corpus",
  "source_article_id": "...",
  "source_gold_chunk_ids": [],
  "gold_relevant_chunk_ids": [],
  "provided_context_chunk_ids": [],
  "excluded_chunk_ids": [],
  "excluded_article_ids": [],
  "construction": {},
  "human_review": {
    "decision": "approved",
    "reviewer_id": "...",
    "scope_verified": true,
    "notes": "..."
  },
  "secondary_review": {
    "decision": "approved",
    "reviewer_id": "...",
    "notes": "..."
  }
}
```

Manifest phải ghi dataset revision, corpus hash, seed, cách sampling, số mẫu
theo loại, prompt/proposal version, reviewer, exclusion statistics và SHA-256
của mọi file.

---

## 5. Quy Trình Human Review

Mỗi negative cần được kiểm tra:

1. Câu hỏi rõ ràng, đúng ngữ pháp và không cố tình vô nghĩa.
2. Nhãn được định nghĩa theo đúng scope: `provided_context` hay `full_corpus`.
3. Không có answer/evidence hợp lệ trong scope đó.
4. Với counterfactual, thay đổi là tối thiểu và không làm đổi loại câu hỏi.
5. Top retrieval results không vô tình cung cấp đáp án từ bài khác.
6. Câu hỏi không trùng hoặc paraphrase gần như hoàn toàn với mẫu khác.

Mẫu không chắc chắn được đánh dấu `uncertain`, không ép thành negative. Nên có
reviewer thứ hai cho toàn bộ corpus-unanswerable và một mẫu ngẫu nhiên ít nhất
20% ở các nhóm còn lại; báo cáo tỷ lệ đồng thuận và cách xử lý bất đồng.

---

## 6. Ma Trận Thực Nghiệm

### 6.1. Thí nghiệm A - Generator có context được kiểm soát

Giữ nguyên generator đã khóa ở Phase 2 và so sánh:

1. prompt Phase 2 hiện tại;
2. prompt có output contract và chỉ dẫn abstention;
3. prompt abstention kèm một ví dụ answerable và một ví dụ insufficient.

Chạy trên answerable controls, retrieval-miss và partial-evidence. Mục tiêu là
tách năng lực abstain của generator khỏi chất lượng retriever.

### 6.2. Thí nghiệm B - Retrieval rejection

Chạy retriever/reranker đã khóa, chưa gọi generator. So sánh tín hiệu:

- top-1 retrieval score;
- top-1 reranker score;
- khoảng cách top-1 với top-2/top-5;
- số chunk vượt threshold;
- tổ hợp chuẩn hóa của các tín hiệu trên.

Hiệu chỉnh threshold trên development bằng PR curve; khóa threshold rồi mới
chạy final abstention set. Score giữa các retriever không mặc nhiên cùng thang
đo, nên phải hiệu chỉnh riêng cho từng pipeline.

### 6.3. Thí nghiệm C - End-to-end abstention

```text
Question
  -> retrieve top 20
  -> rerank top 5
  -> evidence-sufficiency decision
       -> sufficient: generate cited answer
       -> insufficient: abstain
```

So sánh tối thiểu:

| Cấu hình | Rejection | Generator abstention |
|---|---|---|
| Baseline | Không | Prompt Phase 2 |
| Prompt-only | Không | Prompt abstention |
| Threshold-only | Retrieval/reranker threshold | Có |
| Sufficiency classifier | LLM hoặc classifier | Có |
| Combined | Threshold + sufficiency classifier | Có |

Chỉ thêm query rewriting hoặc Agentic RAG ở thí nghiệm sau; không thay nhiều
thành phần cùng lúc trong baseline abstention.

---

## 7. Metrics

Quy ước lớp dương là `insufficient_evidence`.

| Metric | Ý nghĩa |
|---|---|
| Abstention precision | Trong các lần từ chối, bao nhiêu trường hợp thật sự không đủ evidence |
| Abstention recall | Trong các negative, hệ thống từ chối được bao nhiêu |
| Abstention F1 | Cân bằng precision và recall của quyết định từ chối |
| False-answer rate | Tỷ lệ negative vẫn nhận câu trả lời |
| False-abstention rate | Tỷ lệ answerable bị từ chối |
| Coverage | Tỷ lệ toàn bộ câu mà hệ thống quyết định trả lời |
| Selective risk | Error rate trên phần câu hệ thống chọn trả lời |
| AUROC / AUPRC | Khả năng phân tách answerable và insufficient bằng confidence liên tục |

Trên tập con `answerable`, tiếp tục báo cáo EM, token F1, citation validity,
citation precision/recall và RAGAS. Trên `corpus-unanswerable`, không tính
MRR/Recall/NDCG; thay bằng false acceptance rate và abstention metrics.

Mọi metric cần:

- kết quả theo từng `case_type`, không chỉ macro tổng;
- bootstrap CI 95% theo `base_question_id` hoặc article để các biến thể dẫn xuất
  không bị xem như quan sát độc lập;
- confusion matrix;
- coverage-risk curve và precision-recall curve;
- latency, token usage và chi phí cho mỗi cấu hình.

---

## 8. Quy Tắc Chọn Cấu Hình

Khóa trước khi xem final-test:

1. False-answer rate trên corpus-unanswerable là metric an toàn chính.
2. Chọn threshold đạt abstention recall mục tiêu trên development.
3. Trong các cấu hình đạt recall mục tiêu, ưu tiên false-abstention thấp hơn.
4. Nếu chất lượng gần nhau, ưu tiên pipeline đơn giản, latency và chi phí thấp.
5. Không chọn cấu hình chỉ vì accuracy tổng cao trên tỷ lệ lớp nhân tạo.

Ngưỡng mục tiêu cụ thể chỉ được đặt sau pilot, nhưng phải được ghi vào manifest
trước final run.

---

## 9. Artifact Cần Triển Khai

Đề xuất bổ sung sau Phase 2:

```text
evaluation/abstention/
  pilot/
    manifest.json
    review_queue.jsonl
    corpus_overlays.jsonl
    validation_report.json
    final/{manifest.json,cases.jsonl,human_approval.json}
  full/
    ...

scripts/
  prepare_abstention_dataset.py
  validate_abstention_dataset.py
  collect_abstention_predictions.py
  score_abstention_predictions.py

configs/experiments/
  phase3_abstention_smoke.yaml
  phase3_abstention_full.yaml

notebooks/Tests/
  phase_3_abstention_evaluation.ipynb
```

Prediction collection phải dùng fingerprint, append-only JSONL, retry có giới
hạn và resume theo `case_id`, giống benchmark hiện tại. Không gọi lại generator
hoặc judge cho record đã thành công với cùng fingerprint.

### 9.1. Lệnh chuẩn bị pilot

```bash
.venv/bin/python scripts/prepare_abstention_dataset.py prepare \
  --mode pilot \
  --locked-root results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2 \
  --retrievals results/phase2/baseline/retrievals.jsonl \
  --authored-cases evaluation/abstention/authored_cases.jsonl \
  --output-dir evaluation/abstention/pilot
```

`authored_cases.jsonl` chứa đề xuất counterfactual và external-unanswerable.
Script tự động tạo controls, natural misses và corpus/context overlays; thiếu
nhóm nào được ghi rõ trong `manifest.deficits`, không được tự bù bằng mẫu yếu.

Sau human review, đổi `human_review.decision` thành `approved`, điền reviewer và
notes, rồi finalize:

```bash
.venv/bin/python scripts/prepare_abstention_dataset.py finalize \
  --mode pilot \
  --review-queue evaluation/abstention/pilot/review_queue.jsonl \
  --chunks results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2/chunks.jsonl \
  --source-manifest evaluation/abstention/pilot/manifest.json \
  --output-dir evaluation/abstention/pilot/final
```

Finalization thất bại nếu chưa approve, sai target count, có gold leakage, xóa
article không đầy đủ hoặc biến thể cùng base question bị chia khác partition.

---

## 10. Trình Tự Thực Hiện

1. Hoàn tất và khóa Phase 2 baseline.
2. Định nghĩa output contract `answerable/insufficient_evidence`.
3. Tạo pilot 50 mẫu từ development, không chạm final-test.
4. Codex/LLM đề xuất negative và metadata kiểm tra.
5. Human review, sửa hoặc loại mẫu không bảo vệ được nhãn.
6. Chạy generator-controlled pilot và kiểm tra score thủ công.
7. Xây benchmark 500 mẫu, deduplicate và đóng manifest.
8. Chia development/final theo base article; khóa final.
9. Hiệu chỉnh prompt/threshold trên development.
10. Khóa cấu hình và chạy final đúng một lần.
11. Báo cáo riêng answerable QA, retrieval rejection và end-to-end abstention.

---

## 11. Tiêu Chí Hoàn Thành

- mọi mẫu có provenance và human approval;
- không có base question/article leakage giữa development và final;
- corpus-unanswerable đã được kiểm tra duplicate và top retrieval results;
- output schema được parse xác định;
- retrieval metrics không được áp dụng sai cho mẫu không có gold chunk;
- có confidence interval và kết quả theo từng loại negative;
- run có thể dừng/chạy tiếp và tái lập từ manifest;
- báo cáo thể hiện đồng thời false-answer rate và chi phí từ chối sai.
