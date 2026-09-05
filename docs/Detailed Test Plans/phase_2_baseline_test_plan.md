# Kế Hoạch Chi Tiết - Giai Đoạn 2A: Baseline RAG End-to-End

## 1. Mục tiêu

Thiết lập một baseline end-to-end có thể tái lập cho pipeline RAG đã khóa sau
Phase 1:

```text
Câu hỏi resolved
  -> BGE-M3 learned sparse retrieval (top 20)
  -> BGE-large reranking (top 5)
  -> Gemini 3.1 Flash-Lite
  -> Câu trả lời kèm citation
  -> Deterministic metrics + RAGAS/GLM judge
```

Baseline trả lời bốn câu hỏi:

1. Hệ thống trả lời đúng đến mức nào trên câu hỏi NewsQA đã resolved?
2. Câu trả lời có bám sát context và citation có trỏ đúng evidence không?
3. Lỗi đến từ retrieval hay generation?
4. Chi phí, latency và độ ổn định của một lần chạy là bao nhiêu?

Phase 2A chỉ đo **một cấu hình đã khóa**. Mọi lựa chọn prompt và context depth
được thực hiện ở Phase 2B trong
[`phase_2_generation_tuning_plan.md`](phase_2_generation_tuning_plan.md).

## 2. Hợp đồng thực nghiệm chung của Phase 2

### 2.1. Dataset và artifact

| Thuộc tính | Giá trị khóa |
|---|---|
| Raw dataset | `MatchaMacchiato/newsqa_200_11064_v2.0.0` |
| Raw revision | `b81c8db6847a23272665946c0c43c72e9a212fd9` |
| Locked artifact repo | `ThomasAnderson2009/newsqa-rag-phase2-locked-v2` |
| Artifact revision | tag `locked-bge-m3-512-64-deduplicated-v2` |
| Artifact commit | `bb73e682f472933c212f2c6a3f9575c652b280fd` |
| Artifact ZIP | `artifacts/locked-bge-m3-512-64-deduplicated-v2/locked-bge-m3-512-64-deduplicated-v2.zip` |
| ZIP SHA-256 | `fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24` |
| Corpus | 11.064 bài báo; 22.766 chunks |
| Primary evaluation set | 200 bài báo; 1.152 semantic-deduplicated `resolved` questions |
| Supplementary set | 1.336 full `resolved` questions, chỉ dùng sensitivity analysis |

Artifact gồm `chunks.jsonl`, deduplicated `testset_resolved.jsonl`, BGE-M3
sparse index, Chroma fallback và `bundle_manifest.json`. Deduplication đã được
human-review: 184 câu trùng ngữ nghĩa trong cùng bài được gộp vào 155 clusters;
representative giữ hợp nhất `accepted_answers`, evidence spans và relevant
chunk IDs. Phase 2 tải artifact đã khóa, không rebuild index trong mỗi run.

Artifact `locked-bge-m3-512-64-v2` cũ chứa full set 1.336 câu nên không phải
input chính thức của Phase 2. Artifact deduplicated đã được build và kiểm tra:
toàn bộ `relevant_chunk_ids` tồn tại, BGE-M3 postings và Chroma đều có 22.766
chunk IDs, và các file index khớp byte-for-byte với parent manifest. Không dùng
deduplicated testset v1 gắn với corpus 19.263 chunks.

### 2.2. Partition

Repository chia theo `article_key`, shuffle với seed `42`, sau đó lấy 50 bài
đầu cho development:

| Partition | Số bài | Số câu hỏi resolved | Mục đích |
|---|---:|---:|---|
| Development | 50 | 281 | Baseline và tuning |
| Held-out final | 150 | 871 | Đánh giá một lần sau khi khóa Phase 2B |
| Tổng primary | 200 | 1.152 | Semantic-deduplicated resolved set |

Đây là cùng article-level split đã dùng ở Phase 1. `281` không phải giới hạn
`n_eval`: đó là toàn bộ semantic targets thuộc 50 bài development sau
deduplication; 871 câu còn lại thuộc held-out. Mỗi run phải lưu danh sách
`article_ids` và `question_ids` để không phụ thuộc vào thứ tự file.

### 2.3. Cấu hình pipeline

| Thành phần | Giá trị khóa |
|---|---|
| Question variant | `resolved` |
| Chunking | recursive, size `512`, overlap `64` |
| Retriever | BGE-M3 learned sparse (`BAAI/bge-m3`) |
| Candidate pool | `top_k=20` |
| Reranker | `BAAI/bge-reranker-large`, batch size `8` |
| Context | 5 chunks đầu sau rerank |
| Generator | `gemini-3.1-flash-lite` |
| Generation | `temperature=0`, `max_tokens=512` |
| Generation key | `GEMINI_API_KEY_1` |
| Judge | `accounts/fireworks/models/glm-5p3-flash` |
| Judge provider | Fireworks AI, OpenAI-compatible endpoint |
| Judge key | `FIREWORKS_API_KEY` |
| Judge runtime | timeout `300s`, 1 worker, 3 SDK retries |
| Random seed | `42` |

Generator và judge dùng hai model, provider và credential khác nhau. Secret
không được ghi vào notebook output, manifest hoặc log.

### 2.4. Căn cứ khóa retrieval

Trên 281 câu resolved development của Phase 1, cấu hình 512/64 + BGE-M3 +
BGE-large đạt:

| Hit@5 | MRR@5 | NDCG@5 | Recall@5 | P50 latency |
|---:|---:|---:|---:|---:|
| 0,9573 | 0,8797 | 0,8976 | 0,9555 | 512,7 ms |

Đây là bằng chứng để chọn pipeline, không phải kết quả baseline Phase 2. Phase
2 phải xác nhận lại retrieval metrics trên partition 281 câu của locked
deduplicated artifact trước khi generation.

## 3. Metrics

### 3.1. Retrieval và answer

| Nhóm | Metrics |
|---|---|
| Retrieval | Hit Rate, Recall, MRR, NDCG tại `k=1,5,10,20` |
| Answer xác định | Exact Match, token F1 với `accepted_answers` |
| Answer semantics | RAGAS Answer Correctness, Answer Relevancy |

### 3.2. Grounding và citation

| Nhóm | Metrics |
|---|---|
| Grounding | Faithfulness, Context Precision, Context Recall |
| Citation | Validity, Precision, Recall, F1, Answer Citation Coverage |

Citation validity chỉ xác nhận index tồn tại. Citation precision/recall mới đo
chunk được trích có khớp gold evidence hay không. RAGAS là LLM-as-a-Judge, do
đó phải báo cáo model, provider, prompt/fingerprint, phiên bản và coverage.

### 3.3. Vận hành và thống kê

- Coverage: expected, successful, failed, missing, retry count.
- Latency: mean, P50, P90, P95 cho retrieve, rerank, generate, judge và total.
- Usage: input/output tokens và chi phí ước tính theo bảng giá được pin tại
  thời điểm chạy.
- Báo cáo question-level micro average và article-level macro average.
- Tính 95% confidence interval bằng article-cluster bootstrap.

Không loại câu retrieval miss khỏi điểm end-to-end: gold answer vẫn tồn tại
trong corpus, nên đây là lỗi của pipeline. Chỉ tách nhóm `gold_in_top5` và
`gold_not_in_top5` để chẩn đoán retrieval và generation.

## 4. Chế độ chạy

| Mode | Quy mô | Mục đích |
|---|---:|---|
| Preflight | 1 request/model | Kiểm tra model ID, endpoint, credential và output khác rỗng |
| Smoke | 5 câu cố định | Chạy hết retrieve -> judge và kiểm tra schema |
| RAGAS pilot | 25 câu cố định | Kiểm tra timeout, parse, chi phí và score coverage |
| Development baseline | 281 câu | Baseline chính và mốc so sánh cho Phase 2B |
| Held-out final | 871 câu | Chỉ chạy sau khi khóa winner Phase 2B |

Smoke, pilot và development phải dùng các ID được sinh một lần với seed `42`.
Pilot được phân tầng theo article, loại câu hỏi và trạng thái gold evidence
trong top 5; không chọn lại sau khi xem điểm.

## 5. Quy trình baseline

1. **Preflight artifact:** pin Git commit và HF commit/tag; kiểm tra ZIP và
   từng file theo `bundle_manifest.json`; xác nhận 22.766 chunks, 1.152 câu.
2. **Preflight model:** gọi thử Gemini và GLM. GLM phải dùng Fireworks endpoint,
   `max_tokens>=512`, HTTP thành công và `message.content` khác rỗng.
3. **Khóa partition:** tạo và lưu `partitions.json`; xác nhận 281/871 câu.
4. **Smoke:** chạy 5 câu qua toàn bộ pipeline, deterministic scorer và RAGAS.
5. **Retrieval trace:** chạy 281 câu một lần; lưu top 20, reranked top 5, gold
   mapping và latency. Phase 2B tái sử dụng trace này.
6. **Generation:** sinh 281 answer. Giữ khoảng cách request phù hợp quota của
   free key; retry exponential backoff; cache theo question + prompt + context
   + model fingerprint.
7. **Deterministic scoring:** tính retrieval, EM/F1, citation và coverage trước
   khi gọi judge.
8. **RAGAS pilot:** chạy 25 câu với 1 worker. Chỉ tiếp tục nếu không có schema
   error, output rỗng hoặc lỗi hàng loạt.
9. **Full judge:** chấm 281 câu generation thành công, cho phép resume và không
   chấm lại record đã thành công.
10. **Báo cáo:** tổng hợp overall, article macro, retrieval strata, CI, latency,
    cost và failure analysis. Đóng băng baseline trước Phase 2B.

`predictions.jsonl`, `retrievals.jsonl`, `attempts.jsonl` và
`judge_results.jsonl` là append-only. Duplicate successful records là lỗi;
resume phải bỏ qua record đã thành công.

## 6. Điều kiện chấp nhận

- Artifact hash, source revision, Git commit, config và prompt được ghi đầy đủ.
- Development partition đúng 50 bài/281 câu; held-out 871 câu chưa bị truy cập.
- Generation coverage tối thiểu 95%; mục tiêu 100% sau retry/resume.
- RAGAS coverage tối thiểu 95% trên generation thành công.
- Citation index không hợp lệ được ghi nhận, không bị bỏ qua âm thầm.
- Mỗi score truy ngược được bằng `question_id`, `article_key`, context IDs và
  run fingerprint.
- Báo cáo baseline không chọn lại retriever, reranker hoặc chunking.

## 7. Đầu ra bắt buộc

- `run_manifest.json`, `environment.json`, `partitions.json`;
- `retrievals.jsonl`, `predictions.jsonl`, `attempts.jsonl`;
- `judge_results.jsonl`, `report.json`;
- bảng per-question và summary dạng CSV/JSON;
- đồ thị quality, grounding, citation, latency và cost;
- danh sách failure theo stage và retrieval stratum.

## 8. Điều kiện triển khai trước khi chạy

Code đánh giá đã có nhánh Fireworks, timeout 300 giây, retry và giới hạn output
tối thiểu cho GLM. Tuy nhiên, notebook Phase 2 hiện tại vẫn cần được đồng bộ:

- tải tag `locked-bge-m3-512-64-deduplicated-v2` và kiểm tra đúng SHA-256;
- dùng `accounts/fireworks/models/glm-5p3-flash` qua Fireworks và
  `FIREWORKS_API_KEY` cho judge;
- mô tả artifact và expected coverage cũ.

Phải cập nhật notebook Kaggle/Colab theo hợp đồng trong tài liệu này trước smoke
run. Preflight phải thất bại rõ ràng nếu partition count, artifact hash, model,
provider hoặc endpoint không khớp; không được tự động fallback sang provider khác.
