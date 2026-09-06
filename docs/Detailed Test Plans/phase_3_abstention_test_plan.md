# Kế Hoạch Thực Nghiệm Chi Tiết - Giai Đoạn 3: Abstention

## 1. Mục tiêu

Đánh giá khả năng hệ thống từ chối trả lời khi context hoặc toàn corpus không có
đủ bằng chứng. Đây là benchmark riêng; không trộn negative vào Phase 1 vì các
metric MRR, Recall và NDCG yêu cầu có gold chunk.

Ba câu hỏi nghiên cứu:

1. Pipeline Phase 2 hiện tại tự abstain tốt đến mức nào?
2. Prompt có output contract có giảm câu trả lời không được hỗ trợ hay không?
3. Reranker-score gate có cải thiện thêm mà không từ chối quá nhiều câu answerable hay không?

## 2. Dataset 200 case

Nguồn nội bộ là `heldout_reserve`: 587 câu thuộc 100 bài chưa dùng để chọn hoặc
đánh giá final Phase 2. Chia article với seed `42`: 70 bài development và 30 bài
final. Split được stratify tối thiểu theo natural retrieval miss đã quan sát để
đủ quota 7/3, sau đó mới tạo các negative tổng hợp. Base question và mọi negative
dẫn xuất luôn cùng partition.

| Case type | Development | Final | Tổng |
|---|---:|---:|---:|
| `answerable_control` | 56 | 24 | 80 |
| `natural_retrieval_miss` | 7 | 3 | 10 |
| `controlled_context_ablation` | 16 | 6 | 22 |
| `removed_article` | 16 | 6 | 22 |
| `external_unanswerable` | 15 | 7 | 22 |
| `counterfactual` | 15 | 7 | 22 |
| `partial_weak_evidence` | 15 | 7 | 22 |
| **Tổng** | **140** | **60** | **200** |

Quy mô 200 đủ để so sánh kết quả abstention tổng thể trong phạm vi đồ án. Kết
quả từng subtype chỉ mang tính chẩn đoán vì final strata có 3-7 case.

### Quy tắc tạo case

- Control phải có gold chunk trong top 3 của retrieval trace đã khóa.
- Natural miss phải thiếu gold trong top 5 và không có accepted-answer text.
- Context ablation loại gold, chunk overlap, answer span và paraphrase đủ để suy ra đáp án.
- Removed article loại toàn bộ physical/canonical chunk trước khi rerank lại.
- External dùng 22 bài NewsQA khác nhau không thuộc corpus 11.064 bài; human-review top 20.
- Counterfactual chỉ thay một subject, event, location, date hoặc quantity.
- Partial evidence giữ đúng chủ đề/sự kiện nhưng thiếu fact được hỏi.
- Ít nhất 60 bài nội bộ khác nhau; mỗi external case dùng một bài khác nhau.

Mỗi dòng authored proposal dùng một trong hai contract tối thiểu:

```json
{"case_type":"counterfactual","base_question_id":"...","question":"...","construction":{"changed_field":"date"}}
```

```json
{"case_type":"external_unanswerable","base_question_id":"external-...","source_article_id":"withheld-...","partition":"development","question":"...","construction":{"source":"withheld_newsqa"}}
```

`changed_field` chỉ nhận `subject`, `event`, `location`, `date` hoặc `quantity`.
External proposal phải phân bổ đúng 15 development và 7 final.

## 3. Review và chống leakage

Codex đề xuất authored cases và metadata; con người quyết định cuối cùng. Mọi
case phải được review. `removed_article`, `external_unanswerable` và
`counterfactual` cần reviewer thứ hai; các nhóm khác cần secondary review trên
mẫu seed cố định ít nhất 20%.

Finalization kiểm tra:

- đúng 200 case và đúng quota 140/60;
- không có base question/article leakage giữa partitions;
- không còn answer/evidence trong scope được đánh giá;
- mọi article overlay loại đủ chunk;
- external article là duy nhất và không có duplicate/near-duplicate trong corpus;
- manifest chứa source hashes, seed, reviewer và SHA-256 của artifact.

## 4. Ba cấu hình

### B0 - Baseline Phase 2

Đọc winner manifest sau Phase 2B, giữ nguyên prompt, context depth, Gemini model
và reasoning. Không dùng rejection gate. Chỉ canonical response
`I cannot find this information in the provided context.` không kèm citation
được tính là abstention.

### B1 - Structured abstention prompt

Giữ retrieval/context như B0, chỉ thay prompt và bắt buộc một trong hai JSON:

```json
{"answerability":"answerable","answer":"Natalie Cole","citations":[1]}
```

```json
{"answerability":"insufficient_evidence","answer":null,"citations":[]}
```

Output sai schema retry tối đa ba lần; sau đó vẫn được ghi là failure, không loại
khỏi denominator.

### B2 - B1 cộng score gate

Dùng `top1_reranker_score`; khi triển khai, pipeline có thể abstain trước
generation nếu score thấp hơn threshold. Trong experiment, B2 được suy ra từ B1:
prediction bị gate reject được đổi thành abstention, prediction còn lại được giữ
nguyên. Vì vậy B2 không phát sinh thêm Gemini request. Gate chỉ áp dụng cho
end-to-end cases; controlled-context cases giữ nguyên prediction B1.

Trên development, quét mọi score quan sát được. Chỉ giữ threshold có
false-abstention rate không quá 10%; sau đó chọn false-answer rate thấp nhất,
tie-break bằng Abstention F1 cao hơn rồi threshold thấp hơn. Threshold được khóa
trước final.

## 5. Trình tự chạy

1. Khóa Phase 2 winner manifest.
2. Dùng notebook `14a_phase_3_abstention_preparation_kaggle.ipynb` để thu source
   retrieval, tạo review queue từ `heldout_reserve` và authored proposals.
3. Review, secondary review, validate và finalize dataset.
4. Thu case-specific retrieval trace sau khi áp dụng overlay.
5. Smoke một case mỗi type, tổng 7 case.
6. Dùng notebook `14b_phase_3_abstention_evaluation_colab.ipynb` để chạy B0 và
   B1 trên 140 development case.
7. Calibrate B2 từ output B1; score B0/B1/B2.
8. Chọn và khóa winner chỉ bằng development.
9. Chạy B0 và B1 trên 60 final case đúng một lần; derive B2 với threshold đã khóa.
10. Công bố final mà không chỉnh prompt hoặc threshold.

Tổng tối đa: `2 x 140 + 2 x 60 = 400` Gemini generation requests. Primary
abstention metrics không cần LLM judge.

## 6. Metrics và quy tắc chọn

Primary metrics: Abstention Precision/Recall/F1, false-answer rate,
false-abstention rate, confusion matrix, coverage và selective risk.

Guardrails trên answerable controls: EM, token F1, Citation Validity/F1,
generation success rate, latency, token và chi phí.

Báo cáo overall, theo case type, và tách `controlled_context` với `end_to_end`.
Bootstrap CI 95% theo base question/article để biến thể cùng nguồn không được xem
là quan sát độc lập.

Cấu hình hợp lệ khi:

1. generation success rate ít nhất 98%;
2. false-abstention rate không quá 10%;
3. answerable token F1 giảm không quá 0,02 so với B0;
4. Citation Validity giảm không quá 0,01.

Trong các cấu hình hợp lệ, chọn false-answer rate thấp nhất. Nếu chênh dưới 0,02,
ưu tiên pipeline đơn giản hơn theo thứ tự B0, B1, B2.

## 7. Artifact và lệnh chuẩn

Tạo proposal queue:

```bash
.venv/bin/python scripts/prepare_abstention_dataset.py prepare \
  --mode compact_200 \
  --locked-root results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2 \
  --question-ids-file phase2b_preparation/question_ids/heldout_reserve.json \
  --development-articles 70 \
  --retrievals phase3/source_reserve_retrievals.jsonl \
  --authored-cases evaluation/abstention/authored_cases.jsonl \
  --output-dir evaluation/abstention/compact_200
```

Sau review, finalize tạo riêng `development_cases.jsonl` và `final_test_cases.jsonl`:

```bash
.venv/bin/python scripts/prepare_abstention_dataset.py finalize \
  --mode compact_200 \
  --review-queue evaluation/abstention/compact_200/review_queue.jsonl \
  --chunks results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2/chunks.jsonl \
  --source-manifest evaluation/abstention/compact_200/manifest.json \
  --output-dir evaluation/abstention/compact_200/final
```

Thu retrieval sau overlay:

```bash
.venv/bin/python scripts/collect_abstention_retrievals.py \
  --cases evaluation/abstention/compact_200/final/cases.jsonl \
  --chunks results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2/chunks.jsonl \
  --sparse-index results/datasets/phase2/data/locked-bge-m3-512-64-deduplicated-v2/bge_m3_sparse.pkl \
  --config configs/config.yaml \
  --run-dir results/phase3/retrievals --progress
```

Prediction collection dùng `--policy phase2_baseline` hoặc
`--policy structured_abstention`. Threshold được khóa bằng
`scripts/calibrate_abstention_threshold.py`; winner được khóa bằng
`scripts/select_abstention_policy.py`.

`prepare` đồng thời tạo `review_queue.jsonl` và
`review_queue_readable.json`. Reviewer có thể sửa các trường review trực tiếp
trong bản JSON phân cấp; lệnh `finalize --review-queue` chấp nhận cả hai định dạng.

Runner chuẩn cho development/final là `scripts/run_phase3_abstention.py`. Chế độ
final bắt buộc `--winner-decision`; nếu threshold không được nhúng trong decision
thì phải truyền thêm `--threshold-decision`.

Mọi run dùng fingerprint, append-only JSONL, retry giới hạn và resume theo
`case_id`. Final-test runner phải từ chối chạy nếu chưa có locked winner và
threshold decision tương ứng.
