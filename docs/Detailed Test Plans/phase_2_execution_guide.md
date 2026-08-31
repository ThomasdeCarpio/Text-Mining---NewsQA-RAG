# Phase 2: End-to-End RAG Baseline

## Mục tiêu

Thiết lập một baseline end-to-end duy nhất sau khi khóa retrieval ở Phase 1.
Baseline chạy trên tập `resolved` của development partition; final-test tiếp tục
được giữ kín để tránh điều chỉnh hệ thống theo dữ liệu kiểm thử.

## Cấu hình khóa

| Thành phần | Cấu hình |
|---|---|
| Corpus | 11.064 bài báo |
| Câu hỏi | Resolved, deduplicated, development (50 bài, dự kiến 281 câu) |
| Chunking | Recursive `512/64`, khóa từ Round 3 |
| Retrieval | BGE-M3 sparse, `top_k=20` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 |
| Generator | `gemini-3.1-flash-lite`, tối đa 512 output tokens |
| RAGAS judge | `gemini-3.7-flash`, default medium thinking |
| Judge runtime | batch 5, 1 worker, tối đa 5 attempts |

Notebook thực thi: `notebooks/Tests/09_phase_2_e2e_baseline_kaggle.ipynb`.
Kaggle cần ba secret:

- `HF_TOKEN`: đọc private evaluation dataset;
- `GEMINI_API_KEY_1`: free-project key chỉ dùng cho 281 generation requests;
- `GEMINI_API_KEY`: paid-project key chỉ dùng cho RAGAS judging.

Generation requests được giãn tối thiểu `4,2` giây để phù hợp giới hạn 15 RPM.
Hai Gemini key chỉ được inject vào đúng subprocess và không nằm trong checkpoint.

## Smoke run bắt buộc

Notebook mặc định `RUN_MODE='smoke'`. Chế độ này chạy toàn bộ pipeline trên đúng
5 câu được chọn ổn định bằng seed `42`:

1. BGE-M3 retrieval và MiniLM reranking;
2. 5 Gemini 3.1 Flash-Lite generations bằng free key;
3. deterministic QA/citation scoring;
4. đủ 5 RAGAS metrics bằng Gemini 3.7 paid key;
5. summary, failure diagnostics và raw per-question traces.

Tải file `phase2_e2e_baseline_smoke_results.zip` để kiểm tra trước. Bundle gồm
`predictions.jsonl`, `retrievals.jsonl`, `attempts.jsonl`,
`judge_results.jsonl`, `deterministic_scores.jsonl`, `report.json`, manifests,
experiment spec và `comparison.csv`.

Chỉ đổi `RUN_MODE='full'` sau khi smoke result được duyệt. Smoke và full dùng
experiment ID, result directory và checkpoint khác nhau nên không làm nhiễm cache
prediction/judge của nhau.

## Quy trình

1. Đọc và xác minh kết quả Phase 1; khóa `512/64 + BGE-M3 sparse + MiniLM` theo
   protocol amendment được ghi trong `phase_2_baseline_test_plan.md`.
2. Tải raw evaluation release riêng tư từ Hugging Face, rebuild chunk corpus và
   tạo đúng một BGE-M3 index.
3. Chạy retrieval, reranking và Gemini generation. `predictions.jsonl` và
   `attempts.jsonl` cho phép dừng/chạy tiếp mà không gọi lại câu đã thành công;
   generation dùng riêng free key và pacing `4,2` giây.
4. Tính EM, token F1, citation metrics, retrieval metrics, latency và token usage.
5. Chạy RAGAS pilot 25 câu. Sau khi kiểm tra pilot, chạy tiếp toàn bộ câu generation
   thành công bằng cùng judge fingerprint.
6. Gộp score RAGAS theo từng câu, tính mean, article macro và bootstrap CI 95%,
   rồi xuất báo cáo và ZIP kết quả.

## Metrics báo cáo

- Retrieval: Hit Rate, MRR, Recall, NDCG tại các giá trị K hợp lệ.
- Generation xác định: Exact Match, token F1.
- Citation: validity, precision, recall, F1 và answer citation coverage.
- RAGAS: faithfulness, answer relevancy, context precision, context recall,
  answer correctness.
- Vận hành: success/failed/missing, latency p50/p90/p95, token usage và chi phí
  generation tương đương theo bảng giá đã khóa trong experiment spec.

## Điều kiện chấp nhận

- Có đúng một baseline run; không chạy direct-LLM trong thí nghiệm đầu tiên.
- Generation success rate tối thiểu 95%.
- RAGAS coverage tối thiểu 95% số generation thành công.
- Mỗi score RAGAS có thể truy ngược đến `question_id`, model và judge fingerprint.
- Không lưu API key trong notebook output hoặc artifact.
- Free generator key và paid judge key phải khác nhau; nên thuộc hai Google
  projects khác nhau nếu cần quota tách biệt.
- Không sử dụng final-test trước khi khóa các quyết định Phase 2 tiếp theo.
