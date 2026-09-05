# Phase 2: End-to-End RAG Baseline

## Mục tiêu

Thiết lập một baseline end-to-end duy nhất sau khi khóa retrieval ở Phase 1.
Baseline chạy trên tập `resolved` của development partition; final-test tiếp tục
được giữ kín để tránh điều chỉnh hệ thống theo dữ liệu kiểm thử.

## Cấu hình khóa

| Thành phần | Cấu hình |
|---|---|
| Corpus | 11.064 bài báo |
| Câu hỏi | Resolved, deduplicated, development (50 bài, 281 câu) |
| Chunking | Recursive `512/64`, khóa từ Round 3 |
| Retrieval | BGE-M3 sparse, `top_k=20` |
| Reranker | `BAAI/bge-reranker-large`, top 5, batch 8 |
| Generator | `gemini-3.1-flash-lite`, `reasoning_effort=minimal`, tối đa 512 output tokens |
| RAGAS judge | `accounts/fireworks/models/glm-5p3-flash` qua Fireworks |
| Judge reasoning | `reasoning_effort=low`, tối đa 2.048 output tokens |
| Judge runtime | timeout 300 giây, batch 1, 1 worker, 3 SDK retries |

Notebook thực thi:

- Kaggle: `notebooks/Tests/09_phase_2_e2e_baseline_kaggle.ipynb`;
- Google Colab: `notebooks/Tests/11_phase_2_e2e_baseline_colab.ipynb`.

Cả hai môi trường cần hai secret:

- `GEMINI_API_KEY_1`: free-project key chỉ dùng cho 281 generation requests;
- `FIREWORKS_API_KEY`: key dùng cho RAGAS judging.

Artifact là public nên `HF_TOKEN` không bắt buộc. Có thể khai báo token read-only
để tăng độ ổn định khi tải, nhưng token không được ghi vào output.

Generation requests được giãn tối thiểu `4,2` giây để phù hợp giới hạn 15 RPM.
Hai provider key chỉ được inject vào đúng subprocess và không nằm trong checkpoint.

Artifact chính thức là public dataset repo
`ThomasAnderson2009/newsqa-rag-phase2-locked-v2`, tag
`locked-bge-m3-512-64-deduplicated-v2`, commit
`bb73e682f472933c212f2c6a3f9575c652b280fd`. File ZIP tại
`artifacts/locked-bge-m3-512-64-deduplicated-v2/locked-bge-m3-512-64-deduplicated-v2.zip`
có SHA-256
`fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24`.
Bundle chứa 22.766 chunks và 1.152 câu resolved đã semantic-deduplicate.
Notebook kiểm tra checksum và manifest trước khi chạy; artifact không hợp lệ
làm run dừng, không fallback sang rebuild từ raw NewsQA.

## Smoke run bắt buộc

Notebook mặc định `RUN_MODE='smoke'`. Chế độ này chạy toàn bộ pipeline trên đúng
5 câu được chọn ổn định bằng seed `42`:

1. BGE-M3 retrieval và BGE-large reranking;
2. 5 Gemini 3.1 Flash-Lite generations bằng free key;
3. deterministic QA/citation scoring;
4. chấm bằng GLM-5.3-Flash với `reasoning_effort=low`;
5. xuất metric coverage, latency, token/cost và score;
6. kiểm tra đủ metric trước khi chuyển sang full mode.

Judge đã được khóa sau smoke ablation trên cùng 5 câu: `low` và `high` đều có
đủ coverage, nhưng `low` dùng ít output token hơn và cho Answer Correctness hợp
lý hơn ở trường hợp khác biệt. File ablation được giữ làm bằng chứng lựa chọn,
không còn là runtime gate của notebook.

Tải file `phase2_e2e_baseline_smoke_results.zip` để kiểm tra trước. Bundle gồm
`predictions.jsonl`, `retrievals.jsonl`, `attempts.jsonl`,
`judge_results.jsonl`, `deterministic_scores.jsonl`, `report.json`, manifests,
experiment spec và `comparison.csv`.

Chỉ đổi `RUN_MODE='full'` sau khi smoke result được duyệt. Smoke và full dùng
experiment ID, result directory và checkpoint khác nhau nên không làm nhiễm cache
prediction/judge của nhau.

## Quy trình

1. Đọc và xác minh kết quả Phase 1; khóa
   `512/64 + BGE-M3 sparse + BGE-large` theo
   `phase_2_baseline_test_plan.md`.
2. Tải locked artifact từ Hugging Face theo immutable tag, kiểm tra checksum và
   rebase manifest paths cho môi trường đang chạy.
3. Chạy retrieval, reranking và Gemini generation. `predictions.jsonl` và
   `attempts.jsonl` cho phép dừng/chạy tiếp mà không gọi lại câu đã thành công;
   generation dùng riêng free key và pacing `4,2` giây.
4. Tính EM, token F1, citation metrics, retrieval metrics, latency và token usage.
5. Chạy RAGAS pilot 25 câu bằng GLM-5.3-Flash `low`. Sau khi kiểm tra pilot,
   chạy tiếp toàn bộ câu generation thành công bằng cùng judge fingerprint.
6. Gộp score RAGAS theo từng câu, tính mean, article macro và bootstrap CI 95%,
   rồi xuất báo cáo và ZIP kết quả.

Kaggle giữ checkpoint trong working output. Colab sao lưu checkpoint và result
bundle vào `MyDrive/newsqa_phase2/`; cả hai đều resume theo run fingerprint và
không gọi lại generation/judgment đã thành công.

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
- Generator và judge phải dùng đúng provider/key đã khai báo; không fallback
  ngầm sang model hoặc provider khác.
- Không sử dụng final-test trước khi khóa các quyết định Phase 2 tiếp theo.
