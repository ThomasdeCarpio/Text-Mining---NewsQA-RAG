# Kế Hoạch Thực Nghiệm Chi Tiết - Giai Đoạn 2A: Baseline RAG End-to-End

---

## 1. Mục Tiêu của Baseline

### 1.1. Mục tiêu cốt lõi

Thiết lập một **mốc tham chiếu end-to-end có thể tái lập** cho hệ thống RAG sau
khi đã khóa pipeline truy xuất ở Giai đoạn 1. Baseline đo toàn bộ chuỗi:

```text
Câu hỏi resolved
    -> Chunk 512/64
    -> BGE-M3 sparse retrieval (top 20)
    -> MiniLM cross-encoder reranking (top 5)
    -> Gemini 3.1 Flash-Lite generation
    -> Câu trả lời có citation
    -> Deterministic metrics + RAGAS judge
```

Baseline trả lời ba câu hỏi nghiên cứu:

1. Khi đã nhận đúng pipeline retrieval, hệ thống tạo câu trả lời chính xác và
   bám sát bằng chứng ở mức nào?
2. Citation do mô hình sinh ra có hợp lệ và trỏ đúng gold evidence chunks không?
3. Chi phí vận hành, độ trễ và tỷ lệ hoàn thành của pipeline end-to-end là bao nhiêu?

### 1.2. Phạm vi

- Chạy **một cấu hình RAG duy nhất**, không phải model tournament.
- Chỉ dùng tập `resolved` development để thiết lập baseline.
- Chưa so sánh direct LLM, generator khác, context depth khác hoặc Agentic RAG.
- Không sử dụng final-test để chọn prompt, model hoặc tham số.
- Kết quả baseline là mốc để thiết kế các ablation Phase 2 tiếp theo.

---

## 2. Thiết Kế Thực Nghiệm

### 2.1. Biến được khóa

| Thành phần | Giá trị khóa | Lý do |
|---|---|---|
| Dataset variant | `resolved`, deduplicated | Giảm nhiễu do câu hỏi thiếu chủ thể; phù hợp use case hỏi tin tức độc lập |
| Partition | Development, seed `42` | Cho phép tiếp tục phát triển mà không làm rò rỉ final-test |
| Corpus | 11.064 bài báo | Mô phỏng truy xuất trong corpus có nhiều distractor |
| Retrieval artifact | Private HF tag `phase2-bge-m3-512-64-v1` | Khóa chunks, resolved testset và BGE-M3 postings; kiểm tra checksum trước run |
| Chunking | Recursive, size `512`, overlap `64` | Điểm cân bằng tốt nhất về coverage, ranking và latency trong Round 3 |
| Retriever | BGE-M3 learned sparse | Phương pháp retrieval được chọn từ Phase 1 |
| Initial retrieval | `top_k=20` | Cung cấp đủ candidate cho reranker |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Điểm cân bằng chất lượng - latency tốt hơn reranker lớn |
| Context đưa vào LLM | Top 5 chunks | Giữ cố định retrieval evidence budget |
| Generator | `gemini-3.1-flash-lite` | Model sinh baseline |
| Generator credential | Free-project `GEMINI_API_KEY_1`, interval `4,2` giây | 281 requests nằm trong daily quota và không vượt 15 RPM |
| Output limit | 512 tokens | Đủ cho câu trả lời NewsQA ngắn và citation, tránh output dư thừa |
| Generation prompt | Prompt RAG citation hiện tại của `RAGAgent` | Đo đúng implementation đang dùng trong ứng dụng |
| RAGAS judge | `gemini-3.7-flash` | Judge khác generator, giảm self-evaluation bias |
| Judge credential | Paid-project `GEMINI_API_KEY` | RAGAS có thể tạo nhiều LLM calls cho mỗi question |
| Judge execution | Batch 5, 1 worker, tối đa 5 attempts | Giảm lỗi quota và giữ run ổn định |

### 2.2. Bằng chứng từ kết quả Phase 1

Toàn bộ các run Phase 1 có coverage `100%` trên 281 câu resolved development.
Ba vòng đưa đến quyết định theo chuỗi sau:

| Vòng | Kết luận | Bằng chứng chính |
|---|---|---|
| Round 1 | Chọn BGE-M3 sparse | MRR@5 `0,8131`, cao nhất trong tám retriever không rerank |
| Round 2 | Giữ sparse thay vì dense/hybrid | Sparse + MiniLM đạt MRR@5 `0,8390`, cao hơn hybrid + MiniLM `0,7969` và dense + MiniLM `0,7839` |
| Round 2 | Chọn MiniLM làm cấu hình vận hành | Tại sparse 512/64, MiniLM đạt Hit@5 `95,02%` ở `200,0 ms`; BGE-large đạt `95,37%` nhưng cần `617,3 ms` |
| Round 3 | Khóa chunk 512/64 | Với MiniLM: Hit@5 `95,02%`, MRR@5 `0,8390`, NDCG@5 `0,8658`, P50 `190,3 ms` |

Nếu mục tiêu duy nhất là tối đa hóa MRR, `512/64 + BGE-large` là cấu hình tốt
nhất với MRR@5 `0,8815`. Tuy nhiên, so với `512/64 + MiniLM`, nó chỉ tăng
Hit@5 `0,36` điểm phần trăm trong khi P50 latency tăng khoảng `3,12` lần. Phase
2 baseline vì vậy tiếp tục dùng MiniLM như quyết định Pareto đã đặt ra ở Round 2.

Nguồn kết quả:

- `results/phase1/round1.csv`;
- `results/phase1/round2.csv`;
- `results/phase1/round3.csv`.

### 2.3. Khóa chunking và protocol amendment

Trong nhánh MiniLM của Round 3:

| Chunking | Hit@5 | MRR@5 | NDCG@5 | Recall@5 | P50 latency |
|---|---:|---:|---:|---:|---:|
| 256/32 | 92,88% | 0,8164 | 0,8425 | 92,53% | 139,0 ms |
| **512/64** | **95,02%** | 0,8390 | **0,8658** | **94,84%** | **190,3 ms** |
| 1024/128 | 92,53% | **0,8393** | 0,8612 | 92,53% | 193,9 ms |

Quy tắc xếp hạng ban đầu ưu tiên MRR tuyệt đối nên sẽ chọn `1024/128` vì chênh
lệch `0,00024`. Tuy nhiên, hai bootstrap CI 95% gần như trùng nhau:

- `512/64`: `[0,8057; 0,8736]`;
- `1024/128`: `[0,8037; 0,8755]`.

CI chồng lấp không phải một equivalence test chính thức, nhưng kết quả hiện tại
không chứng minh được lợi ích thực tế của mức chênh MRR này. Trước khi chạy bất
kỳ generation experiment hoặc xem final-test, protocol được sửa minh bạch như
sau:

1. Nếu chênh MRR@5 nhỏ hơn `0,001` và CI 95% chồng lấp, xem hai cấu hình là
   **đồng hạng theo ngưỡng thực dụng**, không diễn giải là tương đương thống kê.
2. Trong nhóm đồng hạng, ưu tiên Hit Rate@5, Recall@5 và NDCG@5 cao hơn, sau đó
   mới xét latency.

Theo quy tắc sửa đổi, **chunk 512/64 được khóa**. Nó tăng Hit@5 `2,49` điểm phần
trăm và Recall@5 `2,31` điểm phần trăm so với 1024/128, đồng thời NDCG cao hơn
và P50 thấp hơn. Amendment này phải được ghi trong `retrieval_lock.json` và báo
cáo cuối để tránh trình bày nó như một tiêu chí đã định trước từ ban đầu.

Sau thời điểm khóa, không được đổi chunking, retriever hoặc reranker dựa trên
kết quả generation baseline.

### 2.4. Tập dữ liệu thực nghiệm

| Thuộc tính | Giá trị |
|---|---:|
| Nguồn canonical | Hugging Face private dataset `ThomasAnderson2009/newsqa-rag-evaluation` |
| Revision | `v1.0.0` |
| Tổng corpus | 11.064 bài báo |
| Evaluation articles | 200 bài |
| Tổng câu hỏi sau semantic deduplication | 1.152 câu |
| Development articles | 50 bài |
| Development questions dự kiến | 281 câu resolved |
| Held-out articles | 150 bài còn lại |

Việc chia partition theo **article**, không theo từng question. Vì vậy các câu
hỏi của cùng một bài báo không xuất hiện đồng thời ở development và final-test.

---

## 3. Hệ Thống Metric Đánh Giá

### 3.1. Retrieval metrics

| Metric | Ý nghĩa | Vai trò trong baseline |
|---|---|---|
| Hit Rate@5 | Tỷ lệ câu có ít nhất một gold evidence chunk trong top 5 | Xác định generator có nhận được ít nhất một bằng chứng đúng hay không |
| MRR@5 | Trung bình nghịch đảo thứ hạng của gold chunk đầu tiên | Đo bằng chứng đúng được ưu tiên sớm đến mức nào |
| Recall@5 | Tỷ lệ gold chunks được tìm thấy trong top 5 | Đo độ phủ khi câu hỏi có nhiều evidence chunks |
| NDCG@5 | Chất lượng thứ tự của toàn bộ gold chunks trong top 5 | Phân biệt các danh sách có cùng Hit Rate nhưng thứ tự khác nhau |

Retrieval metrics được giữ trong báo cáo Phase 2 để tách lỗi retrieval khỏi lỗi
generation; chúng không được dùng để chọn lại retriever sau khi baseline bắt đầu.

### 3.2. Answer metrics xác định

| Metric | Cách tính | Giới hạn |
|---|---|---|
| Exact Match (EM) | So khớp câu trả lời sau chuẩn hóa chữ thường, khoảng trắng và dấu câu với mọi `accepted_answers` | Khắt khe với diễn đạt khác nhưng cùng nghĩa |
| Token F1 | Harmonic mean giữa token precision và recall so với accepted answer tốt nhất | Phù hợp hơn EM cho câu trả lời dài hoặc paraphrase nhẹ |

EM và Token F1 không đủ để đo tính đúng ngữ nghĩa của câu trả lời tự nhiên, do
đó phải được báo cáo cùng RAGAS và citation metrics.

### 3.3. Citation metrics

| Metric | Ý nghĩa |
|---|---|
| Citation Validity | Tỷ lệ citation index như `[1]`, `[2]` thực sự tồn tại trong context đã cung cấp |
| Citation Precision | Tỷ lệ cited chunks thuộc tập gold evidence chunks |
| Citation Recall | Tỷ lệ gold evidence chunks được citation bao phủ |
| Citation F1 | Trung bình điều hòa của citation precision và recall |
| Answer Citation Coverage | Tỷ lệ câu trả lời có ít nhất một citation hợp lệ |

Citation hợp lệ về cú pháp chưa chắc hỗ trợ đúng nội dung. Vì vậy validity phải
được đọc cùng precision, recall và RAGAS faithfulness.

### 3.4. RAGAS metrics

| Metric | Câu hỏi metric trả lời |
|---|---|
| Faithfulness | Các phát biểu trong câu trả lời có được contexts hỗ trợ không? |
| Answer Relevancy | Câu trả lời có trực tiếp giải quyết câu hỏi không? |
| Context Precision | Các contexts được đưa vào có tập trung vào thông tin liên quan không? |
| Context Recall | Contexts có bao phủ đủ thông tin cần thiết từ ground truth không? |
| Answer Correctness | Câu trả lời có đúng về ngữ nghĩa so với gold answer không? |

RAGAS là **LLM-as-a-Judge**, do đó không được xem là ground truth tuyệt đối.
Báo cáo phải lưu model judge, judge fingerprint, phiên bản RAGAS, coverage và
bootstrap confidence interval 95%.

### 3.5. Metrics vận hành

| Nhóm | Metrics |
|---|---|
| Coverage | expected, recorded, successful, failed, missing, success rate |
| Latency | Mean, P50, P90, P95, max cho retrieve, rerank, LLM và total |
| Usage | Input tokens, output tokens, total tokens |
| Cost | Chi phí generation ước tính theo giá được khóa trong experiment spec |

Không so sánh latency giữa các run nếu phần cứng hoặc điều kiện API khác nhau.
Môi trường, package versions, Git commit và GPU phải được lưu cùng kết quả.

---

## 4. Quy Trình Thực Nghiệm

### Bước 1: Preflight

1. Pin Git commit, Hugging Face revision và seed.
2. Kiểm tra Kaggle secrets `HF_TOKEN`, `GEMINI_API_KEY_1` và
   `GEMINI_API_KEY`. `GEMINI_API_KEY_1` phải là free generator key;
   `GEMINI_API_KEY` là paid judge key. Hai key không được giống nhau.
3. Gọi thử cả `gemini-3.1-flash-lite` và `gemini-3.7-flash` qua direct REST và
   OpenAI-compatible endpoint.
4. Kiểm tra GPU, dung lượng đĩa, dependency và model access trước khi build index.

### Bước 2: Khóa retrieval

1. Đọc và lưu hash của ba file kết quả Phase 1.
2. Xác nhận cấu hình khóa là `512/64 + BGE-M3 sparse + MiniLM`.
3. Ghi kết quả, CI và protocol amendment tại Mục 2.3 vào `retrieval_lock.json`.
4. Không thay đổi cấu hình trong cùng baseline.

### Bước 3: Materialize corpus và index

1. Tải raw evaluation release từ Hugging Face private repository.
2. Rebuild chunks với chunk size `512`, overlap `64`, strategy `recursive`.
3. Map lại gold evidence spans sang chunk IDs của corpus mới.
4. Build đúng một BGE-M3 sparse index trên toàn bộ 11.064 bài.
5. Chạy variant-manifest preflight để kiểm tra hash của config, chunks, testset
   và index trước khi inference.

### Bước 4: Thu thập RAG predictions

Với mỗi question:

1. Retrieve top 20 chunks.
2. Rerank và giữ top 5 contexts.
3. Gửi question + numbered contexts tới generator.
4. Lưu answer, citation indices, citation chunk IDs, contexts, timings và usage.
5. Nếu request lỗi tạm thời, retry theo exponential backoff, tối đa 5 attempts.
6. Giãn thời điểm bắt đầu các generation requests ít nhất `4,2` giây, kể cả
   retry, để giữ tốc độ dưới 15 RPM.

`predictions.jsonl`, `retrievals.jsonl` và `attempts.jsonl` là append-only. Khi
resume, câu đã thành công không được gọi lại.

Trước full run, phải chạy một experiment ID riêng trên đúng 5 câu (`n_eval=5`,
seed `42`). Smoke run phải đi qua toàn bộ retrieval, reranking, generation,
deterministic scoring và RAGAS; không được dùng mock response hoặc bỏ qua judge.

### Bước 5: Chấm deterministic metrics

Chạy scorer ngay sau generation để kiểm tra:

- coverage và failed stages;
- retrieval metrics;
- EM và Token F1;
- citation metrics;
- latency và token usage.

Nếu generation success rate dưới 95%, phải điều tra quota/API/config trước khi
chi tiền cho full RAGAS judging.

### Bước 6: RAGAS pilot

1. Chọn cố định 25 câu bằng seed `42` từ các generation thành công.
2. Chạy đủ năm RAGAS metrics bằng cùng judge model dự kiến cho full run.
3. Kiểm tra score rows không rỗng, output hợp lệ và không có lỗi quota hàng loạt.
4. Lưu pilot vào cùng `judge_results.jsonl` và cùng judge fingerprint.

Pilot chỉ kiểm tra pipeline judge, không dùng để thay đổi answer hoặc loại câu
khó khỏi tập đánh giá.

### Bước 7: Full RAGAS và tổng hợp

1. Resume judge trên các câu generation thành công còn lại.
2. Retry các judge records bị exhausted sau khi quota được khôi phục.
3. Gộp RAGAS scores vào `deterministic_scores.jsonl` theo `question_id`.
4. Tính mean, article macro average và bootstrap CI 95%.
5. Xuất report, comparison CSV, plots, logs và ZIP kết quả.

---

## 5. Kiểm Soát Sai Lệch và Tính Tái Lập

| Rủi ro | Biện pháp kiểm soát |
|---|---|
| Data leakage | Chỉ development được dùng; final-test giữ kín đến khi khóa Phase 2 |
| Self-judge bias | Generator và judge là hai model khác nhau |
| Cherry-picking | Đánh giá toàn bộ câu generation thành công; báo cáo coverage và failures |
| Question-level leakage | Partition theo article, không partition từng question |
| API interruption | Append-only cache, retry log và checkpoint có thể resume |
| Key/quota contamination | Inject free key chỉ vào generation subprocess và paid key chỉ vào judge subprocess |
| Judge stochasticity | Khóa model, seed, metric set, judge fingerprint và báo cáo CI 95% |
| Config drift | Pin Git commit, dataset revision, manifest hash và experiment spec |
| Protocol drift | Ghi rõ amendment chọn 512/64 trước Phase 2 và không thay đổi sau khi xem generation |
| Hidden cost | Lưu token usage và bảng giá sử dụng khi chạy experiment |

Kết quả quality không có ngưỡng "đạt" được đặt trước vì mục tiêu của baseline là
**đo mốc hiện tại**, không chứng minh một mức chất lượng đã chọn sẵn.

---

## 6. Phân Tích Lỗi

Mỗi trường hợp lỗi được phân vào một trong các nhóm:

| Nhóm lỗi | Dấu hiệu |
|---|---|
| Pipeline/API failure | Prediction hoặc judge record có trạng thái `exhausted` |
| Retrieval failure | Không có gold chunk trong reranked top 5 |
| Context dilution | Có gold chunk nhưng context precision thấp |
| Generation correctness failure | Retrieval hit nhưng EM/F1 và answer correctness thấp |
| Hallucination/faithfulness failure | Answer có nội dung không được contexts hỗ trợ |
| Citation omission | Answer không có citation hợp lệ |
| Citation misattribution | Citation hợp lệ về index nhưng không trỏ tới gold chunk |
| Gold-data limitation | Answer hợp lý nhưng accepted answer hoặc evidence mapping chưa bao phủ |

Báo cáo cần hiển thị ít nhất các ca có `answer_correctness` thấp nhất, retrieval
misses, answer không có valid citation và thống kê failed attempts theo stage.

---

## 7. Điều Kiện Chấp Nhận Run

Run được xem là **hoàn chỉnh về mặt kỹ thuật** khi:

1. Có đúng một cấu hình baseline `512/64 + BGE-M3 sparse + MiniLM` và đúng resolved-development partition.
2. Expected question count là 281, trừ smoke test được ghi rõ.
3. Generation success rate đạt tối thiểu 95%.
4. RAGAS coverage đạt tối thiểu 95% số generation thành công.
5. Không có question ID thành công bị ghi trùng.
6. Per-question deterministic và RAGAS scores truy ngược được bằng `question_id`.
7. Report lưu Git commit, config, dataset revision, model names và coverage.
8. Không có API key trong notebook output, logs hoặc result bundle.
9. Final-test chưa được sử dụng.

Trước các điều kiện full-run trên, smoke run được duyệt khi có đúng 5 expected
questions, không có missing record, mọi generation thành công được RAGAS chấm,
và bundle chứa đủ raw traces để kiểm tra thủ công. Smoke metrics không được dùng
như ước lượng chất lượng chính thức vì cỡ mẫu quá nhỏ.

Nếu coverage chưa đạt, run được giữ để resume; không được impute score bằng 0
hoặc chỉ báo cáo nhóm câu thành công mà không công bố coverage.

---

## 8. Artifact và Kết Quả Kỳ Vọng

### 8.1. Artifact chính

| File | Nội dung |
|---|---|
| `retrieval_lock.json` | Cấu hình 512/64 + BGE-M3 + MiniLM, Phase 1 metrics, CI và protocol amendment |
| `experiment_spec.yaml` | Toàn bộ cấu hình baseline, runtime và pricing |
| `run_manifest.json` | Fingerprint, question IDs và provenance của run |
| `retrievals.jsonl` | Retrieval/reranking traces theo câu hỏi |
| `predictions.jsonl` | Generated answers, citations, contexts, latency và usage |
| `attempts.jsonl` | Audit log của mọi API attempt |
| `judge_results.jsonl` | Per-question RAGAS scores và judge fingerprint |
| `deterministic_scores.jsonl` | Per-question retrieval, QA, citation và merged RAGAS scores |
| `report.json` | Báo cáo tổng hợp machine-readable |
| `report_summary.txt` | Bản tóm tắt dễ đọc |
| `comparison.csv` | Mean, article macro, CI 95%, latency và estimated cost |
| `phase2_smoke_resume_checkpoint.zip` / `phase2_full_resume_checkpoint.zip` | Artifact nhỏ để resume đúng run mode trên session mới |
| `phase2_e2e_baseline_smoke_results.zip` / `phase2_e2e_baseline_full_results.zip` | Bundle kết quả theo run mode để tải và chia sẻ |

### 8.2. Kết quả dùng trong báo cáo

1. Bảng baseline gồm retrieval, answer, citation và năm RAGAS metrics.
2. Bootstrap CI 95% và article macro score cho các metric theo câu hỏi.
3. Bảng latency breakdown và token/cost usage.
4. Biểu đồ chất lượng generation/citation/RAGAS.
5. Danh sách failure cases đại diện và phân tích nguyên nhân theo pipeline stage.
6. Một baseline đã khóa để so sánh công bằng với context-depth ablation,
   generator comparison, direct LLM và các kiến trúc RAG nâng cao sau này.

---

## 9. Công Cụ Thực Thi

- Kaggle notebook: `notebooks/Tests/09_phase_2_e2e_baseline_kaggle.ipynb`
- Colab notebook: `notebooks/Tests/11_phase_2_e2e_baseline_colab.ipynb`
- One-time artifact builder: `notebooks/Tests/10_build_phase_2_locked_index_colab.ipynb`
- Private artifact publisher: `scripts/publish_phase2_index_artifact.py`
- Hướng dẫn chạy ngắn: `docs/Detailed Test Plans/phase_2_execution_guide.md`
- Generation collector: `scripts/collect_benchmark_predictions.py`
- Deterministic scorer: `scripts/score_benchmark_predictions.py`
- RAGAS judge: `scripts/judge_benchmark_predictions.py`
- Experiment summarizer: `scripts/summarize_experiments.py`
