# Kế Hoạch Chi Tiết - Giai Đoạn 2B: Tối Ưu Generation

## 1. Mục tiêu và phạm vi

Phase 2B tối ưu **prompt và cấu hình sinh**, không huấn luyện lại trọng số của
Gemini. Vì vậy tên học thuật đúng là *prompt/configuration tuning*, không phải
supervised fine-tuning.

Mục tiêu là tìm cấu hình generation có Answer Correctness cao hơn baseline mà
không làm giảm grounding và citation. Mọi thí nghiệm tái sử dụng hợp đồng dataset,
partition, retrieval và metrics trong
[`phase_2_baseline_test_plan.md`](phase_2_baseline_test_plan.md).

Ngoài phạm vi:

- thay embedding, retriever, reranker hoặc chunking;
- query rewriting và Agentic RAG;
- huấn luyện trọng số model;
- dùng held-out để sửa prompt hay chọn tham số;
- trộn bài toán abstention/open-domain vào benchmark NewsQA answerable.

Nếu sau này fine-tune trọng số, cần một phase riêng với train/validation/test
tách biệt và model/provider hỗ trợ training.

## 2. Hợp đồng cố định

| Thành phần | Giá trị |
|---|---|
| Artifact repo | `ThomasAnderson2009/newsqa-rag-phase2-locked-v2` |
| Artifact revision | `locked-bge-m3-512-64-deduplicated-v2` |
| Artifact commit | `bb73e682f472933c212f2c6a3f9575c652b280fd` |
| Artifact ZIP | `artifacts/locked-bge-m3-512-64-deduplicated-v2/locked-bge-m3-512-64-deduplicated-v2.zip` |
| ZIP SHA-256 | `fc5d67b7acf6e8be0205ce00b8069b3b6c8dcce853f8671f2feb3887b2707a24` |
| Primary set | 1.152 semantic-deduplicated resolved questions |
| Development | 50 bài, 281 câu resolved, seed `42` |
| Held-out pool | 150 bài, 871 câu resolved; chưa dùng khi tuning |
| Final held-out subset | 50 bài unseen, 284 câu; sample theo article với seed `46` |
| Held-out reserve | 100 bài, 587 câu; giữ nguyên cho mở rộng sau |
| Supplementary set | 1.336 full resolved questions; không dùng chọn winner |
| Retrieval | BGE-M3 learned sparse, `top_k=20` |
| Reranking | `BAAI/bge-reranker-large`, top 5 |
| Generator | `gemini-3.1-flash-lite`, `temperature=0`, `reasoning_effort=minimal` |
| Generator key | `GEMINI_API_KEY_1` |
| Judge | `accounts/fireworks/models/glm-5p3-flash`, `reasoning_effort=low`, tối đa 2.048 output tokens |
| Judge key | `FIREWORKS_API_KEY` |
| Seed | `42` |

Retrieval chạy một lần. Mỗi biến thể generation nhận cùng ranked contexts và
cùng question IDs. Khi thử context depth, chỉ cắt danh sách top 5 đã cache;
không retrieve/rerank lại.

Mọi tuning score và quy tắc chọn winner dùng primary deduplicated set. Full set
chỉ được chạy với cấu hình đã khóa để phân tích độ nhạy với alternate wording;
kết quả đó phải được báo cáo riêng và không thay đổi winner.

## 3. Câu hỏi nghiên cứu

1. Prompt grounding rõ ràng có tăng faithfulness mà không giảm correctness?
2. Định dạng answer ngắn có phù hợp gold span của NewsQA hơn prompt hiện tại?
3. Citation contract chặt hơn có tăng citation F1?
4. Đưa 1, 3 hay 5 context vào generator cho trade-off tốt nhất giữa quality,
   latency và token cost?

Mỗi câu hỏi chỉ thay đổi một nhóm biến trong một vòng. Generator model,
temperature và retrieval được giữ cố định để có thể quy kết khác biệt.

## 4. Ma trận thí nghiệm đăng ký trước

### 4.1. Vòng P - Prompt

Dùng `context_depth=5`, `max_tokens=512` cho bốn prompt:

| ID | Prompt contract | Giả thuyết |
|---|---|---|
| `P0` | Prompt citation hiện tại của `RAGAgent` | Baseline |
| `P1` | Chỉ dùng context; không được suy diễn ngoài bằng chứng | Tăng faithfulness |
| `P2` | Trả lời trực tiếp, ngắn gọn theo NewsQA; sau đó citation | Tăng EM/F1 và relevancy |
| `P3` | Kết hợp P1/P2; citation `[i]` bắt buộc cho mỗi khẳng định chính | Cân bằng quality và grounding |

Prompt có thể nói "không đủ thông tin trong context" để tránh bịa đặt, nhưng
NewsQA không có gold unanswerable. Do đó không dùng abstention rate làm metric
chọn winner trong Phase 2. Khả năng abstain được đánh giá riêng ở Phase 3.

Không đưa few-shot vào vòng chính: ví dụ làm tăng token, có nguy cơ data leakage
và tạo thêm một biến gây nhiễu. Few-shot chỉ là ablation sau nếu tất cả zero-shot
prompt không ổn định định dạng.

### 4.2. Vòng C - Context depth

Lấy hai prompt tốt nhất từ Vòng P và thử:

| Biến | Giá trị |
|---|---|
| `context_depth` | `1`, `3`, `5` |
| Thứ tự context | Giữ nguyên thứ tự BGE-large reranking |
| Retrieval pool | Giữ `top_k=20` |
| Max output | Giữ `512` |

Đây là ma trận tối đa 6 cấu hình. Record trùng với baseline được tái sử dụng,
không gọi Gemini hoặc GLM lại.

### 4.3. Biến không quét

- `temperature=0` để đảm bảo tái lập.
- `max_tokens=512` giữ cố định. Chỉ mở ablation `{128, 256, 512}` nếu baseline
  ghi nhận output bị cắt hoặc chi phí output bất thường.
- Không so sánh generator model trong cùng prompt experiment. Nếu cần, khóa
  prompt/context winner trước rồi tạo một model-ablation riêng.
- Không thử `top_n>5` vì locked trace và artifact chỉ khóa 5 context sau rerank.

## 5. Thiết kế mẫu và thứ tự chạy

### 5.1. Các tập development

| Tập | Số câu | Cách dùng |
|---|---:|---|
| Smoke | 5 | Kiểm tra mỗi prompt/config mới |
| Screening | 80 | Quét P và C với chi phí giới hạn |
| Judge calibration | 20 | RAGAS sớm cho mỗi cấu hình screening |
| Full development | 281 | Xác nhận hai finalist |
| Held-out final | 284 từ 50 bài unseen | Đánh giá winner đúng một lần |

Tập 80 và 20 được tạo một lần với seed `42`, phân tầng theo article, question
type và `gold_in_top5`. ID được commit/lưu trong experiment manifest. Không đổi
mẫu sau khi xem kết quả. Không giảm thấp hơn `80/20`: đây là screening để loại
cấu hình yếu, không phải final inference, nhưng mẫu nhỏ hơn sẽ quá nhạy với vài
article và độ nhiễu của LLM judge.

Final subset được chọn từ 150 bài held-out bằng thứ tự hash seed `46`, lấy đúng
50 bài đầu. Với artifact đã khóa, các bài này chứa `284` câu. Chọn theo article
thay vì ép đúng 281 câu để giữ độc lập với development và tránh lấy dở một cụm
câu hỏi của cùng bài.

### 5.2. Trình tự

1. Hoàn tất và đóng băng baseline P0 trên 281 câu.
2. Smoke 5 câu cho P1-P3; loại lỗi schema, citation và output rỗng.
3. Chạy P0-P3 trên cùng 80 câu; deterministic score trên 80 và RAGAS trên
   cùng 20 câu.
4. Chọn hai prompt theo quy tắc ở Mục 7.
5. Chạy context depth `{1,3,5}` với hai prompt trên cùng 80/20 câu.
6. Chọn hai cấu hình finalist; chạy mỗi finalist trên toàn bộ 281 câu và chấm
   đầy đủ deterministic + RAGAS.
7. Khóa winner, prompt text, context depth, model fingerprint và giá API.
8. Chạy winner trên 284 câu thuộc 50 bài held-out unseen đúng một lần.
9. Công bố kết quả held-out mà không tiếp tục sửa cấu hình.

## 6. Metrics và phân tích

Dùng đầy đủ metrics của baseline. Vai trò trong tuning:

| Vai trò | Metrics |
|---|---|
| Primary quality | RAGAS Answer Correctness |
| Grounding guardrail | Faithfulness |
| Citation guardrail | Citation F1 và Citation Validity |
| Supporting quality | EM, token F1, Answer Relevancy |
| Context diagnosis | Context Precision, Context Recall |
| Efficiency | Input/output tokens, cost, generation P50/P95 latency |

Mỗi so sánh dùng paired per-question difference. Báo cáo:

- mean difference và 95% article-cluster bootstrap CI;
- question-level micro và article-level macro;
- overall, `gold_in_top5` và `gold_not_in_top5`;
- success/score coverage và failure count;
- cost và latency trên cùng môi trường.

Không diễn giải CI chồng lấp là hai phương pháp tương đương. Nếu cần tuyên bố
equivalence, phải đăng ký equivalence margin và dùng equivalence test riêng.

## 7. Quy tắc chọn winner

Quy tắc được khóa trước khi xem kết quả tuning:

1. Một cấu hình chỉ hợp lệ nếu so với P0:
   - Faithfulness không giảm quá `0,02`;
   - Citation F1 không giảm quá `0,01`;
   - Citation Validity không giảm quá `0,01`;
   - generation và RAGAS coverage đạt ngưỡng của baseline plan.
2. Trong các cấu hình hợp lệ, chọn Answer Correctness cao nhất.
3. Nếu chênh Answer Correctness nhỏ hơn `0,01`, chọn token cost thấp hơn.
4. Nếu chi phí chênh dưới 5%, chọn generation P95 latency thấp hơn.
5. Nếu vẫn đồng hạng, chọn cấu hình đơn giản hơn: ít context hơn, prompt ngắn
   hơn và ít ràng buộc định dạng hơn.

Ngưỡng là practical guardrail, không phải khẳng định ý nghĩa thống kê. CI và
effect size vẫn phải được báo cáo.

## 8. Kiểm soát tính học thuật

- Preregister prompt IDs, exact prompt text, subset IDs, metrics và selection
  rule trước full development run.
- Không sửa gold answer, evidence hoặc resolved question trong Phase 2.
- Không chọn câu hỏi dựa trên output của generator/judge.
- Dùng cùng cached retrieval trace cho mọi cấu hình.
- Cache key bao gồm dataset revision, question, exact prompt, ordered contexts,
  generator model và decoding config.
- Lưu raw answer và raw judge output để audit LLM-as-a-Judge.
- Judge khác provider với generator; ghi endpoint và model fingerprint.
- Primary benchmark lấy từ 1.152 semantic targets: tuning dùng 281 development
  và final report dùng 284 held-out đã đăng ký trước. Full set 1.336 câu chỉ
  dùng cho sensitivity analysis sau khi đã khóa cấu hình; không đưa các câu
  trùng trở lại objective dùng để chọn prompt.
- Nếu RAGAS disagreement quan trọng, human-review một mẫu lỗi có phân tầng;
  không sửa điểm từng câu tùy ý.
- Không dùng held-out cho prompt debugging, threshold tuning hay retry có chọn
  lọc theo điểm.
- Audit thủ công mù cấu hình trên 30 cặp output development của P0 và winner.
  Hai reviewer chấm correctness, grounding và citation; báo cáo tỷ lệ đồng thuận
  và cách giải quyết bất đồng để kiểm tra độ tin cậy của LLM judge.
- Chạy lặp lại 25 câu cố định cho P0 và winner để ước lượng độ ổn định của API,
  ngay cả khi `temperature=0`; không dùng lần lặp tốt hơn để thay kết quả chính.

## 9. Yêu cầu hệ thống và provenance

Experiment interface cần cho phép khai báo và fingerprint:

- `prompt_id` và exact `system_prompt`;
- `context_depth`;
- `temperature` và `max_tokens`;
- fixed question-ID file;
- shared retrieval cache;
- generator/judge provider và model;
- retry, timeout và minimum request interval.

Mỗi run phải resume được. Successful record không được ghi trùng; failed record
giữ stage, error type, attempt count và timestamp. Kết quả mỗi vòng gồm
per-question JSONL, summary CSV/JSON, manifest, environment và chi phí.

Thực thi được chia thành các notebook nhỏ để chạy độc lập trên Kaggle/Colab:

| Notebook | Nền tảng | Trách nhiệm |
|---|---|---|
| `13a_phase_2b_0_preparation_kaggle.ipynb` | Kaggle | Khóa subset và đóng gói baseline |
| `13b`/`13c`/`13d_phase_2b_1_prompt_*_colab.ipynb` | Colab | Chạy riêng P1, P2, P3 trên 80/20 |
| `13e`/`13f_phase_2b_2_depth_*_colab.ipynb` | Colab | Chạy hai prompt finalist ở depth 1 hoặc 3; depth 5 tái sử dụng Phase 2B.1 |
| `13g_phase_2b_3_finalist_1_kaggle.ipynb` | Colab | Xác nhận finalist 1 P2/depth 5 trên 281 câu; tên file cũ được giữ để tương thích |
| `13h_phase_2b_3_finalist_2_colab.ipynb` | Colab | Xác nhận finalist 2 P2/depth 3 trên 281 câu |
| `13i_phase_2b_4_heldout_final_kaggle.ipynb` | Kaggle | Chạy winner trên 284 câu held-out |

Phase 2B.0 xuất `phase2b_preparation_bundle.zip`. Mọi notebook sau phải nạp
đúng bundle này và kiểm tra lại hash của toàn bộ subset. Các biến
`PROMPT_FINALISTS`, `FINALIST_CONFIG` và `LOCKED_WINNER` chỉ được điền từ decision
record của vòng trước; notebook không tự chọn winner theo một scalar metric.
Mỗi notebook có checkpoint/result ZIP riêng để tránh một session lỗi làm mất
toàn bộ tournament.

Các notebook mặc định `EXECUTE_API_CALLS=False`. Với Colab, mỗi notebook cho
phép đặt `GEMINI_SECRET_NAME` riêng; chạy song song chỉ tăng throughput khi các
key thuộc project/quota độc lập. Tất cả vẫn dùng chung `FIREWORKS_API_KEY` và
phải tuân theo rate limit của tài khoản đó.

Exact prompt text được quản lý tại
`configs/experiments/phase2_generation_prompts.yaml`. Collector nhận
`--source-retrievals`, `--prompt-id`, `--system-prompt-file` và
`--context-depth`; vì vậy các run generation tái sử dụng cùng ranked trace thay
vì gọi lại retriever/reranker. Judge nhận `--question-ids-file` để mọi cấu hình
screening được chấm trên đúng cùng 20 câu.

## 10. Điều kiện hoàn thành Phase 2

- Baseline 281 câu và hai finalist có deterministic/RAGAS coverage hợp lệ.
- Winner được chọn đúng quy tắc đăng ký trước, không dựa trên held-out.
- Held-out final 284 câu từ 50 bài unseen được chạy đúng một lần sau khi khóa
  cấu hình; 587 câu còn lại không được dùng để sửa winner.
- Báo cáo nếu winner tốt hơn, không khác rõ ràng, hoặc tệ hơn baseline; không chỉ
  công bố kết quả có lợi.
- Cấu hình winner được đóng gói để app và benchmark cùng nạp một artifact/prompt.
- Trong báo cáo sử dụng thuật ngữ "tối ưu prompt/cấu hình sinh"; không tuyên bố
  đã fine-tune trọng số Gemini.
