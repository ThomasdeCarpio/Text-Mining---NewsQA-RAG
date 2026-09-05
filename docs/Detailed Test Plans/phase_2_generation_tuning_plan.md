# Kế Hoạch Chi Tiết — Giai Đoạn 2B: Đo và Tinh Chỉnh Chất Lượng Sinh

Tài liệu này trả lời hai câu hỏi:

1. **Cần dựng những gì để đo baseline sinh câu trả lời?** — generator
   `gemini-3.1-flash-lite`, judge RAGAS chạy trên `GLM 5.3 Flash`.
2. **Còn tinh chỉnh được gì?** — danh sách các núm vặn còn tự do, xếp theo
   tỉ lệ giá trị/chi phí, kèm giả thuyết và quy tắc quyết định cho từng núm.

Quy trình baseline chi tiết (bước preflight → smoke → full → score → judge)
nằm ở [`phase_2_baseline_test_plan.md`](phase_2_baseline_test_plan.md); tài
liệu này chỉ ghi phần **khác đi** và phần **chưa từng có**.

> **Tất cả đo đạc trong tài liệu này chạy trên tập tinh chỉnh** — 281 câu
> resolved của 50 bài development. 150 bài held-out không được đụng tới cho
> tới khi mọi lựa chọn dưới đây đã chốt. Xem
> [`phase_1_retrieval_test_plan.md`](phase_1_retrieval_test_plan.md) §5.

---

## 1. Trần Chất Lượng: Sinh Không Thể Vượt Retrieval

Trước khi đo generation phải biết mức trần. Với cấu hình đã khóa, đo trên
tập tinh chỉnh:

| Đại lượng | Giá trị | Ý nghĩa cho generation |
|---|---:|---|
| `Hit@5` sau rerank | `0,9573` | **95,7%** số câu có bằng chứng vàng nằm trong 5 context đưa cho LLM |
| `Recall@5` | `0,9555` | phần bằng chứng thực sự lọt vào 5 context |
| `Hit@1` | `0,8221` | 82,2% câu có bằng chứng ngay ở context đầu |
| `Hit@10` trước rerank | `0,9395` | trần của `top_k=20` — reranker đã vượt mức này |

**Hệ quả trực tiếp**: khoảng **4,3%** số câu (~12 câu) **không thể** trả lời
đúng từ context. Đó là sàn lỗi của generation, không phải lỗi của LLM. Mọi
báo cáo answer accuracy phải tách hai nhóm:

- **Answerable-from-context** (~269 câu): đo năng lực sinh thực sự.
- **Evidence-missing** (~12 câu): ở đây hành vi **đúng** là từ chối trả lời,
  không phải bịa. Đo bằng abstention rate, không đo bằng EM/F1.

Gộp hai nhóm rồi báo một con số F1 duy nhất sẽ thưởng cho model biết đoán mò.

---

## 2. Biến Khóa Cho Baseline 2B

Bảng này **thay thế** §2.1 của `phase_2_baseline_test_plan.md` ở ba dòng đã
thay đổi sau Giai đoạn 1 và sau khi đổi judge.

| Thành phần | Giá trị khóa | Thay đổi so với bản cũ |
|---|---|---|
| Retrieval artifact | Bundle từ `notebooks/public/14_export_locked_index_kaggle.ipynb` (corpus v2.0.0) | **thay** tag `phase2-bge-m3-512-64-v1` dựng trên v1.0.0 |
| Reranker | `BAAI/bge-reranker-large`, `top_n=5` | **thay** `ms-marco-MiniLM-L-6-v2` |
| Generator | `gemini-3.1-flash-lite`, `temperature=0`, `max_tokens=512` | giữ nguyên |
| RAGAS judge | **`GLM 5.3 Flash`** | **thay** `gemini-3.7-flash` |
| Judge credential | Key Z.ai/Zhipu riêng, **không** dùng chung key generator | — |
| Retriever / chunking / `top_k` | BGE-M3 sparse · recursive `512/64` · `20` | giữ nguyên |

Đổi judge sang GLM còn **cải thiện** thiết kế: judge và generator giờ khác
hẳn nhà cung cấp, không chỉ khác kích thước trong cùng họ Gemini. Rủi ro
self-evaluation bias giảm thật, không phải giảm trên danh nghĩa.

---

## 3. Việc Phải Làm Trước Khi Chạy: Định Tuyến GLM

**Đây là blocker, không phải tùy chọn.** Judge GLM chưa chạy được với code
hiện tại. Hai lỗi trong `common/newsqa_rag/evaluation/metrics.py`:

1. **Nhánh fallback không nhận `base_url`.** `_ragas_judge()` có nhánh riêng
   cho `gemini` và `deepseek`; mọi model khác rơi vào:

   ```python
   chat = ChatOpenAI(model=llm_model, temperature=0)
   ```

   Không truyền `base_url`, nên request đi tới `api.openai.com` chứ không tới
   endpoint Z.ai. Judge sẽ lỗi xác thực, hoặc tệ hơn là gọi nhầm một model
   OpenAI nếu `OPENAI_API_KEY` tình cờ hợp lệ.

2. **`provider="auto"` ưu tiên DeepSeek.** Nhánh chọn provider là
   `provider == "deepseek" or (provider == "auto" and bool(os.getenv("DEEPSEEK_API_KEY")))`.
   Nếu `.env` còn `DEEPSEEK_API_KEY`, judge GLM sẽ **âm thầm** chạy bằng
   DeepSeek và vẫn ghi `judge_model` là GLM vào manifest. Đây là lỗi làm hỏng
   provenance mà không báo lỗi.

**Cách sửa tối thiểu**: thêm một nhánh `glm`/`zhipu` dùng
`OPENAI_BASE_URL`-style config riêng, và bắt `provider` phải khớp tường minh
với tiền tố của `--judge-model` thay vì đoán từ biến môi trường nào đang tồn
tại. Preflight phải in ra `base_url` thực tế mà judge dùng.

**Xác minh model ID.** Chuỗi ID chính xác của GLM 5.3 Flash phải được kiểm
bằng một call thật ở bước preflight (tương tự `scripts/verify_gemini_models.py`)
và ghi vào manifest. Không hard-code một ID chưa gọi thử bao giờ.

---

## 4. Metric Cho Baseline Sinh

Ba tầng, chạy đúng thứ tự này — tầng rẻ chặn lỗi trước khi tiêu tiền tầng đắt.

### 4.1. Deterministic (chi phí 0đ)

| Metric | Vì sao cần |
|---|---|
| Exact Match, token-F1 | so với `answer` vàng; chuẩn NewsQA |
| Abstention rate | tách theo nhóm answerable / evidence-missing ở §1 |
| Answer length | phát hiện model lan man hoặc cụt |
| Citation precision / recall / F1 | câu có trích dẫn đúng chunk chứa bằng chứng không |
| Hallucinated-citation rate | trích dẫn index không tồn tại trong context |

### 4.2. RAGAS (LLM-as-a-Judge, GLM 5.3 Flash)

`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`,
`answer_correctness`. `answer_relevancy` cần embedding — chạy local bằng
`all-MiniLM-L6-v2`, không tốn API.

RAGAS **không phải** ground truth. Báo cáo phải kèm judge model, judge
fingerprint, phiên bản RAGAS, coverage và CI 95%.

### 4.3. Vận hành

Latency tách tầng (retrieve / rerank / generate), token in-out, số attempt,
tỉ lệ request lỗi.

---

## 5. Danh Sách Tinh Chỉnh, Xếp Theo Giá Trị Trên Chi Phí

Mỗi mục ghi: giả thuyết → biến thể → chi phí → metric quyết định.

### 5.1. Prompt (đòn bẩy lớn nhất, chi phí thấp nhất)

Prompt hiện tại là prompt citation mặc định của `RAGAgent` — chưa từng được
tinh chỉnh lần nào. Với 281 câu × `gemini-3.1-flash-lite`, mỗi biến thể là
một lần chạy generation rẻ; đây là núm nên vặn trước.

| Biến thể | Giả thuyết |
|---|---|
| Cấm suy diễn ngoài context | giảm hallucination, tăng `faithfulness` |
| Cho phép nói "không đủ thông tin" | sửa đúng ~12 câu evidence-missing ở §1 |
| Ép định dạng câu trả lời ngắn | NewsQA answer là span ngắn; văn dài làm tụt EM/F1 |
| Ép định dạng citation `[i]` chặt | tăng citation precision, giảm citation ảo |
| Few-shot 2–3 ví dụ | ổn định định dạng, đổi lại tốn token đầu vào |

**Quyết định**: giữ biến thể có `answer_correctness` cao nhất mà không làm
tụt `faithfulness`. Thay đổi prompt phải ghi vào manifest — prompt là một
biến thực nghiệm, không phải chi tiết cài đặt.

### 5.2. Số context đưa vào LLM (`rerank_top_n`)

Đang khóa `5`. Đây là đánh đổi hai chiều rõ ràng và **đã có sẵn số retrieval**
để dự đoán:

| `top_n` | `Hit@n` | Kỳ vọng |
|---:|---:|---|
| 3 | `0,9324` | ít nhiễu, ít token, mất 2,5% coverage |
| 5 | `0,9573` | mốc hiện tại |
| 7 | *cần notebook 12* | thêm coverage, thêm nhiễu và token |

Coverage tăng dần nhưng **không** đảm bảo answer tốt hơn: context thừa làm
loãng sự chú ý của model. Đây chính là lý do phải đo end-to-end chứ không suy
ra từ `Hit@n`. Chi phí: 3 lần chạy generation.

### 5.3. Hierarchical chunking — trick riêng cho generation

Notebook 12 đang chờ chạy. Giá trị thật của hierarchical **nằm ở tầng sinh**,
không phải tầng truy xuất: khớp bằng **child** nhỏ cho chính xác, rồi đưa
**parent** lớn cho LLM đọc để không cắt cụt ngữ cảnh. `parent_id` đã có sẵn
trong metadata chunk.

Đây là biến thể có khả năng cải thiện cao nhất trong nhóm "kiến trúc", vì nó
tấn công đúng điểm yếu của chunking cố định: ranh giới cắt rơi vào giữa câu
trả lời.

### 5.4. Query rewriting — phần thưởng lớn nhất đang bỏ trống

Chênh lệch giữa câu hỏi `original` và `resolved` là **0,4164 MRR@5** trung
bình trên 8 retriever. Đó là khoảng cách lớn nhất đo được trong toàn bộ
Giai đoạn 1, lớn hơn mọi khác biệt giữa các model.

Người dùng thật gõ câu hỏi ở dạng gần `original`. Pipeline hiện **chưa có**
bước viết lại câu hỏi. Một bước rewriting rẻ (một call LLM giải quyết đại
từ/mỏ neo trước khi truy xuất) là ứng viên có kỳ vọng lợi ích cao nhất trong
cả danh sách.

Đo bằng: chạy retrieval trên câu `original` **đã qua rewriting**, so với
`original` thô (sàn) và `resolved` thủ công (trần).

### 5.5. Reranker — có nên đổi lại MiniLM

Đã khóa `bge-reranker-large` vì độ chính xác truy xuất. Nhưng ở tầng
end-to-end, câu hỏi khác đi: khi generator chỉ đọc 5 context, chênh lệch
`0,041 MRR@5` giữa hai reranker có biến mất không?

| Reranker | MRR@5 | P50 |
|---|---:|---:|
| `bge-reranker-large` | `0,8797` | `513 ms` |
| `ms-marco-MiniLM-L-6-v2` | `0,8387` | `189 ms` |

Nếu answer F1 **không** khác biệt có ý nghĩa thống kê, MiniLM tiết kiệm
~324 ms mỗi query. Chỉ đổi khi có bằng chứng; mặc định giữ bản đã khóa.

### 5.6. `top_k` — núm chưa từng được quét

`20` là giá trị kế thừa từ cấu hình Vòng 2, **chưa bao giờ** được quét riêng.
`Hit@10` trước rerank là `0,9395`, tức reranker đang làm việc tốt trên pool
hiện có. Thử `30` và `50`: chi phí là thời gian rerank tăng tuyến tính, lợi
ích là trần recall cao hơn. Đây là thử nghiệm rẻ nhất còn lại ở tầng truy xuất.

### 5.7. Tham số sinh

`temperature=0` giữ nguyên — cần tái lập. `max_tokens=512` nhiều khả năng
thừa cho span ngắn của NewsQA; giảm xuống sẽ cắt chi phí mà gần như không
ảnh hưởng chất lượng. Đo trước bằng phân bố answer length ở §4.1.

---

## 6. Thứ Tự Thực Hiện

```
1. Sửa định tuyến GLM (§3)          ─┐ blocker
2. Chạy notebook 14, có artifact v2 ─┘
3. Notebook 12: hierarchical + @7            → chốt chunking & top_n
4. Baseline 2B: 281 câu, prompt hiện tại     → mốc so sánh
5. Quét prompt (§5.1)                        → đòn bẩy lớn nhất
6. Quét top_n {3,5,7} (§5.2)
7. Query rewriting (§5.4)                    → phần thưởng lớn nhất
8. Nếu còn ngân sách: top_k (§5.6), reranker (§5.5)
9. CHỐT TOÀN BỘ CẤU HÌNH
10. Chạy held-out 150 bài đúng MỘT lần       → robustness
```

Bước 4 phải chạy **trước** mọi bước tinh chỉnh: không có mốc thì không biết
núm nào có tác dụng.

Bước 10 chỉ được chạy sau bước 9. Không chọn, không tinh chỉnh, không "chạy
thử xem sao" trên held-out — tập giữ kín mất giá trị ngay khi có một quyết
định dựa trên nó.

---

## 7. Điều Kiện Chấp Nhận Run

1. Judge thực sự gọi GLM: manifest ghi đúng `judge_model`, `base_url` và
   provider; preflight in ra endpoint thật.
2. Generator và judge là hai model khác nhau, hai credential khác nhau.
3. Coverage generation `100%` trên 281 câu, không thiếu record.
4. RAGAS coverage tối thiểu `95%` số generation thành công.
5. Điểm số tách theo nhóm answerable / evidence-missing (§1).
6. Mọi biến thể prompt được ghi nguyên văn vào manifest.
7. Truy ngược được per-question bằng `question_id` cho cả deterministic và
   RAGAS.
8. Không có kết quả nào đến từ held-out.
