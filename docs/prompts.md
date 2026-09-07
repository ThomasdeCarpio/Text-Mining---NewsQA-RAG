# Prompt của hệ thống — nguyên văn

> Nguồn duy nhất: [`configs/experiments/phase2_generation_prompts.yaml`](../configs/experiments/phase2_generation_prompts.yaml)
> Dùng ở: [Phase 2 §7](reports/phase2/report.md) · Tóm tắt: [reports/report.md](reports/report.md)

Trang này chép nguyên văn mọi prompt đã thực sự chạy, để báo cáo tự kiểm chứng được. Không diễn giải, không rút gọn.

---

## 1. Một request gửi cho Gemini trông như thế nào

Mỗi lần sinh câu trả lời là hai phần: **system prompt** (chọn từ P0–P3, cố định cho cả run) và **user message** (ghép từ context + câu hỏi).

`common/newsqa_rag/llm.py:130`:

```python
context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
```

Nên user message có dạng:

```text
Context:
[1] <toàn văn đoạn hạng 1>

[2] <toàn văn đoạn hạng 2>

[3] <toàn văn đoạn hạng 3>

Question: Who was the prime minister at the time?

Answer:
```

Chỉ số `[1]`, `[2]`, `[3]` ở đây **chính là** chỉ số mà model dùng khi trích dẫn. Citation Validity đo xem model có trả về chỉ số nằm ngoài khoảng này không.

### `context_depth` — cái được tinh chỉnh ở Phase 2B.2

`context_depth` là **số đoạn văn được dán vào khối trên**, lấy từ đầu danh sách 5 đoạn mà reranker của Phase 1 đã xếp hạng.

`common/newsqa_rag/agents/rag_agent.py:82`:

```python
generation_chunks = ranked_chunks[:context_depth]
```

Đúng một phép cắt danh sách. **Không truy xuất lại, không xếp hạng lại** — nên chênh lệch giữa depth 1, 3 và 5 quy hết được cho phần sinh.

| depth | Model thấy | Đánh đổi |
| ---: | :--- | :--- |
| 1 | `[1]` | Ít token nhất, ít nhiễu nhất — nhưng 38/281 câu mất sạch bằng chứng |
| 3 | `[1] [2] [3]` | Cân bằng — nhưng 7/281 câu mất sạch bằng chứng |
| 5 | `[1]`…`[5]` | Không mất bằng chứng — tốn token nhất |

Con số "mất sạch bằng chứng" đến từ đường cong Hit@k mà Phase 1 đã đo (Hit@1 0,8221 · Hit@3 0,9324 · Hit@5 0,9573), không phải từ ước lượng.

---

## 2. Bốn system prompt của Phase 2B

Cả bốn kết thúc bằng cùng một câu từ chối chuẩn hóa:
`I cannot find this information in the provided context.`
Đây là chuỗi được đếm là *abstention*, và là hạt giống của Phase 3.

### P0 — `baseline`

Giả thuyết đăng ký: *prompt RAG có citation hiện hành của `RAGAgent`; dùng làm control.*

```text
You are a helpful assistant answering questions based on provided context.
Answer concisely and only based on the given context. Cite supporting context
using its bracketed number, for example [1]. Every factual claim must have a
citation, and you must not cite context that does not support the claim. If
the answer is not in the context, say: 'I cannot find this information in the
provided context.'
```

### P1 — `strict_grounding`

Giả thuyết đăng ký: *ràng buộc bằng chứng tường minh sẽ tăng faithfulness.*

```text
Answer the question using only facts explicitly stated in the numbered
context. Do not add background knowledge, assumptions, or plausible details.
Give a concise answer and cite the supporting context with [n] after every
factual claim. Cite only context that directly supports that claim. If the
context does not contain the answer, say: 'I cannot find this information in
the provided context.'
```

**Kết quả: thất bại.** Baseline đã đạt Faithfulness 0,9761 — không còn dư địa. P1 nhắm vào một vấn đề không tồn tại và mất correctness khi làm vậy (0,7118 → 0,6885).

### P2 — `concise_answer_type` ⭐ **winner**

Giả thuyết đăng ký: *một câu trả lời duy nhất, đúng loại thông tin được hỏi, sẽ tăng correctness và token F1.*

```text
Answer using only the numbered context. Give one short answer sentence
containing only the information type requested by the question, such as a
person, place, date, year, number, object, event, or explanation. Include
multiple items only when the question explicitly asks for them. Do not repeat
the question, list alternative answers, or add background details. Place a
supporting citation [n] immediately after the answer. If the answer is not in
the context, say: 'I cannot find this information in the provided context.'
```

**Vì sao thắng:** nó khớp với hình dạng của nhãn. Gold NewsQA là span ngắn — một cái tên, một mốc thời gian, một con số. Đây cũng là lý do Answer Relevancy giảm: metric ấy thưởng câu trả lời đầy đủ ngữ cảnh, còn dữ liệu này thưởng câu trả lời cụt.

### P3 — `event_disambiguated_concise`

Giả thuyết đăng ký: *chọn đoạn khớp sự kiện trước khi trả lời sẽ giảm việc trộn thông tin giữa các bài, mà vẫn giữ grounding và citation.*

```text
Answer using only facts explicitly stated in the numbered context. First
identify the passage that best matches all identifying details in the
question, including its subject, event, location, and time. Do not combine
facts, figures, or dates from different events, articles, or reporting
periods. Give one short answer sentence containing only the requested
information type, and include multiple items only when explicitly asked.
Place a supporting citation [n] immediately after the answer and cite only
the passage that directly supports it. If no passage contains a defensible
answer, say: 'I cannot find this information in the provided context.'
```

**Kết quả: đúng hướng, chưa đủ mạnh.** Citation F1 và Answer Relevancy tốt nhất trong vòng screening, nhưng correctness chỉ 0,7190 so với 0,8363 của P2. Đáng chú ý: P3 **chứa cả chỉ thị của P2** cộng thêm phần chọn đoạn, mà correctness lại thấp hơn hẳn — có thể vì chỉ thị dài làm loãng mệnh lệnh súc tích. RAGAS chỉ chấm 20 câu nên đây là quan sát, chưa phải kết luận.

---

## 3. Kiểm chứng: prompt trong repo đúng là prompt đã chạy

Mỗi run lưu kèm bản sao `prompts/p{0,1,2,3}.txt`. Đối chiếu 8 bản sao đó với YAML trong repo:

| Prompt | Độ dài | SHA-256 (16 ký tự đầu) | Đối chiếu |
| :--- | ---: | :--- | :--- |
| P0 | 402 ký tự | `7f0e0c11a80ed535` | ✅ khớp cả 8 bản |
| P1 | 397 ký tự | `26e42db3fb6fd78d` | ✅ khớp cả 8 bản |
| P2 | 525 ký tự | `09b2508dbf7af601` | ✅ khớp cả 8 bản |
| P3 | 652 ký tự | `118a424180c7b2ee` | ✅ khớp cả 8 bản |

Không bản nào lệch. Prompt không bị sửa giữa chừng giữa các thí nghiệm.

---

## 4. Một chỗ P3 lệch với kế hoạch đăng ký trước

| Nguồn | P3 là gì |
| :--- | :--- |
| [`phase_2_generation_tuning_plan.md`](Detailed%20Test%20Plans/phase_2_generation_tuning_plan.md) §4.1 | "Kết hợp P1/P2; citation `[i]` bắt buộc cho mỗi khẳng định chính" |
| `phase2_generation_prompts.yaml` (đã chạy) | `event_disambiguated_concise` — chọn đoạn khớp sự kiện trước khi trả lời |

Bản đã chạy là bản trong YAML. Việc sửa **có cơ sở** — nó nhắm vào phát hiện distractor collision của EDA §7 — nhưng test plan là văn bản đăng ký trước nên cần một ghi chú sửa đổi có ngày tháng thay vì sửa lùi. **P3 không được chọn**, nên độ lệch này không ảnh hưởng tới winner.

---

## 5. Prompt của Phase 3 — chưa chạy

Plan abstention đăng ký cấu hình **B1** bắt model trả về đúng một trong hai JSON:

```json
{"answerability":"answerable","answer":"Natalie Cole","citations":[1]}
```

```json
{"answerability":"insufficient_evidence","answer":null,"citations":[]}
```

Output sai schema được retry tối đa ba lần, sau đó vẫn ghi là failure và **không** bị loại khỏi mẫu số. Prompt đầy đủ sẽ được chép vào đây khi Phase 3 chạy.

---

## 6. Judge

RAGAS gọi `accounts/fireworks/models/glm-5p3-flash` qua Fireworks với `reasoning_effort=low`, tối đa 2.048 output token. Prompt do thư viện RAGAS sinh, không do nhóm viết, nên được nhận diện bằng fingerprint thay vì chép nguyên văn:

| | |
| :--- | :--- |
| Judge fingerprint | `a4c91b2a3436adfa0c2ae4665cbc18f8bcc958b18210e99bd2f022481afa6532` |
| Metrics | faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness |

**Judge phải khác provider với generator.** EDA §9 phát hiện notebook cũ đặt `JUDGE_MODEL = GENERATOR_MODEL`, tức LLM tự chấm bài của chính nó.
