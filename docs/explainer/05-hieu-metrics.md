# Hiểu các chỉ số đánh giá (metrics) — bản cho người mới

Chương này giải thích **mấy con số điểm** mà hệ thống dùng để tự chấm mình, bằng
ngôn ngữ đời thường. Không cần biết toán. Muốn công thức + file nào tính cái gì,
xem `docs/evaluation.md` (mục 6–7).

## Ý tưởng lớn: ta đang đo 2 việc tách rời

Một câu trả lời RAG tốt cần **hai** thứ đúng, và ta đo từng thứ riêng:

1. **Tìm có trúng đoạn không?** — *retrieval*. Hệ thống lôi ra đúng đoạn văn bản
   chứa đáp án chưa? (Nếu tìm sai đoạn thì AI có giỏi mấy cũng trả lời bậy.)
2. **Viết câu trả lời có đúng không?** — *generation*. Từ đoạn đã tìm, AI viết
   ra câu trả lời có chính xác, có bịa không?

> Tách 2 phần ra để biết **lỗi nằm ở đâu**: tìm sai, hay tìm đúng mà viết sai.

## Ví dụ xuyên suốt

Câu hỏi: **"Tỷ lệ thất nghiệp được báo cáo là bao nhiêu?"** — đáp án đúng: **"3.9%"**.

Hệ thống trả về 5 đoạn, xếp hạng từ liên quan nhất → ít nhất. Đoạn **thật sự
chứa "3.9%"** nằm ở **hạng 2**:

```
Hạng 1:  [đoạn nói chung về nền kinh tế]        ✗ không chứa đáp án
Hạng 2:  [đoạn có "...tỷ lệ thất nghiệp 3.9%"]  ✓ ĐÚNG
Hạng 3:  [đoạn về lãi suất]                      ✗
Hạng 4:  ...                                     ✗
Hạng 5:  ...                                     ✗
```

Ta sẽ chấm ví dụ này bằng từng chỉ số bên dưới.

## "@k" nghĩa là gì?

Nhiều chỉ số có đuôi `@k` (đọc: "at k"), ví dụ `Hit Rate@5`. Nó nghĩa là **chỉ
xét k đoạn xếp hạng cao nhất**. `@5` = chỉ nhìn top 5. Vì trên thực tế người dùng
(và AI) chỉ đọc vài đoạn đầu, nên ta đo chất lượng trong top 5, top 10 — không
phải toàn bộ kho.

---

## Nhóm 1 — Chấm phần TÌM KIẾM (retrieval)

### Hit Rate@k — "có lọt vào top k không?"
- **Nôm na:** Trong k đoạn đầu, có **ít nhất 1** đoạn đúng không? Trúng = 1, trượt = 0.
- **Ví dụ:** đoạn đúng ở hạng 2 → `Hit@5 = 1` (có trong top 5), nhưng `Hit@1 = 0`
  (hạng 1 không phải đoạn đúng).
- **Cao = tốt.** 0.85 nghĩa là 85% câu hỏi tìm được đoạn đúng trong top k. Đây là
  chỉ số dễ hiểu nhất, nhìn đầu tiên.

### Recall@k — "vớt được bao nhiêu phần đoạn đúng?"
- **Nôm na:** Một câu hỏi có thể có nhiều đoạn đúng. Recall = **tìm được mấy phần**
  trong số đó (trong top k). Tìm được 3/4 đoạn đúng → 0.75.
- **Ví dụ:** có 1 đoạn đúng, tìm được nó → `Recall@5 = 1.0`.
- **Cao = tốt.** Khác Hit Rate ở chỗ: Hit Rate chỉ hỏi "có trúng phát nào không",
  Recall hỏi "trúng được **bao nhiêu**".

### MRR — "đoạn đúng nằm hạng mấy?"
- **Nôm na:** MRR (Mean Reciprocal Rank) = `1 / hạng của đoạn đúng đầu tiên`. Đoạn
  đúng càng lên **đầu** thì điểm càng cao.
- **Ví dụ:** đoạn đúng ở hạng 2 → `MRR = 1/2 = 0.5`. Nếu ở hạng 1 → 1.0; hạng 4 → 0.25.
- **Cao = tốt.** Quan trọng vì AI thường đọc kỹ vài đoạn đầu nhất — đẩy đoạn đúng
  lên top 1 tốt hơn nhiều so với nhét nó ở hạng 5.

### NDCG@k — "thứ tự xếp hạng có đẹp không?"
- **Nôm na:** Giống MRR nhưng "khó tính" hơn: thưởng nhiều khi đoạn đúng ở hạng
  cao, **phạt** khi đẩy đoạn đúng xuống thấp. Điểm từ 0 đến 1.
- **Ví dụ:** đoạn đúng ở hạng 2 → NDCG ≈ 0.63.
- **Cao = tốt.** Đây là chỉ số "xịn" nhất để đánh giá **chất lượng thứ tự** của cả
  danh sách kết quả.

> Cả 4 chỉ số này tính hoàn toàn bằng **so sánh ID đoạn** (không cần AI, không tốn
> tiền). Chúng chỉ đúng khi `relevant_chunk_ids` trong bộ test **khớp** với đoạn
> trong kho — xem cảnh báo ở `docs/evaluation.md` mục 7.

---

## Nhóm 2 — Chấm CÂU TRẢ LỜI (so với đáp án mẫu)

### Exact Match (EM) — "giống hệt đáp án không?"
- **Nôm na:** Câu AI trả lời có **trùng khít** đáp án mẫu không (sau khi bỏ hoa
  thường, dấu câu)? Trùng = 1, khác = 0.
- **Ví dụ:** AI trả "3.9%" vs đáp án "3.9%" → EM = 1. AI trả "khoảng 3.9 phần trăm"
  → EM = 0 (dù ý đúng!).
- **Cao = tốt, nhưng khắt khe.** EM rất nghiêm — chỉ hợp với đáp án ngắn gọn.

### F1 — "trùng được bao nhiêu từ?"
- **Nôm na:** Đếm số **từ chung** giữa câu AI và đáp án mẫu, cân bằng giữa "nói
  thừa" và "nói thiếu". Điểm 0 đến 1.
- **Ví dụ:** "tỷ lệ 3.9 phần trăm" vs "3.9 phần trăm" → chung nhiều từ → F1 cao dù
  không giống hệt. Dễ tính hơn EM cho câu dài.
- **Cao = tốt.** F1 khoan dung hơn EM — đây là chỉ số chuẩn của bộ NewsQA.

---

## Nhóm 3 — Dùng AI chấm AI (RAGAS / LLM-as-a-judge)

Mấy chỉ số ở đây **không đếm từ** được, phải nhờ **một con AI khác đọc và chấm**
(nên **tốn tiền API**, dùng framework Ragas). Điểm 0 đến 1.

### Faithfulness (độ trung thực) — "có bịa không?"
- **Nôm na:** Mọi câu AI nói có **suy ra được** từ đoạn văn bản đã đưa cho nó
  không? Nếu AI thêm thông tin không có trong nguồn → đó là **bịa** (hallucination),
  điểm tụt.
- **Cao = tốt.** Đây là chỉ số **quan trọng nhất** cho tin tức — chống bịa.

### Answer Relevance (độ liên quan) — "có trả lời đúng câu hỏi không?"
- **Nôm na:** Câu trả lời có **đi thẳng vào** câu hỏi không, hay lan man/lạc đề/
  né tránh?
- **Cao = tốt.** Một câu trả lời có thể trung thực (không bịa) nhưng vẫn lạc đề —
  chỉ số này bắt lỗi đó.

### Context Precision / Recall — "chấm chất lượng đoạn đã tìm"
- **Context Precision:** trong các đoạn đưa cho AI, bao nhiêu phần **thật sự hữu
  ích**? (nhiễu ít = cao).
- **Context Recall:** các đoạn đó có **đủ** thông tin để trả lời không?
- **Cao = tốt** cho cả hai.

> **RAG Triad** = gộp 3 góc: Context (tìm có đúng) + Faithfulness (viết có bịa) +
> Answer Relevance (có đúng câu hỏi). Ba cái này ổn thì cả hệ thống ổn.

---

## Nhóm 4 — Chẩn đoán phụ (không cần đáp án)

- **Dedup Rate (tỷ lệ trùng lặp):** bao nhiêu phần đoạn bị **trùng y hệt** nhau
  trong kho. **Thấp = tốt** (0 là không trùng). Trùng nhiều = tốn chỗ, nhiễu kết quả.
- **Semantic Integrity (tính toàn vẹn):** bao nhiêu phần đoạn **kết thúc trọn câu**
  (không bị cắt ngang giữa câu). **Cao = tốt** — đoạn bị cắt cụt làm AI khó hiểu.
- **Delta MRR (∆MRR):** dành cho bước **rerank** (sắp xếp lại). `= MRR sau khi rerank
  − MRR trước`. **Dương = tốt** (rerank đã đẩy đoạn đúng lên cao hơn); âm = rerank
  làm tệ đi.

---

## Bảng tra nhanh

| Chỉ số | Đo cái gì | Điểm tốt | Chấm module nào | Tốn tiền? |
|---|---|---|---|---|
| Hit Rate@k | Có đoạn đúng trong top k | Cao (→1) | Retriever | Không |
| Recall@k | Vớt được mấy phần đoạn đúng | Cao (→1) | Retriever | Không |
| MRR | Đoạn đúng nằm hạng mấy | Cao (→1) | Retriever | Không |
| NDCG@k | Thứ tự xếp hạng đẹp cỡ nào | Cao (→1) | Retriever / Reranker | Không |
| ∆MRR | Rerank có cải thiện không | Dương (>0) | Reranker | Không |
| Exact Match | Trùng khít đáp án | Cao (→1) | Generator | Không |
| F1 | Trùng bao nhiêu từ | Cao (→1) | Generator | Không |
| Faithfulness | Có bịa không | Cao (→1) | Generator | **Có** |
| Answer Relevance | Có đúng câu hỏi không | Cao (→1) | Generator | **Có** |
| Context Precision/Recall | Chất lượng đoạn đã tìm | Cao (→1) | Retriever | **Có** |
| Dedup Rate | Đoạn trùng lặp | **Thấp** (→0) | Chunking | Không |
| Semantic Integrity | Đoạn có trọn câu | Cao (→1) | Chunking | Không |

Muốn biết **file nào tính chỉ số nào** và **sửa module nào để cải thiện chỉ số
nào**, xem `docs/evaluation.md` mục 7.3.
