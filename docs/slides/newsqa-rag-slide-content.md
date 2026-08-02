# Nội dung thuyết trình NewsQA-RAG

- Thời lượng: **15 phút**
- Số slide đề xuất: **15**
- Trọng tâm: **Data và Evaluation**
- Mỗi slide nên dùng ít chữ, tối đa 3–5 ý; phần chi tiết nằm trong mục “Nội dung nói”.

> Thông điệp chính: hệ thống thực hiện factual QA có dẫn nguồn trên một thư viện nhiều bài báo. Đóng góp quan trọng của project không chỉ là pipeline RAG, mà còn là cách chuyển NewsQA sang bài toán library-scale và đánh giá riêng từng module lẫn toàn pipeline.

---

## Slide 1 — Xây dựng và đánh giá NewsQA-RAG

### Chữ trên slide

**Xây dựng và đánh giá NewsQA-RAG**  
Hỏi đáp có dẫn nguồn trên một thư viện gồm nhiều bài báo

### Nội dung nói

- Đây không chỉ là một chatbot; project tập trung đo chất lượng của từng phần trong hệ thống RAG.
- Khác NewsQA gốc, hệ thống không được cung cấp sẵn bài báo chứa đáp án.
- Bài trình bày tập trung nhiều nhất vào dữ liệu và phương pháp đánh giá.

### Gợi ý hình

- Sơ đồ: câu hỏi → thư viện bài báo → câu trả lời kèm citation.
- Hoặc screenshot giao diện chat/retrieval playground.

---

## Slide 2 — Định nghĩa bài toán

### Chữ trên slide

- **Input:** câu hỏi + thư viện bài báo.
- **Output:** câu trả lời ngắn + nguồn/chunk làm bằng chứng.
- **Hiện tại:** factual question answering.
- **Tiếp theo:** câu hỏi cần tổng hợp nhiều nguồn.

### Nội dung nói

- Bài toán cốt lõi là tìm đúng bằng chứng trong toàn bộ corpus trước khi sinh câu trả lời.
- Hệ thống tốt phải đồng thời: tìm đúng, xếp đúng thứ tự, trả lời đúng và dẫn nguồn đúng.
- Vì vậy chúng ta sẽ đánh giá đồng thời từng module và toàn bộ hệ thống với metric cụ thể, để chỉ ra điểm mạnh từng phần, cũng như đánh giá tổng thể của hệ thống.
---

## Slide 3 — Các module và input/output

### Chữ trên slide

| Module | Input | Output |
|---|---|---|
| Preprocessing | Article | Clean article |
| Chunking | Clean article | Chunks + metadata |
| Indexing | Chunks | Dense/BM25 index |
| Retrieval | Question | Top-10 chunks |
| Reranking | Question + top-10 | Top-5 chunks |
| Generation | Question + top-5 | Answer + citations |
| Evaluation | Predictions + labels | Metrics + error analysis |

### Nội dung nói

- Các phần như preprocessing, chunking, index thì được chuẩn bị trước
- Khi người dùng hỏi 1 câu hỏi hệ thống sẽ bắt đầu truy xuất từ database thông qua các thuật toán truy xuất
- Một điều lưu ý về dữ liệu là Metadata phải giữ được article ID, chunk ID và vị trí trong đoạn văn bản để bước sau phục vụ đánh giá.
- Các Module được thiết kế theo OOP để có thể dễ dàng thay thế và đánh giá độc lập.
act trung gian: corpus → chunks → index → ranked chunks → answer.

---

## Slide 4 — Kiến trúc có thể mở rộng

### Chữ trên slide

**Các thành phần có thể thay thế độc lập**

- Embedding model
- Retriever: Dense / BM25 / Hybrid
- Reranker: No-op / Cross-Encoder
- Generator và Judge
- Chunking strategy

### Nội dung nói

- Các module được định nghĩa theo từng class, có interface thống nhất, giúp việc thêm mô hình mới dễ dàng mà không viết lại toàn pipeline.
- Khi một module đã chạy xong, các module phía sau có thể tận dụng lại kết quả này mà không cần phải chạy lại toàn bộ pipeline
- Lưu ý khi demo: cấu hình live chat hiện có thể khác cấu hình benchmark; cần ghi rõ retriever/reranker thực sự đang dùng.

---

## Slide 5 — Vì sao NewsQA gốc chưa phù hợp?

### Chữ trên slide

**NewsQA gốc**

- Câu hỏi được viết khi đã biết bài báo.
- Nhiều câu phụ thuộc ngữ cảnh của bài.
- Phù hợp đánh giá đọc hiểu trong một document.

**Bài toán mới**

- Tìm evidence trong **11.064 bài báo**.
- Câu hỏi phải tự đứng độc lập.
- Có thêm nhiều bài nhiễu.

### Nội dung nói

- Vì NewsQA ban đầu được dùng để đánh giá các mô hình cũ, các mô hình bert khi mà cửa sổ ngữ cảnh không lớn và còn hạn chế, nên các chunk cần phải đủ nhỏ khoản vài chục token so với 512 token như hiện tại, nên dữ liệu bạn đầu chỉ được thiết kế để đánh giá khả năng truy xuất chunk trong từng document.

- Ví dụ câu hỏi gốc: “Who is protesting?”
- Sau chỉnh sửa: “Who is protesting the results of the presidential election in Gabon?”
- Việc làm rõ câu hỏi không chỉ là làm sạch văn bản; nó thay đổi benchmark thành bài toán retrieval thực tế hơn.

---

## Slide 6 — Xây dựng bộ benchmark mới

### Chữ trên slide

- **638** bài ứng viên được kiểm tra.
- Chọn **200** bài để đánh giá.
- **1.340** câu hỏi ban đầu.
- Loại **4** mẫu không hợp lệ → **1.336** mẫu được chấm.
- Thêm **10.864** bài nhiễu.
- Corpus cuối: **11.064 bài**, **19.263 chunks**.
- Random seed: **42**.

### Nội dung nói

- Từ dữ liệu newsqa gốc, chúng tôi chọn ra tập 200 bài để làm câu hỏi và ground-truth evidence. Đồng thời thì cũng dùng các phương pháp LLM để điều chỉnh lại câu hỏi cho cụ thể hơn để có thể search khi bài báo nằm trong corpus nhiều bài báo.

- Ngoài ra tất cả các bài trong tập train ~10k bài được chọn làm corpus để test, chúng tôi còn thêm các bài khác vào để đóng vai trò distractor để kiểm tra khả năng tìm kiếm những bài mục tiêu ở quy mô thư viện.

- Mọi cấu hình benchmark dùng chung corpus, chunker và quy tắc gán evidence để so sánh công bằng.

---

## Slide 7 — EDA và đặc trưng của thư viện

### Chữ trên slide

| Đặc trưng | Giá trị |
|---|---:|
| Eval articles | 200 — **1,8%** |
| Distractor articles | 10.864 — **98,2%** |
| Tổng chunks | **19.263** |
| Trung bình chunks/article | **1,74** |
| Trung bình tokens/chunk | **383,9** |
| Max tokens/chunk | **504** |
| Chunk limit | **512** |
| Chunk compliance | **100%** |
| Metadata completeness | **100%** |
| Deduplication rate | **0,07%** |

### Nội dung nói

- Corpus rất mất cân bằng có chủ đích: phần lớn tài liệu là distractor.
- Chunk trung bình khá dài nhưng vẫn nằm dưới giới hạn 512 token.
- Self-retrieval sanity check trên 50 mẫu đạt Recall@1 = **0,96**.
- Trong 200 bài eval: độ dài bài trung bình **2.947 ký tự**, p95 **4.201**; trung bình **6,7 câu hỏi/bài**, tối đa **23**.

---

## Slide 8 — Chất lượng dữ liệu sau hiệu chỉnh

### Chữ trên slide

- **1.078** câu hỏi được làm rõ.
- **298** đáp án được sửa.
- **4** mẫu bị loại.

| Nhãn chất lượng | Số mẫu |
|---|---:|
| Non-standalone | 979 |
| Uncertain | 99 |
| Standalone | 258 |
| Invalid | 4 |

### Nội dung nói

- Sau khi dùng LLM để đánh giá bộ test (200 bài báo nói trên) thì chúng tôi rút ra được các số liệu sau. Cung cấp system prompt  [agent đi tìm phần này].
- Các lỗi phổ biến nhất: thiếu chủ thể **688**, thiếu sự kiện cụ thể **139**, tham chiếu chung chung **98**, evidence sai **87**, malformed **71**.
- Ngoài ra có đáp án bị cắt **59**, unresolved coreference **58**, thiếu địa điểm **55**, đáp án sai **35**.
- Đây là lý do cần audit và sửa dữ liệu trước khi so sánh mô hình.

---

## Slide 9 — Khung đánh giá từng module trên toàn hệ thống

### Chữ trên slide

1. **Corpus:** compliance, duplicate, metadata.
2. **Retrieval:** Hit, Recall, MRR, NDCG.
3. **Reranking:** thay đổi MRR/NDCG trước và sau rerank.
4. **Generation:** Exact Match, token F1.
5. **Citation:** validity, precision, recall, F1.
6. **Operation:** latency, coverage, failure rate.

### Nội dung nói

- Đánh giá theo từng thành phần giúp chúng ta phân biệt và xác định lỗi một cách trực quan hơn, ví dụ như lỗi “không tìm thấy evidence” với lỗi “đã có evidence nhưng trả lời sai” có thể được hiểu là sai sót ở 2 module khác nhau, cái đầu tiên là lỗi của retriever và cái thứ 2 là lỗi của generator.

- Pipeline cuối phải báo cả chất lượng và độ ổn định của toàn bộ hệ thống khi vận hành, được đánh giá bằng những trải nghiệm thật của thành viên trong nhóm ở 1 số bài test, và đc đánh giá theo phương pháp Ragas ở bộ test có kích thước lớn hơn.

---

## Slide 10 — Retrieval được chấm như thế nào?

### Chữ trên slide

**Chuyển evidence sang chunk ID**

Một chunk là relevant nếu khoảng ký tự của nó giao với evidence:

```text
chunk_start < evidence_end
AND
evidence_start < chunk_end
```

Ví dụ:

- Relevant chunks: **{C12, C13}**
- Kết quả xếp hạng: **[C7, C12, C4, C13, C9]**
- Hit@5 = **1**
- MRR@5 = **1/2 = 0,50**
- Recall@5 = **2/2 = 1,00**
- NDCG@5 ≈ **0,65**

### Nội dung nói

- NewsQA cung cấp evidence bằng vị trí ký tự trong bài; production retrieval trả về chunk ID.
- Vì vậy, ta ánh xạ mọi chunk giao với evidence thành tập `relevant_chunk_ids`, rồi so sánh chính xác với `retrieved_ids`.
- **Hit@k:** chỉ hỏi có ít nhất một evidence trong top-k hay không.
- **MRR@k:** nghịch đảo vị trí của evidence đầu tiên; evidence càng lên đầu càng tốt.
- **Recall@k:** bao nhiêu relevant chunks đã được lấy ra.
- **NDCG@k:** thưởng cho việc xếp nhiều relevant chunks ở vị trí cao, có chiết khấu theo rank.
- Tính metric theo từng câu hỏi rồi lấy trung bình trên **1.336 mẫu**.

### Ý nghĩa và giới hạn

- Hit cao là điều kiện cần, chưa chứng minh được câu trả lời được sinh ra là đúng.
- MRR chỉ quan tâm chunk liên quan đầu tiên.
- Recall phụ thuộc cách chia chunk và số lượng chunk liên quan.
- NDCG ở đây dùng binary relevance, chưa phản ánh mức độ evidence mạnh/yếu.
- Vì vậy phải đọc các chỉ số trên cùng với các chỉ số QA F1, citation metrics, coverage và latency.

---

## Slide 11 — Thiết lập thực nghiệm retrieval

### Chữ trên slide

- **2 phiên bản câu hỏi:** Original / Resolved.
- **3 retriever:** Dense / BM25 / Hybrid.
- **2 reranker:** No-op / Cross-Encoder.
- Tổng cộng: **12 cấu hình × 1.336 câu hỏi**.
- Chunk size **512**, overlap **64**.
- Retrieve top-10 → rerank top-5.
- Dense embedding: MiniLM, **384 dimensions**.
- Hybrid RRF weights: Dense **0,7**, BM25 **0,3**.
- Cross-Encoder: `ms-marco-MiniLM-L-6-v2`.

### Nội dung nói

- Với các chuẩn đầu vào và đầu ra đã được định nghĩa rõ ràng. Chúng tôi dùng factory design pattern, để giúp quản lý các phiên bản cũng như method khác nhau cho từng module một cách hiệu quả.

- Từng cấu hình sẽ chạy trên cùng benchmark để so sánh với nhau.

- Cần báo cáo cả metric chất lượng lẫn latency vì Cross-Encoder tăng chi phí đáng kể.

- Nói rõ là retriever chưa có cài gì đặc biệt, kể cả system promt cho query, chỉ đi embed rồi truy vấn cả querry gốc.

### Gợi ý hình

- Ma trận 2 × 3 × 2 cho 12 cấu hình.
- Sơ đồ top-10 → Cross-Encoder → top-5.

---

## Slide 12 — Kết quả benchmark retrieval

### Chữ trên slide

Kết quả trên **resolved questions**:

| Retriever | Reranker | Hit@5 | MRR@5 | Latency |
|---|---|---:|---:|---:|
| Dense | No-op | 0,6759 | 0,5206 | 24,7 ms |
| Dense | Cross-Encoder | 0,7253 | 0,6470 | 390,1 ms |
| BM25 | No-op | 0,7919 | 0,6561 | 77,0 ms |
| **BM25** | **Cross-Encoder** | **0,8256** | **0,7369** | 422,3 ms |
| Hybrid | No-op | 0,7320 | 0,5955 | 107,4 ms |
| Hybrid | Cross-Encoder | 0,7672 | 0,6766 | 476,7 ms |

### Nội dung nói

- Cross-Encoder cải thiện cả ba retriever, nhưng latency tăng rõ rệt.
- **BM25 + Cross-Encoder** là cấu hình tốt nhất hiện tại về Hit@5 và MRR@5.
- Hybrid chưa vượt BM25; nguyên nhân có thể là trọng số RRF hoặc dense embedding chưa phù hợp.
- Không nên mặc định kiến trúc phức tạp hơn sẽ tốt hơn; cần dựa vào benchmark.

### Gợi ý hình

- Grouped bar chart Hit@5/MRR@5 cho 6 cấu hình.
- Một callout “Best quality: BM25 + CE”; thêm marker latency.

---

## Slide 13 — Tác động của việc hiệu chỉnh dữ liệu

### Chữ trên slide

So sánh Hit@5 khi đều dùng Cross-Encoder:

| Retriever | Original | Resolved | Thay đổi |
|---|---:|---:|---:|
| Dense | 0,3256 | 0,7253 | **+39,97 điểm %** |
| BM25 | 0,3683 | 0,8256 | **+45,73 điểm %** |
| Hybrid | 0,3570 | 0,7672 | **+41,02 điểm %** |

### Nội dung nói

- Đây là **data effect**, không phải model gain.
- Câu hỏi khi được làm rõ chủ thể, sự kiện và địa điểm giúp cả lexical lẫn semantic retrieval trở nên tốt hơn.
- Mức tăng lớn cho thấy formulation của benchmark có thể ảnh hưởng nhiều hơn việc đổi mô hình.
- Khi báo kết quả phải ghi rõ đang dùng original hay resolved questions.


---

## Slide 14 — Đánh giá end-to-end và các lưu ý

### Chữ trên slide

Thiết lập: Resolved + Hybrid + Cross-Encoder + DeepSeek V4 Flash

| Chỉ số | Kết quả |
|---|---:|
| Retrieval Hit@5 | 0,7657 |
| Retrieval MRR@5 | 0,6751 |
| Request thành công | 760 / 1.336 |
| Coverage | **56,89%** |
| QA F1 — toàn bộ | 0,1693 |
| QA F1 — request thành công | 0,2976 |
| EM — request thành công | 0,0368 |
| Citation F1 — toàn bộ | 0,3835 |
| Citation validity/coverage | 0,4963 |
| Mean latency | 2.391,4 ms |
| P95 latency | 4.804,6 ms |

### Nội dung nói

- Có **576** request lỗi, chủ yếu là `APIConnectionError`; vì vậy chưa thể coi đây là kết quả end-to-end hoàn chỉnh.
- Luôn báo song song điểm trên toàn bộ tập, điểm trên request thành công và coverage.
- Pilot 50 mẫu có generation chạy đủ: QA F1 **0,2899**, Citation F1 **0,6833**.
- RAGAS mới có coverage **40/50**, nhưng record điểm đang trống; **chưa công bố RAGAS score**.
- Retrieval tốt không đảm bảo generation đúng; đây là hai nguồn lỗi riêng.

### Gợi ý hình

- Funnel: 1.336 expected → 760 success / 576 failed.
- Bốn KPI lớn: Hit@5, QA F1, Citation F1, Coverage; đặt cảnh báo cạnh coverage.

---

## Slide 15 — Kết luận và kế hoạch 5 tuần

### Chữ trên slide

**Kết luận**

- Data formulation là yếu tố ảnh hưởng lớn nhất.
- Cross-Encoder tạo cải thiện ổn định.
- BM25 + Cross-Encoder là baseline tốt nhất hiện tại.
- Phải báo đồng thời quality, coverage và latency.

**Nguồn lực:** 4 người × 3–4 giờ/tuần × 5 tuần = **60–80 giờ**.

### Kế hoạch

| Tuần | Milestone |
|---|---|
| 1 | Retry 576 lỗi, sửa judge/RAGAS, khóa baseline tái lập được |
| 2 | Thử chunk 256/512/768, overlap; MiniLM vs MPNet/BGE; tạo tập multi-source nhỏ |
| 3 | Tune RRF weights, thử reranker thứ hai, chọn top-2 cấu hình |
| 4 | So sánh generator; cải thiện prompt citation; prototype multi-source retrieval |
| 5 | Full end-to-end; independent judge; failure analysis; bootstrap 95% CI; hoàn thiện demo |

### Nội dung nói

- Phân bổ đề xuất: Data/Eval **30%**, Retrieval **25%**, Generation/Agent **25%**, Integration **15%**, UI/UX **5%**.
- Ưu tiên đầu tiên là benchmark đáng tin cậy và có thể tái lập.
- Agentic/multi-source QA chỉ nên mở rộng sau khi baseline một lượt retrieval đã ổn định.
- UI/UX tập trung vào hiển thị citation, evidence và trạng thái lỗi thay vì nhiều tính năng trang trí.

### Gợi ý hình

- Timeline 5 tuần hoặc swimlane bốn thành viên.
- Roadmap từ “reliable benchmark” → “better modules” → “multi-source demo”.

---

# Nguồn số liệu trong repo

| Nội dung | Nguồn tham khảo |
|---|---|
| Corpus integrity và review | `integrity_report.json` |
| EDA tập được chọn | Selection manifest / EDA outputs |
| Cấu hình benchmark | Variant manifest |
| Corpus và chunk diagnostics | `newsqa_200_11064_corpus.json` |
| 12 retrieval benchmark | `reports/benchmarks/{original,resolved}_*/report.json` |
| End-to-end benchmark | Full resolved report và attempts |
| Công thức metric | `src/evaluation/metrics.py` |
| Ánh xạ evidence → chunk ID | `src/evaluation/testset.py` |
| Mô tả dataset/evaluation | `docs/evaluation_dataset.md`, `docs/benchmarking.md` |

# Các lưu ý bắt buộc khi trình bày

- Không gọi mức tăng Original → Resolved là “model improvement”; đây là ảnh hưởng của dữ liệu.
- Không nói Hybrid là tốt nhất: kết quả hiện tại cho thấy **BM25 + Cross-Encoder** tốt nhất.
- Không công bố RAGAS score khi score records còn trống.
- Khi nói về generation phải nêu coverage **56,89%** và 576 request lỗi.
- Retrieval Hit không đồng nghĩa answer đúng.
- Chưa khẳng định khác biệt có ý nghĩa thống kê cho đến khi chạy paired test hoặc bootstrap confidence interval.
