# Cấu trúc báo cáo NewsQA RAG

## 0. Quy ước biên soạn

- **Ngôn ngữ:** tiếng Việt; giữ thuật ngữ tiếng Anh khi cần định nghĩa chính xác.
- **Văn phong:** học thuật, ngắn gọn, ưu tiên bảng, hình và số liệu hơn mô tả dài.
- **Trọng tâm:** Phase 1 và Phase 2 chiếm khoảng 50--60% nội dung chính.
- **Phạm vi kết luận:** Phase 2C và Phase 3 chỉ là khảo sát bổ sung hoặc hạn chế, không phải bằng chứng chính để chọn cấu hình.
- Mọi kết quả phải ghi rõ tập dữ liệu, số câu, số bài, cấu hình và chế độ `screening`, `development` hoặc `held-out`.
- Không cập nhật cấu hình cuối theo model ablation cho đến khi thí nghiệm hoàn tất.

## 1. Phần đầu

### 1.1. Trang bìa

- Tên đề tài: **Hệ thống hỏi đáp tin tức có trích dẫn sử dụng RAG**.
- Tên hệ thống: **NewsLens**.
- Môn học, giảng viên, nhóm và thành viên.

### 1.2. Tóm tắt

Viết khoảng 200--300 từ, gồm:

- Bài toán hỏi đáp trên kho tin tức.
- Pipeline retrieval, reranking và generation.
- Bộ NewsQA được rà soát để tạo benchmark.
- Thiết kế thực nghiệm Phase 1--2.
- Cấu hình tốt nhất đã xác định trước model ablation.
- Các thí nghiệm one-shot và generator model bổ sung theo phản hồi giảng viên.

### 1.3. Mục lục và danh mục

- Mục lục.
- Danh mục hình và bảng.
- Danh mục từ viết tắt: RAG, MRR, nDCG, RAGAS, CI95, AC.

## 2. Bài toán và hệ thống

### 2.1. Giới thiệu

- Bối cảnh hỏi đáp trên kho bài báo.
- Khó khăn: truy xuất đúng bằng chứng, trả lời đúng loại thông tin và cung cấp trích dẫn kiểm chứng được.
- Mục tiêu:
  - Sinh câu trả lời dựa trên corpus.
  - Cung cấp citation tới context.
  - Cho phép đánh giá độc lập từng thành phần của pipeline.
- Phạm vi: RAG pipeline, chưa phải Agentic RAG.
- Đóng góp:
  - Dataset đánh giá đã được rà soát và chống trùng lặp.
  - Quy trình thực nghiệm phân tầng cho retrieval và generation.
  - Pipeline có cache, checkpoint và provenance.
  - Phân tích zero-shot, one-shot và generator model.

### 2.2. Cơ sở lý thuyết và công trình liên quan

- Kiến trúc RAG.
- Sparse, dense và hybrid retrieval.
- Cross-encoder reranking.
- Zero-shot và one-shot prompting.
- Đánh giá retrieval và generation bằng RAGAS.
- Chỉ trình bày công thức được dùng trong thực nghiệm.

### 2.3. Phát biểu bài toán

- **Input:** câu hỏi ngôn ngữ tự nhiên về tin tức.
- **Output:** câu trả lời ngắn kèm citation `[n]` tới context.
- **Corpus:** bài báo CNN thuộc NewsQA và distractor articles.

Các câu hỏi nghiên cứu:

- **RQ1:** Retriever và reranker nào tìm bằng chứng tốt nhất?
- **RQ2:** Zero-shot prompt và context depth ảnh hưởng thế nào đến generation?
- **RQ3:** Một ví dụ one-shot có cải thiện chất lượng so với P2 zero-shot không?
- **RQ4:** Generator model ảnh hưởng thế nào đến chất lượng, độ trễ và chi phí?

### 2.4. Kiến trúc hệ thống

- Ingestion và chunking.
- Sparse/dense indexing.
- Retrieval và reranking.
- Context construction.
- Answer generation.
- Citation parsing và reference construction.
- Offline evaluation.

Hình cần có:

1. Pipeline từ câu hỏi đến câu trả lời có trích dẫn.
2. Sequence diagram của một request.
3. Sơ đồ tách online inference và offline evaluation.

## 3. Dữ liệu và phương pháp đánh giá

### 3.1. Dữ liệu

- Nguồn NewsQA và liên kết Hugging Face.
- Dữ liệu ban đầu: article, question, answer span và metadata.
- Chọn 200 evaluation articles và distractor corpus.
- Quy trình rà soát:
  - Sửa câu không standalone.
  - Sửa đáp án sai hoặc bị cắt.
  - Loại câu không có gold defensible.
  - Deduplicate sau khi tạo câu resolved.
- Dataset khóa:
  - 11.064 bài báo.
  - 1.152 câu resolved deduplicated.
  - Recursive chunking 512/64 cho pipeline chính.

Bảng và hình:

- Số bài/câu qua từng bước làm sạch.
- Phân bố issue code.
- Ví dụ lỗi đáp án "35.000 Canadian troops".
- Sơ đồ từ raw NewsQA đến locked artifact.

### 3.2. Chia tập

Chia theo article để tránh leakage giữa các câu cùng bài:

| Tập | Bài | Câu | Vai trò |
|---|---:|---:|---|
| Development | 50 | 281 | Screening và chọn cấu hình |
| Held-out generation | 50 | 284 | Đánh giá cấu hình đã chốt |
| Phase 1 final-test retrieval | 150 | 871 | Kiểm tra retrieval trên bài chưa thấy |
| Reserve | 100 | 587 | Đánh giá bổ sung, không dùng để chọn cấu hình |

### 3.3. Nguyên tắc thực nghiệm

- Mỗi vòng chỉ thay đổi một nhóm thành phần.
- Reuse cùng retrieval trace khi đánh giá generation.
- Screening chỉ chọn ứng viên; development mới dùng để quyết định.
- Held-out chỉ đánh giá khả năng tổng quát hóa, không chọn lại cấu hình.
- Lưu configuration fingerprint, SHA-256, cache và checkpoint cho từng run.

### 3.4. Metrics và kiểm định

- **Retrieval:** MRR@5, nDCG@5, Hit@5, Recall@5 và latency.
- **Generation:** Answer Correctness, Faithfulness, Answer Relevancy, Exact Match và Token F1.
- **Citation:** Citation F1 và Citation Validity.
- **Vận hành:** coverage, latency, token usage và chi phí.
- **Thống kê:** paired bootstrap; cluster bootstrap theo article cho Phase 2; CI95 và mức ý nghĩa 0,05.
- **Quyết định Phase 2:** kiểm tra guardrail faithfulness, citation và coverage trước khi so Answer Correctness.

## 4. Phase 1: Retrieval

### 4.1. Thiết kế

- Mục tiêu: chọn retriever và reranker.
- Giữ cố định corpus, câu hỏi và chunking.
- Ba bước:
  1. Retriever screening.
  2. Reranker và hybrid retrieval.
  3. Final-test trên bài chưa thấy.

### 4.2. Retriever screening

- So sánh BM25 variants, BGE-M3 learned sparse và các dense embedding models.
- Ưu tiên MRR@5; sau đó nDCG@5, Hit@5 và latency.
- Trình bày bảng các cấu hình chính.
- Vẽ scatter plot MRR--latency hoặc nDCG--latency.
- Báo cáo CI95 cho các so sánh quan trọng; không khẳng định thứ hạng khi CI chưa phân định.

### 4.3. Reranker và hybrid

- So sánh không rerank, MiniLM cross-encoder và BGE-large reranker.
- Chạy trên sparse tốt nhất, dense tốt nhất và hybrid RRF.
- Bảng có hai cột MRR/nDCG cho từng reranker.
- Biểu đồ mức tăng chất lượng và chi phí latency.
- Ghi nhận hybrid không chứng minh được lợi ích trong thiết lập đã thử.

### 4.4. Final-test và cấu hình retrieval khóa

- Đánh giá trên 871 câu/150 bài chưa thấy.
- Báo cáo MRR@5, nDCG@5, Hit@5 và Recall@5.
- So sánh reranker với không rerank bằng paired bootstrap.
- Cấu hình đầu ra Phase 1:
  - BGE-M3 sparse.
  - Top-k 20.
  - BGE-large reranker.
  - Top 5 context.

## 5. Phase 2: Generation

### 5.1. Baseline end-to-end

- Cấu hình: retrieval khóa + P0 + depth 5 + Gemini 3.1 Flash-Lite.
- Kết quả trên 281 development questions.
- Bảng đầy đủ generation và citation metrics.
- Phân tích:
  - Faithfulness cao nhưng Answer Correctness còn hạn chế.
  - EM thấp do gold answers ngắn.
  - Câu trả lời dài hoặc sai loại thông tin được hỏi.
  - Tách retrieval failure khỏi generation failure.
- Dùng audit 30 câu correctness thấp làm bằng chứng định tính.

### 5.2. Zero-shot prompt screening

- Trình bày P0--P3 và giả thuyết của từng prompt.
- Screening trên cùng 80 câu; RAGAS dùng cùng subset 20 câu.
- Giữ retrieval trace và depth 5 cố định.
- Bảng AC, Faithfulness, Citation F1 và Citation Validity.
- Screening chỉ chọn prompt triển vọng, không tạo kết luận cuối.

### 5.3. Context-depth screening

- Ma trận P0/P2 × depth 1/3/5.
- Depth 5 tái sử dụng kết quả vòng prompt.
- Trình bày Hit@depth và số câu mất gold so với depth 5.
- Chạy P0-D5, P2-D3 và P2-D5 trên đủ 281 câu.
- Áp guardrail trước khi chọn theo Answer Correctness.

### 5.4. Zero-shot finalist và held-out

- Bảng đầy đủ ba finalist.
- Delta và CI95 của P2-D5 so với P0-D5.
- Giải thích P2-D3 bị loại dù AC cao hơn vì không đạt guardrail.
- Báo cáo P2-D5 trên 284 held-out questions.
- Không dùng held-out để chọn lại cấu hình.
- Phân tầng `gold in top 5` và `gold not in top 5`.

### 5.5. One-shot prompt

- Động cơ: bổ sung một ví dụ hành vi theo phản hồi giảng viên.
- P2-1S kế thừa P2 và thêm một ví dụ synthetic không thuộc development/held-out.
- Ví dụ minh họa input, câu trả lời ngắn và citation đúng.
- So sánh paired P2-1S-D5 với P2-D5 trên cùng development traces.
- Dùng kết quả P2-D5 authoritative đã lưu; không generate hoặc judge lại control.
- Báo cáo:
  - AC, Faithfulness và Answer Relevancy.
  - Token F1, Citation F1 và Citation Validity.
  - Latency, token usage và chi phí.
  - Delta và cluster-bootstrap CI95.
- Held-out P2-1S chỉ dùng để mô tả độ ổn định dev--held-out, không chọn lại prompt.
- Phân tích riêng sự suy giảm Answer Relevancy và ảnh hưởng của câu trả lời cực ngắn.

### 5.6. Generator model ablation

Giữ cố định:

- 281 development questions.
- Retrieval/reranking traces.
- P2-1S và depth 5.
- Judge và scoring protocol.

So sánh:

- Gemini 3.1 Flash-Lite: control đã lưu.
- Gemini 3.5 Flash-Lite.
- Gemini 3.6 Flash.

Không gọi lại hoặc judge lại Gemini 3.1. Bảng kết quả gồm:

- AC, Faithfulness và Answer Relevancy.
- Token F1, Citation F1 và Citation Validity.
- Generation coverage.
- P50/P95 latency.
- Input/output tokens và chi phí ước tính.

Dùng paired comparison vì các model trả lời cùng câu và cùng context. Nếu kết quả chưa hoàn tất, ghi `pending`; không điền số suy đoán. Đây là ablation trên development và không tự động thay đổi cấu hình triển khai.

## 6. Ứng dụng và khả năng tái lập

### 6.1. Ứng dụng

- Kiến trúc backend/frontend.
- Luồng user question → answer → references.
- Cấu hình runtime và external model services.
- Cache, retry và failure handling.
- Hình giao diện và ví dụ câu hỏi--đáp án--nguồn.
- Liên kết video demo.

### 6.2. Khả năng tái lập

- Git commit và configuration fingerprint.
- Hugging Face dataset/index artifacts.
- Notebook theo từng phase.
- Checkpoint và resumable JSONL.
- Manifest và SHA-256.
- Hướng dẫn tối thiểu để tái chạy Phase 1 và Phase 2.

## 7. Thảo luận

### 7.1. Tổng hợp theo câu hỏi nghiên cứu

- Trả lời lần lượt RQ1--RQ4 bằng số liệu đã hoàn tất.
- Phân biệt rõ:
  - Cải thiện quan sát được.
  - Khác biệt có CI95 không chứa 0.
  - Cấu hình qua guardrail.
  - Kết quả chỉ mang tính mô tả.
- Không gọi P2-1S hoặc generator mới là winner nếu protocol chưa hoàn tất.

### 7.2. Hạn chế

- Development chỉ gồm 281 câu/50 bài.
- Gold evidence không bao phủ mọi bằng chứng có thể hợp lệ.
- NewsQA có gold answer ngắn, tạo ưu thế cho câu trả lời ngắn.
- RAGAS phụ thuộc judge model và provider.
- API latency và chi phí thay đổi theo thời điểm.
- Corpus chỉ gồm CNN NewsQA, hạn chế khả năng khái quát sang nguồn khác.
- One-shot example có thể thiên lệch format đáp án.
- Generator ablation chỉ thực hiện trên development.

### 7.3. Khảo sát bổ sung

Đặt Phase 2C và Phase 3 tại đây hoặc trong phụ lục:

- Phase 2C: paragraph, sentence và hierarchical chunking.
- Phase 3: các chính sách abstention.
- Chỉ trình bày ngắn mục tiêu, thiết lập và kết quả quan sát.
- Ghi rõ thiết kế chưa đủ mạnh để xem là thực nghiệm chính.
- Không dùng kết quả để mở rộng kết luận hoặc thay đổi cấu hình khóa.

## 8. Kết luận và hướng phát triển

- Tóm tắt pipeline và các cấu hình đã được kiểm chứng.
- Trả lời ngắn các câu hỏi nghiên cứu.
- Hướng tiếp theo:
  - Thêm generator models.
  - Thử nhiều one-shot/few-shot examples.
  - Dataset đa nguồn tin.
  - Blind human evaluation.
  - Thiết kế abstention độc lập và chặt chẽ hơn.
  - Agentic RAG như phần mở rộng, không phải trạng thái hiện tại.

## 9. Tài liệu tham khảo và phụ lục

### 9.1. Tài liệu tham khảo

- NewsQA paper và Hugging Face dataset.
- RAG, BM25, BGE-M3 và cross-encoder.
- RAGAS.
- Gemini và GLM API/model documentation.

### 9.2. Phụ lục

- System prompts P0--P3 và P2-1S nguyên văn.
- RAGAS judge prompts.
- Bảng kết quả đầy đủ.
- Guardrail definitions.
- Configuration manifests và artifact hashes.
- Ví dụ citation đúng/sai.
- Chi tiết Phase 2C và Phase 3.
- Hướng dẫn chạy notebook và ứng dụng.

## 10. Phân bổ nội dung đề xuất

| Phần | Tỷ trọng |
|---|---:|
| Giới thiệu, lý thuyết và kiến trúc | 15--20% |
| Dữ liệu và phương pháp đánh giá | 15--20% |
| Phase 1 retrieval | 20--25% |
| Phase 2 generation | 30--35% |
| Ứng dụng và tái lập | 5--10% |
| Thảo luận, hạn chế và kết luận | 10--15% |

Phase 2C và Phase 3 không tính vào tỷ trọng nội dung chính; đưa vào khảo sát bổ sung hoặc phụ lục.
