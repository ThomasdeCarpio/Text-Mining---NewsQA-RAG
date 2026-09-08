# TỔNG KẾT PHASE 2C - ABLATION CHIẾN LƯỢC CHUNKING

## 1. Bối cảnh

Phase 2B đã xác định cấu hình chính:

- chunking `C0`: recursive `512/64`;
- retrieval: BGE-M3 learned sparse, `top_k=20`;
- reranker: `BAAI/bge-reranker-large`, lấy tối đa 5 context;
- prompt `P2`: trả lời trực tiếp, ngắn và đúng loại thông tin được hỏi;
- context depth `D5`;
- generator: `gemini-3.1-flash-lite`, reasoning `minimal`;
- judge: `GLM-5.3 Flash`, reasoning `low`.

C0-P2-D5 đã được chạy trên 284 câu thuộc 50 bài held-out trước khi Phase 2C
được thiết kế. Kết quả vẫn còn một số retrieval miss và citation chưa tối ưu.
Vì vậy Phase 2C được bổ sung như một **post-hoc exploratory extension** để kiểm
tra xem cấu trúc chunk tốt hơn có cải thiện đầu vào cho generator hay không.

Phase 2C không dùng output của 284 câu held-out để chọn chunking strategy. Mọi
screening và lựa chọn trong Phase 2C chỉ dùng development/screening đã khóa.

## 2. Câu hỏi nghiên cứu

1. Sentence-aware hoặc paragraph-aware chunking có bảo toàn evidence tốt hơn
   recursive chunking không?
2. Retrieve child nhỏ rồi mở rộng sang parent có cải thiện đồng thời retrieval
   precision và context completeness không?
3. Một strategy mới có tăng Answer Correctness mà không làm giảm Faithfulness,
   Citation F1 và Citation Validity vượt guardrail không?
4. Lợi ích có đủ bù cho chi phí index, token và độ phức tạp runtime không?

## 3. Dataset và các strategy

- Corpus: 11.064 bài báo.
- Evaluation set: 1.152 câu `resolved`, semantic-deduplicated.
- Development: 281 câu thuộc 50 bài, seed `42`.
- Screening: 80 câu; RAGAS judge dùng cùng 20 câu calibration.
- Phase 2 held-out đã mở: 284 câu thuộc 50 bài.
- Generation reserve chưa chạy: 587 câu thuộc 100 bài.

| ID | Strategy | Đơn vị retrieve/rerank | Context cho generator | Số chunks | Index MiB |
|---|---|---|---|---:|---:|
| C0 | Recursive `512/64` | Chunk 512 | Chunk đã retrieve | 22.766 | 64,90 |
| C1 | Sentence-aware | Nhóm câu <= 512 | Nhóm câu đã retrieve | 22.014 | 63,38 |
| C2 | Paragraph-aware | Nhóm đoạn <= 512 | Nhóm đoạn đã retrieve | 22.018 | 63,23 |
| C3 | Hierarchical | Child `256/32` | Parent `512/64` | 49.218 child + 22.766 parent | 76,58 |

C3 tạo nhiều hơn C0 khoảng 116% retrieval units và index lớn hơn khoảng 18%.
Sau rerank, child được ánh xạ và backfill thành tối đa 5 parent khác nhau; số
citation trỏ đến thứ tự parent thực sự đưa vào prompt.

## 4. Quy trình thực nghiệm

### 4.1. Retrieval screening

C0-C3 được chạy retrieval-only trên cùng 281 câu. Với C3, metric dùng để so sánh
end-to-end là metric sau khi child được mở rộng thành delivered parent.

| Strategy | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | P50 tổng |
|---|---:|---:|---:|---:|---:|
| C0 | 0,9573 | 0,9537 | **0,8791** | 0,8963 | 578,7 ms |
| C1 | **0,9680** | **0,9591** | 0,8796 | 0,8979 | 598,7 ms |
| C2 | 0,9609 | 0,9555 | **0,8817** | **0,8989** | 601,6 ms |
| C3 parent | 0,9466 | 0,9431 | 0,8466 | 0,8700 | **354,0 ms** |

Kết luận:

- C1 và C2 tăng nhẹ Hit@5/Recall@5 nhưng effect nhỏ.
- C3 parent thấp hơn C0 khoảng 0,0107 Hit@5 và 0,0106 Recall@5 nhưng vẫn đủ
  điều kiện đi tiếp ở screening, nơi margin loại sớm là 0,02.
- C3 có latency thấp trong run này, nhưng tạo index và runtime phức tạp hơn.
  Latency không được dùng để bù cho suy giảm quality ở vòng cuối.

### 4.2. Generation screening

Mỗi strategy mới được chạy với P2-D5 trên cùng 80 câu; RAGAS tính trên 20 câu.
C0-P2-D5 từ Phase 2B được dùng làm control.

| Strategy | QA F1 | Answer Correctness | Faithfulness | Citation F1 | Citation Validity |
|---|---:|---:|---:|---:|---:|
| C0 | **0,3999** | **0,8363** | **1,0000** | 0,8133 | **0,9750** |
| C1 | 0,3783 | 0,7331 | 0,9433 | **0,8196** | **0,9750** |
| C2 | 0,3801 | 0,8076 | 0,9100 | **0,8196** | 0,9625 |
| C3 | 0,3791 | 0,8094 | 0,9750 | 0,7592 | 0,9625 |

Chỉ C3 được xem là alternative đủ triển vọng để phân tích thêm vì nó giữ
Faithfulness tốt hơn C1/C2 và có Answer Correctness gần C0. Citation F1 thấp là
rủi ro cần kiểm tra lại trên full development. Các RAGAS score ở vòng này chỉ có
20 mẫu nên không được dùng một mình để tuyên bố winner.

### 4.3. C3 context-depth screening

| Depth | QA F1 | Answer Correctness | Faithfulness | Citation F1 | Citation Validity | Input tokens |
|---:|---:|---:|---:|---:|---:|---:|
| D1 | **0,5834** | 0,8451 | 0,9000 | 0,5250 | 0,6250 | **46.281** |
| D3 | 0,5649 | **0,8761** | 0,9500 | 0,7479 | **0,9750** | 113.126 |
| D5 | 0,3791 | 0,8094 | **0,9750** | **0,7592** | 0,9625 | 181.342 |

- D1 trả lời ngắn và đạt QA F1 cao nhưng loại quá nhiều evidence, khiến citation
  quality sụp giảm; cấu hình này bị loại.
- D3 có Answer Correctness screening cao nhất và dùng ít token hơn D5.
- D5 giữ nhiều context hơn và có Faithfulness/Citation F1 cao hơn D3.
- C3-D3 và C3-D5 được đưa vào full-development finalists.

## 5. Kết quả bốn finalists trên development

Cả bốn cấu hình đạt coverage 281/281. C0-D3/D5 được tái sử dụng từ Phase 2B;
C3-D3/D5 được chạy mới với cùng question IDs, prompt, generator và judge.

| Cấu hình | Hit@5 | QA F1 | Answer Correctness | Answer Relevancy | Faithfulness | Citation F1 | Citation Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0-P2-D3 | **0,9573** | 0,5549 | 0,7711 | 0,5694 | 0,9561 | **0,8381** | 0,9751 |
| **C0-P2-D5** | **0,9573** | 0,4191 | 0,7350 | **0,7092** | **0,9801** | **0,8391** | **0,9858** |
| C3-P2-D3 | 0,9466 | **0,5573** | **0,7730** | 0,5689 | 0,9674 | 0,8108 | 0,9786 |
| C3-P2-D5 | 0,9466 | 0,4285 | 0,7415 | 0,6983 | 0,9653 | 0,8116 | 0,9822 |

### 5.1. C3-D3 so với C3-D5

C3-D3 là cấu hình tốt hơn nếu chỉ xét hai C3 finalists:

- Answer Correctness cao hơn 0,0315, CI95 `[+0,0161; +0,0474]`;
- QA F1 cao hơn 0,1288;
- Faithfulness chênh không đáng kể: +0,0021;
- Citation F1 gần như bằng nhau: -0,0008;
- giảm khoảng 38% generation input tokens và 30% tổng chi phí ước tính.

### 5.2. C3 so với control cùng depth

| So sánh | Delta Correctness | Delta Faithfulness | Delta Citation F1 | Delta Hit@5 |
|---|---:|---:|---:|---:|
| C3-D3 trừ C0-D3 | +0,0019 | +0,0113 | **-0,0273** | -0,0107 |
| C3-D5 trừ C0-D5 | +0,0065 | **-0,0148** | **-0,0275** | -0,0107 |

- Correctness gain của C3 so với C0 cùng depth rất nhỏ và CI95 chứa `0`.
- Citation F1 giảm khoảng 0,027 ở cả hai depth, vượt guardrail 0,01. Paired
  article-cluster CI95 đều loại `0`, nên đây không chỉ là khác biệt do làm tròn.
- C3-D5 còn giảm Faithfulness 0,0148; paired CI95 xấp xỉ
  `[-0,0299; -0,0021]`.
- Hit@5 giảm 3 câu trên 281 câu, tương đương khoảng 0,0107.

Citation F1 thấp không chủ yếu do citation numbering sai: Citation Validity của
C3 vẫn khoảng 0,98. Vấn đề chính là parent được trích dẫn không khớp gold context
tốt bằng recursive chunks, làm citation precision/recall giảm.

## 6. Áp dụng guardrail và chọn winner

Một strategy mới chỉ hợp lệ nếu so với C0:

1. Coverage đầy đủ.
2. Hit@5 và Recall@5 không giảm quá 0,01.
3. Faithfulness không giảm quá 0,02.
4. Citation F1 không giảm quá 0,01.
5. Citation Validity không giảm quá 0,01.

C3-D3 và C3-D5 đều không đạt Citation F1 guardrail. Mức tăng Answer Correctness
so với C0 cùng depth cũng không đủ rõ để biện minh cho suy giảm citation và độ
phức tạp artifact/runtime.

C0-D3 có Answer Correctness cao hơn C0-D5, nhưng đã bị loại ở Phase 2B vì
Faithfulness giảm khoảng 0,024, vượt margin 0,02; Citation Validity cũng thấp hơn.
Không được thay đổi guardrail sau khi đã quan sát kết quả.

**Quyết định cuối: giữ C0-P2-D5.** C3-D3 là sensitivity/accuracy alternative,
không phải winner chính thức.

## 7. Generalization result được giữ lại

Vì Phase 2C không thay đổi winner, kết quả held-out đã có của C0-P2-D5 vẫn là
kết quả generalization cuối:

| Partition | Bài | Câu | QA F1 | Answer Correctness | Faithfulness | Citation F1 | Citation Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development | 50 | 281 | 0,4191 | 0,7350 | 0,9801 | 0,8391 | 0,9858 |
| Held-out | 50 | 284 | 0,4313 | 0,7157 | 0,9249 | 0,7289 | 0,9437 |

Answer Correctness giảm 0,0193 trên held-out. Grounding và citation giảm mạnh
hơn, phù hợp với phân tích rằng held-out có nhiều retrieval miss hơn development.

Không chạy final reserve 587 câu do giới hạn thời gian. Tập này được giữ cho
future validation và không được mô tả là đã hoàn thành.

## 8. Ý nghĩa

1. Thay recursive boundary bằng sentence/paragraph boundary không tạo cải thiện
   end-to-end đủ ổn định.
2. Hierarchical retrieval có thể cung cấp context đầy đủ và đạt correctness cao,
   nhưng parent expansion làm citation alignment kém chính xác hơn.
3. Nhiều context hơn không luôn tăng correctness; D3 thường cho câu trả lời ngắn
   và chính xác hơn D5, nhưng D5 bảo toàn evidence và citation tốt hơn.
4. Strategy phức tạp chỉ nên thay control khi cải thiện primary metric mà vẫn
   qua các grounding/citation guardrail.
5. Kết quả Phase 2C củng cố lựa chọn C0-P2-D5 thay vì cung cấp bằng chứng để thay
   production configuration.

## 9. Giới hạn và cách báo cáo

- Phase 2C được thiết kế sau khi đã xem Phase 2 held-out, nên phải gọi là
  post-hoc exploratory extension, không phải phần pre-registered ban đầu.
- Development 281 câu được dùng qua nhiều vòng; adaptive overfitting vẫn có thể
  xảy ra.
- RAGAS screening chỉ chấm 20 câu; kết luận cuối dựa trên full 281-question
  finalists và paired article-cluster analysis.
- Các run cố định số context nhưng không áp đặt một hard token budget giống hệt
  giữa mọi strategy. Input-token usage được báo cáo như outcome; khác biệt độ dài
  context là một phần của treatment và cũng là limitation khi diễn giải.
- Không dùng latency giữa các runtime khác nhau như bằng chứng chính. Quality,
  provenance và paired metrics được ưu tiên.
- Không được tuyên bố Phase 2C đã chạy trên 587-question reserve.

## 10. Artifact chính

- `retrieval screening/phase2c_retrieval_screening.csv`
- `generation screening/{c1,c2,c3}/summary.json`
- `depth screening/{d1,d3}/summary.json`; D5 tái sử dụng C3 generation screening
- `finalists/{c3_d3,c3_d5}/summary.json`
- `finalists/{c3_d3,c3_d5}/finalist_lock.json`
- C0 controls: `../phase2/configuration tunning/finalists/{p2_d3,p2_d5}/`
- Held-out C0-P2-D5: `../phase2/configuration tunning/held out test/`

