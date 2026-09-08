# NewsQA RAG — Tóm tắt

> Chi tiết: **[Phase 1](phase1/report.md)** · **[Phase 2](phase2/report.md)** · [EDA](../eda/eda_report.md) · [Prompt nguyên văn](../prompts.md) · [Báo cáo gộp hai phase](report_detail.md)
> Bản nộp: [`docs/latex/`](../latex/) — slide (27 trang) và báo cáo (9 trang), số sinh tự động

Hệ thống hỏi–đáp trên 11.064 bài báo CNN. Người dùng hỏi một câu, hệ thống tìm đoạn văn chứa câu trả lời rồi để LLM viết đáp án kèm trích dẫn.

```
Câu hỏi → [ tìm đoạn văn ] → [ viết đáp án ] → đáp án + citation
             PHASE 1              PHASE 2
```

Dự án chia hai giai đoạn vì hai việc đó hỏng theo hai cách khác nhau và phải đo bằng hai loại thước khác nhau.

| | Trạng thái | Chốt được gì |
| :--- | :--- | :--- |
| EDA — khảo sát dữ liệu | ✅ | Biên độ nhập nhằng nhãn chunk 7,0–24,5%; phục hồi 4.603 bài bị cắt |
| **Phase 1** — chọn cách lấy đoạn văn | ✅ khóa | BGE-M3 sparse + bge-reranker-large + chunk 512/64 |
| **Phase 1 — final-test** | ✅ **đã chạy** | **Hit@5 0,8978 trên 871 câu / 150 bài** |
| **Phase 2** — chọn cách hỏi LLM | ✅ chọn xong | Prompt P2, 5 đoạn context |
| **Phase 2 — held-out** | ✅ **đã chạy** | **AC 0,7157 trên 284 câu — số công bố** |
| Phase 2C — chọn cách cắt đoạn | ✅ đã chạy | Không chiến lược nào qua guardrail; giữ nguyên recursive 512/64 |
| Phase 3 — biết từ chối trả lời | ✅ đã chạy | Không chính sách nào qua guardrail; prompt Phase 2 đã đủ |

---

# PHASE 1 — Chọn cách lấy đoạn văn

**Việc của nó:** trong 22.766 đoạn văn, tìm ra 5 đoạn nhiều khả năng chứa câu trả lời nhất. Không gọi LLM, không tốn tiền API.

**Đã thử:** 23 cấu hình, chia 3 vòng để khỏi phải chạy đủ 72 tổ hợp.

| Vòng | Thử gì | Thắng |
| :--- | :--- | :--- |
| 1 | 4 mô hình dense × 4 mô hình sparse | **BGE-M3 sparse** |
| 2 | 3 retriever × 3 reranker | **+ bge-reranker-large** |
| 3 | 3 kích thước chunk (256/512/1024) | 512/64 — *không ai thắng rõ* |
| + | Ablation: contextual chunking | *giúp dense, hại sparse — không đổi quyết định* |

**Kết quả cấu hình khóa** (281 câu development, tập `resolved`):

| Hit@1 | Hit@5 | nDCG@5 | Recall@5 | Latency P50 |
| ---: | ---: | ---: | ---: | ---: |
| 0,8221 | **0,9573** | 0,8976 | 0,9555 | 513 ms |

**Vì sao chọn như vậy:**

- **Sparse thắng dense +0,1634 nDCG@5** (CI95 [+0,1231; +0,2080]) — không phải chênh lệch nhỏ. EDA giải thích: câu hỏi NewsQA bám vào tên riêng, ngày tháng, con số — thứ mà khớp từ vựng bắt được còn embedding thì làm nhòe đi.
- **Reranker thêm +0,0659 nDCG@5.** EDA đo được mỗi câu hỏi có trung vị **25 đoạn đối thủ** cùng chủ đề, nên lọc lại là bắt buộc chứ không phải tùy chọn.
- **Chunk size là kết quả null.** Ba kích thước chồng lấn nhau; chọn 512 vì lý do vận hành, không vì nó thắng.
- **Contextual chunking:** thêm 160 ký tự ngữ cảnh bài báo vào đầu mỗi chunk giúp dense (+0,0293) nhưng hại sparse (−0,0208) — cả hai đều có ý nghĩa thống kê, cả hai đều dưới ngưỡng nhiễu 7,0%. Sparse vẫn thắng trên cả hai kho, nên quyết định không đổi.

**Nghiệm thu — 871 câu / 150 bài chưa từng chạm tới**, chạy đúng một lần, chỉ truy xuất, coverage 871/871. Bản ghi giao thức mang ba cờ: `winner_locked_before_heldout`, `confirmatory_only`, `no_post_heldout_reselection`.

| | Dev (281 câu) | **Final-test (871 câu)** | Δ | CI95 final-test | Hai CI chồng lấn? |
| :--- | ---: | ---: | ---: | :--- | :-- |
| Hit@1 | 0,8221 | 0,7543 | −0,0678 | [0,7161; 0,7939] | có |
| Hit@3 | 0,9324 | **0,8611** | −0,0713 | [0,8298; 0,8904] | **không** |
| Hit@5 | 0,9573 | **0,8978** | −0,0595 | [0,8722; 0,9210] | **không** |
| MRR@5 | 0,8797 | 0,8112 | −0,0685 | [0,7803; 0,8423] | có |
| nDCG@5 | 0,8976 | **0,8313** | −0,0663 | [0,8019; 0,8598] | **không** |
| Recall@5 | 0,9555 | **0,8955** | −0,0600 | [0,8693; 0,9199] | **không** |

Bốn trong sáu metric có hai khoảng tin cậy tách rời: **tập development dễ hơn phần còn lại của kho, và đây là số đo chứ không phải phỏng đoán.** Hệ quả: mọi con số *tuyệt đối* của Phase 1 đo trên development lạc quan khoảng 0,06–0,07. Các *so sánh* (sparse ↔ dense, có ↔ không reranker) không bị ảnh hưởng, vì chúng ghép cặp trong cùng một tập.

Reranker vẫn có lợi trên dữ liệu khó hơn: ΔnDCG@5 **+0,0758** (CI95 [+0,0504; +0,1017]), ΔMRR@5 +0,0843, ΔnDCG@1 +0,1102 — cả ba khoảng đều không chứa 0, và giá trị đo ở vòng 2 (+0,0659) nằm trong khoảng của final-test.

Latency P50 tổng 550,0 ms (dev 513 ms), trong đó rerank 474,0 ms.

---

# PHASE 2 — Chọn cách hỏi LLM

**Việc của nó:** với 5 đoạn văn Phase 1 đưa sang, tìm cách ra lệnh cho Gemini sao cho nó trả lời đúng mà không bịa. **Không huấn luyện lại gì cả** — chỉ đổi hai thứ: nội dung prompt, và số đoạn context (`context_depth` = cắt `ranked_chunks[:n]`, không truy xuất lại).

**Điểm xuất phát** (prompt gốc P0, 5 context, 281 câu):

| Answer Correctness | Exact Match | Faithfulness |
| ---: | ---: | ---: |
| 0,6308 | **0,0000** | 0,9761 |

Ba số này đọc cùng nhau ra một chẩn đoán: **hệ thống không bịa, nó chỉ nói dài.** Model biết đáp án nhưng gói trong một đoạn văn, trong khi đáp án chuẩn của NewsQA là một span rất ngắn — một cái tên, một mốc thời gian, một con số.

**Đã thử:** 4 prompt × 3 mức độ sâu context. Mỗi prompt nhắm vào một nhóm lỗi đã đếm được từ audit tay 30 câu điểm thấp. [Nội dung nguyên văn](../prompts.md).

| Prompt | Nhắm nhóm lỗi | Cỡ nhóm | Kết quả |
| :--- | :--- | ---: | :--- |
| P1 — siết grounding | Hallucination | ~0 | ❌ nhắm vào vấn đề không tồn tại |
| **P2 — trả lời ngắn, đúng answer type** | Đúng nhưng quá dài (10) + sai answer type (1) | **11/30** | ✅ thắng |
| P3 — chọn đúng đoạn trước khi trả lời | *(không nhắm nhóm lỗi nào của audit; xuất phát từ phát hiện distractor collision của EDA §7)* | — | ⚠️ đúng hướng, chưa đủ mạnh |

**Chọn:** prompt **P2** với **5 đoạn context**.

| | P0 (gốc) | **P2-depth5 (chọn)** | P2-depth3 (loại) |
| :--- | ---: | ---: | ---: |
| Answer Correctness | 0,6308 | **0,7350** | *0,7711* |
| Exact Match | 0,0000 | **0,0925** | *0,2633* |
| Faithfulness | 0,9761 | **0,9801** | 0,9561 ❌ |
| Citation Validity | 0,9893 | 0,9858 | 0,9751 ❌ |

**Vì sao chọn P2-depth5 dù P2-depth3 điểm cao hơn:** Phase 2 có **4 điều kiện khóa trước khi nhìn thấy kết quả**. P2-depth3 trượt 2 trong 4. Kiểm định theo cặp xác nhận **cả hai mức tụt đều là thật** (CI95 không chứa 0). Nới ngưỡng lúc này chính là hành vi mà việc đăng ký trước sinh ra để ngăn.

---

# HELD-OUT — Con số được phép công bố

284 câu / 50 bài chưa từng bị chạm tới, chạy **đúng một lần** bằng cấu hình đã khóa. Quyết định winner ký duyệt **13:31Z**, run bắt đầu **14:23Z** — thứ tự thời gian là bằng chứng, và 4/5 hash artifact kiểm lại khớp.

| | Dev (281) | **Held-out (284)** | macro |
| :--- | ---: | ---: | ---: |
| Answer Correctness | 0,7350 | **0,7157** | 0,7230 |
| Exact Match | 0,0925 | 0,1092 | — |
| Token F1 | 0,4191 | 0,4313 | 0,4370 |
| Faithfulness | 0,9801 | 0,9249 | 0,9318 |
| Citation F1 | 0,8391 | 0,7289 | 0,7315 |
| Citation Validity | 0,9858 | 0,9437 | 0,9483 |
| **Hit@5** (truy xuất) | 0,9573 | **0,8768** | — |

Chi phí toàn bộ run: **0,94 USD**. Latency P50 1.958 ms. Micro và macro lệch dưới 0,008 ở mọi metric.

**Phát hiện quan trọng nhất — prompt không bị overfit:**

| Held-out | n | Answer Corr. | Citation F1 |
| :--- | ---: | ---: | ---: |
| `gold_in_top5` | 249 | **0,7689** | 0,8313 |
| `gold_not_in_top5` | 35 | 0,3375 | **0,0000** |

Trên các câu **truy xuất làm đúng việc**, held-out đạt 0,7689 (CI95 [0,7391; 0,7978]) — khoảng này **chứa** giá trị development 0,7597, tức hai bên **không phân biệt được**. Cái thay đổi là tỉ lệ truy xuất trượt: 12/281 → 35/284, gần ba lần.

> Chênh lệch giữa development và held-out đến từ nhóm `gold_not_in_top5` lớn hơn, không từ việc prompt kém tổng quát.

---

# PHASE 2C — Cách cắt đoạn có quan trọng không?

Vòng 3 của Phase 1 nói *kích thước* chunk không quan trọng. 2C hỏi tiếp: *cấu trúc*
thì sao? Bốn chiến lược, cùng 281 câu development, cùng prompt P2.

| ID | Chiến lược | Đơn vị retrieve | Context cho LLM | Số đơn vị |
| :-- | :--- | :--- | :--- | ---: |
| `C0` | Recursive 512/64 *(control)* | chunk 512 | chính chunk đó | 22.766 |
| `C1` | Theo câu | nhóm câu ≤512 | chính nhóm đó | 22.014 |
| `C2` | Theo đoạn văn | nhóm đoạn ≤512 | chính nhóm đó | 22.018 |
| `C3` | Phân cấp | child 256/32 | **parent** chứa child | 49.218 + 22.766 |

C1 và C2 bị loại ở vòng sàng lọc sinh (80 câu, RAGAS chấm 20). Chỉ C3 vào chung
kết, với hai mức depth.

**Kết quả: không cấu hình nào qua hết 6 guardrail. Winner giữ nguyên C0-P2-D5.**

| So với C0-P2-D5 | Answer Correctness | Citation F1 | Hit@5 | Recall@5 |
| :--- | ---: | ---: | ---: | ---: |
| C3-P2-D3 | **+0,0380** ✅ | −0,0284 ❌ | −0,0107 ❌ | −0,0125 ❌ |
| C3-P2-D5 | +0,0065 | −0,0275 ❌ | −0,0107 ❌ | −0,0125 ❌ |

C3-P2-D3 là **cấu hình cao điểm nhất của cả dự án** trên development (AC 0,7730,
CI95 của chênh lệch [+0,0183; +0,0575]) và vẫn bị loại vì trượt ba guardrail.

Cơ chế Citation F1 tụt: không phải đánh số citation sai — Citation Validity của C3
vẫn khoảng 0,98. Là đoạn *parent* được trích dẫn khớp gold context kém hơn chunk
recursive.

> Hai phán quyết truy xuất là **áp luật đã đăng ký trước**, không phải kết luận
> thống kê: CI95 của Hit@5 [−0,0283; +0,0072] và Recall@5 [−0,0300; +0,0054] đều
> chứa 0. Ba trong năm ngưỡng số của 2C nằm dưới độ phân giải của cỡ mẫu.

2C được thiết kế **sau khi** đã mở held-out, nên là phần mở rộng khám phá hậu
kiểm, không phải phần đăng ký trước ban đầu. Tập dự trữ 587 câu chưa chạy.

---

# PHASE 3 — Hệ thống có biết im lặng không?

Bài toán đảo trục: metric chính không còn là "trả lời đúng tới đâu" mà là
**tỉ lệ trả lời bừa khi không đủ bằng chứng**. Bộ dữ liệu phải dựng riêng: 200
case (140 development / 60 final-test), gồm 87 câu trả lời được làm control và 6
loại câu không trả lời được.

| | Chính sách | Là gì |
| :-- | :--- | :--- |
| `B0` | control | Đúng prompt P2 của Phase 2, không sửa một chữ |
| `B1` | schema | Bắt model trả JSON có trường `answerability` |
| `B2` | schema + cổng | B1, cộng cổng chặn theo điểm reranker |

**Kết quả: B0 thắng — không đổi gì cả.**

| Development, 140 câu | B0 | B1 / B2 |
| :--- | ---: | ---: |
| False-answer rate | 5,06% | **1,27%** |
| Token F1 các câu trả lời được | **0,4490** | 0,3420 ❌ |
| Abstention F1 | 0,9740 | **0,9936** |

B1 hạ false-answer rate gần bốn lần nhưng làm token F1 tụt **−0,1070**
(CI95 [−0,1563; −0,0631]), gấp hơn năm lần ngưỡng 0,02.

**Nguyên nhân, và đây là chỗ đáng chú ý:** prompt B1 bỏ mất câu lệnh quy định
*hình dạng đáp án* của P2 và chỉ còn "concise answer". Đáp án dài trở lại — trung
bình 12,5 → 16,3 từ — tức đúng nhóm lỗi mà Phase 2 đã chữa quay về. B1 vì thế đổi
**hai** thứ cùng lúc, và thí nghiệm không tách được hai nguyên nhân.

**B2 là kết quả null theo đúng nghĩa:** hiệu chuẩn quét 107 ngưỡng, điểm tốt nhất
chính là ngưỡng thấp nhất — tức *tắt cổng*. Không ngưỡng nào hạ được false-answer
rate mà vẫn giữ false-abstention ≤ 10%.

Final-test 60 câu, chạy sau khi đã khóa chính sách và ngưỡng từ development:
B0 abstention F1 **0,9697** · false-answer 5,88% · false-abstention **0,0%** ·
citation validity 1,0000.

---

# Ba lần liên tiếp, luật thắng điểm số

| Phase | Cấu hình cao điểm nhất | Cấu hình được nhận | Vì sao bị loại |
| :--- | :--- | :--- | :--- |
| 2B | P2-depth3 | P2-depth5 | trượt 2 guardrail |
| 2C | C3-depth3 | C0-depth5 *(không đổi)* | trượt 3 guardrail |
| 3 | B1 / B2 | B0 *(không đổi)* | trượt guardrail token F1 |

Nếu chỉ chọn theo điểm cao nhất thì cả ba lần đều chọn sai. Mặt kia của chuyện
này: hai trong ba lần, phán quyết nằm sát biên độ đo được — xem phần *Chưa được
phép kết luận*.

# Hai lần chạy độc lập, cùng một tập câu hỏi

Tập held-out của Phase 2 (284 câu / 50 bài) hóa ra là **tập con** của final-test Phase 1 (871 câu / 150 bài) — đủ cả 284 câu, đủ cả 50 bài. Cùng bộ câu hỏi đó đã được chấm hai lần, bởi hai lần chạy khác nhau.

| Hit@5 trên chính 284 câu đó | |
| :--- | ---: |
| Lần chạy Phase 1 (nằm trong 871 câu) | 0,8768 |
| Lần chạy Phase 2 (284 câu riêng) | 0,8768 |

Bằng nhau tới bốn chữ số — bằng chứng tái lập được cho nhánh sparse, đối lập với chỗ dense trôi 0,0143 giữa hai lần chạy.

Và 50 bài Phase 2 bốc trúng **không** khó bất thường:

| Trong 871 câu final-test | n | Hit@5 | CI95 |
| :--- | ---: | ---: | :--- |
| 50 bài Phase 2 dùng làm held-out | 284 | 0,8768 | [0,8333; 0,9181] |
| 100 bài còn lại | 587 | 0,9080 | [0,8786; 0,9385] |

Hai khoảng chồng lấn. Không phải Phase 2 bốc phải cụm khó — **cả vùng held-out đều khó hơn development**. Số câu mất sạch bằng chứng ở top-5: 12/281 (4,3%) trên dev, 89/871 (10,2%) trên final-test.

# Chỗ Phase 1 giúp Phase 2

Phát hiện phương pháp đáng kể nhất của dự án: **Phase 1 đã trả lời sẵn một câu hỏi của Phase 2, miễn phí.**

| Số context | Hit@k | Số câu mất sạch bằng chứng (/281) |
| ---: | ---: | ---: |
| 1 | 0,8221 | **38** |
| 3 | 0,9324 | **7** |
| 5 | 0,9573 | 0 |

Phase 2 quan sát được **đúng 7 câu** trả về "không tìm thấy thông tin" ở depth 3, 2 trong đó có bằng chứng ở hạng 4. Con số khớp chính xác.

Held-out xác nhận cùng cơ chế ở quy mô lớn hơn: với Hit@5 = 0,8768, khoảng 12% số câu **không có đường nào** để trả lời đúng, bất kể prompt viết thế nào. **Trần của tầng sinh nằm ở tầng truy xuất** — đòn bẩy còn lại là query rewriting, không phải prompt.

---

# Chưa được phép kết luận

**Phase 1** — không phân định được mô hình dense nào tốt nhất, và dense chưa tái lập được giữa các lần chạy (trôi 0,0143 > khoảng cách giữa các mô hình 0,0094). Chunk size là kết quả null; hybrid là "không chứng minh được có lợi", không phải "đã chứng minh có hại".

**Phase 2C** — ngưỡng guardrail đặt không kèm tính cỡ mẫu: Hit@5 và Recall@5 có tỉ lệ báo động giả 13,3% và 12,8%, Citation F1 là 23,8%. Chỉ phán quyết Citation F1 là vững cả hai đường (trượt ngưỡng *và* CI95 không chứa 0). 2C là phần mở rộng hậu kiểm, và development 281 câu đã dùng qua nhiều vòng nên vẫn có thể overfit thích nghi.

**Phase 3** — B1 đổi hai thứ cùng lúc (thêm schema *và* bỏ ràng buộc hình dạng đáp án) nên không tách được nguyên nhân. Loại case `natural_retrieval_miss` chỉ có 2 case development và 1 case final-test, không mang thông tin. Bộ dữ liệu có một lần sửa sau duyệt, 3 dòng, cùng một câu hỏi mơ hồ về cháy rừng.

**Phase 2** — AC 0,7157 là **cận dưới**: audit 30 câu điểm thấp nhất thấy 23/30 thực ra đúng về ngữ nghĩa, và giám khảo bất đồng với người duyệt ở 18/30 câu. (Biên độ 7,0–24,5% của EDA là sai số của việc chấm điểm theo **nhãn chunk** — nó áp cho Hit@k, nDCG và Citation F1, không áp cho Answer Correctness.) Judge cũng là một LLM, đã tách khỏi generator nhưng **chưa hiệu chuẩn với nhãn người**. Answer Relevancy giảm (0,7092 → 0,6485) là hệ quả có chủ đích của P2, không phải suy thoái.

---

# Còn thiếu gì

| # | Việc | Phase | Ghi chú |
| ---: | :--- | :--- | :--- |
| 1 | Audit mù 30 cặp P0 ↔ winner, **hai người chấm độc lập** | 2 | **Cần chia việc giữa hai người** |
| 2 | Lặp 25 câu × 2 cấu hình đo độ ổn định API | 2 | |
| 3 | Hiệu chuẩn judge trên 20 câu đã dành sẵn | 2 | Tập đã chuẩn bị |
| 4 | ~~Chạy held-out 150 bài final-test~~ | 1 | ✅ đã chạy — xem phần nghiệm thu Phase 1 |
| 5 | Chạy lại vòng 1 trên chỉ mục đã đóng gói (cố định số dense) | 1 | Rẻ — bỏ qua bước dựng chỉ mục |
| 6 | Chạy 2C trên tập dự trữ 587 câu | 2C | Chưa chạy, giữ cho kiểm chứng sau |

Ngoài ra test plan Phase 1 §6 yêu cầu **4 biểu đồ 300 DPI** (so sánh embedding, dumbbell ΔMRR@5, Pareto accuracy–latency, phân rã latency) — hiện chưa có cái nào; `scripts/generate_retrieval_figures.py` chưa nối được vào CSV các vòng.
