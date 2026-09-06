# NewsQA RAG — Tóm tắt

> Chi tiết: **[Phase 1](phase1/report.md)** · **[Phase 2](phase2/report.md)** · [EDA](../eda/eda_report.md) · [Báo cáo gộp hai phase](report_detail.md)

Hệ thống hỏi–đáp trên 11.064 bài báo CNN. Người dùng hỏi một câu, hệ thống tìm đoạn văn chứa câu trả lời rồi để LLM viết đáp án kèm trích dẫn.

```
Câu hỏi → [ tìm đoạn văn ] → [ viết đáp án ] → đáp án + citation
             PHASE 1              PHASE 2
```

Dự án chia hai giai đoạn vì hai việc đó hỏng theo hai cách khác nhau và phải đo bằng hai loại thước khác nhau.

| | Trạng thái | Chốt được gì |
| :--- | :--- | :--- |
| EDA — khảo sát dữ liệu | ✅ | Ngưỡng nhiễu nhãn 7,0–24,5%; phục hồi 4.603 bài bị cắt |
| **Phase 1** — chọn cách lấy đoạn văn | ✅ khóa | BGE-M3 sparse + bge-reranker-large + chunk 512/64 |
| **Phase 2** — chọn cách hỏi LLM | ✅ chọn xong | Prompt P2, 5 đoạn context |
| Phase 2 — held-out | ❌ chưa chạy | *số công bố nằm ở đây* |
| Phase 3 — biết từ chối trả lời | ⛔ bị chặn | chờ duyệt tay 200 case |

---

# PHASE 1 — Chọn cách lấy đoạn văn

**Việc của nó:** trong 22.766 đoạn văn, tìm ra 5 đoạn nhiều khả năng chứa câu trả lời nhất. Không gọi LLM, không tốn tiền API.

**Đã thử:** 23 cấu hình, chia 3 vòng để khỏi phải chạy đủ 72 tổ hợp.

| Vòng | Thử gì | Thắng |
| :--- | :--- | :--- |
| 1 | 4 mô hình dense × 4 mô hình sparse | **BGE-M3 sparse** |
| 2 | 3 retriever × 3 reranker | **+ bge-reranker-large** |
| 3 | 3 kích thước chunk (256/512/1024) | 512/64 — *không ai thắng rõ* |

**Kết quả cấu hình khóa** (281 câu development, tập `resolved`):

| Hit@1 | Hit@5 | nDCG@5 | Recall@5 | Latency P50 |
| ---: | ---: | ---: | ---: | ---: |
| 0,8221 | **0,9573** | 0,8976 | 0,9555 | 510 ms |

**Vì sao chọn như vậy:**

- **Sparse thắng dense +0,1634 nDCG@5** (CI95 [+0,1231; +0,2080]) — không phải chênh lệch nhỏ. EDA giải thích: câu hỏi NewsQA bám vào tên riêng, ngày tháng, con số — thứ mà khớp từ vựng bắt được còn embedding thì làm nhòe đi.
- **Reranker thêm +0,0659 nDCG@5.** EDA đo được mỗi câu hỏi có trung vị **25 đoạn đối thủ** cùng chủ đề, nên lọc lại là bắt buộc chứ không phải tùy chọn.
- **Chunk size là kết quả null.** Ba kích thước chồng lấn nhau; chọn 512 vì lý do vận hành, không vì nó thắng.

---

# PHASE 2 — Chọn cách hỏi LLM

**Việc của nó:** với 5 đoạn văn Phase 1 đưa sang, tìm cách ra lệnh cho Gemini sao cho nó trả lời đúng mà không bịa. **Không huấn luyện lại gì cả** — chỉ đổi hai thứ: nội dung prompt, và số đoạn context đưa vào.

**Điểm xuất phát** (prompt gốc P0, 5 context, 281 câu):

| Answer Correctness | Exact Match | Faithfulness |
| ---: | ---: | ---: |
| 0,6308 | **0,0000** | 0,9761 |

Ba số này đọc cùng nhau ra một chẩn đoán: **hệ thống không bịa, nó chỉ nói dài.** Model biết đáp án nhưng gói trong một đoạn văn, trong khi đáp án chuẩn của NewsQA là một span rất ngắn — một cái tên, một mốc thời gian, một con số.

**Đã thử:** 4 prompt × 3 mức độ sâu context. Mỗi prompt nhắm vào một nhóm lỗi đã đếm được từ audit tay 30 câu điểm thấp.

| Prompt | Nhắm nhóm lỗi | Cỡ nhóm | Kết quả |
| :--- | :--- | ---: | :--- |
| P1 — siết grounding | Hallucination | ~0 | ❌ nhắm vào vấn đề không tồn tại |
| **P2 — trả lời ngắn, đúng answer type** | Đúng nhưng quá dài | **11/30** | ✅ thắng |
| P3 — chọn đúng đoạn trước khi trả lời | Trộn bài, trộn mốc thời gian | 3/30 | ⚠️ đúng hướng, chưa đủ mạnh |

**Chọn:** prompt **P2** với **5 đoạn context**.

| | P0 (gốc) | **P2-depth5 (chọn)** | P2-depth3 (loại) |
| :--- | ---: | ---: | ---: |
| Answer Correctness | 0,6308 | **0,7350** | *0,7711* |
| Exact Match | 0,0000 | **0,0925** | *0,2633* |
| Faithfulness | 0,9761 | **0,9801** | 0,9561 ❌ |
| Citation Validity | 0,9893 | 0,9858 | 0,9751 ❌ |

**Vì sao chọn P2-depth5 dù P2-depth3 điểm cao hơn:**

Phase 2 có **4 điều kiện được khóa trước khi nhìn thấy kết quả** — vì ở tầng sinh, sai một chút nghĩa là hệ thống nói điều không có trong bằng chứng, chứ không chỉ là xếp hạng kém. P2-depth3 trượt 2 trong 4: nó làm tụt Faithfulness và Citation Validity quá ngưỡng.

Kiểm định theo cặp xác nhận **cả hai mức tụt đều là thật** (CI95 không chứa 0), không phải nhiễu đo. Nới ngưỡng lúc này chính là hành vi mà việc đăng ký trước sinh ra để ngăn.

P2-depth5 tăng correctness +0,1043 (CI95 [+0,0844; +0,1240]) mà **không** làm giảm grounding. Cái giá: tốn input token nhiều hơn depth 3 khoảng 60%.

---

# Chỗ Phase 1 giúp Phase 2

Phát hiện phương pháp đáng kể nhất của dự án: **Phase 1 đã trả lời sẵn một câu hỏi của Phase 2, miễn phí.**

Phase 2 cần biết cắt bớt context tốn gì. Đường cong Hit@k mà Phase 1 đã đo cho biết chính xác:

| Số context | Hit@k | Số câu mất sạch bằng chứng (/281) |
| ---: | ---: | ---: |
| 1 | 0,8221 | **38** |
| 3 | 0,9324 | **7** |
| 5 | 0,9573 | 0 |

Phase 2 quan sát được **đúng 7 câu** trả về "không tìm thấy thông tin" ở depth 3, 2 trong đó có bằng chứng ở hạng 4 — bị cắt mất. Con số khớp chính xác.

Bài học: đọc Hit@k trước, rồi mới chọn depth để thử. Đỡ được cả một vòng chi tiền API.

---

# Chưa được phép kết luận

**Phase 1** — không phân định được mô hình dense nào tốt nhất, và dense chưa tái lập được giữa các lần chạy (trôi 0,0143 > khoảng cách giữa các mô hình 0,0094). Chunk size là kết quả null; hybrid là "không chứng minh được có lợi", không phải "đã chứng minh có hại".

**Phase 2** — **chưa có số nào để công bố**: mọi số trên đo ở tập tinh chỉnh nên lệch lạc quan theo cấu trúc. P2-depth5 thắng vì nó **hợp lệ**, không phải vì chất lượng cao nhất. Và AC 0,6308 của baseline là **cận dưới** — EDA cho thấy 7,0–24,5% câu bị chấm sai oan, người duyệt thấy 23/30 câu điểm thấp thực ra đúng.

---

# Còn thiếu gì

| # | Việc | Phase | Ghi chú |
| ---: | :--- | :--- | :--- |
| 1 | Bổ sung phân tầng `gold_in_top5` và article-macro | 2 | **Không tốn API** — dữ liệu đã có |
| 2 | Chạy lại vòng 1 trên chỉ mục đã đóng gói (cố định số dense) | 1 | Rẻ — bỏ qua bước dựng chỉ mục |
| 3 | Tạo `phase2b_winner_decision.json` | 2 | Điều kiện để chạy held-out |
| 4 | Audit mù 30 cặp P0 ↔ winner, **hai người chấm độc lập** | 2 | **Cần chia việc giữa hai người** |
| 5 | Lặp 25 câu × 2 cấu hình đo độ ổn định API | 2 | |
| 6 | Chạy held-out 284 câu, **đúng một lần** | 2 | Số công bố |
| 7 | Chạy held-out 150 bài final-test | 1 | Nghiệm thu Phase 1 |
| 8 | Duyệt tay dataset abstention 200 case | 3 | 3 loại cần reviewer thứ hai |

Ngoài ra test plan Phase 1 §6 yêu cầu **4 biểu đồ 300 DPI** (so sánh embedding, dumbbell ΔMRR@5, Pareto accuracy–latency, phân rã latency) — hiện chưa có cái nào; `scripts/generate_retrieval_figures.py` chưa nối được vào CSV các vòng.
