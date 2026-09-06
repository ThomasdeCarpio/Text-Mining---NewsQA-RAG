# NewsQA RAG — Tóm tắt Phase 1 + Phase 2

> Chi tiết đầy đủ: **[report_detail.md](report_detail.md)** · Phase 1: [phase1/report.md](phase1/report.md) · EDA: [../eda/eda_report.md](../eda/eda_report.md)

## Hệ thống đang là gì

```
Câu hỏi → BGE-M3 sparse (top 20) → bge-reranker-large (top 5) → Gemini 3.1 Flash-Lite → đáp án + citation
          └─────────── Phase 1 khóa ───────────┘   └────── Phase 2 tinh chỉnh ──────┘
```

Chunk 512/64. Corpus 11.064 bài, 22.766 chunk. Đánh giá trên 1.152 câu `resolved` đã khử trùng lặp, chia theo bài báo: **281 câu development** (tinh chỉnh) / 284 câu held-out / 587 câu dự trữ.

## Trạng thái

| | Trạng thái | Kết quả |
| :--- | :--- | :--- |
| EDA | ✅ | Ngưỡng nhiễu nhãn 7,0–24,5%; phục hồi 4.603 bài bị cắt |
| Phase 1 — chọn retrieval | ✅ khóa | 23 cấu hình, 3 vòng → BGE-M3 + bge-reranker-large + 512/64 |
| Phase 2A — baseline | ✅ | Answer Correctness 0,6308 |
| Phase 2B — tinh chỉnh prompt & depth | ✅ | **Winner: P2-depth5**, AC 0,7350 |
| Phase 2B.4 — held-out | ❌ chưa chạy | *số công bố nằm ở đây* |
| Phase 3 — abstention | ⛔ bị chặn | chờ duyệt tay 200 case |

## Ba kết quả chính

**1. Sparse thắng dense áp đảo.** +0,1634 nDCG@5, CI95 [+0,1231; +0,2080]. Không phải chênh lệch nhỏ. Reranker thêm +0,0659 nữa. Cấu hình khóa đạt **Hit@5 0,9573 / nDCG@5 0,8976** trên tập `resolved`.

**2. Prompt đúng dạng đáng giá hơn mọi thứ khác ở tầng sinh.** Baseline có AC 0,6308 nhưng **Exact Match 0,0000** — model biết đáp án nhưng nói dài. Gold NewsQA là span ngắn. Prompt P2 ép trả lời trực tiếp, đúng answer type:

| | P0 (baseline) | **P2-depth5 (winner)** |
| :--- | ---: | ---: |
| Answer Correctness | 0,6308 | **0,7350** (+0,1043) |
| Exact Match | 0,0000 | **0,0925** |
| Token F1 | 0,2648 | **0,4191** |
| Faithfulness | 0,9761 | **0,9801** |
| Citation F1 | 0,8025 | **0,8391** |

Mọi mức tăng đều có ý nghĩa thống kê (bootstrap gom cụm bài báo, CI95 không chứa 0), và grounding **không giảm**.

**3. Guardrail đã chặn một cấu hình điểm cao hơn.** P2-depth3 có AC **0,7711** — cao hơn winner — nhưng bị loại vì làm tụt Faithfulness (−0,0200) và Citation Validity (−0,0142) quá ngưỡng đăng ký trước. Kiểm định theo cặp xác nhận **cả hai mức tụt đều là thật**, không phải nhiễu đo.

> Bộ quy tắc được khóa **trước khi** nhìn thấy kết quả, nên nó phân xử được đúng tình huống này: một cấu hình đổi tính trung thực lấy điểm số. Nới ngưỡng lúc này chính là hành vi mà việc đăng ký trước sinh ra để ngăn.

## Phát hiện đáng kể nhất về phương pháp

**Phase 1 đã định giá sẵn câu hỏi của Phase 2, miễn phí.** Đường cong Hit@k của cấu hình khóa cho biết chính xác việc cắt context tốn bao nhiêu bằng chứng:

| Depth | Hit@k | Số câu mất sạch bằng chứng (/281) |
| ---: | ---: | ---: |
| 1 | 0,8221 | **38** |
| 3 | 0,9324 | **7** |
| 5 | 0,9573 | 0 |

Phase 2 quan sát được **đúng 7 câu abstention không citation** ở depth 3, 2 trong đó có gold ở rank 4 — bị cắt mất. Con số khớp chính xác. Depth không phải cái núm miễn phí, và lẽ ra có thể dự đoán kết quả trước khi chi tiền API.

## Vì sao chọn như vậy — nối với EDA

| Phát hiện EDA | Dẫn tới |
| :--- | :--- |
| 7,0–24,5% câu có distractor trả lời được nhưng bị chấm sai | Ngưỡng nhiễu 7,0%; và giải thích vì sao người duyệt thấy 23/30 câu điểm thấp thực ra đúng ⇒ **AC 0,6308 là cận dưới**, không phải năng lực thật |
| Câu hỏi `original` thiếu mỏ neo định danh (28,4% vs 59,5% có mỏ neo) | Chọn sparse; và giải thích vì sao BGE-M3 hơn BM25 **chỉ trên** `original` |
| Trung vị 25 chunk đối thủ mỗi câu | Reranker là bắt buộc, không phải tùy chọn |
| Bằng chứng nằm ở 16% đầu bài | Hit@1 cao tới 0,8221; truncation không gây hại |
| Notebook cũ đặt judge = generator | Phase 2 tách hẳn hai provider |

## Còn thiếu gì

| # | Việc | Chặn |
| ---: | :--- | :--- |
| 1 | Tạo `phase2b_winner_decision.json` | Held-out |
| 2 | **Audit mù 30 cặp P0 ↔ winner, hai người chấm độc lập** | Đóng Phase 2 — cần chia việc |
| 3 | Lặp 25 câu × 2 cấu hình đo độ ổn định API | Đóng Phase 2 |
| 4 | Phân tầng `gold_in_top5` + article macro — **không tốn API**, dữ liệu đã có | Đóng Phase 2 |
| 5 | Chạy held-out 284 câu, **đúng một lần** | Số công bố |
| 6 | Chạy lại vòng 1 trên chỉ mục đã đóng gói (cố định số dense) | Phase 1 §4.3 |
| 7 | Chạy Phase 1 held-out trên 150 bài final-test | Nghiệm thu Phase 1 |
| 8 | Duyệt dataset abstention 200 case | Phase 3 |

## Chưa được phép kết luận

- **Chưa có số nào để công bố.** Mọi số trên đo ở tập tinh chỉnh, lệch lạc quan theo cấu trúc. Số công bố là số held-out.
- **P2-depth5 thắng vì nó hợp lệ**, không phải vì nó chất lượng cao nhất. Depth 3 có correctness cao hơn có ý nghĩa.
- **Không phân định được best dense**, và dense chưa tái lập được giữa các lần chạy (drift 0,0143 > khoảng cách giữa các model 0,0094).
- **Chunk size là kết quả null** — ba kích thước chồng lấn nhau. Chọn 512 vì lý do vận hành.
- **Phase 1 chưa chạy held-out** như exec guide yêu cầu.
