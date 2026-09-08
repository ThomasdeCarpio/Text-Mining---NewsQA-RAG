# Phase 3 — abstention

Hệ thống có im lặng được khi không đủ bằng chứng không. Metric chính là
**false-answer rate**, không phải Answer Correctness.

**Kết quả: B0 thắng — chính prompt P2 của Phase 2, không sửa gì.**

Kế hoạch đăng ký trước: [`phase_3_abstention_test_plan.md`](../../Detailed%20Test%20Plans/phase_3_abstention_test_plan.md).

## Ba chính sách

| | Là gì |
| :-- | :--- |
| `B0` | Đúng prompt P2 của Phase 2, không sửa một chữ |
| `B1` | Bắt model trả JSON có trường `answerability` |
| `B2` | B1, cộng cổng chặn theo điểm reranker của đoạn hạng 1 |

B1 và B2 hạ false-answer rate 5,06% → 1,27% nhưng làm token F1 của các câu trả
lời được tụt −0,1070, gấp hơn năm lần ngưỡng 0,02, nên không hợp lệ.

## Có gì ở đây

| File | Nội dung |
| :--- | :--- |
| `phase3_final_results.json` | Kết quả đầy đủ: 200 case × 3 chính sách, per-case, predictions, đường cong hiệu chuẩn 107 ngưỡng |
| `policy_comparison.csv` | Bảng tóm tắt 6 dòng — 2 partition × 3 chính sách |
| `case_types.csv` | Lỗi quyết định theo từng loại case |
| `policy_significance.json` | CI95 của chênh lệch token F1 B1 − B0, và thống kê cổng B2 |

## Hai chỗ phải nêu khi trích dẫn

1. **B1 có confound.** Prompt B1 vừa thêm schema `answerability`, vừa **bỏ mất**
   câu lệnh quy định hình dạng đáp án của P2 (chỉ còn "concise answer"). Độ dài
   đáp án trung bình 12,5 → 16,3 từ. Thí nghiệm không tách được hai nguyên nhân.
2. **B2 là kết quả null theo đúng nghĩa.** Hiệu chuẩn quét 107 ngưỡng và chọn
   ngưỡng thấp nhất — tức *tắt cổng*. Không ngưỡng nào hạ được false-answer rate
   mà vẫn giữ false-abstention ≤ 10%.

Mã chạy ra kết quả này nằm ngoài repo (`phase3_run.py`, `phase3_metrics.py` của
Thắng), khác với `scripts/run_phase3_abstention.py` đang có ở đây.
