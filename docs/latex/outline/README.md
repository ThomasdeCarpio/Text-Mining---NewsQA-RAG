# Dàn ý báo cáo

Đặc tả nội dung cho báo cáo LaTeX ở [`../report/`](../report/). Viết mục nào thì
tra file tương ứng trước.

| File | Trả lời câu hỏi | Đọc khi |
| :--- | :--- | :--- |
| [`00_scope_and_style.md`](00_scope_and_style.md) | Viết cho ai, dài bao nhiêu, giọng thế nào | Trước khi viết bất kỳ mục nào |
| [`01_terms.md`](01_terms.md) | Thuật ngữ nào phải định nghĩa, và định nghĩa ra sao | Khi viết §0 |
| [`02_outline.md`](02_outline.md) | Có mục nào, mỗi mục phải chứng minh gì | Khi viết mục đó |
| [`03_claims.md`](03_claims.md) | Được phép nói mạnh đến đâu, dựa vào artifact nào | Mỗi lần viết một phát biểu có số |
| [`04_stats.md`](04_stats.md) | Giả thuyết là gì, cỡ mẫu phân giải được tới đâu | Khi viết §2, và mỗi lần định dùng chữ "có ý nghĩa" |
| [`05_guardrails.md`](05_guardrails.md) | Toàn bộ guardrail của cả bốn phase | Khi viết §2, §4, §5 |
| [`06_decisions.md`](06_decisions.md) | Những chỗ còn chờ quyết | Trước khi bắt đầu viết |

## Ba nhãn trạng thái

Mọi mục trong thư mục này mang một trong ba nhãn:

- **✅ ĐÃ CHỐT** — có nguồn: test plan, artifact, hoặc đã được duyệt.
- **📌 SỰ KIỆN ĐO ĐƯỢC** — số tính từ artifact. Đúng hoặc sai, không thương lượng.
- **❓ CHỜ QUYẾT** — viết dưới dạng **câu hỏi kèm phương án**, không viết thành
  khẳng định. Toàn bộ gom ở [`06_decisions.md`](06_decisions.md).

Nguyên tắc: thứ chưa được quyết thì không xuất hiện dưới dạng câu khẳng định.

## Thứ tự làm

1. ✅ Chốt người đọc, độ dài, phạm vi
2. ✅ Thuật ngữ, dàn ý, sổ phát biểu, phân tích thống kê, bảng guardrail
3. ✅ Bổ sung Phase 2C (§5) và Phase 3 (§7); thêm §1 mở đầu, §9 kết luận, và
   đoạn *Chốt lại* cho từng mục — 08/09/2026
4. ⬜ **Quyết D1, D2, D3, D7, D14** ở [`06_decisions.md`](06_decisions.md) —
   năm cái này chặn việc bắt đầu viết. **D14 chặn nặng nhất**: §1.2 để trống cho
   tới khi có đề bài. D5, D6, D8, D13 ảnh hưởng bố cục, nên quyết cùng lúc.
5. ⬜ Viết `report/main.tex` theo [`02_outline.md`](02_outline.md), mục nào tra
   file nấy. §0 và §2 viết trước vì các mục sau trỏ ngược về chúng.
6. ⬜ Chỉnh slide cho khớp báo cáo mới — slide hiện **chưa có** 2C và Phase 3.

Việc kỹ thuật không chặn viết, làm xen kẽ được: D9 (`.gitattributes` cho hash
khớp lại), D11 (đo trung vị độ dài đáp án), D13 (đưa mã Phase 3 vào repo).

## Báo cáo có mấy mục

§0 thuật ngữ · §1 mở đầu · §2 luật phán quyết · §3 Phase 1 · §4 Phase 2 ·
§5 Phase 2C · §6 held-out · §7 Phase 3 · §8 giới hạn · §9 kết luận.

## Thư mục này không sinh ra file build

Không được `\input` vào LaTeX và không ảnh hưởng `make`.
