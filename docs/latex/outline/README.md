# Dàn ý báo cáo

Bốn file dựng nội dung **trước khi** viết LaTeX. Lý do có thư mục này: bản báo
cáo đầu được viết thẳng, không qua dàn ý, và hỏng hai chỗ — thuật ngữ dùng mà
không định nghĩa, và phát biểu bằng ẩn dụ ở chỗ đáng lẽ phải có số đo. Rà lại
thì tìm thêm được **hai con số bị phát biểu sai phạm vi**.

| File | Trả lời câu hỏi | Đọc khi |
| :--- | :--- | :--- |
| [`00_scope_and_style.md`](00_scope_and_style.md) | Viết cho ai, dài bao nhiêu, giọng thế nào | Trước khi viết bất kỳ mục nào |
| [`01_terms.md`](01_terms.md) | Thuật ngữ nào phải định nghĩa, và định nghĩa ra sao | Khi viết §0 |
| [`02_outline.md`](02_outline.md) | Có mục nào, mỗi mục phải chứng minh gì | Khi viết mục đó |
| [`03_claims.md`](03_claims.md) | Được phép nói mạnh đến đâu, dựa vào artifact nào | Mỗi lần viết một phát biểu có số |
| [`04_stats.md`](04_stats.md) | Giả thuyết là gì, và cỡ mẫu phân giải được tới đâu | Khi viết §2, và mỗi lần định dùng chữ "có ý nghĩa" |
| [`05_guardrails.md`](05_guardrails.md) | Toàn bộ guardrail của cả bốn phase, chép từ test plan | Khi viết §2, §4, §5 |

## Thứ tự làm

1. ~~Chốt người đọc, độ dài, phạm vi~~ ✅
2. ~~Liệt kê thuật ngữ, dàn ý, sổ phát biểu~~ ✅ ← **đang ở đây**
3. Sửa hai phát biểu sai (`03_claims.md` mục S1, S2) trong các báo cáo hiện có
4. Đo hai số còn thiếu: trung vị độ dài đáp án chuẩn; `judge_disagreement` 18/30
5. Viết lại `report/main.tex` theo `02_outline.md`
6. Chỉnh slide cho khớp báo cáo mới

## Năm thứ đã tìm ra khi dựng dàn ý

**Biên độ 7,0% / 24,5% đang bị dùng sai tầng.** EDA §7 đo nó trên **chấm điểm
truy xuất** — tìm text đáp án trong các chunk *không* được gán nhãn. Báo cáo đang
trích nó để nói Answer Correctness là cận dưới. Sai tầng. Cái chống lưng đúng cho
AC là audit 30 câu.

**Cỡ nhóm lỗi "3/30" của P3 không tồn tại.** File annotation không có nhóm lỗi
nào là trộn bài / trộn sự kiện. Nguồn đúng của giả thuyết P3 là phát hiện
distractor collision của EDA §7, không phải audit.

**Không chỗ nào phát biểu giả thuyết, và không chỗ nào kiểm cỡ mẫu.** Báo cáo kết
luận "có ý nghĩa thống kê" mà không viết H₀, và không hỏi 281 câu / 50 bài phân
giải được khoảng cách nhỏ tới đâu. Đo ra thì **hai trong ba ngưỡng guardrail nằm
dưới độ phân giải của chính thiết kế**, và nhóm `gold_not_in_top5` (n = 12) không
đủ để kết luận gì. `04_stats.md`.

**Một chỗ tôi tự nói quá.** Tôi viết bootstrap theo câu của Phase 1 là "lỏng hơn"
và gom cụm sẽ làm khoảng tin cậy rộng ra. Đo rồi thì design effect chạy 0,72–1,36
— vì so *hiệu* nên độ khó của bài triệt tiêu trong phép trừ. Đã hạ giọng.

**EDA đã có sẵn `## 0. Terms used in this report`.** Một bảng thuật ngữ đầy đủ,
viết mộc. Báo cáo Phase 1/2 bỏ mất quy ước đó — `01_terms.md` nối lại, và đánh
dấu mục nào đã có ở EDA (✅) để khỏi định nghĩa lệch nhau giữa hai tài liệu.

## Thứ này không sinh ra file build

`docs/latex/outline/` là tài liệu làm việc. Nó không được `\input` vào LaTeX và
không ảnh hưởng `make`.
