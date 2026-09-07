# 06 — Những chỗ chờ bạn quyết

Mỗi mục là **một câu hỏi**, kèm phương án và hệ quả. Quyết xong thì đánh dấu
✅ và ghi lựa chọn vào chính mục đó; các file còn lại cập nhật theo sau.

Xếp theo thứ tự nên giải quyết: chặn việc viết báo cáo trước, chi tiết sau.

---

## Nhóm A — Chặn việc bắt đầu viết

### D1. Thang mức A/B/C/D dùng để làm gì, có giữ không?

**Nó là gì.** Hệ nhãn dùng trong [`03_claims.md`](03_claims.md): mỗi phát biểu
được gán một mức, quy định **được nói mạnh tới đâu**.

| Mức | Nghĩa | Trong báo cáo được viết là |
| :-: | :--- | :--- |
| **A** | Vượt cả hai điều kiện: > ngưỡng 7,0% *và* CI95 không chứa 0 | "hơn", "thắng", "chứng minh được" |
| **B** | CI95 không chứa 0 nhưng khoảng cách < 7,0% | "đo được", cấm nói "chắc chắn" |
| **C** | Đúng theo dữ liệu nhưng không có CI, hoặc n quá nhỏ | "quan sát thấy", kèm n |
| **D** | CI95 chứa 0 | "không phân định được", cấm nói "tương đương" |

**Mục đích:** để lúc viết không phải nhớ lại từng con số, cứ tra bảng là biết
được dùng động từ nào.

**Nhược điểm:** thêm một hệ ký hiệu riêng phải nhớ, trong khi mỗi dòng trong sổ
đã ghi sẵn Δ, CI và n — ba dữ kiện đó tự nói.

**Phương án:**
- **(a)** Giữ nguyên bốn mức.
- **(b)** Rút còn hai: *nói mạnh được* / *chỉ nêu như quan sát*.
- **(c)** Bỏ hẳn, giữ lại cột "cấm viết gì" cho vài chỗ dễ sa bẫy (2.5, 3.4).
  *(nghiêng về cái này — sổ chỉ có ~30 phát biểu, cột CI và n làm được việc đó)*

> **Quyết:** ⬜ *(chưa)*

---

### D2. Vấn đề so sánh bội — xử lý thế nào?

**Nó là gì.** Chạy càng nhiều phép so sánh thì càng dễ vớ được một kết quả "có ý
nghĩa" hoàn toàn do may. Với α = 0,05, cứ 20 phép so sánh vô thưởng vô phạt thì
trung bình có 1 cái đạt ngưỡng.

**Phase 2B đã chạy bao nhiêu?** 7 metric × 2 finalist so với P0, cộng vòng sàng
lọc prompt và depth. Cỡ vài chục phép so sánh.

**Cách xử lý thông thường:** siết α lại (Bonferroni: chia α cho số phép so), hoặc
kiểm soát tỉ lệ phát hiện sai (FDR).

**Hiện tại dự án không làm gì cả.** Lý do có thể bào chữa: metric chính được
**đăng ký trước đúng một cái** (Answer Correctness) nên nó không nằm trong "vài
chục phép so"; các metric còn lại là ngưỡng guardrail hoặc chỉ để mô tả.

**Áp thử vào dữ liệu thật** ([`04_stats.md`](04_stats.md) mục 1): m = 14, ngưỡng
$z$ lên 2,914, và **đúng hai kết luận đổi** — Faithfulness (2,37) và Citation
Validity (2,11) của p2_d3, tức **đúng hai phép đã loại P2-depth3**. Mười hai phép
còn lại không đổi. Winner **không** lật, vì guardrail là phép so ngưỡng chứ không
phải kiểm định.

**Phương án:**
- **(a)** Không hiệu chỉnh, nhưng §2 nói rõ là không, kèm lý do; §7 nêu rằng hai
  phán quyết loại d3 không sống sót qua Bonferroni.
  *(nghiêng về cái này — trung thực hơn là im, mạnh hơn là hiệu chỉnh máy móc rồi
  mất luôn mấy kết luận chắc chắn)*
- **(b)** Hiệu chỉnh Bonferroni cho các so sánh phụ, giữ nguyên metric chính.
- **(c)** Không nhắc tới. → *tệ nhất: thầy hỏi mà báo cáo không có chữ nào.*

> **Quyết:** ⬜ *(chưa)*

---

### D3. Phân tích độ nhạy của luật chọn winner — có đưa vào không?

**Nó là gì.** Luật đã đăng ký so **điểm trung bình** của hiệu với ngưỡng. Phát
biểu luật theo một cách khác — *chỉ loại khi cận trên CI95 nằm dưới ngưỡng* —
thì **winner lật từ P2-depth5 sang P2-depth3**.

**Đây không phải luật thay thế được ai đề xuất**, mà là một phản-thực dựng lên để
đo xem kết quả nhạy tới mức nào với cách phát biểu luật.

**Vì sao đáng cân nhắc đưa vào:** nó cho thấy phán quyết sát biên tới đâu, và
biến câu "guardrail đã làm đúng việc" (tự khen) thành một sự kiện kiểm chứng được.

**Vì sao có thể không nên:** dễ bị đọc thành "nhóm biết có cách chọn ra kết quả
tốt hơn mà không dùng", nếu viết không khéo.

**Phương án:**
- **(a)** Đưa vào §7 (giới hạn), như một phân tích độ nhạy.
- **(b)** Đưa vào §4 ngay sau đoạn chốt.
- **(c)** Không đưa vào.

> **Quyết:** ⬜ *(chưa)*

---

### D4. §5 Phase 2C — chờ số của Thắng tới bao giờ?

Khung đã dựng theo test plan (C0–C3, 6 guardrail). Chưa có số.

**Phương án:**
- **(a)** Chờ. Nếu tới hạn chưa có thì nộp bản có khung + ghi "đang chạy".
- **(b)** Nộp không có §5, để 2C thành phần mở rộng sau.
- **(c)** Hỏi Thắng xem bao giờ xong rồi quyết.

> **Quyết:** ⬜ *(chưa)*

---

## Nhóm B — Ảnh hưởng bố cục, quyết trước khi viết

### D5. §0 thuật ngữ đặt ở đâu?

- **(a)** Thân bài, ngay sau mở đầu — giống EDA. Người đọc gặp `resolved` ở §1 thì đã biết.
- **(b)** Phụ lục cuối, §1 chỉ trỏ tới.

*Đề xuất: (a).*

> **Quyết:** ⬜ *(chưa)*

### D6. Ablation contextual chunking để ở đâu?

Nó và Phase 2C cùng hỏi *"cách biểu diễn chunk có quan trọng không?"*.

- **(a)** Giữ ở §3 (Phase 1) + một câu bắc cầu sang §5.
- **(b)** Gộp vào §5 cùng 2C thành một mục "chiến lược chunking".

*Đề xuất: (a) — ablation chạy trước, harness khác, và chính nó mở ra 2C.*

> **Quyết:** ⬜ *(chưa)*

### D7. Phase 3 — mục riêng hay một đoạn?

- **(a)** Một đoạn trong §7, vì chưa có kết quả.
- **(b)** Mục riêng mô tả kế hoạch.

> **Quyết:** ⬜ *(chưa)*

### D8. Có làm 4 biểu đồ 300 DPI mà test plan Phase 1 §6 yêu cầu không?

Hiện **chưa có cái nào**. 10–15 trang thì đủ chỗ cho 2–3 hình.
Bốn hình được yêu cầu: so sánh embedding · dumbbell ΔMRR@5 · Pareto
accuracy–latency · phân rã latency.

- **(a)** Làm đủ 4.
- **(b)** Làm 2 cái có giá trị nhất cho lập luận.
- **(c)** Không làm, ghi vào phần việc còn lại.

> **Quyết:** ⬜ *(chưa)*

---

## Nhóm C — Việc kỹ thuật, không chặn viết

### D9. Sửa `.gitattributes` để hash khớp lại

Git đổi LF → CRLF nên file bằng chứng trong repo băm ra khác giá trị đã ghi.
Câu "4/5 hash khớp" trong báo cáo **hiện đang sai** với thứ trong repo.

Sửa: thêm `docs/reports/phase2/** -text` rồi checkout lại.

- **(a)** Sửa ngay.
- **(b)** Để sau.

> **Quyết:** ⬜ *(chưa)*

### D10. Phần diễn giải đã push lên `main` ✅ *(đã sửa)*

Ba chỗ đã sửa trong commit `7f30d10`:

- biên độ 7,0–24,5% được gắn lại đúng tầng (nhãn chunk, không phải Answer Correctness);
- nhóm lỗi "3/30" của P3 thay bằng nguồn thật là EDA §7;
- "held-out cao hơn dev" hạ xuống "không phân biệt được", kèm CI95.

Không còn gì phải quyết ở mục này.

### D11. Hai số còn thiếu, không tốn API

- **Trung vị độ dài đáp án chuẩn** (token) — cần để thay câu ẩn dụ "khớp với hình
  dạng của nhãn" bằng số thật.
- ~~**`judge_disagreement` 18/30**~~ ✅ đã đưa vào `docs/reports/report.md`.

- **(a)** Đo trung vị độ dài đáp án ngay.
- **(b)** Để sau.

> **Quyết:** ⬜ *(chưa)*

### D12. Đo SE trước khi áp guardrail Phase 2C

Kinh nghiệm 2B: ngưỡng 0,01 cho Citation F1 có tỉ lệ báo động giả 21,3%. Ngưỡng
0,01 cho Hit@5 / Recall@5 của 2C có thể vướng đúng bẫy đó. 2C **chưa chạy xong**
nên vẫn kịp.

- **(a)** Báo cho Thắng ngay, trước khi 2C chốt kết quả.
- **(b)** Chờ có kết quả C1 rồi tính.

> **Quyết:** ⬜ *(chưa)*

---

## Đã chốt — không hỏi lại

| | Nội dung | Chốt lúc nào |
| :-- | :--- | :--- |
| ✅ | Người đọc: thầy biết IR/NLP, không biết dự án | bạn chọn |
| ✅ | Độ dài 10–15 trang | bạn chọn |
| ✅ | Báo cáo có §5 Phase 2C, chừa khung đầy đủ | bạn chọn |
| ✅ | $H_0$, $H_1$ hai phía, $\alpha = 0{,}05$, bác bỏ khi CI95 không chứa 0 | bạn duyệt |
| ✅ | Toàn bộ luật viết ở [`00_scope_and_style.md`](00_scope_and_style.md) | bạn duyệt sau khi tự sửa |
| ✅ | Đoạn chốt §4 về lý do chọn P2-depth5 + ba ràng buộc cách viết | bạn duyệt |
