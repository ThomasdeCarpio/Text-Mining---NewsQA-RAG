# 02 — Dàn ý báo cáo

Mục tiêu 10–15 trang. Mỗi mục ghi rõ: **phải chứng minh được điều gì**, người đọc
**đã phải biết gì** trước đó, **bằng chứng** nào chống lưng, và **không** được
nhét gì vào.

Ngân sách trang là để giữ tỉ lệ, không phải để đếm chính xác.

---

## §0 — Thuật ngữ · ~1 trang

**Phải làm được:** người đọc gặp `resolved`, `gold chunk`, `guardrail`,
`context_depth` ở các mục sau mà không phải đoán.

**Nội dung:** [`01_terms.md`](01_terms.md).

**Không nhét vào:** kết quả. §0 chỉ định nghĩa.

---

## §1 — Bài toán, dữ liệu, và cách chia việc · ~1,5 trang

**Phải làm được:**
1. Hệ thống nhận gì, trả gì (câu hỏi → 5 đoạn → đáp án ngắn + citation `[n]`).
2. Dữ liệu là gì: 11.064 bài, 200 bài đánh giá, 1.152 câu sau khử trùng lặp.
3. Vì sao tách phase — hai kiểu hỏng khác nhau, hai loại thước đo khác nhau.
4. `original` vs `resolved`, và vì sao báo cáo theo `resolved`.

**Bằng chứng:** EDA §1, §6; `subset_manifest.json`.

**Cần cho mục sau:** §3 và §4 đều dựa vào việc người đọc hiểu tại sao truy xuất
và sinh được đo tách nhau.

**Không nhét vào:** chi tiết EDA. Chỉ lấy đúng phần cần để hiểu thiết kế thí
nghiệm; phần còn lại trỏ sang báo cáo EDA.

---

## §2 — Cái gì được coi là bằng chứng · ~1,5 trang

> **Mục mới. Bản hiện tại không có, và đây là lý do chính khiến nó lủng củng** —
> luật phán quyết đang nằm rải rác ở ba chỗ, mỗi chỗ phát biểu một kiểu.

**Phải làm được:**
1. **Biên độ nhập nhằng 7,0% / 24,5%** — định nghĩa đúng (xem `01_terms.md` mục C),
   nó đo cái gì, **và nó áp cho metric nào**: điểm **truy xuất**, không phải điểm sinh.
2. **Giả thuyết, phát biểu tường minh** — H₀, H₁ hai phía, α = 0,05, và phân
   biệt ba loại phát biểu: *kiểm định* / *ngưỡng quyết định* / *mô tả*. Bản hiện
   tại trộn cả ba vào cùng chữ "có ý nghĩa". Chi tiết: [`04_stats.md`](04_stats.md) mục 1.
3. **Hai điều kiện để gọi là thắng:** vượt biên độ *và* CI95 không chứa 0.
4. **So sánh theo từng câu hỏi + bootstrap bốc lại theo bài** — vì sao bốc theo bài.
5. **Cỡ mẫu này phân giải được tới đâu** — MDE₈₀ cho từng metric, và **hai trong
   ba ngưỡng guardrail nằm dưới độ phân giải**. Chi tiết: [`04_stats.md`](04_stats.md) mục 2.
6. **Guardrail** — vì sao Phase 2 cần mà Phase 1 không: ở tầng truy xuất, sai là
   xếp hạng kém; ở tầng sinh, sai là nói điều không có trong bằng chứng. Kèm
   **bảng liệt kê đủ guardrail của cả bốn phase** — hiện chưa tài liệu nào có.
   Chi tiết: [`05_guardrails.md`](05_guardrails.md).
7. **Đăng ký trước** — test plan là gì, và độ lệch được ghi lại chứ không sửa lùi.
8. **Giới hạn của chính bộ đo:** giám khảo là LLM, chưa hiệu chuẩn với nhãn người.

**Bằng chứng:** EDA §7; `docs/Detailed Test Plans/`; `paired_significance.json`
(mục `resolution`); `scripts/phase2_paired_comparison.py`.

**Cần cho mục sau:** §3, §4, §5 đều chỉ nói "vượt/không vượt luật ở §2" thay vì
mỗi chỗ giải thích lại.

**Không nhét vào:** kết quả cụ thể.

---

## §3 — Phase 1: chọn cách truy xuất · ~3 trang

**Phải làm được:**
1. **Thiết kế giải đấu phân tầng** 23 cấu hình / 3 vòng thay vì 72 tổ hợp — và
   nêu thẳng cái đánh đổi: **không phát hiện được tương tác giữa các vòng**.
2. **Vòng 1:** sparse thắng dense +0,1634 (CI95 [+0,1231; +0,2080]) — vượt cả hai
   điều kiện. Hai so sánh còn lại: CI chứa 0, **không phân định được**.
3. **Cơ chế, lấy từ EDA:** câu hỏi neo vào tên riêng / ngày / số.
4. **Vòng 2:** reranker +0,0659; EDA đã dự báo bằng trung vị 25 đoạn đối thủ.
   Hybrid: không chứng minh được có lợi.
5. **Vòng 3:** null. Khoảng cách 0,0469 < 7,0%, ba CI chồng lấn. Chọn 512/64 vì
   lý do vận hành.
6. **Ablation contextual chunking:** giúp dense, hại sparse, cả hai dưới ngưỡng.
   Nói rõ đây là *contextual chunking*, không phải *metadata indexing*, và vì sao
   (dataset không có metadata thật).
7. **Cấu hình khóa** + Hit@1/3/5 — con số §4 sẽ dùng lại.

**Bằng chứng:** `round1..3.csv`, `paired_significance.json`, `winner_lock.jsonl`,
`contextual_chunking_ablation.json`.

**Không nhét vào:** phần giải thích lại nDCG; bảng đầy đủ 23 cấu hình (trỏ sang
`docs/reports/phase1/report.md`).

**Cân nhắc:** ablation nên nằm ở §3 hay gộp vào §5 (2C)? Cả hai đều hỏi *"cách
biểu diễn chunk có quan trọng không?"*. **Đề xuất: giữ ở §3 và có một câu bắc
cầu sang §5** — vì nó chạy trước, trên harness khác, và nó là cái mở ra 2C.

---

## §4 — Phase 2: chọn cách hỏi LLM · ~3 trang

**Phải làm được:**
1. **Phạm vi:** chỉ đổi prompt và `context_depth`. Không fine-tune. Retrieval
   đóng băng.
2. **Baseline P0:** AC 0,6308 / EM 0,0000 / Faithfulness 0,9761 — và chẩn đoán
   **bằng số**, không bằng ẩn dụ.
3. **Audit 30 câu điểm thấp** → bảng phân loại lỗi thật (xem `03_claims.md`,
   con số đang bị nói sai).
4. **Bốn prompt**, mỗi cái gắn với một nhóm lỗi *có thật trong bảng*. P1 thất bại
   viết dài bằng P2 thắng.
5. **`context_depth`** — giải thích bằng đoạn code, và **Hit@k của Phase 1 định
   giá sẵn chi phí cắt context**. Đây là chỗ nối hai phase, viết ở đây thay vì
   tách thành mục riêng.
6. **Quyết định winner:** guardrail trước, điểm sau. Đoạn chốt ở dưới.
7. **Hai lát cắt kiểm tra:** phân tầng `gold_in_top5`, và macro theo bài.

### 🔒 Đoạn chốt — đã duyệt, viết theo đúng ý này

> Cả P2-depth3 lẫn P2-depth5 đều tăng Answer Correctness so với P0
> (+0,1403 và +0,1043; cả hai khoảng tin cậy đều không chứa 0). So trực tiếp hai
> ứng viên với nhau thì **depth 3 còn cao hơn depth 5** 0,0360 điểm
> (CI95 [+0,0190; +0,0554]), và cao hơn đáng kể ở Exact Match lẫn Token F1.
>
> Nhưng so với P0, depth 3 **trượt hai trong bốn guardrail**: Faithfulness
> −0,0200 (ngưỡng 0,02) và Citation Validity −0,0142 (ngưỡng 0,01). Depth 5 qua
> cả bốn. Chỉ còn **một cấu hình hợp lệ**, nên winner là P2-depth5.
>
> **P2-depth5 thắng vì nó hợp lệ, không phải vì nó đạt điểm cao nhất.**

**Ba ràng buộc cách viết, bắt buộc:**

1. **Nói rõ depth 3 trượt *hai* guardrail và liệt kê cả hai.** Chỉ nêu
   Faithfulness thì lập luận còn đúng một điểm trượt sát biên, rất dễ bị vặn.
   Hai điểm trượt độc lập thì vững hơn hẳn.
2. **Cấm viết depth 5 "trung thực hơn P0".** Chênh lệch Faithfulness là +0,0040
   với CI95 [−0,0127; +0,0189] — **chứa 0**. Chỉ được viết *"không phân biệt
   được với P0"*. Guardrail đòi "đừng tệ đi quá ngưỡng", **không** đòi "phải tốt
   hơn"; qua guardrail không phải là thắng ở metric đó.
3. **Câu chốt là "hợp lệ duy nhất", không phải "điểm cao nhất".**

Phần *"kết luận này chắc tới đâu"* (công suất 66%, khả năng thổi phồng, luật thay
thế sẽ lật winner) **không** viết ở đây — để riêng ở §7. Trộn hai thứ vào một chỗ
chính là cái làm bản hiện tại đọc lủng củng.

**Bằng chứng:** `report_baseline_p0_d5.json`, `report_finalist_p2_d{3,5}.json`,
`paired_significance.json`, `low_correctness_review_annotations.json`,
`docs/prompts.md`.

**Không nhét vào:** nguyên văn prompt (trỏ sang `docs/prompts.md`).

---

## §5 — Phase 2C: chiến lược chia đoạn · ~2 trang · **CHỜ SỐ**

> Khung dựng sẵn, bảng để trống, điền khi Thắng có kết quả. Notebook đã có ở
> `notebooks/Tests/phase2c/`; kế hoạch ở `phase_2c_chunking_strategy_test_plan.md`.

**Phải làm được:**
1. **Vì sao có 2C:** Vòng 3 nói *kích thước* chunk không quan trọng; ablation gợi
   ý *cấu trúc* chunk thì có. 2C thử trục đó cho tử tế.
2. **Ma trận bốn chiến lược:**

   | ID | Chiến lược | Đơn vị retrieve | Context đưa cho generator |
   | :-- | :--- | :--- | :--- |
   | `C0` | Recursive 512/64 (control) | chunk 512 | chính chunk đó |
   | `C1` | Theo câu | nhóm câu ≤512 | chính nhóm đó |
   | `C2` | Theo đoạn văn | nhóm đoạn ≤512 | chính nhóm đó |
   | `C3` | Phân cấp | child 256/32 | **parent 512/64** chứa child |

3. **Vì sao C3 cần hai tầng đánh giá:** nó retrieve *child* nhưng generator đọc
   *parent*, nên phải đo cả *retrieval relevance* lẫn *delivered-context
   relevance*. Đây là điểm phương pháp thú vị nhất của 2C — nên viết kỹ.
4. **Guardrail của 2C** khác Phase 2B: thêm ràng buộc **Hit@5 và Recall@5 không
   tụt quá 0,01**, vì 2C được phép đụng vào truy xuất còn 2B thì không.
5. **Kết quả** — *(để trống)*
6. **Winner và vì sao** — *(để trống)*

**Bằng chứng cần có khi điền:** `phase2c_comparison.csv`, per-question scores,
bản ghi quyết định winner.

**Không nhét vào:** C4 semantic chunking — kế hoạch xếp nó là exploratory và
**không được dùng để thay winner**.

---

## §6 — Held-out · ~1,5 trang

**Phải làm được:**
1. **Vì sao cần:** 12 cấu hình đã thử trên development rồi chọn cái cao nhất →
   lệch lạc quan theo cấu trúc.
2. **Bằng chứng đăng ký trước:** mốc thời gian ký duyệt < mốc bắt đầu run, và
   chuỗi băm. Viết thành sự kiện có mốc, không thành khẩu hiệu.
3. **Kết quả:** bảng dev ↔ held-out ↔ macro.
4. **Phân tầng** — và phát biểu đúng phạm vi: trên `gold_in_top5`, held-out
   không thấp hơn dev; chênh lệch toàn tập đến từ tỉ lệ truy xuất trượt.
5. **Hệ quả:** AC toàn tập bị chặn trên bởi Hit@5.

**Bằng chứng:** `docs/reports/phase2/heldout/`, `phase2b_winner_decision.json`.

**Không nhét vào:** so sánh A/B trên held-out — không có baseline held-out, và
theo thiết kế thì không được chạy thêm.

---

## §7 — Giới hạn và việc còn lại · ~1,5 trang

**Phải làm được:**

*Giới hạn của kết quả*
- Dense **chưa tái lập được**: trôi 0,0143 > khoảng cách giữa các mô hình 0,0094.
  Nêu nguyên nhân trong mã: thiếu seed HNSW, `batch_size`, `torch.manual_seed`.
- Giải đấu phân tầng không phát hiện được tương tác giữa các vòng.
- Bootstrap không đồng nhất: Phase 1 và ablation theo câu, Phase 2 gom cụm theo bài.
- Biên độ nhập nhằng là **chỉ báo**, chưa ai đọc tay xác nhận.

*Giới hạn của bộ đo*
- Giám khảo là LLM, đã tách khỏi generator nhưng **chưa hiệu chuẩn với nhãn người**
  (20 câu `judge_calibration` đã chuẩn bị, chưa dùng).
- Audit 30 câu là **một người duyệt có AI hỗ trợ**, không phải audit mù hai người
  như kế hoạch yêu cầu.
- Answer Relevancy giảm là **hệ quả có chủ đích**, không phải suy thoái.

*Việc còn lại* — bảng ngắn, 6–7 dòng.

**Không nhét vào:** phần tự khen về việc đã trung thực nêu giới hạn.

---

## Những chỗ chưa quyết

| # | Câu hỏi | Ghi chú |
| ---: | :--- | :--- |
| 1 | §5 (2C) chờ số tới bao giờ? Nếu Thắng chưa xong thì nộp bản có khung rỗng hay bỏ hẳn §5? | Cần hỏi Thắng |
| 2 | Có cần 4 biểu đồ 300 DPI mà test plan Phase 1 §6 yêu cầu không? | Hiện chưa có cái nào. 10–15 trang thì có chỗ cho 2–3 hình |
| 3 | Phase 3 chỉ nhắc một đoạn ở §7, hay có mục riêng? | Đề xuất: một đoạn ở §7, vì chưa có kết quả |
| 4 | Trung vị độ dài đáp án chuẩn — chưa đo | Cần cho §4.2, không tốn API |
