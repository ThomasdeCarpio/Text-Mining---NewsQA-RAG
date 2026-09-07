# 05 — Toàn bộ guardrail, liệt kê đầy đủ

Chép từ test plan, không nhớ lại. Mỗi dòng ghi nguồn để tra ngược.

**Guardrail là gì:** điều kiện một cấu hình phải **qua hết** thì mới được đem ra
so điểm chính. Trượt một cái là bị loại, dù điểm chính cao hơn. Đây là ngưỡng
nhóm tự đặt và đăng ký trước — **không phải kết quả của một phép kiểm định**.
Cả hai plan đều tự ghi rõ: *"Ngưỡng là practical guardrail, không phải khẳng
định ý nghĩa thống kê."*

---

## Phase 1 — **không có guardrail**

Phase 1 không dùng cơ chế này. Nó chỉ có **hai điều kiện để gọi là thắng**:
khoảng cách vượt biên độ nhập nhằng 7,0%, *và* CI95 của hiệu theo cặp không
chứa 0.

Vì sao khác: ở tầng truy xuất, chọn sai chỉ là xếp hạng kém hơn. Ở tầng sinh,
chọn sai nghĩa là hệ thống nói điều không có trong bằng chứng — nên Phase 2 trở
đi mới cần một cơ chế loại thẳng.

> Nguồn: `phase_1_retrieval_test_plan.md`; `docs/reports/phase1/report.md` §1.2.

---

## Phase 2A — Baseline · ngưỡng nghiệm thu

Không phải guardrail chọn winner (2A không chọn gì), mà là điều kiện để một run
được coi là hợp lệ:

| # | Điều kiện | Ngưỡng |
| ---: | :--- | :--- |
| 1 | Generation coverage | ≥ **95%**, mục tiêu 100% sau retry/resume |
| 2 | RAGAS coverage trên các generation thành công | ≥ **95%** |

> Nguồn: `phase_2_baseline_test_plan.md` dòng 192–193.

---

## Phase 2B — Chọn prompt và context depth · **4 điều kiện**

So với **P0**:

| # | Metric | Không được giảm quá | SE đo được | Tỉ lệ **báo động giả** |
| ---: | :--- | ---: | ---: | :--- |
| 1 | **Faithfulness** | `0,02` | 0,0080 | **0,6%** ✅ |
| 2 | **Citation F1** | `0,01` | 0,0126 | **21,3%** 🔴 |
| 3 | **Citation Validity** | `0,01` | 0,0036 | **0,3%** ✅ |
| 4 | **Coverage** (generation + RAGAS) | ≥ 95% | — | ✅ đếm, không ước lượng |

*Báo động giả* = xác suất một cấu hình **thật ra không kém hơn P0** vẫn bị loại vì
nhiễu đo. Chỉ **Citation F1** có vấn đề: cứ 5 cấu hình giống hệt P0 thì hơn 1 cái
bị loại oan. Muốn chữa bằng dữ liệu thì cần 3.485 câu, mà cả tập chỉ có 1.152 —
không chữa được. Chi tiết và cách xử lý: [`04_stats.md`](04_stats.md) mục 2.

Qua hết 4 điều kiện rồi mới xét **Answer Correctness cao nhất**, sau đó ba mức
phá hòa:

| Thứ tự | Áp dụng khi | Chọn theo |
| ---: | :--- | :--- |
| 2 | (sau khi lọc guardrail) | Answer Correctness cao nhất |
| 3 | chênh AC < `0,01` | token cost thấp hơn |
| 4 | chi phí chênh < 5% | generation P95 latency thấp hơn |
| 5 | vẫn hòa | cấu hình đơn giản hơn: ít context, prompt ngắn, ít ràng buộc định dạng |

**Thực tế chỉ dùng tới bước 1 và 2.** Chỉ P2-depth5 qua được 4 guardrail nên
bước 3–5 không kích hoạt.

> 🔴 **Bộ ngưỡng này quyết định winner, và một luật thay thế hợp lý sẽ chọn khác.**
> Nếu đổi sang *"chỉ loại khi CI95 nằm hẳn dưới ngưỡng"* thay vì nhìn điểm trung
> bình, P2-depth3 qua hết guardrail và có AC cao hơn → **winner lật sang
> P2-depth3**. Không đổi luật, vì đổi lúc này là chọn luật cho ra số đẹp hơn.
> Nhưng phải nêu trong báo cáo. Bảng đối chiếu: [`04_stats.md`](04_stats.md) mục 2.

> Nguồn: `phase_2_generation_tuning_plan.md` §7 (dòng 176–188);
> code: `scripts/phase2_paired_comparison.py` hằng `GUARDRAILS`, `MIN_COVERAGE`.
> MDE₈₀: [`04_stats.md`](04_stats.md) mục 2.

---

## Phase 2C — Chọn chiến lược chunking · **5 điều kiện** *(chưa chạy)*

So với **C0**. Nhiều hơn 2B **hai ràng buộc truy xuất**, vì 2C được phép đụng vào
cách cắt chunk còn 2B thì không:

| # | Metric | Không được giảm quá | Có ở 2B? |
| ---: | :--- | ---: | :-: |
| 1 | Coverage (retrieval + generation + judge) | đạt ngưỡng của C0 | ✅ |
| 2 | **Hit@5** | `0,01` | 🆕 **mới** |
| 3 | **Recall@5** | `0,01` | 🆕 **mới** |
| 4 | Faithfulness | `0,02` | ✅ |
| 5 | Citation F1 | `0,01` | ✅ |
| 6 | Citation Validity | `0,01` | ✅ |

Phá hòa: AC cao nhất → chênh AC < `0,01` thì chọn context input token thấp hơn →
token chênh < 5% thì chọn end-to-end P95 thấp hơn → vẫn hòa thì chọn strategy đơn
giản hơn, ít phụ thuộc artifact/runtime hơn.

Plan có tự giải thích vì sao đặt hai mức ngưỡng khác nhau, và đây là lập luận
đúng đắn cần trích vào báo cáo:

> `0,01` trên 281 câu tương đương khoảng 3 câu; `0,02` tương đương khoảng 6 câu.
> Citation Validity dùng margin chặt vì invalid citation trực tiếp làm người dùng
> không kiểm chứng được nguồn; Faithfulness dùng `0,02` vì judge metric liên tục
> có độ nhiễu cao hơn.

**Cảnh báo cho Thắng trước khi chạy.** Kinh nghiệm từ 2B: ngưỡng `0,01` cho
Citation F1 hóa ra có tỉ lệ báo động giả **21,3%** vì SE của metric đó là 0,0126.
Ngưỡng `0,01` cho Hit@5 / Recall@5 của 2C có thể vướng đúng bẫy đó.

Việc nên làm, và làm được ngay khi có kết quả C1 đầu tiên: **tính SE của hiệu
C1 − C0 rồi đối chiếu với ngưỡng**, trước khi áp guardrail cho cả bốn strategy.
Nếu ngưỡng nhỏ hơn ~2×SE thì nó đang loại theo nhiễu.

Cận trên (tính trên chính giá trị, chưa ghép cặp) là SE 0,0121 → MDE₈₀ 0,0339 cho
cả Hit@5 lẫn Recall@5. Hiệu theo cặp sẽ có SE nhỏ hơn nhiều vì C1–C3 trùng C0 ở
phần lớn câu, nên con số thật có thể vẫn ổn — nhưng phải đo mới biết.

> Nguồn: `phase_2c_chunking_strategy_test_plan.md` §9 (dòng 304–326).

---

## Phase 3 — Abstention · **4 điều kiện** *(chưa chạy)*

Bộ guardrail **khác hẳn** ba phase trên, vì bài toán khác: ở đây cái phải chặn là
hệ thống trả lời bừa khi không đủ bằng chứng.

| # | Metric | Ngưỡng |
| ---: | :--- | :--- |
| 1 | Generation success rate | ≥ **98%** *(chặt hơn 95% của 2A/2B)* |
| 2 | **False-abstention rate** | ≤ **10%** — từ chối trong khi thật ra trả lời được |
| 3 | Answerable token F1 | giảm ≤ `0,02` so với B0 |
| 4 | Citation Validity | giảm ≤ `0,01` |

Metric chính: **false-answer rate thấp nhất** (không phải Answer Correctness).
Chênh dưới `0,02` thì ưu tiên pipeline đơn giản hơn, theo thứ tự B0 → B1 → B2.

> Nguồn: `phase_3_abstention_test_plan.md` §6 (dòng 144–152).

---

## Đọc ngang bốn phase

| | Phase 1 | 2A | 2B | 2C | 3 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| Có guardrail loại thẳng? | ❌ | — | ✅ 4 | ✅ 6 | ✅ 4 |
| Coverage tối thiểu | — | 95% | 95% | bằng C0 | **98%** |
| Faithfulness | — | — | 0,02 | 0,02 | — |
| Citation F1 | — | — | 0,01 | 0,01 | — |
| Citation Validity | — | — | 0,01 | 0,01 | **0,01** |
| Hit@5 / Recall@5 | — | — | — | **0,01** | — |
| Token F1 (answerable) | — | — | — | — | **0,02** |
| False-abstention | — | — | — | — | **10%** |
| Metric chính | nDCG@5 | — | Answer Correctness | Answer Correctness | **false-answer rate** |

Ba điều đọc ra được, và cả ba nên vào §2 báo cáo:

1. **Ba metric lặp lại ở mọi phase có tầng sinh:** Faithfulness, Citation F1,
   Citation Validity — luôn cùng mức 0,02 / 0,01 / 0,01. Đây là "bộ ba bất khả
   xâm phạm" của dự án: dù đang tối ưu prompt hay chunking, không được đổi chúng
   lấy điểm.
2. **Guardrail bám theo thứ mà phase đó được phép đụng.** 2B không đụng truy xuất
   nên không có guardrail truy xuất; 2C đụng nên phải thêm Hit@5 và Recall@5.
3. **Phase 3 đảo trục hoàn toàn** — metric chính không còn là "trả lời đúng đến
   đâu" mà là "trả lời bừa ít đến đâu", và ngưỡng coverage siết từ 95% lên 98%.

## Việc phải làm

1. Bảng "đọc ngang" ở trên nên vào §2 báo cáo — hiện chưa có chỗ nào liệt kê đủ.
2. Nêu ở §7 hai điều: ngưỡng 2B đặt **không kèm tính toán cỡ mẫu**, và Citation F1
   có tỉ lệ báo động giả 21,3% ([`03_claims.md`](03_claims.md) mục S5).
3. §4 nêu chuyện **luật thay thế sẽ lật winner** — thay cho câu tự khen
   "guardrail đã làm đúng việc của nó".
4. Nhắn Thắng đo SE trước khi áp guardrail 2C.
