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

> **Bộ ngưỡng này quyết định winner.** Phát biểu luật bằng khoảng tin cậy thay vì
> bằng điểm trung bình sẽ cho P2-depth3 qua hết guardrail, và vì d3 có AC cao hơn
> nên winner đổi. Có đưa phân tích độ nhạy này vào báo cáo không ❓ — xem
> [`06_decisions.md`](06_decisions.md) D3.

> Nguồn: `phase_2_generation_tuning_plan.md` §7 (dòng 176–188);
> code: `scripts/phase2_paired_comparison.py` hằng `GUARDRAILS`, `MIN_COVERAGE`.
> MDE₈₀: [`04_stats.md`](04_stats.md) mục 2.

---

## Phase 2C — Chọn chiến lược chunking · **6 điều kiện** 📌 *(đã chạy)*

So với **C0**. Nhiều hơn 2B **hai ràng buộc truy xuất**, vì 2C được phép đụng vào
cách cắt chunk còn 2B thì không:

| # | Metric | Không được giảm quá | Có ở 2B? | C3-D3 | C3-D5 |
| ---: | :--- | ---: | :-: | :--- | :--- |
| 1 | Coverage (retrieval + generation + judge) | đạt ngưỡng của C0 | ✅ | 281/281 ✅ | 281/281 ✅ |
| 2 | **Hit@5** | `0,01` | 🆕 **mới** | −0,0107 ❌ | −0,0107 ❌ |
| 3 | **Recall@5** | `0,01` | 🆕 **mới** | −0,0125 ❌ | −0,0125 ❌ |
| 4 | Faithfulness | `0,02` | ✅ | −0,0128 ✅ | −0,0148 ✅ |
| 5 | Citation F1 | `0,01` | ✅ | −0,0284 ❌ | −0,0275 ❌ |
| 6 | Citation Validity | `0,01` | ✅ | −0,0071 ✅ | −0,0036 ✅ |

Phá hòa: AC cao nhất → chênh AC < `0,01` thì chọn context input token thấp hơn →
token chênh < 5% thì chọn end-to-end P95 thấp hơn → vẫn hòa thì chọn strategy đơn
giản hơn, ít phụ thuộc artifact/runtime hơn. **Không kích hoạt lần nào** — không
có cấu hình nào qua hết guardrail nên không có gì để phá hòa.

Plan có tự giải thích vì sao đặt hai mức ngưỡng khác nhau, và đây là lập luận
đúng đắn cần trích vào báo cáo:

> `0,01` trên 281 câu tương đương khoảng 3 câu; `0,02` tương đương khoảng 6 câu.
> Citation Validity dùng margin chặt vì invalid citation trực tiếp làm người dùng
> không kiểm chứng được nguồn; Faithfulness dùng `0,02` vì judge metric liên tục
> có độ nhiễu cao hơn.

**C3 trượt ba guardrail, không phải một.** `notes_thang.md` §6 chỉ nêu Citation F1.
Hai ràng buộc truy xuất cũng trượt, và cả hai đều là ràng buộc **mới của 2C** —
tức đúng cái mà 2C thêm vào để canh phần nó được phép đụng, đã bắt được cái nó
sinh ra để bắt. Báo cáo phải nêu đủ ba.

### Ngưỡng `0,01` của 2C nằm ở đâu so với độ phân giải 📌

Đây là câu hỏi D12 đặt ra trước khi có số. Giờ đo được:

| Guardrail | Ngưỡng $T$ | SE đo được | Tỉ lệ **báo động giả** |
| :--- | ---: | ---: | :--- |
| **Hit@5** | 0,010 | 0,0090 | **13,3%** 🔴 |
| **Recall@5** | 0,010 | 0,0088 | **12,8%** 🔴 |
| **Citation F1** | 0,010 | 0,0140 | **23,8%** 🔴 |
| Faithfulness | 0,020 | 0,0098 | 2,1% ✅ |
| Citation Validity | 0,010 | 0,0049 | 2,0% ✅ |

Ba trong năm ngưỡng số của 2C nằm dưới tầm phân giải của cỡ mẫu — nhiều hơn 2B,
nơi chỉ Citation F1 có vấn đề. Phải nêu ở §7 khi báo cáo kết quả 2C.

**Hệ quả cho cách viết:** hai phán quyết truy xuất là **áp luật**, không phải kiểm
định. CI95 của cả hai đều **chứa 0** (Hit@5 [−0,0283; +0,0072], Recall@5
[−0,0300; +0,0054]), nên không được viết "C3 truy xuất kém hơn C0". Chỉ được viết
*"C3 tụt quá ngưỡng đã đăng ký trước"*. Riêng Citation F1 thì CI95
[−0,0568; −0,0022] **không** chứa 0, nên phán quyết đó vững cả hai đường.

> Tái lập:
> `python scripts/phase2_paired_comparison.py --scores-dir docs/reports/phase2c/scores --baseline c0_d5_development.jsonl --plan 2c`
>
> Nguồn: `phase_2c_chunking_strategy_test_plan.md` §9 (dòng 304–326);
> `docs/reports/phase2c/paired_significance.json`.

---

## Phase 3 — Abstention · **4 điều kiện** 📌 *(đã chạy)*

Bộ guardrail **khác hẳn** ba phase trên, vì bài toán khác: ở đây cái phải chặn là
hệ thống trả lời bừa khi không đủ bằng chứng.

Ba chính sách được so, tất cả trên cùng 140 câu development:

| | Chính sách | Là gì |
| :-- | :--- | :--- |
| `B0` | **control** | Đúng prompt P2 đã thắng Phase 2B, không sửa một chữ |
| `B1` | schema | Bắt model trả JSON có trường `answerability`, tự khai là trả lời được hay không |
| `B2` | schema + cổng | B1, cộng thêm: nếu điểm reranker của đoạn hạng 1 dưới ngưỡng thì ép từ chối |

| # | Metric | Ngưỡng | B0 | B1 | B2 |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 1 | Generation success rate | ≥ **98%** *(chặt hơn 95% của 2A/2B)* | 99,3% ✅ | 99,3% ✅ | 99,3% ✅ |
| 2 | **False-abstention rate** | ≤ **10%** — từ chối trong khi thật ra trả lời được | 0,0% ✅ | 0,0% ✅ | 0,0% ✅ |
| 3 | Answerable token F1 | giảm ≤ `0,02` so với B0 | — | **−0,1070** ❌ | **−0,1070** ❌ |
| 4 | Citation Validity | giảm ≤ `0,01` | — | 0,0000 ✅ | 0,0000 ✅ |

Metric chính: **false-answer rate thấp nhất** (không phải Answer Correctness).
Chênh dưới `0,02` thì ưu tiên pipeline đơn giản hơn, theo thứ tự B0 → B1 → B2.

**Kết quả: B0 thắng — không đổi gì cả.** B1 và B2 giảm false-answer rate rất mạnh
(5,06% → 1,27%) nhưng trượt guardrail 3, nên không hợp lệ. Đây **lại đúng hình
dạng của phán quyết Phase 2B**: cấu hình ăn điểm ở metric chính bị loại vì đánh
đổi một thứ đã khóa từ trước.

Khác 2B ở một điểm quan trọng: phán quyết này **không sát biên**. Mức tụt −0,1070
(CI95 [−0,1563; −0,0631], SE 0,0243) lớn gấp hơn **5 lần** ngưỡng 0,02 và gấp 4,4
lần SE của chính nó. Không có chuyện thiếu công suất ở đây.

> Tái lập: `docs/reports/phase3/policy_significance.json`.
>
> Nguồn: `phase_3_abstention_test_plan.md` §6 (dòng 144–152);
> `docs/reports/phase3/policy_comparison.csv`.

---

## Đọc ngang bốn phase

| | Phase 1 | 2A | 2B | 2C | 3 |
| :--- | :-: | :-: | :-: | :-: | :-: |
| Có guardrail loại thẳng? | ❌ | — | ✅ 4 | ✅ 6 | ✅ 4 |
| Có cấu hình nào qua hết? | — | — | 1/2 | **0/3** | **1/3** (chính là control) |
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

4. **Ba phase liên tiếp, guardrail đều thắng điểm số.** 2B loại P2-depth3, 2C loại
   cả C3-D3 lẫn C3-D5, Phase 3 loại B1/B2 — lần nào cấu hình bị loại cũng là cấu
   hình cao điểm hơn ở metric chính. Đây là phát hiện xuyên suốt đáng đưa lên §2,
   không phải ba sự cố rời rạc.

## Việc phải làm

1. Bảng "đọc ngang" ở trên vào §2 báo cáo.
2. §7 nêu: ngưỡng đặt không kèm tính toán cỡ mẫu ở cả 2B lẫn 2C. Ở 2B, Citation F1
   có tỉ lệ báo động giả 21,3%; ở 2C, **ba trong năm** ngưỡng số dưới tầm phân giải.
3. D3 ở [`06_decisions.md`](06_decisions.md). *(D12 đã đóng — số đã đo, xem trên.)*
