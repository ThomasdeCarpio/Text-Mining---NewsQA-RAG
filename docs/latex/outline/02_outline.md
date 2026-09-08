# 02 — Dàn ý báo cáo

Mục tiêu 10–15 trang. Mỗi mục ghi rõ: **phải chứng minh được điều gì**, người đọc
**đã phải biết gì** trước đó, **bằng chứng** nào chống lưng, và **không** được
nhét gì vào.

Ngân sách trang là để giữ tỉ lệ, không phải để đếm chính xác.

## Quy ước: mỗi mục kết bằng một đoạn **Chốt lại**

Từ §3 trở đi, mỗi mục kết bằng **2–3 câu** theo đúng ba ý này, không thêm:

1. **Mục này đã chốt được gì** — một câu, có số.
2. **Phải trả giá bằng gì** — cái bị loại, cái không phân định được, hoặc cái
   chưa đo. Không có mục nào không phải trả giá gì.
3. **Nó đưa sang mục sau cái gì** — một câu bắc cầu.

Ba ý này đã ghi sẵn ở cuối mỗi mục bên dưới, mục nào tra mục đó.

**Cấm:** tóm tắt lại toàn bộ mục, lặp lại bảng, hoặc tự khen. Nếu đoạn Chốt lại
chỉ nói lại những gì vừa đọc thì bỏ nó đi.

---

## §0 — Thuật ngữ · ~1 trang

**Phải làm được:** người đọc gặp `resolved`, `gold chunk`, `guardrail`,
`context_depth` ở các mục sau mà không phải đoán.

**Nội dung:** [`01_terms.md`](01_terms.md).

**Không nhét vào:** kết quả. §0 chỉ định nghĩa.

---

## §1 — Mở đầu: mục tiêu và cách chia việc · ~2 trang

> Đây là mục **duy nhất** người đọc chắc chắn đọc hết. Nó phải trả lời được
> *"đồ án này định làm gì, và làm thế nào để biết là đã làm được"* — trước khi
> có bất kỳ con số nào.

**Phải làm được, theo đúng thứ tự này:**

### 1.1 Bối cảnh — vì sao bài toán này khó

Hệ thống nhận gì, trả gì: câu hỏi → 5 đoạn văn → đáp án ngắn kèm trích dẫn `[n]`.
Cái khó không nằm ở việc gọi LLM, mà ở chỗ có **hai tầng hỏng độc lập**: truy xuất
lấy nhầm đoạn, và sinh viết sai dù đoạn đúng đã nằm trong tay. Một điểm số cuối
cùng không phân biệt được hai thứ đó.

### 1.2 Mục tiêu đồ án ❓ *(cần bạn điền — xem [`06_decisions.md`](06_decisions.md) D14)*

Chưa có tài liệu nào trong repo phát biểu mục tiêu đồ án theo lời của đề bài.
`master_test_plan.md` mở đầu bằng *kế hoạch thực nghiệm*, không bằng *mục tiêu*.
Cần hai thứ, và cả hai phải lấy từ đề bài chứ không suy ra:

- **Mục tiêu chung** — một câu, dạng "xây dựng và đánh giá …".
- **Tiêu chí chấm của môn học** — thầy chấm cái gì: hệ thống chạy được, hay quy
  trình thực nghiệm, hay cả hai, tỉ lệ ra sao.

*Cho tới khi có, mục này để trống chứ không được đoán.*

### 1.3 Mục tiêu cụ thể — bốn câu hỏi đồ án đặt ra

Bốn câu này **suy ra được từ bốn phase đã chạy**, nên viết được ngay:

| | Câu hỏi | Trả lời ở |
| :-- | :--- | :-- |
| Q1 | Cách truy xuất nào lấy được đoạn chứa đáp án tốt nhất? | §3 |
| Q2 | Cách ra lệnh cho LLM thế nào để nó trả lời đúng mà không bịa? | §4 |
| Q3 | Cách **cắt** văn bản thành đoạn có ảnh hưởng tới kết quả cuối không? | §5 |
| Q4 | Hệ thống có biết im lặng khi không đủ bằng chứng không? | §7 |

Và một câu hỏi xuyên suốt, quan trọng hơn cả bốn câu trên: **làm sao biết một cải
tiến là thật, chứ không phải là nhiễu đo hoặc là hệ quả của việc thử quá nhiều
lần?** Đó là nội dung §2, và là phần đóng góp phương pháp của đồ án.

### 1.4 Thế nào là đạt

Ba tiêu chí, và cả ba đều **đã chốt trước khi chạy**, không phải đặt ra sau khi
nhìn kết quả:

1. Mọi quyết định chọn cấu hình phải dựa trên **luật đăng ký trước**, không phải
   trên điểm số nhìn thấy sau.
2. Con số công bố phải đo trên tập **chưa từng dùng để tinh chỉnh**.
3. Mỗi cải tiến phải nói rõ **có phân định được với nhiễu hay không**.

Đây cũng chính là ba thứ §2 định nghĩa và §3–§7 thi hành.

### 1.5 Dữ liệu

11.064 bài báo CNN (200 bài có câu hỏi, 10.864 bài làm nhiễu); 1.152 câu sau khử
trùng lặp ngữ nghĩa; chia 50 bài development / 50 bài held-out / 100 bài dự trữ.
`original` vs `resolved` là gì, và vì sao báo cáo theo `resolved`.

### 1.6 Cách chia việc

Vì sao tách phase: hai tầng hỏng ở 1.1 cần hai loại thước đo khác nhau. Sơ đồ một
dòng: `EDA → Phase 1 (truy xuất) → Phase 2 (sinh) → Phase 2C (chia đoạn) → held-out → Phase 3 (từ chối)`.

**Bằng chứng:** EDA §1, §6; `subset_manifest.json`; `master_test_plan.md` §1.

**Cần cho mục sau:** §3 và §4 đều dựa vào việc người đọc hiểu tại sao truy xuất
và sinh được đo tách nhau. §8 sẽ đối chiếu ngược lại với 1.4.

**Không nhét vào:** chi tiết EDA; kết quả; và **không được liệt kê công nghệ
dùng** (BGE-M3, Gemini…) như một mục "công nghệ sử dụng" — chúng là *kết quả* của
§3 và §4, không phải tiền đề.

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

### Chốt lại §3

1. Truy xuất chốt ở BGE-M3 sparse + bge-reranker-large + chunk 512/64, Hit@5
   0,9573 trên 281 câu development.
2. Trả giá: chỉ **hai** trong nhiều so sánh phân định được (sparse ↔ dense,
   thêm reranker); kích thước chunk và lựa chọn giữa các mô hình dense là kết
   quả null, và dense chưa tái lập được giữa hai lần chạy.
3. Bàn giao sang §4: một tập 5 đoạn **đóng băng**, cộng một bảng Hit@k sẽ được
   dùng lại để định giá chi phí của việc cắt bớt context.

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

### Chốt lại §4

1. Prompt P2 với 5 đoạn context tăng Answer Correctness +0,1043 so với P0
   (CI95 [+0,0844; +0,1240]) và qua cả bốn guardrail.
2. Trả giá: cấu hình điểm cao hơn (P2-depth3, +0,0360 AC) bị loại vì trượt hai
   guardrail; và P2-depth5 **không** trung thực hơn P0 — chỉ là không tệ đi.
3. Bàn giao sang §5: prompt và depth đã khóa, nên biến duy nhất còn lại để thử
   là **cách cắt văn bản thành đoạn**.

---

## §5 — Phase 2C: chiến lược chia đoạn · ~2 trang 📌 *(đã có số)*

**Phải làm được:**
1. **Vì sao có 2C:** Vòng 3 nói *kích thước* chunk không quan trọng; ablation gợi
   ý *cấu trúc* chunk thì có. 2C thử trục đó cho tử tế.
2. **Nói ngay từ câu đầu rằng 2C là phần mở rộng khám phá hậu kiểm** — thiết kế
   sau khi held-out đã mở. Không được trình bày như thể nó nằm trong đăng ký
   trước ban đầu. Đây là điều `notes_thang.md` §9 tự yêu cầu, và nó đúng.
3. **Ma trận bốn chiến lược:**

   | ID | Chiến lược | Đơn vị retrieve | Context đưa cho generator | Số đơn vị | Index |
   | :-- | :--- | :--- | :--- | ---: | ---: |
   | `C0` | Recursive 512/64 (control) | chunk 512 | chính chunk đó | 22.766 | 64,90 MiB |
   | `C1` | Theo câu | nhóm câu ≤512 | chính nhóm đó | 22.014 | 63,38 MiB |
   | `C2` | Theo đoạn văn | nhóm đoạn ≤512 | chính nhóm đó | 22.018 | 63,23 MiB |
   | `C3` | Phân cấp | child 256/32 | **parent 512/64** chứa child | 49.218 + 22.766 | 76,58 MiB |

4. **Vì sao C3 cần hai tầng đánh giá:** nó retrieve *child* nhưng generator đọc
   *parent*, nên phải đo cả *retrieval relevance* lẫn *delivered-context
   relevance*. Đây là điểm phương pháp thú vị nhất của 2C — nên viết kỹ.
5. **Quy trình ba vòng thu hẹp dần**, và nêu thẳng cỡ mẫu từng vòng vì chúng rất
   khác nhau:

   | Vòng | Chạy gì | n | Kết quả |
   | :--- | :--- | ---: | :--- |
   | Sàng lọc truy xuất | C0–C3, chỉ retrieval | 281 | cả bốn qua, margin loại sớm 0,02 |
   | Sàng lọc sinh | C1, C2, C3 với P2-D5 | 80 *(RAGAS 20)* | chỉ C3 đi tiếp |
   | Sàng lọc depth | C3 với D1/D3/D5 | 80 *(RAGAS 20)* | D1 loại; D3 và D5 vào chung kết |
   | Chung kết | C3-D3, C3-D5 trên development đầy đủ | 281 | cả hai bị loại |

6. **Guardrail của 2C** khác Phase 2B: thêm ràng buộc **Hit@5 và Recall@5 không
   tụt quá 0,01**, vì 2C được phép đụng vào truy xuất còn 2B thì không.
7. **Kết quả:** C3-D3 là cấu hình **cao điểm nhất toàn dự án** trên development
   (AC 0,7730, cao hơn production +0,0380, CI95 [+0,0183; +0,0575]), và nó **bị
   loại**. Trượt ba guardrail: Citation F1 −0,0284, Hit@5 −0,0107, Recall@5
   −0,0125. Winner giữ nguyên **C0-P2-D5**.
8. **Cơ chế Citation F1 tụt:** không phải đánh số citation sai — Citation Validity
   của C3 vẫn khoảng 0,98. Là parent được trích dẫn khớp gold context kém hơn
   chunk recursive, nên precision/recall của citation giảm. Chi tiết ở
   [`03_claims.md`](03_claims.md) 4.8.

### 🔒 Ràng buộc cách viết cho §5

1. **Cấm viết "C3 truy xuất kém hơn C0".** CI95 của cả Hit@5 lẫn Recall@5 đều
   **chứa 0**. Chỉ được viết *"tụt quá ngưỡng đã đăng ký trước"* — đó là áp luật,
   không phải kết luận thống kê. Xem [`03_claims.md`](03_claims.md) 4.6.
2. **Cấm viết C1/C2 "kém hơn".** Chúng bị loại trên 80 câu, RAGAS chỉ chấm 20.
   Phải nêu kèm n.
3. **Cấm viết 2C đã chạy trên reserve 587 câu.** Chưa chạy.

**Bằng chứng:** `docs/reports/phase2c/paired_significance.json`,
`phase2c_retrieval_screening.csv`, `screening/*.json`, `notes_thang.md`.

**Không nhét vào:** C4 semantic chunking — kế hoạch xếp nó là exploratory và
**không được dùng để thay winner**; và nó không được chạy.

### Chốt lại §5

1. Không chiến lược chunking nào qua hết sáu guardrail; winner giữ nguyên
   C0-P2-D5.
2. Trả giá: C3-D3 là cấu hình cao điểm nhất toàn dự án (+0,0380 AC so với
   production) và vẫn bị loại — lần thứ hai trong đồ án luật thắng điểm số.
3. Bàn giao sang §6: cấu hình production không đổi sau 2C, nên con số held-out
   đã chạy vẫn là kết quả tổng quát hóa cuối cùng, không phải chạy lại.

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

### Chốt lại §6

1. Con số công bố: Answer Correctness 0,7157 trên 284 câu chưa từng bị chạm tới,
   chạy đúng một lần sau khi quyết định đã đóng băng.
2. Trả giá: Hit@5 tụt xuống 0,8768, tức khoảng 12% số câu không có đường nào để
   trả lời đúng — trần của tầng sinh nằm ở tầng truy xuất, không ở prompt.
3. Bàn giao sang §7: hệ thống trả lời tốt khi có bằng chứng; câu còn lại là nó
   làm gì **khi không có**.

---

## §7 — Phase 3: dạy hệ thống biết từ chối · ~1,5 trang 📌 *(đã có số)*

> Mục này có tồn tại hay không phụ thuộc **D7** ở
> [`06_decisions.md`](06_decisions.md). Phương án còn lại là gói thành một đoạn
> trong §8.

**Phải làm được:**
1. **Bài toán đảo trục:** ba phase trước hỏi *"trả lời đúng tới đâu"*. Phase 3
   hỏi *"có im lặng được khi không đủ bằng chứng không"*. Metric chính đổi từ
   Answer Correctness sang **false-answer rate**, và ngưỡng coverage siết từ 95%
   lên 98%.
2. **Bộ dữ liệu không tự nhiên có sẵn** — phải dựng: 200 case, 140 development /
   60 final-test, gồm 87 câu trả lời được làm control và 6 loại câu *không* trả
   lời được. Nêu bảng loại case, và nêu thẳng rằng `natural_retrieval_miss` chỉ
   có 2 + 1 case nên ô đó không mang thông tin.
3. **Ba chính sách:** B0 (chính prompt P2 của Phase 2, không sửa) · B1 (bắt trả
   JSON có trường `answerability`) · B2 (B1 + cổng chặn theo điểm reranker).
4. **Kết quả — B0 thắng, không đổi gì:** B1 giảm false-answer rate 5,06% → 1,27%
   (4 lỗi → 1 lỗi trên 79 câu) nhưng làm token F1 của các câu trả lời được tụt
   **−0,1070**, gấp hơn năm lần ngưỡng 0,02.
5. **Cơ chế, và đây là chỗ đáng viết nhất:** prompt B1 bỏ mất câu lệnh quy định
   *hình dạng đáp án* của P2 và chỉ còn "concise answer". Đáp án dài trở lại —
   trung bình 12,5 → 16,3 từ — tức **đúng nhóm lỗi mà Phase 2 đã chữa** quay về.
   Ví dụ cụ thể: đáp án `Robert Park [1].` (F1 1,00) thành một câu 24 từ kể lại
   cả bối cảnh (F1 0,15).
6. **B2 là kết quả null theo đúng nghĩa:** hiệu chuẩn quét 107 ngưỡng, và điểm
   tốt nhất chính là ngưỡng thấp nhất — tức **tắt cổng**. Không ngưỡng nào hạ
   được false-answer rate mà vẫn giữ false-abstention ≤ 10%. Điểm reranker không
   mang đủ thông tin để làm cổng.
7. **Final-test 60 câu, chạy sau khi đã khóa:** B0 abstention F1 0,9697,
   false-abstention 0,0%, citation validity 1,0000. Mã từ chối hiệu chuẩn trên
   tập final — trích một dòng để chứng minh.

### 🔒 Ràng buộc cách viết cho §7

1. **Phải nêu confound.** B1 đổi hai thứ cùng lúc: thêm schema *và* bỏ ràng buộc
   hình dạng đáp án. Thí nghiệm này **không** tách được "schema có hại" khỏi
   "mất câu lệnh hình dạng có hại". Viết B1 thất bại mà giấu chuyện đó là nói quá.
2. **Cấm gọi Phase 3 là "chưa xong".** Nó xong rồi, và cho kết quả âm. Kết quả âm
   có kiểm soát là kết quả.
3. **Phải nêu lần sửa dữ liệu sau duyệt** (3 dòng, một câu hỏi mơ hồ về cháy rừng).

**Bằng chứng:** `docs/reports/phase3/policy_comparison.csv`,
`policy_significance.json`, `phase3_final_results.json`,
`docs/Detailed Test Plans/phase_3_abstention_test_plan.md`.

**Không nhét vào:** so sánh Phase 3 với Phase 2 bằng điểm số — hai bộ dữ liệu
khác nhau, hai metric chính khác nhau.

### Chốt lại §7

1. Không chính sách abstention nào được nhận: B0 — chính prompt của Phase 2 —
   thắng, với false-answer rate 5,88% trên final-test.
2. Trả giá: B1 hạ được false-answer rate xuống gần bốn lần nhưng phá hỏng hình
   dạng đáp án (−0,1070 token F1), và thí nghiệm không tách được hai nguyên nhân.
3. Bàn giao sang §8: ba phase liên tiếp cùng cho một dạng kết quả, và đó là thứ
   đáng nói nhất của đồ án.

---

## §8 — Giới hạn và việc còn lại · ~1,5 trang

**Phải làm được:**

*Giới hạn của kết quả*
- Dense **chưa tái lập được**: trôi 0,0143 > khoảng cách giữa các mô hình 0,0094.
  Nêu nguyên nhân trong mã: thiếu seed HNSW, `batch_size`, `torch.manual_seed`.
- Giải đấu phân tầng không phát hiện được tương tác giữa các vòng.
- Bootstrap không đồng nhất: Phase 1, ablation và Phase 3 bốc theo câu; Phase 2
  và 2C gom cụm theo bài.
- Biên độ nhập nhằng là **chỉ báo**, chưa ai đọc tay xác nhận.
- **Ngưỡng guardrail đặt không kèm tính cỡ mẫu, ở cả 2B lẫn 2C.** Ở 2B, ngưỡng
  Citation F1 có tỉ lệ báo động giả 21,3%; ở 2C, ba trong năm ngưỡng số nằm dưới
  tầm phân giải (13,3% / 12,8% / 23,8%). Chi tiết:
  [`04_stats.md`](04_stats.md) mục 2.
- **2C là phần mở rộng khám phá hậu kiểm**, không phải phần đăng ký trước; và
  development 281 câu đã bị dùng qua nhiều vòng nên vẫn có thể overfit thích nghi.
- **Phase 3 có confound trong thiết kế B1** — xem §7, ràng buộc 1.
- Tập reserve 587 câu của 2C và final-test 150 bài của Phase 1 đều **chưa chạy**.

*Giới hạn của bộ đo*
- Giám khảo là LLM, đã tách khỏi generator nhưng **chưa hiệu chuẩn với nhãn người**
  (20 câu `judge_calibration` đã chuẩn bị, chưa dùng).
- Audit 30 câu là **một người duyệt có AI hỗ trợ**, không phải audit mù hai người
  như kế hoạch yêu cầu.
- Answer Relevancy giảm là **hệ quả có chủ đích**, không phải suy thoái.

*Việc còn lại* — bảng ngắn, 6–7 dòng.

**Không nhét vào:** phần tự khen về việc đã trung thực nêu giới hạn.

### Chốt lại §8

Một câu duy nhất: giới hạn lớn nhất của đồ án không phải là điểm số, mà là
**bộ đo** — giám khảo là LLM chưa hiệu chuẩn với nhãn người, nên mọi con số
tuyệt đối đều là ước lượng, còn mọi con số **so sánh** thì vững hơn vì cùng một
giám khảo chấm cả hai phía.

---

## §9 — Kết luận · ~1 trang

> Không phải bản tóm tắt báo cáo. Người đọc vừa đọc xong tám mục; đừng kể lại.

**Phải làm được, đúng bốn ý, theo thứ tự:**

### 9.1 Trả lời thẳng bốn câu hỏi ở §1.3

Mỗi câu **một dòng**, có số, không giải thích lại:

| | Câu hỏi | Trả lời |
| :-- | :--- | :--- |
| Q1 | Truy xuất thế nào? | BGE-M3 sparse + bge-reranker-large, Hit@5 0,9573 — sparse hơn dense +0,1634 nDCG@5 |
| Q2 | Ra lệnh cho LLM thế nào? | Prompt P2, 5 đoạn context, AC +0,1043 so với baseline |
| Q3 | Cách cắt đoạn có quan trọng không? | **Không đủ để đổi cấu hình** — không chiến lược nào qua hết guardrail |
| Q4 | Có biết im lặng không? | Biết sẵn: false-answer rate 5,88%; **không** chính sách bổ sung nào được nhận |

### 9.2 Kết quả công bố

Answer Correctness **0,7157** trên 284 câu held-out, chạy đúng một lần. Kèm dải
sàn–trần: đây là **cận dưới**, vì audit 30 câu điểm thấp thấy 23/30 thực ra đúng
về ngữ nghĩa và giám khảo bất đồng với người duyệt ở 18/30 câu.

### 9.3 Phát hiện phương pháp — phần đáng nói nhất

Ba phase liên tiếp cùng cho một dạng kết quả:

| Phase | Cấu hình cao điểm nhất | Cấu hình được nhận | Vì sao |
| :--- | :--- | :--- | :--- |
| 2B | P2-depth3 | P2-depth5 | trượt 2 guardrail |
| 2C | C3-depth3 | C0-depth5 *(không đổi)* | trượt 3 guardrail |
| 3 | B1 / B2 | B0 *(không đổi)* | trượt guardrail token F1 |

**Ba lần liên tiếp, cấu hình ăn điểm ở metric chính bị loại vì đánh đổi một thứ
đã khóa từ trước.** Nếu đồ án chỉ chọn theo điểm cao nhất thì cả ba lần đều chọn
sai. Đây là lập luận cho việc đăng ký trước, và nó là **kết quả đo được**, không
phải là quan điểm.

Nói thẳng cả mặt kia: ba lần đó, hai lần phán quyết nằm sát biên độ đo được
(§8), và một lần thì không. Không được viết như thể cả ba đều chắc như nhau.

### 9.4 Việc tiếp theo, và nó đi từ đâu ra

Đúng **ba** việc, mỗi việc trỏ về một con số trong báo cáo — không phải danh sách
mong muốn:

1. **Query rewriting**, vì §6 cho thấy trần nằm ở Hit@5 chứ không ở prompt.
2. **Hiệu chuẩn giám khảo với nhãn người**, vì §8 cho thấy đó là giới hạn lớn nhất.
3. **Tách lại thí nghiệm B1**, vì §7 có confound chưa gỡ được.

**Không nhét vào:** con số mới; lời cảm ơn; câu "hướng phát triển trong tương lai"
không gắn với số nào.

---

## Những chỗ chưa quyết

| # | Câu hỏi | Ghi chú |
| ---: | :--- | :--- |
| 1 | ~~§5 (2C) chờ số tới bao giờ?~~ | ✅ số đã về ngày 08/09 |
| 2 | Có cần 4 biểu đồ 300 DPI mà test plan Phase 1 §6 yêu cầu không? | Hiện chưa có cái nào. 10–15 trang thì có chỗ cho 2–3 hình |
| 3 | Phase 3: mục riêng (§7) hay một đoạn trong §8? | Đã có kết quả final-test nên **D7** phải quyết lại. Để mục riêng thì báo cáo dài thêm khoảng 1,5 trang |
| 4 | Trung vị độ dài đáp án chuẩn — chưa đo | Cần cho §4.2, không tốn API |
| 5 | Mã Phase 3 nằm ngoài repo | Bản chạy thật là `phase3_run.py` / `phase3_metrics.py` của Thắng, khác `scripts/run_phase3_abstention.py` đang có. Xem **D13** |
