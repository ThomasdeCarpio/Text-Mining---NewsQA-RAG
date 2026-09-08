# 04 — Giả thuyết và cỡ mẫu

Nội dung cho §2 của báo cáo. Số lấy từ `paired_significance.json` (mục
`resolution`), sinh bởi `scripts/phase2_paired_comparison.py`.

---

## 1. Giả thuyết ✅ *(đã duyệt)*

Mọi so sánh trong báo cáo có cùng một dạng. Với cấu hình mốc A và cấu hình B chạy
trên cùng bộ câu hỏi, gọi $d_i$ là hiệu điểm ở câu hỏi thứ $i$:

$$d_i = m(B, q_i) - m(A, q_i)$$

- $H_0:\ \mathbb{E}[d] = 0$ — hai cấu hình không khác nhau ở metric $m$.
- $H_1:\ \mathbb{E}[d] \neq 0$ — **hai phía**, vì ta quan tâm cả chiều tụt lẫn chiều tăng.
- Mức ý nghĩa $\alpha = 0{,}05$.
- Phép kiểm: khoảng tin cậy percentile bootstrap 95% cho $\mathbb{E}[d]$. Bác bỏ
  $H_0$ khi khoảng đó không chứa 0. Không dùng p-value; khoảng tin cậy hai phía ở
  mức 95% tương đương phép kiểm hai phía ở $\alpha = 0{,}05$.

### Ba loại phát biểu, đừng gọi chung một tên

| Loại | Ví dụ | Có $H_0$ không? |
| :--- | :--- | :--- |
| **Kiểm định** | Sparse hơn dense? P2 hơn P0? | Có, như trên |
| **Ngưỡng quyết định (guardrail)** | Faithfulness không được tụt quá 0,02 | **Không.** Ngưỡng nhóm tự đặt, đăng ký trước. Không có $H_0$, không có $\alpha$ |
| **Mô tả** | AC held-out = 0,7157 | Không. Một số đo, kèm khoảng tin cậy để biết độ rộng |

Dùng chung chữ "có ý nghĩa" cho cả ba là nguồn nhầm lẫn chính.

### Luật chọn winner, viết dưới dạng hình thức

Gọi $\bar{m}_X$ là điểm trung bình của cấu hình $X$ ở metric $m$. Theo
`phase_2_generation_tuning_plan.md` §7, cấu hình $X$ **hợp lệ** khi và chỉ khi cả
bốn điều sau đúng:

$$\bar{f}_X - \bar{f}_{P_0} \ge -0{,}02 \quad \text{(Faithfulness)}$$
$$\bar{c}_X - \bar{c}_{P_0} \ge -0{,}01 \quad \text{(Citation F1)}$$
$$\bar{v}_X - \bar{v}_{P_0} \ge -0{,}01 \quad \text{(Citation Validity)}$$
$$\text{coverage}_X \ge 0{,}95$$

Trong các cấu hình hợp lệ, chọn $\arg\max$ Answer Correctness.

**Về hình thức:** luật so **ước lượng điểm của hiệu trung bình** với $-T$. Nó
không dùng khoảng tin cậy ở bất kỳ đâu.

### So sánh bội 📌 *(số đo)*

Chạy càng nhiều phép kiểm, càng dễ vớ phải một kết quả "có ý nghĩa" hoàn toàn do
may. `paired_significance.json` chứa **14 phép** (2 finalist × 7 metric). Nếu cả
14 đều thật sự không có khác biệt, xác suất ít nhất một cái trông có ý nghĩa là
**51%**.

Bonferroni siết $\alpha$ từ 0,05 xuống $0{,}05/14$, tức ngưỡng $z$ từ 1,960 lên
**2,914**. Áp vào dữ liệu thật thì **đúng hai kết luận đổi**:

| Phép so sánh | Δ | $|\Delta|/SE$ | $\alpha=0{,}05$ | Bonferroni |
| :--- | ---: | ---: | :--- | :--- |
| p2_d5 Answer Correctness | +0,1043 | 10,09 | có ý nghĩa | giữ |
| p2_d5 Citation F1 | +0,0366 | 2,91 | có ý nghĩa | giữ (sát) |
| **p2_d3 Faithfulness** | **−0,0200** | **2,37** | có ý nghĩa | 🔴 **mất** |
| **p2_d3 Citation Validity** | **−0,0142** | **2,11** | có ý nghĩa | 🔴 **mất** |

Mười hai phép còn lại không đổi.

Hai phép bị mất đúng là hai phép đã loại P2-depth3. Cộng với công suất 66% / 56%
và hệ số thổi phồng ×1,23 / ×1,33 ở dưới, có **ba đường độc lập** cùng chỉ ra
rằng phân định d3 với d5 là mắt xích yếu nhất.

**Không lật winner:** guardrail là phép so ngưỡng, không phải kiểm định, nên d3
vẫn trượt vì điểm trung bình −0,0200 và −0,0142 nằm dưới ngưỡng. Cái bị ảnh
hưởng là **câu chống lưng** trong báo cáo, không phải phán quyết.

Bonferroni cũng quá thận trọng ở đây: 7 metric không độc lập (AC, EM, Token F1
đo những thứ liên quan nhau), và metric chính đã được đăng ký trước đúng một cái.

Cách xử lý ❓ — xem [`06_decisions.md`](06_decisions.md) D2.

---

## 2. Cỡ mẫu này phân giải được tới đâu 📌 *(số đo)*

Câu hỏi đúng không phải "281 có đủ lớn không?" mà **"281 câu / 50 bài phân biệt
được khoảng cách nhỏ tới mức nào?"**

### Hai con số dễ nhầm

Lấy Faithfulness, P2-depth3 so với P0 làm ví dụ. Phân bố của 281 hiệu số $d_i$:

```
tụt mạnh (≤ −0,5)     11 câu    3,9%
tụt nhẹ                5 câu    1,8%
đúng bằng 0          252 câu   89,7%
tăng nhẹ              12 câu    4,3%
tăng mạnh (≥ +0,5)     1 câu    0,4%
```

| | Độ lệch chuẩn của **từng câu** | **SE** — độ lệch chuẩn của **số trung bình** |
| :--- | ---: | ---: |
| Giá trị | 0,1602 | 0,0084 |
| Nói về | 281 số riêng lẻ nhấp nhô cỡ nào | Δ sẽ xê dịch cỡ nào nếu làm lại thí nghiệm |
| Quan sát trực tiếp được? | Có | Không — phải giả lập bằng bootstrap |

Hẹp hơn khoảng 19 lần vì trung bình hóa: $SE \approx 0{,}1602/\sqrt{281} = 0{,}0096$,
sát với 0,0084 mà bootstrap cho ra (lệch chút vì có gom cụm theo bài).

**Khoảng tin cậy và MDE nói về cột phải, không nói về cột trái.**

### MDE₈₀ tính thế nào

1. Có 281 hiệu số $d_i$.
2. Bootstrap: bốc lại 50 bài (có hoàn lại) 2.000 lần, mỗi lần tính trung bình →
   2.000 giá trị Δ giả lập.
3. **SE** = độ lệch chuẩn của 2.000 giá trị đó.

Rồi công thức công suất chuẩn, hai phía, $\alpha = 0{,}05$, công suất $0{,}80$:

$$MDE_{80} = (z_{0{,}975} + z_{0{,}80}) \times SE = (1{,}960 + 0{,}842) \times SE = 2{,}802 \times SE$$

Nghĩa: khoảng cách **thật** phải lớn ít nhất chừng đó thì mới bắt được 80% số lần.

### P2-depth5 so với P0

| Metric | Δ | SE | DEFF | n hiệu lực | MDE₈₀ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Answer Correctness | +0,1043 | 0,0103 | 0,72 | 388 | **0,0290** |
| Exact Match | +0,0925 | 0,0180 | 1,08 | 260 | 0,0504 |
| Token F1 | +0,1542 | 0,0122 | 0,81 | 346 | 0,0343 |
| Faithfulness | +0,0040 | 0,0080 | 1,27 | 221 | 0,0225 |
| Citation F1 | +0,0366 | 0,0126 | 1,20 | 235 | 0,0352 |
| Citation Validity | −0,0036 | 0,0036 | 1,04 | 270 | 0,0100 |
| Answer Relevancy | −0,0858 | 0,0183 | 1,36 | 207 | 0,0512 |

Khoảng cách chính (+0,1043) lớn gấp **3,6 lần** MDE₈₀ của nó — kết luận vững.

### Gom cụm theo bài ảnh hưởng bao nhiêu

**Design effect** = $(SE_\text{bốc theo bài} / SE_\text{bốc theo câu})^2$. Kiểm
chéo bằng hệ số tương quan trong cụm (ICC); hai phương pháp khớp nhau:

| Metric | ICC | DEFF lý thuyết | DEFF bootstrap |
| :--- | ---: | ---: | ---: |
| Answer Correctness | −0,056 | 0,74 | 0,72 |
| Faithfulness | +0,110 | 1,50 | 1,27 |
| Citation F1 | +0,052 | 1,24 | 1,20 |

Không có quy luật chung: với Faithfulness thì gom cụm làm khoảng tin cậy **rộng
ra** (ICC dương), với Answer Correctness thì **hẹp lại** (ICC âm). Lý do ICC nhỏ:
đây là **hiệu** giữa hai cấu hình, nên độ khó của bài ảnh hưởng lên cả hai như
nhau và triệt tiêu trong phép trừ.

Phase 1 và ablation bốc theo câu, Phase 2 bốc theo bài — không đồng nhất giữa các
phase, cần ghi ở §7.

### Ngưỡng guardrail có nằm trong tầm đo không

Guardrail là lằn ranh quyết định, không phải phép kiểm định, nên câu hỏi đúng là
**hai tỉ lệ sai**:

- **Báo động giả** — cấu hình thật ra *không* kém hơn P0 nhưng đo ra tụt quá
  ngưỡng nên bị loại oan: $P(\hat{d} \le -T \mid d_\text{thật} = 0) = \Phi(-T/SE)$
- **Bỏ sót** — cấu hình tụt đúng bằng ngưỡng nhưng đo ra chưa tới: luôn bằng
  **50%** với mọi ngưỡng. Đây là tính chất cố hữu của mọi lằn ranh.

| Guardrail | Ngưỡng $T$ | SE | Báo động giả | Số câu cần để $MDE_{80}=T$ |
| :--- | ---: | ---: | ---: | ---: |
| Faithfulness | 0,020 | 0,0080 | **0,6%** | 355 |
| **Citation F1** | 0,010 | 0,0126 | **21,3%** 🔴 | **3.485** |
| Citation Validity | 0,010 | 0,0036 | **0,3%** | 281 |

**Citation F1 là ngưỡng có vấn đề:** cứ 5 cấu hình *giống hệt P0* thì hơn 1 cái
bị loại oan. Chữa bằng cách tăng dữ liệu cần 3.485 câu, cả tập đã khử trùng lặp
chỉ có 1.152 — không chữa được bằng dữ liệu.

Rủi ro này không kích hoạt lần nào: Citation F1 của cả hai finalist đều dương
(+0,0366 và +0,0356), cách xa ngưỡng.

Cách xử lý ❓ — xem [`06_decisions.md`](06_decisions.md) D2 và D12.

### Hai phán quyết loại P2-depth3 đều thiếu công suất

| | Δ | CI95 | MDE₈₀ | Công suất thật | Hệ số thổi phồng | Hiệu ứng thật ước tính |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Faithfulness | −0,0200 | [−0,0372; −0,0048] | 0,0232 | **66%** | ×1,23 | ≈ −0,0162 |
| Citation Validity | −0,0142 | [−0,0282; −0,0034] | 0,0177 | **56%** | ×1,33 | ≈ −0,0107 |

Cả hai nằm trong vùng **có ý nghĩa nhưng thiếu công suất**. Khi một thiết kế
thiếu công suất mà vẫn ra kết quả có ý nghĩa, thường là vì mẫu rơi vào phía cực
đoan — các mẫu cho hiệu ứng nhỏ hơn thì không đạt ngưỡng nên không được nhìn
thấy. Hệ quả: **độ lớn quan sát được bị thổi phồng có hệ thống**.

Hệ số thổi phồng tính bằng mô phỏng 200.000 lần: giả sử hiệu ứng thật đúng bằng
cái đã quan sát, chỉ giữ lại những lần đạt ngưỡng có ý nghĩa.

Faithfulness thật của P2-depth3 có thể chỉ khoảng −0,016, tức **dưới** ngưỡng
0,02. Toàn bộ phán quyết quy về **khoảng một chục câu trong 281**.

Phân tích độ nhạy của luật ❓ — xem [`06_decisions.md`](06_decisions.md) D3.

### Nhóm `gold_not_in_top5` quá nhỏ để kết luận

| Nhóm | n | Δ (AC) | MDE₈₀ |
| :--- | ---: | ---: | ---: |
| Toàn tập | 281 | +0,1043 | 0,0336 |
| `gold_in_top5` | 269 | +0,1075 | 0,0354 |
| **`gold_not_in_top5`** | **12** | **+0,0325** | **0,0847** |

Khoảng cách quan sát nhỏ hơn **một nửa** thứ mà n = 12 phát hiện nổi. Nêu như
quan sát kèm n, không nêu như kết luận.

### Phase 2C: ba trong sáu guardrail dưới tầm phân giải 📌

Cùng cỡ mẫu (281 câu / 50 bài), cùng cách bootstrap, nhưng 2C có sáu guardrail
thay vì bốn. Baseline là C0-P2-D5, ứng viên là C3-P2-D3:

| Guardrail | Ngưỡng $T$ | Δ đo được | CI95 | SE | Báo động giả | Phán quyết |
| :--- | ---: | ---: | :--- | ---: | ---: | :--- |
| **Hit@5** | 0,010 | −0,0107 | [−0,0283; +0,0072] | 0,0090 | **13,3%** 🔴 | trượt |
| **Recall@5** | 0,010 | −0,0125 | [−0,0300; +0,0054] | 0,0088 | **12,8%** 🔴 | trượt |
| **Citation F1** | 0,010 | −0,0284 | [−0,0568; −0,0022] | 0,0140 | **23,8%** 🔴 | trượt |
| Faithfulness | 0,020 | −0,0128 | [−0,0321; +0,0053] | 0,0098 | 2,1% ✅ | qua |
| Citation Validity | 0,010 | −0,0071 | [−0,0184; +0,0000] | 0,0049 | 2,0% ✅ | qua |

Ba phán quyết loại C3 rơi vào ba tình huống **khác nhau**, và báo cáo phải phân
biệt cả ba:

1. **Citation F1** — trượt ngưỡng *và* CI95 không chứa 0. Vững cả hai đường.
2. **Hit@5, Recall@5** — trượt ngưỡng nhưng CI95 **chứa 0**. Chỉ là áp luật.
   Δ đo được (−0,0107 và −0,0125) còn nhỏ hơn MDE₈₀ của chính chúng (0,0252 và
   0,0247), nên với cỡ mẫu này không thể phân biệt "C3 truy xuất kém hơn thật"
   với "C3 giống hệt C0". Cả hai ngưỡng đều được đặt ở `0,01` mà không ai tính
   trước SE.
3. **Faithfulness, Citation Validity** — qua ngưỡng, và cũng không phân định được.
   Qua guardrail không phải là "tốt bằng".

**So với 2B:** 2B có 1/3 guardrail dưới tầm phân giải, 2C có 3/5. Nguyên nhân
giống nhau — ngưỡng chọn theo trực giác "khoảng 3 câu trên 281" chứ không theo
tính toán cỡ mẫu — nhưng 2C nặng hơn vì hai ngưỡng truy xuất mới cũng đặt ở `0,01`.

Điều **không** kéo theo: winner không đổi. Không cấu hình nào qua hết dù đọc luật
kiểu gì, vì Citation F1 trượt ở cả hai cách đọc.

### Phase 3: phán quyết duy nhất không sát biên 📌

Ba phase trước đều có ít nhất một phán quyết nằm sát ngưỡng. Phase 3 thì không:

| | Δ | CI95 | SE | Ngưỡng | $|\Delta|/T$ | $|\Delta|/SE$ |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Token F1, B1 − B0, development (n=61) | −0,1070 | [−0,1563; −0,0631] | 0,0243 | 0,02 | **5,4×** | **4,41** |
| Token F1, B1 − B0, final-test (n=26) | −0,1769 | [−0,2686; −0,0943] | 0,0436 | 0,02 | **8,8×** | **4,06** |

MDE₈₀ là 0,0681 (dev) và 0,1221 (final) — **cao hơn** ngưỡng 0,02 rất nhiều, tức
thiết kế này *không* phân giải nổi một khoảng cách cỡ ngưỡng. Nhưng khoảng cách
thật lớn gấp năm lần ngưỡng nên chuyện đó không thành vấn đề: cỡ mẫu chỉ cần đủ
để bắt cái có thật, không cần đủ để bắt mọi thứ.

Đây là cách nói đúng về công suất, và nên viết vào §2: **"n có đủ không?" không
có câu trả lời chung — nó phụ thuộc khoảng cách đang muốn bắt.** Cùng n = 61 mà
đủ thừa cho khoảng cách 0,107 và không đủ cho khoảng cách 0,02.

Bootstrap ở đây bốc lại **theo câu**, không gom cụm theo bài, vì bộ 200 case của
Phase 3 không tổ chức theo bài như Phase 2. Không đồng nhất với Phase 2, phải ghi
ở §7.

### Held-out: độ rộng của các số công bố

Không có baseline trên held-out nên không ghép cặp được; đây là khoảng tin cậy
của chính giá trị trung bình.

| Nhóm | n | AC | CI95 | nửa khoảng |
| :--- | ---: | ---: | :--- | ---: |
| Toàn tập | 284 | 0,7157 | [0,6799; 0,7490] | ±0,0345 |
| `gold_in_top5` | 249 | 0,7689 | [0,7391; 0,7978] | ±0,0293 |
| `gold_not_in_top5` | 35 | 0,3375 | [0,2235; 0,4591] | ±0,1178 |

Giá trị dev trên `gold_in_top5` là **0,7597** — **nằm trong** khoảng tin cậy của
held-out. Phát biểu đúng:

> Trên nhóm truy xuất đưa được bằng chứng ra, held-out (0,7689, CI95
> [0,7391; 0,7978]) **không phân biệt được** với development (0,7597) — không có
> dấu hiệu prompt bị overfit.

Chênh lệch +0,0092 nằm sâu trong sai số. Không viết "held-out cao hơn dev".
