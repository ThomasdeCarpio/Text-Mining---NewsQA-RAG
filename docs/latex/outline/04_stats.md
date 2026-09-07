# 04 — Giả thuyết và cỡ mẫu

Hai chỗ báo cáo hiện tại **không có**, mà đáng lẽ phải có:

1. Không chỗ nào **phát biểu giả thuyết**. Chỉ có "CI95 không chứa 0 → có ý nghĩa".
2. Không chỗ nào hỏi **281 câu / 50 bài có đủ để kiểm định không**.

File này viết ra cả hai. Số lấy từ `paired_significance.json` (mục `resolution`,
sinh bởi `scripts/phase2_paired_comparison.py`).

---

## 1. Giả thuyết, phát biểu tường minh

Mọi so sánh trong báo cáo có cùng một dạng. Với hai cấu hình A (mốc đối chứng)
và B, trên cùng bộ câu hỏi, gọi $d_i$ là hiệu điểm ở câu hỏi thứ $i$:

$$d_i = m(B, q_i) - m(A, q_i)$$

- $H_0:\ \mathbb{E}[d] = 0$ — hai cấu hình không khác nhau ở metric $m$.
- $H_1:\ \mathbb{E}[d] \neq 0$ — **hai phía**, vì với guardrail ta quan tâm cả
  chiều tụt lẫn chiều tăng.
- Mức ý nghĩa $\alpha = 0{,}05$.
- Phép kiểm: khoảng tin cậy percentile bootstrap 95% cho $\mathbb{E}[d]$. Bác bỏ
  $H_0$ khi khoảng đó không chứa 0. Không dùng p-value; khoảng tin cậy hai phía
  ở mức 95% tương đương phép kiểm hai phía ở $\alpha = 0{,}05$.

**Không phải mọi so sánh đều là kiểm định giả thuyết.** Phải phân biệt rõ ba loại,
vì báo cáo hiện tại đang trộn chúng:

| Loại | Ví dụ | Có $H_0$ không? |
| :--- | :--- | :--- |
| **Kiểm định** | Sparse hơn dense? P2 hơn P0? | Có, như trên |
| **Ngưỡng quyết định (guardrail)** | Faithfulness không được tụt quá 0,02 | **Không.** Đây là ngưỡng do nhóm tự đặt, đăng ký trước. Không có $H_0$, không có $\alpha$ |
| **Mô tả** | AC held-out = 0,7157 | Không. Chỉ là một số đo, kèm khoảng tin cậy để biết độ rộng |

Trộn ba loại này là lý do bản hiện tại đọc lủng củng: cùng một chữ "có ý nghĩa"
được dùng cho cả ba.

### Vấn đề so sánh bội

Phase 2B chạy 7 metric × nhiều cấu hình. Báo cáo **không** hiệu chỉnh
(Bonferroni, FDR). Lý do bào chữa được: metric chính được **đăng ký trước** đúng
một cái (Answer Correctness), các metric còn lại là guardrail (ngưỡng, không phải
kiểm định) hoặc mô tả. Nhưng phải nói ra, không lờ đi.

---

## 2. Cỡ mẫu này phân giải được tới đâu

Câu hỏi đúng không phải "281 có đủ lớn không?" mà **"281 câu / 50 bài phân biệt
được khoảng cách nhỏ tới mức nào?"**. Trả lời được bằng chính dữ liệu đã có.

- **SE** = độ lệch chuẩn của phân phối bootstrap.
- **MDE₈₀** = khoảng cách nhỏ nhất mà thiết kế này phát hiện được 80% số lần,
  ở $\alpha = 0{,}05$ hai phía: $2{,}802 \times SE$.
- **Design effect (DEFF)** = $(SE_\text{bốc theo bài} / SE_\text{bốc theo câu})^2$.
  Lớn hơn 1 nghĩa là gom cụm theo bài làm khoảng tin cậy rộng ra.
- **n hiệu lực** = $n / DEFF$.

### P2-depth5 so với P0, 281 câu / 50 bài

| Metric | Δ | SE | DEFF | n hiệu lực | MDE₈₀ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Answer Correctness | +0,1043 | 0,0103 | 0,72 | 388 | **0,0290** |
| Exact Match | +0,0925 | 0,0180 | 1,08 | 260 | 0,0504 |
| Token F1 | +0,1542 | 0,0122 | 0,81 | 346 | 0,0343 |
| Faithfulness | +0,0040 | 0,0080 | 1,27 | 221 | 0,0225 |
| Citation F1 | +0,0366 | 0,0126 | 1,20 | 235 | 0,0352 |
| Citation Validity | −0,0036 | 0,0036 | 1,04 | 270 | 0,0100 |
| Answer Relevancy | −0,0858 | 0,0183 | 1,36 | 207 | 0,0512 |

**Đọc bảng này:** khoảng cách chính (+0,1043 Answer Correctness) lớn gấp **3,6
lần** MDE₈₀ của nó. Kết luận đó vững. Ngược lại, một khoảng cách cỡ 0,02 ở
Answer Relevancy thì thiết kế này **không** phân giải được.

### MDE₈₀ đến từ đâu

Ba bước, không có gì huyền bí:

1. Ta có 281 hiệu số $d_i$ (điểm P2-depth5 trừ điểm P0 ở từng câu).
2. **Bootstrap:** bốc lại 50 bài (có hoàn lại) 2.000 lần, mỗi lần tính trung bình
   của các $d_i$ bốc được → có 2.000 giá trị trung bình.
3. **SE** = độ lệch chuẩn của 2.000 giá trị đó. Nó nói: nếu lặp lại cả thí nghiệm
   trên một mẫu khác, trung bình đo được sẽ xê dịch cỡ này.

Rồi công thức công suất chuẩn, kiểm định hai phía, $\alpha = 0{,}05$, công suất
$1-\beta = 0{,}80$:

$$MDE_{80} = (z_{0{,}975} + z_{0{,}80}) \times SE = (1{,}960 + 0{,}842) \times SE = 2{,}802 \times SE$$

Nghĩa: nếu khoảng cách **thật** đúng bằng $MDE_{80}$ thì phép kiểm bắt được nó
80% số lần. Nhỏ hơn thì tỉ lệ trượt cao hơn 20%.

### ⚠️ Nhưng so MDE₈₀ với ngưỡng guardrail là phép so **thô**

Bản trước của file này kết luận "hai trong ba ngưỡng nằm dưới độ phân giải" bằng
cách so thẳng $MDE_{80}$ với ngưỡng $T$. **So như vậy không đúng câu hỏi.** Ngưỡng
guardrail không phải một phép kiểm định — nó là một lằn ranh quyết định. Câu hỏi
đúng là hai tỉ lệ sai:

- **Báo động giả** — cấu hình thật ra *không* kém hơn P0, nhưng đo ra tụt quá
  ngưỡng nên bị loại oan: $P(\hat{d} \le -T \mid d_\text{thật} = 0) = \Phi(-T/SE)$
- **Bỏ sót** — cấu hình thật sự tụt đúng bằng ngưỡng, nhưng đo ra chưa tới nên
  lọt: luôn bằng **50%**, với mọi ngưỡng. Đây là tính chất cố hữu của mọi lằn
  ranh, không phải khuyết điểm riêng của bộ ngưỡng này.

| Guardrail | Ngưỡng $T$ | SE | Báo động giả | Cần bao nhiêu câu để $MDE_{80}=T$ |
| :--- | ---: | ---: | ---: | ---: |
| Faithfulness | 0,020 | 0,0080 | **0,6%** ✅ | 355 |
| **Citation F1** | 0,010 | 0,0126 | **21,3%** 🔴 | **3.485** |
| Citation Validity | 0,010 | 0,0036 | **0,3%** ✅ | 281 ✅ |

**Chỉ Citation F1 hỏng, không phải hai cái.** Với ngưỡng 0,01 mà SE là 0,0126,
cứ 5 cấu hình *giống hệt P0* thì có hơn 1 cái bị loại oan. Muốn chữa bằng cách
tăng cỡ mẫu thì cần **3.485 câu** — trong khi cả tập đã khử trùng lặp chỉ có
1.152. Không chữa được bằng dữ liệu.

Faithfulness cần 355 câu (đang có 281, khá gần) và Citation Validity thì 281 câu
là vừa đủ. Hai cái này ổn.

**May là chuyện đó chưa xảy ra:** Citation F1 của P2-depth5 là +0,0366 và của
P2-depth3 là +0,0356 — cả hai đều dương và cách xa ngưỡng, nên rủi ro 21,3% không
kích hoạt lần nào. Nhưng đó là **may, không phải do thiết kế**.

### 🔴 Và đây mới là phát hiện lớn: đổi luật là đổi luôn winner

Có một luật chặt hơn và rất dễ bào chữa: *chỉ loại một cấu hình khi **khoảng tin
cậy** của nó nằm hẳn dưới ngưỡng*, thay vì chỉ nhìn điểm trung bình. Áp luật đó
vào chính dữ liệu đã có:

| P2-depth3 | Δ | CI95 | Luật đã đăng ký | Luật chặt hơn |
| :--- | ---: | :--- | :-: | :-: |
| Faithfulness | −0,0200 | [−0,0372; **−0,0048**] | 🔴 TRƯỢT | ✅ QUA |
| Citation F1 | +0,0356 | [−0,0011; +0,0730] | ✅ QUA | ✅ QUA |
| Citation Validity | −0,0142 | [−0,0282; **−0,0034**] | 🔴 TRƯỢT | ✅ QUA |
| **Kết luận** | | | **BỊ LOẠI** | **HỢP LỆ** |

Lý do: cận trên của cả hai khoảng (−0,0048 và −0,0034) đều **cao hơn** ngưỡng
(−0,02 và −0,01). Tức là dữ liệu chứng minh được "có tụt", nhưng **không** chứng
minh được "tụt quá ngưỡng".

Mà P2-depth3 có Answer Correctness **0,7711** so với **0,7350** của P2-depth5.

> **Đổi sang luật chặt hơn thì winner lật từ P2-depth5 sang P2-depth3.**

Đây chính là lý do **không được đổi luật sau khi đã thấy kết quả**. Luật chặt hơn
nghe có vẻ nghiêm ngặt hơn, nhưng chọn nó lúc này đồng nghĩa với chọn cái luật
cho ra con số đẹp hơn — đúng hành vi mà việc đăng ký trước sinh ra để chặn.

**Cách viết vào báo cáo:** giữ P2-depth5 là winner vì luật đã đăng ký được tuân
thủ, **và nói thẳng** rằng một luật thay thế hợp lý sẽ chọn khác, kèm lý do không
áp dụng nó. Điều này thay cho câu tự khen "guardrail đã làm đúng việc của nó" —
nó cho thấy chuyện thật sự xảy ra, và mạnh hơn nhiều.

### Vậy chữa thế nào?

| # | Cách | Chi phí | Áp được cho |
| ---: | :--- | :--- | :--- |
| 1 | **Luôn báo cáo CI bên cạnh phán quyết guardrail** — không đổi luật, chỉ hiện thêm thông tin để người đọc tự thấy phán quyết chắc tới đâu | 0 | Ngay, mọi phase |
| 2 | **Đặt ngưỡng từ SE đo được**: chọn $T \approx 2{,}8 \times SE$ để hai tỉ lệ sai cân bằng | 0, nhưng phải làm **trước** khi chạy | Phase 2C, Phase 3 |
| 3 | **Giảm nhiễu giám khảo** — chấm lặp $k$ lần rồi lấy trung bình, phần nhiễu do giám khảo giảm theo $1/\sqrt{k}$. Bài lặp 25 câu (đang treo ở việc còn lại) sẽ cho biết bao nhiêu phần của SE là do giám khảo | tiền API | Phase 2C trở đi |
| 4 | Tăng cỡ mẫu | Citation F1 cần 3.485 câu / tập chỉ có 1.152 | ❌ không khả thi |

Cách 1 và 2 là thứ nên làm. Cách 2 đặc biệt quan trọng cho **Phase 2C**, vì nó
chưa chạy — vẫn kịp đặt ngưỡng cho tử tế thay vì đặt bằng cảm tính rồi phát hiện
sau.

### 🔴 Nhóm `gold_not_in_top5` quá nhỏ để kết luận bất cứ điều gì

| Nhóm | n | Δ (AC) | MDE₈₀ |
| :--- | ---: | ---: | ---: |
| Toàn tập | 281 | +0,1043 | 0,0336 |
| `gold_in_top5` | 269 | +0,1075 | 0,0354 |
| **`gold_not_in_top5`** | **12** | **+0,0325** | **0,0847** |

Khoảng cách quan sát được (+0,0325) **nhỏ hơn một nửa** thứ mà n = 12 phát hiện
nổi. Phát biểu "đổi prompt gần như không cứu được câu truy xuất trượt" hiện đang
**không kiểm định được** — dữ liệu không phân biệt nổi +0,0325 với 0, mà cũng
không phân biệt nổi với +0,08.

Bên held-out nhóm này lớn hơn (n = 35) nhưng vẫn rộng: AC 0,3375, CI95
[0,2235; 0,4591] — nửa khoảng ±0,118.

**Cách phát biểu đúng:** nêu con số như một **quan sát**, kèm n và khoảng tin cậy,
rồi nói thẳng là chưa kiểm định được. Không được viết như một kết luận.

### Held-out: độ rộng của các số công bố

Không có baseline trên held-out nên không ghép cặp được; đây là khoảng tin cậy
của chính giá trị trung bình.

| Nhóm | n | AC | CI95 | nửa khoảng |
| :--- | ---: | ---: | :--- | ---: |
| Toàn tập | 284 | 0,7157 | [0,6799; 0,7490] | ±0,0345 |
| `gold_in_top5` | 249 | 0,7689 | [0,7391; 0,7978] | ±0,0293 |
| `gold_not_in_top5` | 35 | 0,3375 | [0,2235; 0,4591] | ±0,1178 |

### 🔴 Hệ quả cho phát biểu quan trọng nhất của báo cáo

Giá trị dev trên `gold_in_top5` là **0,7597**. Nó **nằm gọn trong** khoảng tin cậy
của held-out [0,7391; 0,7978].

Nên phát biểu đúng là:

> Trên nhóm truy xuất đưa được bằng chứng ra, held-out (0,7689, CI95
> [0,7391; 0,7978]) **không phân biệt được** với development (0,7597) — không có
> dấu hiệu prompt bị overfit.

**Không** được viết "held-out cao hơn dev". Chênh lệch +0,0092 nằm sâu trong sai
số. Bản hiện tại đang viết quá mạnh ở cả báo cáo lẫn slide.

---

## 3. Chỗ tôi đã nói quá trong `01_terms.md`

Tôi viết rằng bootstrap theo câu của Phase 1 là "lỏng hơn" và gom cụm sẽ làm
khoảng tin cậy rộng ra. **Đo rồi thì không hẳn:** DEFF chạy từ **0,72 đến 1,36**
tùy metric, trung vị quanh 1,08.

Lý do: đây là **hiệu theo cặp**. Độ khó của một bài báo ảnh hưởng lên cả A lẫn B
như nhau nên nó **triệt tiêu trong phép trừ**. Tương quan trong cụm vì thế nhỏ,
và gom cụm gần như không đổi gì.

Phát biểu đúng: gom cụm theo bài là lựa chọn **thận trọng và đúng nguyên tắc**,
nhưng với hiệu theo cặp thì tác động của nó nhỏ (DEFF ≈ 1). Việc Phase 1 bốc theo
câu **không** làm kết luận của Phase 1 sai. Vẫn nên ghi là điểm không đồng nhất
giữa các phase, nhưng đừng dựng nó thành một khiếm khuyết lớn.

---

## 4. Việc phải làm

1. Thêm hai tiểu mục vào §2 báo cáo: **giả thuyết** (mục 1 ở trên) và **cỡ mẫu,
   độ phân giải** (mục 2).
2. Sửa `01_terms.md`: hạ giọng phần gom cụm; thêm mục **MDE**, **design effect**,
   **n hiệu lực**.
3. Sửa `03_claims.md`: hạ 2.10 và 3.4, thêm phát hiện guardrail-dưới-độ-phân-giải.
4. Sửa mọi chỗ đang viết "held-out cao hơn dev" thành "không phân biệt được".
5. §7 thêm hai giới hạn: (a) ngưỡng guardrail đặt không kèm tính toán cỡ mẫu, và
   Citation F1 có tỉ lệ báo động giả 21,3%; (b) một luật thay thế hợp lý sẽ chọn
   winner khác.
6. §4 thay câu tự khen "guardrail đã làm đúng việc" bằng bảng so hai luật ở trên.
7. **Phase 2C: tính SE rồi mới đặt ngưỡng**, ngay khi có kết quả C1 đầu tiên.
8. Chạy bài lặp 25 câu để biết bao nhiêu phần của SE là nhiễu giám khảo.
