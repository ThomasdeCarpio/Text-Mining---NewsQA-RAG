# 00 — Người đọc, độ dài, và luật viết

Chốt trước khi viết một dòng nào của báo cáo.

## Người đọc

**Thầy: biết IR/NLP, không biết dự án này.**

| Không cần giải thích | Bắt buộc giải thích |
| :--- | :--- |
| nDCG, MRR, Hit@k, precision/recall | `gold chunk`, `evidence span`, `distractor` trong ngữ cảnh bộ dữ liệu này |
| BM25, embedding, cross-encoder | `original` vs `resolved`, và vì sao có hai bản |
| khoảng tin cậy | *vì sao* gom cụm theo bài báo, giải thích ngắn gọn  boostrap trong tình huống này là gì |
| RAG là gì | RAGAS đo Faithfulness / Answer Correctness **bằng cách nào** -> dịch các từ này ra tiếng việt |
| prompt engineering là gì | `P0–P3`, `C0–C3`, `context_depth`, `guardrail` |

Nguyên tắc: thuật ngữ chung của ngành thì dùng thẳng; thuật ngữ **do dự án đặt ra
hoặc do dự án gán nghĩa riêng** thì phải định nghĩa. Ranh giới đó là §0 thuật ngữ.

## Độ dài: 10–15 trang

Ngân sách ở [`02_outline.md`](02_outline.md). Bản LaTeX hiện tại 10 trang nhưng
là 10 trang **nén** — cắt hết phần dẫn dắt để nhét bảng vào. Bản mới dài bằng
nhưng thoáng hơn, vì bỏ bảng phụ và thêm chỗ cho lập luận.

## Luật viết

### 1. Định nghĩa trước khi dùng

Không dùng một thuật ngữ ở §3 rồi định nghĩa ở §5. §0 định nghĩa trước, và mỗi
thuật ngữ được nhắc lại ngắn ở chỗ nó xuất hiện lần đầu (đúng cách EDA §0 làm).

### 2. Mỗi con số đi kèm bốn thứ

> giá trị · đo bằng metric nào · trên tập nào, n bằng bao nhiêu · artifact nào sinh ra nó

"AC 0,7157" một mình là vô nghĩa. "Answer Correctness 0,7157 (RAGAS, 284 câu
held-out, `heldout_final_summary.json`)" thì kiểm chứng được.

### 3. Phát biểu đúng phạm vi đã đo

Một biên độ đo trên **truy xuất** không được bê sang một metric của **tầng sinh**.
Một tỉ lệ đếm trên **30 câu điểm thấp nhất** không được phát biểu như tỉ lệ của
toàn tập. Xem [`03_claims.md`](03_claims.md) — đã có hai lỗi loại này trong bản
hiện tại.

### 4. Không dùng ẩn dụ ở chỗ đáng lẽ phải có số đo

Đây là lỗi nhiều nhất của bản hiện tại. Ẩn dụ nghe như một kết luận nhưng không
kiểm chứng được, và người đọc không biết nó dựa trên cái gì.

| Đang viết | Vấn đề | Viết lại thành |
| :--- | :--- | :--- |
| "Hệ thống không bịa. Nó chỉ nói dài." | Hai mệnh đề, không mệnh đề nào có số | "Faithfulness 0,9761 nhưng Exact Match 0,0000: câu trả lời bám bằng chứng nhưng không câu nào trùng khớp span đáp án." |
| "Prompt đúng dạng là **đòn bẩy lớn nhất** ở tầng sinh" | "lớn nhất" so với cái gì? Chưa đo | "Trong bốn prompt đã thử, P2 là prompt duy nhất vượt guardrail và tăng AC (+0,1043)." |
| "Toàn bộ khoảng cách dev → held-out là **câu chuyện của truy xuất**" | Ẩn dụ thay cho số | "Trên nhóm `gold_in_top5`, held-out 0,7689 ≥ dev 0,7597; chênh lệch toàn tập đến từ việc nhóm `gold_not_in_top5` tăng từ 12/281 lên 35/284." |
| "**Trần** của tầng sinh nằm ở tầng truy xuất" | "Trần" chưa được định nghĩa | "Với 35/284 câu không có gold trong context, AC toàn tập bị chặn trên bởi tỉ lệ đó." |
| "P2 khớp với **hình dạng của nhãn**" | Ẩn dụ | "Đáp án chuẩn NewsQA có trung vị *n* token; P2 yêu cầu một câu duy nhất." *(cần đo n)* |
| "EDA **không phải bước trang trí**" | Tự khen, không thông tin | Bỏ. Để phần EDA tự chứng minh bằng việc nó dự báo đúng. |
| "Guardrail **đã làm đúng việc của nó**" | Tự khen | "P2-depth3 có AC cao hơn 0,0361 nhưng trượt hai guardrail nên bị loại." |
| "**Thứ tự thời gian là bằng chứng**" | Khẩu hiệu | "Bản ghi quyết định ký 13:31:30Z, run bắt đầu 14:23:24Z; băm của bản ghi được nhúng trong `heldout_access.json`." |
| "Báo cáo kết quả null **đúng như nó là**, thay vì tô thành chiến thắng" | Tự khen về đạo đức nghiên cứu | Bỏ. Chỉ cần nói khoảng cách 0,0469 dưới ngưỡng và ba CI chồng lấn. |

### 5. Đừng đặt tên mới cho một quy trình

Nếu một thứ phải giải thích mất một đoạn thì nó là **một ý**, không phải một
thuật ngữ. Ý thì kể ra, đừng đúc thành danh từ ghép rồi bắt người đọc nhớ.

Dấu hiệu nhận biết: cụm từ đó **không có trong code, không có trong test plan**,
chỉ có trong văn mình vừa viết.

| Chữ tự chế | Vấn đề | Viết lại |
| :--- | :--- | :--- |
| "**vết truy xuất đóng băng**" | Không có ở đâu trong dự án ngoài văn tôi viết; "vết" không gợi được gì; giải thích lại nén y hệt | "Bước tìm đoạn văn chỉ chạy **một lần**, kết quả được lưu lại và cả 12 cấu hình dùng chung, nên mọi cấu hình đọc đúng cùng 5 đoạn." |
| "**luật phán quyết**" | Ngôn ngữ tòa án | "hai điều kiện để gọi là thắng" |
| "**hiệu theo cặp**" | Dịch sát nhưng không gợi hình | "so sánh theo từng câu hỏi" |
| "**cấu hình khóa**" | "khóa" là động từ bị danh từ hóa | "cấu hình đã chốt" |

Và đừng thả thuật ngữ chưa giải thích *vào trong phần giải thích* — bản đầu định
nghĩa "cấu hình khóa" bằng cụm "được băm SHA-256", tức là giải thích một chữ lạ
bằng một chữ lạ khác.

### 6. Bỏ thiết bị trình chiếu ra khỏi báo cáo

`\keypoint{}` (hộp tô nền) và `\takeaway{}` (dải kết luận cuối slide) là ngôn ngữ
của slide. Trong báo cáo, một đoạn văn có câu chủ đề làm đúng việc đó.

**Giữ `\takeaway{}` trong slide** — ở đó nó đúng chức năng. Đây là luật cho báo
cáo, không phải cho slide.

### 7. Tự khen là số 0 thông tin

Bỏ mọi câu nói về *chất lượng của phương pháp luận của chính mình* ("nghiêm
ngặt", "trung thực", "đúng chuẩn", "đáng tin"). Nếu quy trình chặt thì mô tả quy
trình, người đọc tự kết luận. Ngoại lệ duy nhất: nêu một **giới hạn**, vì cái đó
người đọc không tự thấy được.

### 8. Kết quả âm và kết quả null viết cùng độ dài với kết quả dương

P1 thất bại, chunk size null, hybrid không chứng minh được — ba cái này là kết
quả, không phải phần phụ lục xin lỗi.

## Danh sách kiểm trước khi nộp

- [ ] Mọi thuật ngữ ở §0 đều được dùng; mọi thuật ngữ được dùng đều có ở §0
- [ ] Mọi con số truy được về một artifact đã commit
- [ ] Không con số nào bị phát biểu ngoài tập/metric đã đo nó
- [ ] Không còn câu nào trong bảng "Đang viết" ở trên
- [ ] Mọi phát biểu trong [`03_claims.md`](03_claims.md) đúng mức đã cho phép
- [ ] Con số trong `.tex` đều là macro `\Num...`, không gõ tay
