# 1. Tổng quan: Dự án này làm gì?

## Vấn đề đang giải quyết

Nếu bạn hỏi ChatGPT "Tỷ lệ thất nghiệp tháng trước là bao nhiêu?", nó có thể
**bịa** ra một con số nghe hợp lý, vì nó chỉ đang "nhớ" (hoặc đoán) chứ không
thực sự tra cứu nguồn tin nào. Đây gọi là **hallucination** (ảo giác) — vấn đề
lớn nhất khi dùng LLM cho việc trả lời tin tức thời sự.

Dự án này xây một hệ thống **RAG (Retrieval-Augmented Generation)**: thay vì
để AI trả lời từ trí nhớ, hệ thống sẽ:

1. **Tìm** (Retrieval) những đoạn tin tức thật liên quan đến câu hỏi, trong
   một kho bài báo đã có sẵn.
2. **Đưa** những đoạn đó cho AI đọc.
3. **Yêu cầu** AI trả lời **dựa trên** những đoạn đó, kèm trích dẫn nguồn
   (citation) — để người dùng có thể tự kiểm chứng.

Nói cách khác: AI không được "nhớ", chỉ được "tra cứu rồi tóm tắt".

## Vì sao gọi là "Agentic"?

README gọi đây là hệ thống **Agentic RAG** — nghĩa là thay vì luôn làm đúng
1 bước "tìm rồi trả lời", AI đóng vai trò như một **agent (tác nhân) biết tự
quyết định**: nó có thể tự chọn gọi công cụ tìm kiếm nhiều lần, tìm theo nhiều
cách khác nhau, rồi mới tổng hợp câu trả lời — giống một nhà báo tự đi tra
nhiều nguồn trước khi viết bài, thay vì chỉ đọc 1 tài liệu rồi trả lời ngay.

*(Tính năng "tự quyết định nhiều bước" này **chưa được code** — xem
[04-con-thieu-gi-va-tai-sao.md](04-con-thieu-gi-va-tai-sao.md). Hiện tại phần
tìm kiếm mới chỉ làm được "tìm 1 lần".)*

## Hai loại người dùng

Dự án được thiết kế cho 2 vai trò khác nhau (xem `docs/ui.md` để biết chi tiết
yêu cầu UI):

- **Standard User** (người đọc tin tức): đăng nhập, gõ câu hỏi trong khung
  chat, xem câu trả lời + xem được "nguồn" (đoạn văn bản gốc + link bài báo)
  mà AI đã dùng để trả lời.
- **Admin** (người đánh giá/dev): ngoài chat, còn có thêm 2 trang riêng:
  - **Evaluation Desk**: xem điểm số đánh giá chất lượng hệ thống (ví dụ:
    AI trả lời đúng bao nhiêu %, tìm đúng tài liệu bao nhiêu %).
  - **Retrieval Playground**: gõ 1 câu hỏi, xem trực tiếp hệ thống tìm ra
    những đoạn văn bản nào, điểm số ra sao — dùng để debug/test phần tìm
    kiếm mà không cần AI trả lời, giúp biết được lỗi (nếu có) nằm ở khâu
    tìm kiếm hay khâu AI viết câu trả lời.

## Vì sao có phần "Evaluation" (đánh giá)?

Đây là đồ án học thuật — **trọng tâm không chỉ là "làm ra 1 chatbot chạy
được"**, mà là **đo lường được** chatbot đó tốt tới đâu, bằng số liệu cụ thể
(dùng framework tên **Ragas** — thư viện chuyên đo chất lượng hệ thống RAG:
ví dụ đo xem câu trả lời có "trung thực" với tài liệu gốc không, có tìm đúng
tài liệu không...). Vì vậy hệ thống có nguyên 1 trang Admin riêng chỉ để xem
các con số đánh giá này.

## Trạng thái hiện tại (tóm tắt 1 dòng)

**Phần "tìm tài liệu" (retrieval) đã chạy thật.** Phần "AI đọc tài liệu rồi
viết câu trả lời" (agent + LLM) và phần "đo điểm chất lượng" (evaluation)
**chưa được xây** — hiện tại 2 phần đó chỉ đang trả về dữ liệu giả (mock) để
giao diện có cái để hiển thị trong lúc chờ xây phần lõi. Chi tiết ở
[02-cach-he-thong-hoat-dong.md](02-cach-he-thong-hoat-dong.md).
