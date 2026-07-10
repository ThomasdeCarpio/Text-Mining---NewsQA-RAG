# 4. Còn thiếu gì, và vì sao nó quan trọng?

File này giải thích **hậu quả** nếu từng phần không được xây, để bạn hiểu vì
sao nó nằm trong kế hoạch chứ không phải chỉ để liệt kê việc cần làm. Danh
sách việc cần làm chi tiết theo từng file nằm ở mục **"Roadmap / Remaining
Work"** trong `README.md` — file này là phần **"vì sao"** đi kèm.

Thứ tự dưới đây là thứ tự nên làm — mỗi mục xây trên nền mục trước.

## 1. AI thật sự trả lời câu hỏi (thay vì trả lời giả)

**Thiếu:** `src/llm.py` (kết nối tới GPT), `src/agents/rag_agent.py` (logic
"tìm rồi trả lời"), và việc nối chúng vào `chat_service.py`.

**Vì sao quan trọng:** Đây là phần **lõi nhất** của cả dự án. Hiện tại trang
"News Chat" chỉ là vỏ giao diện — trả lời y hệt nhau bất kể bạn hỏi gì, vì
chưa có gì thật sự đọc câu hỏi của bạn cả. Nếu không làm phần này, dự án
**không có gì để đánh giá** (không có ChatGPT thật để so sánh đúng/sai) —
mọi thứ khác (evaluation, hybrid search...) đều để cải thiện chất lượng của
bước này, nên phải có nó trước.

**Vì sao chưa làm:** Phần **tìm kiếm** (retrieval) — bước bắt buộc phải có
trước khi AI có gì để đọc — mới vừa xong (Retrieval Playground). Bước tiếp
theo tự nhiên là nối tìm kiếm đó vào 1 lệnh gọi AI.

## 2. Tìm kiếm chính xác hơn (Hybrid search + Reranker)

**Thiếu:** `src/indexing/bm25_index.py`, `src/retrieval/hybrid.py`,
`src/retrieval/reranker.py`.

**Vì sao quan trọng:** Cách tìm kiếm hiện tại (`dense.py`) tìm theo **ý
nghĩa** — rất giỏi hiểu "ý" câu hỏi, nhưng có thể bỏ sót khi câu hỏi cần
**khớp chính xác** một con số, tên riêng, hoặc từ hiếm (VD: tìm đúng "3.9%"
hay tên 1 người cụ thể) — vì các con số/tên riêng không mang nhiều "ý
nghĩa" để so vector. Cách tìm theo **từ khoá** (BM25) lại giỏi đúng việc
đó, nhưng dở việc hiểu ý. **Kết hợp cả 2 (hybrid)** giúp hệ thống không bỏ
sót cả 2 kiểu câu hỏi. **Reranker** là 1 bước lọc thêm: sau khi tìm ra ~10-20
đoạn khả nghi, dùng 1 model chuyên biệt để chấm điểm lại và chỉ giữ vài đoạn
tốt nhất — giảm khả năng AI bị "nhiễu" bởi đoạn không liên quan.

**Vì sao chưa làm:** Cần có "dense search" chạy ổn trước để làm nền so
sánh — không thể biết "hybrid tốt hơn dense bao nhiêu %" nếu chưa đo được
dense trước.

## 3. AI biết tự tìm nhiều lần, so sánh nhiều nguồn (Agent nhiều bước)

**Thiếu:** `src/agents/orchestrator.py` (dùng LangGraph).

**Vì sao quan trọng:** Theo `docs/ui.md`, dự án còn muốn trả lời được các
câu như *"Tóm tắt tất cả bài viết mới nhất về động đất Haiti"* hay *"So sánh
phản ứng của WSJ và AP về báo cáo việc làm mới"*. Loại câu hỏi này **không
thể trả lời bằng 1 lần tìm kiếm** — cần tìm nhiều đợt, từ nhiều nguồn, rồi
tổng hợp lại. Đây chính là phần khiến hệ thống xứng đáng gọi là "Agentic"
(tự quyết định các bước) thay vì chỉ 1 quy trình cố định. Không có phần
này, hệ thống chỉ trả lời được câu hỏi "tìm 1 sự kiện trong 1 bài báo"
(single-source fact-finding), chưa làm được phần "so sánh/tổng hợp nhiều
nguồn" (multi-source synthesis).

**Vì sao chưa làm:** Cố tình để sau — README ghi rõ nên làm chắc phần
single-source (mục 1) trước, vì phần multi-source phức tạp hơn và còn cần
sửa lại bộ dữ liệu test (Multi-News dataset) để đánh giá cho đúng.

## 4. Đo điểm chất lượng thật (Evaluation)

**Thiếu:** `src/evaluation/metrics.py` (dùng Ragas), `scripts/run_benchmark.py`.

**Vì sao quan trọng:** Đây là **mục tiêu học thuật chính** của đồ án (theo
Project Overview) — không chỉ "có chatbot chạy được" mà phải **đo được** nó
tốt tới đâu bằng số liệu (VD: bao nhiêu % câu trả lời trung thực với tài
liệu gốc, bao nhiêu % lần tìm đúng tài liệu). Không có phần này thì trang
"Evaluation Desk" — vốn là 1 nửa giao diện dành cho vai trò Admin — mãi mãi
chỉ hiển thị số bịa, không nói lên được điều gì thật về hệ thống. Đây cũng
là dữ liệu cần để **chứng minh** hybrid search / reranker (mục 2) thật sự
tốt hơn dense search, chứ không chỉ đoán.

**Vì sao chưa làm:** Cần có câu trả lời AI thật (mục 1) để có gì mà chấm
điểm — chấm điểm 1 câu trả lời giả thì vô nghĩa.

## 5. Hoàn thiện để dùng "nghiêm túc" hơn (không bắt buộc cho đồ án)

Nhóm việc này **không ảnh hưởng đến việc chatbot có trả lời đúng hay không**
— chỉ quan trọng nếu hệ thống được nhiều người dùng thật cùng lúc, chạy lâu
dài, hoặc public ra ngoài:

- **Gộp cấu hình về 1 chỗ** (`configs/setting.py`): hiện `scripts/ingest.py`
  và `src/services/retrieval_service.py` **mỗi nơi tự khai 1 bản config
  giống nhau** (model embedding, đường dẫn...). Nếu sửa 1 chỗ mà quên sửa
  chỗ kia, 2 phần sẽ dùng model khác nhau mà không ai biết — lỗi rất khó
  phát hiện. Gộp về 1 chỗ đọc `config.yaml` sẽ triệt tiêu rủi ro này.
- **Tài khoản người dùng thật** (thay vì 2 tài khoản viết cứng trong code):
  cần nếu có nhiều người dùng thật, không cần cho demo/nộp bài.
- **Lưu lịch sử chat bền vững** (Redis/DB thay vì RAM): hiện lịch sử chat
  **mất hết mỗi khi tắt/khởi động lại server** — chấp nhận được cho demo,
  không ổn nếu chạy thật.
- **Có thể chọn nhiều đoạn hội thoại** (hiện chỉ có 1 đoạn hội thoại đang
  chạy mỗi lần đăng nhập).
- **Xử lý lỗi/loading tốt hơn ở giao diện** (hiện nếu API lỗi thì chỉ báo
  lỗi đơn giản, chưa có thử lại tự động...).
- **Viết test tự động**: hiện chưa có, nên mỗi lần sửa code phải tự tay
  chạy lại kiểm tra bằng tay.

---

**Tóm lại theo thứ tự ưu tiên nên làm:** (1) AI trả lời thật → (2) tìm kiếm
chính xác hơn → (3) agent nhiều bước → (4) đo điểm thật → (5) phần còn lại,
làm khi có thời gian, không gấp.
