# 01 — Thuật ngữ (dự thảo §0 của báo cáo)

Nối tiếp quy ước đã có ở [`docs/eda/eda_report.md` §0](../../eda/eda_report.md).
Chỉ liệt kê thuật ngữ **do dự án đặt ra hoặc gán nghĩa riêng** — không định nghĩa
lại nDCG, BM25, embedding (xem [`00_scope_and_style.md`](00_scope_and_style.md)).
Bootstrap thì **có** giải thích ngắn, vì báo cáo dùng nó ở mọi so sánh.

**Cách viết mỗi mục:** một câu tả việc thực sự xảy ra, hình dung được; rồi một
câu vì sao nó quan trọng. Không đặt tên mới cho một quy trình. Không thả thuật
ngữ chưa giải thích vào phần giải thích.

Mục ✅ đã có ở EDA §0, chỉ cần nhắc lại một dòng. Mục 🆕 là mới.

---

## A. Dữ liệu và nhãn

| Thuật ngữ | Nghĩa trong báo cáo này | |
| :--- | :--- | :-: |
| **Corpus** | 11.064 bài báo CNN mà hệ thống được phép tìm. | ✅ |
| **Evaluation article** | 200 bài thực sự chứa đáp án của ít nhất một câu hỏi. | ✅ |
| **Distractor** | 10.864 bài **không** chứa đáp án nào. Có mặt để bài toán tìm kiếm giống thực tế — hệ thống thật phải lục trong một đống lớn, không phải trong 200 bài đã chọn sẵn. | ✅ |
| **Chunk** | Bài báo được cắt nhỏ trước khi đem đi tìm. Một chunk là một mẩu, ở đây khoảng 512 token. Hệ thống trả về *chunk*, không trả về cả bài. | ✅ |
| **Gold chunk** | Chunk chứa đáp án đúng. Dùng để chấm một lần tìm kiếm là trúng hay trượt. | ✅ |
| **Evidence span** | Vị trí chính xác của đáp án trong bài, ghi bằng số thứ tự ký tự — ví dụ ký tự 770 đến 791. Vì thế không được sửa text bài báo tùy tiện: sửa một chữ là mọi vị trí phía sau lệch đi. | ✅ |
| **Ground truth / đáp án chuẩn** | Text đáp án chính thức của một câu hỏi. Ở bộ này thường rất ngắn — một cái tên, một mốc thời gian, một con số. | ✅ |
| **`original` / `resolved`** | Hai bản của cùng bộ câu hỏi. `original` là câu như NewsQA viết ra; nhiều câu trong đó viết như đang nói tiếp một cuộc hội thoại (*"Có bao nhiêu người chết?"*). `resolved` là chính những câu đó sau khi nhóm viết lại cho đủ ngữ cảnh (*"Có bao nhiêu người chết trong vụ đánh bom Kandahar?"*). 🆕 Báo cáo lấy số trên `resolved` vì hệ thống thật luôn nhận câu hỏi đầy đủ; `original` chỉ dùng làm cận dưới. | 🆕 |
| **Câu hỏi standalone** | Câu hiểu được khi đứng một mình. *"Ai thắng bầu cử Gabon?"* là standalone; *"Ai thắng?"* thì không. | ✅ |

## B. Cấu trúc thực nghiệm

| Thuật ngữ | Nghĩa trong báo cáo này | |
| :--- | :--- | :-: |
| **Phase 1 / 2 / 2C / 3** | Bốn giai đoạn, mỗi giai đoạn chỉ đổi một thứ. 1 = chọn cách tìm đoạn văn. 2 = chọn cách ra lệnh cho LLM. 2C = chọn cách cắt bài thành đoạn. 3 = dạy hệ thống nói "không tìm thấy" khi bằng chứng không đủ. | 🆕 |
| **Development (281 câu / 50 bài)** | Tập dùng để thử các cấu hình và chọn ra cái tốt nhất. Đã bị chạm rất nhiều lần. | 🆕 |
| **Held-out (284 câu / 50 bài khác)** | Tập giữ kín từ đầu, chạy **đúng một lần** bằng cấu hình đã chốt. Số đem công bố lấy ở đây, không lấy ở development. | 🆕 |
| **Held-out reserve (587 câu / 100 bài)** | Chưa đụng tới. Sẽ là nguồn cho Phase 3. | 🆕 |
| **Cấu hình đã chốt** | Bộ ba retriever + reranker + cách cắt chunk mà Phase 1 chọn ra, và các phase sau **không được đổi nữa**. Bộ chỉ mục của nó được lưu thành một file có mã kiểm tra, để về sau chứng minh được là đã dùng đúng bộ đó chứ không phải một bản khác. | 🆕 |
| **Truy xuất chạy một lần, dùng chung** | Phase 2 thử 12 cấu hình sinh câu trả lời. Bước tìm đoạn văn không phụ thuộc vào prompt, nên nó chỉ chạy **một lần**: kết quả là danh sách 5 đoạn đã xếp hạng cho mỗi câu hỏi, được lưu lại và đưa cho cả 12 cấu hình dùng chung. Nhờ vậy mọi cấu hình đọc **đúng cùng 5 đoạn văn** — nếu điểm khác nhau thì chỉ có thể do prompt hoặc do số đoạn, chứ không thể do lần tìm kiếm này may hơn lần kia. | 🆕 |
| **`context_depth`** | Số đoạn được dán vào prompt, lấy từ đầu danh sách 5 đoạn nói trên. Depth 3 nghĩa là chỉ đưa 3 đoạn xếp hạng cao nhất, bỏ 2 đoạn còn lại. Không tìm lại, không xếp hạng lại — chỉ là cắt bớt danh sách có sẵn. | 🆕 |
| **`P0`–`P3`** | Bốn cách ra lệnh cho LLM mà Phase 2 đem so. P0 là cách đang dùng sẵn, làm mốc đối chứng. Nguyên văn ở [`docs/prompts.md`](../../prompts.md). | 🆕 |
| **`C0`–`C3`** | Bốn cách cắt bài thành đoạn mà Phase 2C đem so. C0 là cách đang dùng, làm mốc. C1 cắt theo câu, C2 theo đoạn văn, C3 tìm bằng mẩu nhỏ rồi đưa cho LLM đoạn lớn chứa mẩu đó. | 🆕 |
| **Đăng ký trước** | Viết ra quy tắc "thế nào là thắng" **trước khi** chạy và nhìn thấy kết quả. Các file trong `docs/Detailed Test Plans/` chính là văn bản đó. Mục đích: không được đổi luật sau khi đã biết cấu hình mình thích đang thua. | 🆕 |
| **Guardrail** | Điều kiện một cấu hình phải qua **trước khi** được đem ra so điểm chính. Ví dụ: so với mốc đối chứng, Faithfulness không được tụt quá 0,02. Cấu hình trượt guardrail thì bị loại, dù điểm chính có cao hơn. 🆕 Đây là ngưỡng do nhóm tự đặt vì lý do thực dụng, không phải kết quả của một phép kiểm định nào. | 🆕 |

## C. Cách chấm điểm

| Thuật ngữ | Nghĩa trong báo cáo này | |
| :--- | :--- | :-: |
| **Closed-world assumption** (giả định thế giới đóng) | Cách chấm điểm coi **đúng một** chunk là đúng, mọi chunk khác là sai — kể cả khi một chunk khác cũng trả lời được câu hỏi. | ✅ |
| **Biên độ nhập nhằng 7,0% / 24,5%** | EDA lấy text đáp án của từng câu hỏi rồi đi tìm nó trong các chunk **không** được gán nhãn. Kết quả: **24,5%** số câu có ít nhất một chunk như vậy chứa nguyên văn đáp án. Nhưng chứa đáp án chưa chắc đã trả lời được câu hỏi, nên EDA lọc thêm: **7,0%** số câu có một chunk vừa chứa đáp án vừa dùng chung ít nhất 2 từ hiếm với câu hỏi — tức nhiều khả năng nó thật sự trả lời được, thường vì cùng một sự kiện được đưa tin ở bài thứ hai.<br><br>Hệ quả: nếu hệ thống trả về bài thứ hai đó, cách chấm gọi là **sai** dù người dùng vẫn hài lòng. Nên **mọi điểm tìm kiếm đều mang sẵn sai số 7,0–24,5%**, và hai hệ thống chênh nhau ít hơn thế thì không kết luận được gì.<br><br>Đây là **chỉ báo, không phải kết quả đọc tay** — EDA tự ghi rõ như vậy.<br><br>🔒 **Gọi đúng một tên: *biên độ nhập nhằng*.** Bản LaTeX cũ gọi nó là *nhãn nhiễu* ở chỗ này và *biên độ nhiễu* ở chỗ khác — ba tên cho một thứ. *Nhãn nhiễu* còn sai nghĩa: nhãn không sai, **tập nhãn không đầy đủ**. Nói "nhãn nhiễu" là tự nhận dữ liệu gán sai, mà EDA không đo điều đó. | 🆕 |
| **False negative** | Hệ thống tìm ra một đáp án đúng nhưng bị chấm là sai, vì nó không phải đoạn được gán nhãn. | ✅ |
| **RAGAS** | Thư viện chấm điểm bằng cách nhờ một LLM khác làm giám khảo. 🆕 Ở đây giám khảo là `glm-5p3-flash` chạy qua Fireworks, **khác hẳn** model sinh câu trả lời (`gemini-3.1-flash-lite` của Google) — vì có lần notebook cũ để model tự chấm bài của chính nó. | 🆕 |
| **Answer Correctness**<br>*độ đúng của câu trả lời* | Giám khảo đọc câu trả lời và đáp án chuẩn rồi cho điểm xem hai bên khớp nhau đến đâu. Đây là **metric chính** của Phase 2. | 🆕 |
| **Faithfulness**<br>*độ bám bằng chứng* | Giám khảo tách câu trả lời thành từng khẳng định, rồi kiểm xem mỗi khẳng định có suy ra được từ mấy đoạn văn đã đưa không. Đây là metric bắt hành vi **bịa**. | 🆕 |
| **Answer Relevancy**<br>*độ trúng trọng tâm* | Giám khảo chấm xem câu trả lời có đúng trọng tâm câu hỏi không. 🆕 Cần cảnh báo: metric này thưởng câu trả lời đầy đủ ngữ cảnh, nên nó **giảm** khi ta cố tình bắt model trả lời cụt hơn. Giảm ở đây không có nghĩa là tệ đi. | 🆕 |
| **Citation Validity**<br>*trích dẫn có hợp lệ không* | LLM được yêu cầu ghi `[1]`, `[2]`… để chỉ ra nó lấy thông tin từ đoạn nào. Metric này đếm xem các số đó có nằm trong khoảng thật không — trích `[7]` trong khi chỉ có 5 đoạn là sai. | 🆕 |
| **Citation F1**<br>*trích dẫn có đúng và có đủ không* | So đoạn mà LLM trích dẫn với đoạn thật sự chứa bằng chứng: nó có trích đúng đoạn không, và có bỏ sót đoạn nào không. | 🆕 |
| **Coverage**<br>*tỉ lệ chạy trót lọt* | Tỉ lệ câu hỏi chạy xong và chấm được. Dưới 95% thì cấu hình bị loại, bất kể điểm cao hay thấp — vì thiếu quá nhiều câu thì điểm trung bình không so được với ai. | 🆕 |
| **Exact Match**<br>*khớp từng chữ* | Câu trả lời có trùng khớp từng chữ với đáp án chuẩn không. Ở đây chỉ dùng để **chẩn đoán**, không dùng để chọn winner: model diễn đạt lại đúng ý vẫn bị 0 điểm. | 🆕 |

## D. Thống kê

| Thuật ngữ | Nghĩa trong báo cáo này | |
| :--- | :--- | :-: |
| **So sánh theo từng câu hỏi** | Vì mọi cấu hình chạy trên **đúng cùng bộ câu hỏi**, ta lấy hiệu điểm trên **từng câu** rồi mới trung bình, thay vì lấy điểm trung bình của A trừ điểm trung bình của B. Cách trước nhạy hơn hẳn: nó loại được ảnh hưởng của việc câu này vốn khó hơn câu kia. | 🆕 |
| **Bootstrap** | Ta chỉ đo được **một lần** trên 281 câu hỏi này. Câu hỏi đặt ra: nếu gặp 281 câu khác thì khoảng cách vừa đo có còn không? Bootstrap trả lời bằng cách giả lập — bốc ngẫu nhiên **có hoàn lại** từ chính dữ liệu đang có để dựng ra 2.000 bộ dữ liệu "giả", tính lại khoảng cách trên từng bộ, rồi xem 95% kết quả rơi vào đâu. Khoảng đó là **khoảng tin cậy 95%**. Nó không chứa số 0 nghĩa là gần như lần giả lập nào cũng cho ra cùng một chiều, nên khoảng cách khó mà chỉ do may rủi. | 🆕 |
| **Bốc lại theo bài báo, không theo câu** | Mỗi lần bootstrap bốc **nguyên một bài kèm toàn bộ câu hỏi của nó**, chứ không bốc từng câu rời. Lý do: mấy câu hỏi thuộc cùng một bài thường cùng đúng hoặc cùng sai — bài dễ thì câu nào cũng dễ. Coi chúng là những lần bốc độc lập sẽ khiến khoảng tin cậy hẹp hơn sự thật, tức là mình tự tin quá mức. 🆕 Đo rồi thì tác động của việc này **nhỏ**: vì ta so *hiệu* giữa hai cấu hình, độ khó của bài triệt tiêu trong phép trừ. Phase 1 bốc theo câu, Phase 2 bốc theo bài — không đồng nhất, nhưng không làm kết luận của Phase 1 sai. Xem [`04_stats.md`](04_stats.md) mục 3. | 🆕 |
| **Micro / macro theo bài** | Micro là trung bình trên tất cả các câu, nên bài nào nhiều câu hỏi thì có tiếng nói lớn hơn. Macro thì trung bình trong từng bài trước, rồi mới trung bình 50 bài với nhau — mỗi bài một phiếu. Hai số lệch nhau nhiều nghĩa là kết quả đang bị một vài bài chi phối. | 🆕 |
| **Hai điều kiện để gọi là thắng** | Một khoảng cách chỉ được coi là thắng khi **(1)** nó lớn hơn biên độ nhập nhằng 7,0%, **và** **(2)** khoảng tin cậy 95% của nó không chứa số 0. Điều kiện (2) một mình là chưa đủ: một khoảng cách có thể đo được rõ ràng nhưng vẫn quá nhỏ so với sai số vốn có của bộ dữ liệu. | 🆕 |
| **MDE₈₀** *(khoảng cách nhỏ nhất phát hiện được)* | Với cỡ mẫu này, một khoảng cách phải lớn tới đâu thì ta mới bắt được nó 80% số lần. Ví dụ MDE₈₀ của Answer Correctness là 0,029: khoảng cách 0,10 thì chắc chắn thấy, khoảng cách 0,01 thì phần lớn các lần sẽ trượt. Dùng để trả lời "cỡ mẫu có đủ không" bằng số thay vì bằng cảm tính. | 🆕 |
| **Design effect** | Bốc lại theo bài làm khoảng tin cậy rộng ra hay hẹp đi bao nhiêu lần so với bốc theo câu. Lớn hơn 1 là rộng ra, tức gom cụm khiến ta thận trọng hơn. | 🆕 |
| **n hiệu lực** | Cỡ mẫu 281 câu "thực chất đáng giá" bao nhiêu câu độc lập, sau khi trừ đi việc nhiều câu dùng chung một bài. | 🆕 |
| **Kết quả null** | Đo rồi nhưng không tách được hai bên ra — khoảng tin cậy chứa số 0. 🆕 Cần nói rõ: điều này **không** chứng minh hai cấu hình bằng nhau, nó chỉ nói dữ liệu hiện có không đủ để phân biệt. Muốn kết luận "bằng nhau" thì phải thiết kế một phép kiểm định khác hẳn ngay từ đầu. | 🆕 |

---

## Việc còn phải làm cho §0

1. **Đo trung vị độ dài đáp án chuẩn** (token) — cần để thay câu ẩn dụ "khớp với
   hình dạng của nhãn" bằng một con số. Xem [`06_decisions.md`](06_decisions.md) D11.
2. Vị trí của §0 trong báo cáo ❓ — xem [`06_decisions.md`](06_decisions.md) D5.
3. Rà chéo với EDA §0 để hai tài liệu không định nghĩa lệch nhau.
