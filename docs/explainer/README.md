# Tài liệu giải thích hệ thống (dành cho người mới đọc project này)

**Đọc bản đẹp:** mở [`index.html`](index.html) trong trình duyệt (double-click
file, hoặc mở qua VS Code bằng "Open with Live Server" / kéo thả vào tab
trình duyệt) — cùng nội dung 4 file bên dưới, trình bày lại kiểu sổ tay đọc
cho có hứng, font/màu nhúng sẵn nên mở offline vẫn đẹp, không cần mạng.
*(Trên GitHub, xem trực tiếp file `.html` chỉ hiện mã nguồn thô — muốn xem
bản render, tải file về mở bằng trình duyệt, hoặc bật GitHub Pages cho repo
rồi vào `https://<user>.github.io/<repo>/docs/explainer/`.)*

Bộ tài liệu này viết cho việc **hiểu bức tranh tổng thể**: dự án làm gì, các
phần nối với nhau ra sao, cái gì đang chạy thật — cái gì đang giả lập, và còn
thiếu gì. Nó **không** đi vào từng dòng code. Nếu bạn cần chi tiết implement
(hàm nào, ký hiệu gì), xem `docs/database.md`, `docs/ingestion_guide.md`,
`docs/indexing_guide.md`, hoặc đọc thẳng code.

Đọc theo thứ tự này:

1. **[01-tong-quan.md](01-tong-quan.md)** — Dự án này để làm gì, cho ai dùng.
2. **[02-cach-he-thong-hoat-dong.md](02-cach-he-thong-hoat-dong.md)** — Một
   câu hỏi của người dùng đi qua những bước nào để ra câu trả lời (hoặc lẽ ra
   phải đi qua, vì vài bước còn chưa code thật).
3. **[03-ban-do-thu-muc.md](03-ban-do-thu-muc.md)** — Mỗi thư mục/file trong
   repo dùng để làm gì, tra cứu nhanh.
4. **[04-con-thieu-gi-va-tai-sao.md](04-con-thieu-gi-va-tai-sao.md)** — Phần
   nào chưa xong, và vì sao nó quan trọng (không phải chỉ là danh sách việc
   cần làm — là giải thích hậu quả nếu không làm).
5. **[05-hieu-metrics.md](05-hieu-metrics.md)** — Mấy con số điểm (Hit Rate,
   MRR, Faithfulness…) nghĩa là gì, giải thích bằng ngôn ngữ đời thường, không
   cần biết toán.

Quy ước trạng thái dùng xuyên suốt 4 file này:

| Ký hiệu | Ý nghĩa |
|---|---|
| ✅ Thật | Code chạy thật, có tác dụng thật (đụng vào file, model, database thật) |
| 🎭 Giả lập (mock) | Có API/UI đầy đủ, nhưng bên trong trả về dữ liệu bịa sẵn, không tính toán gì thật |
| ⛔ Chưa code | File tồn tại nhưng rỗng, hoặc chưa có file |
