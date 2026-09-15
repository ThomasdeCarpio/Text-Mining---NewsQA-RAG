# Trạng thái báo cáo NewsLens

## Phạm vi hiện hành

- Báo cáo sáu chương, mục tiêu 25 trang. Môn học: **Khai thác dữ liệu văn
  bản và ứng dụng**. Giữ kiểu bìa, font, lề 2,5 cm và giãn dòng 1,15 của
  main GitHub `dc04e2d`; Experiment `4d6a4a2` là chuẩn cho thực nghiệm.
- Các tài liệu outline còn lại lưu lịch sử kế hoạch. Khi khác bản hiện hành,
  dùng [README báo cáo](../report/README.md) và nội dung LaTeX đang được input.
- Phạm vi thay đổi: nguồn báo cáo, hướng dẫn build và PDF checker. Không
  sửa ứng dụng, slide, dữ liệu hoặc kết quả; không chạy model/API/benchmark.

## Nội dung đã hoàn thành

- Tích hợp kiến trúc, dữ liệu, thực nghiệm và thảo luận; đối chiếu với mã
  nguồn, tài liệu và kết quả từng câu. Sửa EM P2-D5 thành 0,0925; phân biệt
  đối chứng 2C, dữ liệu/đầu ra sau hiệu chỉnh Phase 3 và khảo sát one-shot.
- Giữ giới hạn diễn giải CI, chất lượng dẫn nguồn, phạm vi held-out và
  điều kiện đo. Phần dữ liệu/lý thuyết đã rút gọn; PDF không có ghi chú
  biên tập. Hình giao diện dùng ảnh gốc, sơ đồ vector đối chiếu từ code.
- Đã duyệt 25 trang, 22 bảng, 5 hình. Kiểm tra font nhúng, lề, Unicode,
  tham chiếu và số liệu đạt; ZIP đã build độc lập và khớp PDF từng trang.

## Dọn trước bàn giao qua nhánh main-report

- Chỉ giữ các section được input. Bỏ entry chuyển tiếp, bản nội dung cũ,
  review đã xử lý và tệp BibTeX sinh tự động; rút gọn README và tracker.
- Giữ mẫu prompt judge trong `report/appendix/ragas_prompts/` vì có giá trị
  đối chiếu và được công cụ xuất prompt sử dụng. Giữ các artifact khoa học
  trong `docs/reports/` và PDF cuối; không đưa backup/ZIP vào commit.
- Đã lưu một bản trước dọn ngoài repository; lịch sử báo cáo gốc còn trong
  Git. So sánh xác nhận toàn bộ 23 input TeX và bibliography không đổi.
- Build từ đúng các tệp đã stage và PDF checker đạt: 25 trang, khớp từng
  pixel với PDF trước dọn ở 108 dpi. `git diff --check` đạt; commit chỉ gồm
  tài liệu báo cáo và checker. Nhánh bàn giao: `main-report`, không push `main`.

## Phần còn điền

Nhóm trưởng và trách nhiệm/sản phẩm bàn giao của từng thành viên; học kỳ
nếu môn học yêu cầu. Không còn ô chờ kết quả thực nghiệm hoặc ảnh demo.
