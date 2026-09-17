# Báo cáo đồ án NewsLens

[PDF](main.pdf) · [Nguồn chính](main.tex) · [Trạng thái](../outline/README.md)

## Build

Từ thư mục gốc repository:

```bash
make -C docs/latex report
# Hoặc dùng Tectonic:
make -C docs/latex report-tectonic TECTONIC=/path/to/tectonic
```

Trên Overleaf: chọn **XeLaTeX**, main document **`report/main.tex`**.
Giữ `report/` cùng cấp với `figures/`, trong đó cần `hcmus-logo.png`,
`chat_interface.png` và `retrieval_playground.png`. Không cần dữ liệu,
model, API, CSV hoặc bước sinh số liệu của slide.

## Cấu trúc

| Nội dung | Nguồn |
|---|---|
| Bìa, nhóm, tóm tắt | `frontmatter/` |
| Sáu chương | `sections/01_introduction.tex` đến `06_conclusion.tex` |
| Kết quả và phân tích từng phase | `results/` |
| Sơ đồ vector | `figures/` |
| Prompt, điều kiện đo và tái lập | `appendices/` |
| Tài liệu tham khảo | `references.bib` |
| Mẫu prompt judge để đối chiếu | `appendix/ragas_prompts/` (không input trong PDF) |

Các section cũ và entry chuyển tiếp đã được bỏ để chỉ có một bộ nguồn đang
sử dụng. Bản Experiment gốc còn trong lịch sử Git tại `4d6a4a2`; bản main
GitHub trước tích hợp tại `dc04e2d`.

## Quy cách và biên tập

- Môn học: **Khai thác dữ liệu văn bản và ứng dụng**.
- A4, một cột, lề 2,5 cm; cỡ chữ 12 pt, giãn dòng 1,15 theo main GitHub.
  Times New Roman nếu có, fallback TeX Gyre Termes. Bìa không số, phần đầu
  dùng số La Mã, nội dung chính bắt đầu từ 1.
- Bản hiện hành: **25 trang, 22 bảng, 5 hình**. Caption bảng ở trên, hình ở
  dưới; mục lục, danh mục và trích dẫn tự động. Không giảm font hoặc thêm
  nội dung đệm để đạt số trang.
- Giữ trọng tâm kiến trúc, thực nghiệm và phân tích. Phần dữ liệu/lý thuyết
  chỉ trình bày các ý cần thiết. Nêu phương pháp hoặc nhận xét chung trước
  ví dụ; không đưa lịch sử biên tập hay ghi chú thao tác vào PDF.
- Phần còn điền: **nhóm trưởng, trách nhiệm và sản phẩm bàn giao của từng
  thành viên**; học kỳ nếu môn học yêu cầu. Không tự gán phân công.

## Nguồn đối chiếu

Các đường dẫn `phase*/` dưới đây thuộc `docs/reports/`.

| Phần | Nguồn chính |
|---|---|
| Dữ liệu, rà soát và chia tập | Experiment `4d6a4a2`, `docs/eda/eda_report.md`, `evaluation/testset.py`, `question_review.py`, `question_dedup.py` trong `common/newsqa_rag/` |
| Kiến trúc và cấu hình | `configs/config.yaml`, `.env.example`, backend `chat_service.py`/`retrieval_service.py`, lõi RAG, chỉ mục sparse và frontend |
| Retrieval và final-test | Experiment `4d6a4a2`, `phase1/` |
| Zero-shot và held-out | `phase2/`, gồm score từng câu, provenance và held-out |
| One-shot và generator | Experiment `4d6a4a2`; các kết quả từng câu được cung cấp cục bộ trong `../test_result/Test Results/` tính từ gốc repo |
| Chunking | `phase2c/notes_thang.md`, screening CSV, finalist scores và `paired_significance.json` |
| Abstention | `phase3/`, `Phase3/PHASE3_RESULTS.md`, bộ chạy và kết quả sau hiệu chỉnh |

Các điểm cần giữ khi sửa tiếp:

- P2-D5 Exact Match là **26/281 = 0,0925**, sửa lỗi in số 0 của bản cũ.
  Không trộn điểm judge từ các lần chấm khác nhau của cùng đầu ra.
- Cấu hình ứng dụng, môi trường mẫu và lần chạy thực nghiệm có vai trò riêng.
  Judge không suy ra từ YAML chung; không gán HNSW cho chỉ mục sparse.
- C0 screening 2C khác đối chứng finalist tái sử dụng từ Phase 2; C3 được
  chấm theo ngữ cảnh cha chuyển cho generator. Phase 3 dùng corpus và bộ
  case riêng, kết quả sau hiệu chỉnh; lỗi sinh được tính vào lỗi quyết định.
- One-shot là khảo sát sau held-out, không thay cấu hình đã khóa. CI chứa 0
  chưa thể khẳng định tương đương; thiếu đoạn gold không đồng nghĩa thiếu mọi
  bằng chứng hợp lý. Giữ các giới hạn này khi rút gọn nhận xét.
- Chỉ báo điều kiện đo được xác nhận trong hồ sơ. Không lấy tên phần cứng,
  trọng số RRF hoặc mặc định hiện tại làm cấu hình thực tế của lần chạy cũ.

## Kiểm tra

```bash
uv run --offline --no-project --with pymupdf python scripts/check_latex_report.py \
  --render-dir /tmp/newslens-report-review
git diff --check
```

Bỏ `--offline` nếu máy chưa có PyMuPDF trong cache. Checker cần PDF và log
cùng lần build; kiểm tra A4, font nhúng, lề, mục lục, kết quả và tham chiếu.
Duyệt trực quan các trang sau khi sửa bố cục. Không cần chạy benchmark hoặc
kiểm thử ứng dụng cho thay đổi chỉ ở báo cáo.

Git giữ PDF để đọc trực tiếp; các tệp build trung gian được ignore. ZIP
Overleaf và bản lưu biên tập đặt ngoài repository, không đưa vào commit.
