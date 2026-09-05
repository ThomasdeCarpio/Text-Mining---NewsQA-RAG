# Documentation map

- `dev-docs/`: kiến trúc và runtime - `architecture.md`, `database.md`, `ui.md`,
  `model_gateway.md`, `crawler.md`, `huggingface_evaluation_dataset.md`;
- `eda/eda_report.md`: EDA của evaluation dataset - dữ liệu bị truncate ra sao,
  original vs resolved, và những gì kết quả retrieval được phép kết luận.
  Chạy lại bằng `notebooks/06_dataset_eda.ipynb` hoặc `scripts/eda/*.py`;
- `figures/`: hình dùng trong các tài liệu ở trên;
- `Detailed Test Plans/` và `master_test_plan.md`: kế hoạch test theo phase;
- `roadmap.md`: checklist những việc còn lại - Phase 1 chưa đóng, Phase 2
  generation, deck thuyết trình, và các rủi ro chưa ai ghi lại.

`archive/` chứa slide, explainer, nội dung thuyết trình cũ và các guide
(`experiments.md`, `evaluation.md`, `evaluation_dataset.md`, `benchmarking.md`).
Chúng được giữ làm tài liệu môn học nhưng không phải source of truth cho
cấu trúc/code hiện tại.
