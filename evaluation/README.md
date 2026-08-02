# Evaluation metadata

Thư mục này **không chứa code chạy evaluation và không chứa report**. Nó chứa
metadata nhỏ cần commit để chứng minh dataset/index nào đã được dùng:

- `manifests/`: article selection, artifact hash, chunking/embedding và Chroma
  collection tương ứng;
- `question_dedup/`: đề xuất trùng câu hỏi và quyết định human approval.

Code nằm ở `backend/newsqa_rag/evaluation/`; cấu hình chạy nằm ở
`configs/experiments/`; kết quả nằm ở `outputs/experiments/`; dữ liệu lớn nằm ở
`data/evaluation/` và được Git bỏ qua.

Không sửa manifest sinh tự động bằng tay. Xem
[`docs/evaluation_dataset.md`](../docs/evaluation_dataset.md) khi cần tạo dataset
mới.
