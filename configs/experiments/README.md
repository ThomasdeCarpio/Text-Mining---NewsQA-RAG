# Experiment configs

Mỗi file YAML trong thư mục này là một experiment mà CLI và Evaluation Desk có
thể chạy. Để test feature mới:

1. copy `newsqa_retrieval_smoke.yaml` sang tên mới;
2. đổi `experiment.id`;
3. giữ baseline trong `fixed`, đặt yếu tố cần so sánh trong `matrix`;
4. chạy `python scripts/run_experiment.py <file> --dry-run`;
5. chạy thật hoặc dùng Evaluation Desk.

Không ghi đè ID cũ. Hướng dẫn và ví dụ:
[`docs/archive/experiments.md`](../../docs/archive/experiments.md).

