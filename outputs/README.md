# Generated outputs

Mọi artifact sinh ra khi chạy script, notebook hoặc app đều nằm ở đây:

- `experiments/`: run registry, traces, metrics và dashboard comparison;
- `benchmarks/`: các benchmark CLI cũ hoặc chạy đơn lẻ;
- `eda/`: kết quả JSON đã cache của `scripts/eda/*.py`;
- `retrieval/figures/`: hình cho retrieval ablation study;
- `frontend/`: build của `app/frontend` (`npm run build`);
- `presentations/`: slide/PPTX đã xuất.

Code không nằm ở đây. Mỗi experiment mới phải dùng một `experiment.id` mới để
không ghi đè kết quả cũ.
