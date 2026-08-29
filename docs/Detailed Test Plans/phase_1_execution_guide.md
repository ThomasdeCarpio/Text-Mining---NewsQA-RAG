# Hướng dẫn chạy Phase 1 Retrieval Tournament

## Chuẩn bị

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/build_retrieval_models_index.py --overwrite
python scripts/validate_phase1_artifacts.py \
  --index-manifest data/evaluation/retrieval_models_index/index_manifest.json
```

Lần đầu cần tải bốn embedding models, BGE-M3 và hai rerankers từ Hugging Face. Phase 1 không cần API key LLM.

## Chạy tournament

Mở `notebooks/06_phase_1_retrieval_tournament.ipynb` và chạy theo thứ tự:

1. Round 1 chạy 4 dense và 4 sparse profiles trên 50 development articles.
2. Tổng hợp kết quả, chọn Best Dense và Best Sparse theo MRR@5; tie-break lần lượt bằng NDCG@5, Hit@5 và P50.
3. Compose hybrid profile và chạy Round 2 gồm 3 retrievers × 3 rerankers.
4. Materialize/index các corpus 256/32, 512/64, 1024/128 và chạy 6 Round 3 runs.
5. Khóa winner trên development, sau đó chạy đúng một lần trên 150 final-test articles.

Mọi experiment dùng `scripts/run_experiment.py`, tự resume và lưu per-question traces. Dùng `scripts/summarize_experiments.py` để tạo `comparison.json` và `comparison.csv`.

## Kiểm tra

```bash
python -m pytest -q
```

Không báo cáo final-test như một phần của model selection. Các biểu đồ phải đọc trực tiếp từ `comparison.json`/`report.json`, không nhập số liệu thủ công.
