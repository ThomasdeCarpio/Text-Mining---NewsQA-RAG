# Phase 1 — nghiệm thu trên final-test

Cấu hình khóa sau vòng 3 của giải đấu, chạy **đúng một lần** trên 150 bài báo
chưa từng dùng để tinh chỉnh. Chỉ truy xuất, không sinh, không gọi API.

- **871 câu `resolved` / 150 bài**, coverage 871/871, 0 lỗi
- Chạy `2026-09-06T18:20:18Z` → `18:29:05Z`, git commit `3ca47cd`
- BGE-M3 learned sparse, `top_k=20` → `BAAI/bge-reranker-large`, `top_n=5`,
  chunk recursive 512/64

## Kết quả

| | Dev (281) | **Final-test (871)** | Δ | CI95 final-test |
| :--- | ---: | ---: | ---: | :--- |
| Hit@1 | 0,8221 | 0,7543 | −0,0678 | [0,7161; 0,7939] |
| Hit@3 | 0,9324 | 0,8611 | −0,0713 | [0,8298; 0,8904] |
| Hit@5 | 0,9573 | **0,8978** | −0,0595 | [0,8722; 0,9210] |
| MRR@5 | 0,8797 | 0,8112 | −0,0685 | [0,7803; 0,8423] |
| nDCG@5 | 0,8976 | 0,8313 | −0,0663 | [0,8019; 0,8598] |
| Recall@5 | 0,9555 | 0,8955 | −0,0600 | [0,8693; 0,9199] |

Hit@3, Hit@5, nDCG@5 và Recall@5 có CI95 của hai tập **không chồng lấn**: tập
development dễ hơn phần còn lại của kho, và đó là số đo chứ không phải phỏng đoán.

Reranker vẫn có lợi trên dữ liệu khó hơn: ΔnDCG@5 +0,0758 (CI95 [+0,0504; +0,1017]).

## Chống chọn lại sau khi nhìn số

`final_protocol.json` mang ba cờ, cùng hash của chunks, index và testset:

```
winner_locked_before_heldout : true
confirmatory_only            : true
no_post_heldout_reselection  : true
```

## Có gì ở đây

| File | Nội dung |
| :--- | :--- |
| `final_comparison.csv` / `.json` | Toàn bộ metric của run, kèm CI95 bootstrap theo câu và article macro |
| `final_protocol.json` | Cấu hình đã khóa, ba cờ chống chọn lại, hash nguồn dữ liệu và index |
| `heldout_question_scores.jsonl` | Điểm truy xuất từng câu, trước và sau rerank |
| `heldout_significance.json` | CI95 gom cụm theo bài, so dev ↔ final-test, và quan hệ tập con với held-out Phase 2 |
| `partitions_preflight.json` | Danh sách bài và câu của từng partition, seed 42 |
| `run_manifest.json` | Fingerprint của run, danh sách 871 question id, đường dẫn artifact |
| `environment.json` | Phiên bản Python và thư viện |
| `report.json` | Báo cáo gốc do harness sinh ra |
| `final_retrieval_metrics.png` | Biểu đồ do run tự vẽ |
| `locked_phase1_heldout_config.yaml`, `phase1-locked-heldout-final.yaml` | Cấu hình và experiment YAML đã dùng |

`predictions.jsonl` và `retrievals.jsonl` (mỗi file ~64 MB) không đưa vào repo;
`heldout_question_scores.jsonl` đã đủ cho mọi con số trong báo cáo.

## Chạy lại

```bash
python scripts/phase1_heldout_summary.py
```

Script tự đối chiếu số nó tính với số run tự ghi trong `final_comparison.json`
trước khi làm gì khác, nên đọc nhầm cột là hỏng ngay tại chỗ.
