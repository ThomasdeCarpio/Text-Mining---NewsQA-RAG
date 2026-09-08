# Phase 2C — ablation chiến lược chia đoạn

Bốn cách cắt văn bản thành đoạn, cùng 281 câu development, cùng prompt P2.
**Kết quả: không cấu hình nào qua hết 6 guardrail; winner giữ nguyên C0-P2-D5.**

Kế hoạch đăng ký trước: [`phase_2c_chunking_strategy_test_plan.md`](../../Detailed%20Test%20Plans/phase_2c_chunking_strategy_test_plan.md).
Tổng kết của người chạy: [`notes_thang.md`](notes_thang.md).

## Có gì ở đây

| Đường dẫn | Nội dung |
| :--- | :--- |
| `paired_significance.json` | So sánh theo cặp, bootstrap gom cụm theo bài, 2000 mẫu, seed 42 |
| `phase2c_retrieval_screening.csv` | Vòng 1 — C0–C3 chỉ chạy retrieval, 281 câu |
| `phase2c_retrieval_screening_eligibility.json` | CI95 của vòng 1 và phán quyết đi tiếp |
| `screening/gen_c{1,2,3}.json` | Vòng 2 — sàng lọc sinh, 80 câu, RAGAS chấm 20 |
| `screening/depth_d{1,3}.json` | Vòng 3 — sàng lọc depth của C3, 80 câu |
| `summary_c3_d{3,5}.json` | Chung kết, 281 câu, coverage 100% |
| `finalist_lock_c3_d{3,5}.json` | Hash cấu hình đã khóa trước khi chạy chung kết |
| `scores/*.jsonl` | Điểm từng câu của bốn cấu hình chung kết |

`scores/c0_d3_development.jsonl` và `c0_d5_development.jsonl` là **bản sao** của
`../phase2/scores/p2_d{3,5}_development.jsonl` — cùng run, chép sang đây để so
sánh 2C chạy được bằng một lệnh trên một thư mục, mà không làm đổi artifact của
Phase 2B.

## Chạy lại

```bash
python scripts/phase2_paired_comparison.py \
    --scores-dir docs/reports/phase2c/scores \
    --baseline c0_d5_development.jsonl \
    --plan 2c \
    --output docs/reports/phase2c/paired_significance.json
```

`--plan 2c` thêm hai guardrail truy xuất (Hit@5, Recall@5 không tụt quá 0,01) mà
Phase 2B không có, vì 2C được phép đụng vào cách cắt chunk còn 2B thì không.

Đổi `--baseline` sang `c0_d3_development.jsonl` để so cùng mức depth, hoặc
`c3_d5_development.jsonl` để so hai mức depth trong nội bộ C3.
