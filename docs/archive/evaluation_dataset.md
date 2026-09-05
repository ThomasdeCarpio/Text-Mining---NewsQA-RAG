# Tạo evaluation dataset mới

Đây là workflow hiếm dùng. Nếu chỉ test retriever, reranker, model hoặc prompt
mới, hãy giữ dataset hiện tại và tạo [experiment YAML](experiments.md).

Chỉ tạo dataset mới khi thay đổi ít nhất một trong các phần sau:

- tập bài báo hoặc câu hỏi;
- đáp án/evidence được xem là ground truth;
- chunking hoặc embedding khiến chunk ID/index thay đổi.

## Dataset hiện tại

`newsqa_200_11064` gồm 200 bài validation để đánh giá và 10.864 bài train làm
distractor, seed `42`. Dữ liệu lớn nằm trong `data/evaluation/...` và không được
Git theo dõi. Metadata kiểm chứng nằm trong `evaluation/manifests/`.

## Build một dataset mới

Các lệnh dưới đây dùng mặc định `newsqa_200_11064`. Muốn tạo biến thể khác,
xem `--help` của từng subcommand và truyền output/manifest path riêng; không ghi
đè bộ đã khóa.

```bash
# 1. Chọn article và question
python scripts/prepare_evaluation_dataset.py stage1 --selection-only

# 2. Chunk corpus, build Chroma/BM25 và tạo baseline testset
python scripts/prepare_evaluation_dataset.py build-baseline

# 3a. Tái sử dụng review tương thích
python scripts/prepare_evaluation_dataset.py migrate-review

# Hoặc 3b. Tạo review mới
python scripts/prepare_evaluation_dataset.py init-review --archive-existing
python scripts/prepare_evaluation_dataset.py prepare-review-packets

# 4. Kiểm tra review và khóa final artifacts
python scripts/prepare_evaluation_dataset.py review-status
python scripts/prepare_evaluation_dataset.py finalize
```

Review proposal được áp dụng bằng:

```bash
python scripts/apply_review_proposals.py --packet PATH_TO_PACKET --proposals PATH_TO_PROPOSALS
```

Human reviewer phải quyết định `mark_standalone`, `approve`, `edit`, `exclude`
hoặc `needs_adjudication`. Còn dòng pending/adjudication thì `finalize` sẽ chặn.

## Sau khi finalize

| Artifact | Công dụng |
| --- | --- |
| `testset_reviewed_original.jsonl` | Câu hỏi gốc dùng để chấm chính |
| `testset_resolved.jsonl` | Cùng câu hỏi nhưng đã làm rõ |
| `chunks.jsonl`, `bm25.pkl` | Corpus/index cho retrieval |
| `integrity_report.json` | Counts và hash kiểm chứng |
| `evaluation/manifests/*.variant.json` | Khóa testset với đúng Chroma/BM25/config |

Tiếp theo, copy một YAML trong `configs/experiments/`, khai báo manifest/testset
mới dưới `dataset.indexes`, chạy `--dry-run`, rồi mới chạy experiment thật.

## Semantic dedup tùy chọn

```bash
python scripts/export_duplicate_question_report.py
python scripts/record_question_dedup_approval.py --reviewer-id ID --approve-all
python scripts/deduplicate_evaluation_dataset.py
```

Chỉ dùng `--approve-all` sau khi đã đọc toàn bộ cluster proposal.
