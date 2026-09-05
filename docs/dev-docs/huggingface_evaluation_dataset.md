# Lưu evaluation dataset private trên Hugging Face

Hugging Face Dataset repository là nơi lưu **canonical source data**. Chroma,
BM25, chunks và `relevant_chunk_ids` được build lại ở mỗi máy theo config đã
chọn; chúng không được upload như ground truth.

> **Trạng thái hiện tại.** Dataset đã được publish **công khai** tại
> [`MatchaMacchiato/newsqa_200_11064_v2.0.0`](https://huggingface.co/datasets/MatchaMacchiato/newsqa_200_11064_v2.0.0),
> ghim tại commit `b81c8db6847a23272665946c0c43c72e9a212fd9`. Các notebook trong
> `notebooks/public/` tải nó ẩn danh, không cần token.
>
> Phần còn lại của tài liệu này là quy trình publish tổng quát; tên organization
> và số version bên dưới chỉ là ví dụ.

Quy trình ban đầu được thiết kế cho private repository vì quyền tái phân phối
phần NewsQA-derived text chưa được xác minh để công khai.

## 1. Chuẩn bị tài khoản và token

1. Tạo Hugging Face organization cho nhóm, ví dụ `hcmus-text-mining`.
2. Mời các thành viên vào organization.
3. Người publish tạo token có quyền ghi repository.
4. Teammate chỉ chạy benchmark tạo token read-only.

Cài project và đăng nhập:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
hf auth login
```

Hoặc đặt biến môi trường, không ghi token vào Git:

```bash
export HF_TOKEN="hf_..."
export NEWSQA_EVAL_REPO_ID="hcmus-text-mining/newsqa-rag-evaluation"
```

## 2. Kiểm tra bundle trước khi upload

Lệnh dry-run đọc dataset local, kiểm tra evidence offsets, review coverage,
dedup approval, tạo checksum và export bundle nhưng không gọi mạng:

```bash
python scripts/publish_evaluation_dataset.py \
  --repo-id "$NEWSQA_EVAL_REPO_ID" \
  --version v1.0.0 \
  --dry-run
```

Bundle nằm tại:

```text
data/evaluation_exports/newsqa_200_11064/v1.0.0/
```

Bundle chỉ gồm articles, original questions, human review, dedup decisions,
selection manifest và `cloud_manifest.json`. Không gồm Chroma/BM25/chunks.

## 3. Publish private release

Sau khi dry-run thành công:

```bash
python scripts/publish_evaluation_dataset.py \
  --repo-id "$NEWSQA_EVAL_REPO_ID" \
  --version v1.0.0
```

CLI luôn tạo/kiểm tra private dataset repository, upload bundle và tạo tag
`v1.0.0`. Tag đã tồn tại sẽ không bị ghi đè. Không dùng lại một tag cho dữ liệu
khác.

Quy ước version:

- `v1.0.1`: sửa metadata hoặc tài liệu, không đổi semantic ground truth;
- `v1.1.0`: sửa question/answer/evidence/review;
- `v2.0.0`: đổi article selection hoặc schema không tương thích.

## 4. Teammate tải và build evaluation database

Teammate cần read token và chạy:

```bash
export HF_TOKEN="hf_..."
export NEWSQA_EVAL_REPO_ID="hcmus-text-mining/newsqa-rag-evaluation"

python scripts/materialize_evaluation_dataset.py \
  --repo-id "$NEWSQA_EVAL_REPO_ID" \
  --revision v1.0.0 \
  --output-root data/evaluation/newsqa_200_11064 \
  --config configs/config.yaml \
  --db-path data/chroma_db
```

Lệnh này:

1. tải đúng release đã pin;
2. kiểm tra SHA-256 của mọi canonical file;
3. tạo lại staging layout;
4. chunk 11.064 articles;
5. map character evidence spans sang chunk IDs;
6. build Chroma và BM25;
7. finalize original/resolved/clarified testsets;
8. áp dụng human-approved semantic deduplication;
9. ghi Hugging Face commit và index fingerprint vào local manifests.

Output quan trọng:

```text
data/evaluation/newsqa_200_11064/final/
data/evaluation/newsqa_200_11064/final_deduplicated/
data/evaluation/newsqa_200_11064/manifests/
data/chroma_db/
```

Chạy lại cùng revision/config sẽ resume nếu output hoàn chỉnh. Nếu một build dở
dang hoặc muốn thay thế output hiện tại, dùng `--overwrite`.

## 5. Thử chunking khác

Tạo một config mới, thay `chunk_size`, `chunk_overlap` hoặc embedding model rồi
build vào output root riêng:

```bash
python scripts/materialize_evaluation_dataset.py \
  --repo-id "$NEWSQA_EVAL_REPO_ID" \
  --revision v1.0.0 \
  --config configs/experiments/chunk_256_32.yaml \
  --output-root data/evaluation/newsqa_200_11064_chunk_256_32 \
  --db-path data/chroma_db
```

Questions, accepted answers, review và character evidence spans được giữ
nguyên. Chunks, Chroma, BM25, `relevant_chunk_ids`, collection name và index
fingerprint được tạo mới.

Để chỉ tạo testset/chunks mà không tải embedding model hoặc build index:

```bash
python scripts/materialize_evaluation_dataset.py \
  --repo-id "$NEWSQA_EVAL_REPO_ID" \
  --revision v1.0.0 \
  --skip-index \
  --no-deduplicate
```

## 6. Dùng bundle local để kiểm tra offline

```bash
python scripts/materialize_evaluation_dataset.py \
  --local-bundle data/evaluation_exports/newsqa_200_11064/v1.0.0 \
  --revision v1.0.0 \
  --output-root /tmp/newsqa-eval-check \
  --skip-index \
  --no-deduplicate
```

## 7. Lỗi thường gặp

| Lỗi | Cách xử lý |
| --- | --- |
| `401/403` | Kiểm tra `HF_TOKEN`, quyền organization và repo ID |
| Repository public | Chuyển repo thành private; publisher sẽ từ chối upload |
| Tag đã tồn tại | Tăng version; không xóa/ghi đè release đã benchmark |
| Checksum mismatch | Xóa Hugging Face cache lỗi và tải lại đúng revision |
| Output đã tồn tại | Dùng output root khác hoặc `--overwrite` có chủ đích |
| Collection đã tồn tại | Dùng cùng build để resume hoặc `--overwrite` |
| Model download timeout | Chạy lại khi Hugging Face model service ổn định |

Mỗi experiment phải lưu `requested_revision`, resolved Hugging Face commit,
`dataset_sha256` và `index_fingerprint` để kết quả có thể tái lập.
