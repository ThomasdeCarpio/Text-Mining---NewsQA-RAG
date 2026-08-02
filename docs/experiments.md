# Test tính năng bằng experiment

## Chọn đúng việc cần làm

| Trường hợp | Việc cần làm |
| --- | --- |
| Đổi retriever, reranker, top K hoặc model | Tạo experiment YAML mới |
| Đổi embedding/chunking và có index mới | Khai báo thêm index trong YAML |
| Chỉ muốn xem kết quả cũ | Bấm **Load saved results** |
| Thay câu hỏi, đáp án đúng hoặc corpus | Tạo evaluation dataset mới |

Phần lớn feature mới thuộc hai dòng đầu. Không tạo lại dataset nếu ground truth
không thay đổi.

## Tạo experiment mới

### 1. Copy smoke config

```powershell
Copy-Item configs/experiments/newsqa_retrieval_smoke.yaml `
  configs/experiments/reranker_v2.yaml
```

Dashboard tự nhận mọi file `.yaml` trong `configs/experiments/`.

### 2. Đặt ID mới

```yaml
experiment:
  id: reranker-v2
  name: Baseline vs reranker v2
  description: Kiểm tra reranker mới trên development set.
```

`experiment.id` phải mới vì nó cũng là tên thư mục kết quả. Không tái sử dụng
ID cũ cho một phép thử khác.

### 3. Chỉ so sánh một thay đổi

Giữ baseline trong `fixed`; đặt đúng yếu tố cần so sánh trong `matrix`:

```yaml
fixed:
  index: baseline
  partition: development
  variant: original
  retriever: hybrid
  retrieval_only: true
  top_k: 10
  rerank_top_n: 5

matrix:
  reranker: [noop, cross-encoder]
```

Ví dụ này tạo 2 run. Nếu để `variant`, `retriever` và `reranker` đều có 2 giá
trị thì sẽ tạo `2 x 2 x 2 = 8` run.

Các trục hiện được hỗ trợ: `index`, `variant`, `partition`, `retriever`,
`reranker`, `top_k`, `rerank_top_n`, `retrieval_only`, `generator_model` và
`reranker_model`.

### 4. Chạy thử cấu hình trước

```bash
python scripts/run_experiment.py configs/experiments/reranker_v2.yaml --dry-run
```

Lệnh này chỉ validate và in các run, không load model. Kiểm tra số run trước khi
chạy thật để tránh vô tình tạo một matrix quá lớn.

### 5. Chạy và xem kết quả

```bash
python scripts/run_experiment.py configs/experiments/reranker_v2.yaml
```

Hoặc mở **Evaluation Desk**, chọn file, bấm **Preview config**, rồi
**Run / resume**. Khi xong, chọn từng run để xem metric và Failure Analysis.

## Test loại feature nào?

### Retriever/reranker mới

Feature phải được CLI `collect_benchmark_predictions.py` nhận trước. Sau đó thêm
tên của nó vào `matrix`. Luôn để baseline và candidate trong cùng experiment.

### Embedding, chunking hoặc index mới

Build index + variant manifest mới, rồi đăng ký cả hai dưới
`dataset.indexes` và so sánh bằng:

```yaml
matrix:
  index: [baseline, candidate]
```

Mỗi index phải trỏ đến đúng config, manifest và testset của chính nó.

### Generator model mới

Đặt `retrieval_only: false`, thêm `generator_model` vào `matrix`, và bật
`judge.enabled` nếu cần RAGAS. Judge model nên khác generator model.

### Thay đổi code không có trục YAML

Chạy baseline trước, implement feature, rồi tạo experiment ID mới và chạy lại.
`environment.json` lưu Git commit để biết mỗi kết quả đến từ phiên bản code nào.
Chỉ thêm một trục YAML mới khi feature đó thực sự cần được chạy lặp lại.

## Đọc dashboard

| Thành phần | Ý nghĩa |
| --- | --- |
| MRR@5 | Gold chunk đầu tiên đứng càng cao càng tốt |
| NDCG@5 | Thưởng khi gold chunks nằm gần đầu danh sách |
| Recall@5 | Tỷ lệ gold chunks xuất hiện trong top 5 |
| P95 latency | 95% câu hỏi hoàn thành nhanh hơn thời gian này |
| Coverage | Tỷ lệ câu hỏi chạy thành công |
| Failure Analysis | Câu hỏi thất bại của run đang chọn và lý do |

`No ground-truth chunk in reranked top 5` chỉ nghĩa là gold chunk không nằm
trong top 5 cuối cùng; không tự động có nghĩa database thiếu dữ liệu.

## Nút trên dashboard

| Nút | Làm gì | Có chạy model? |
| --- | --- | --- |
| Preview config | Validate và hiển thị matrix | Không |
| Run / resume | Chạy phần còn thiếu | Có |
| Rebuild summary | Tính lại comparison từ report có sẵn | Không |
| Load saved results | Đọc kết quả cũ | Không |

Kết quả nằm ở `outputs/experiments/<experiment.id>/`. Dừng giữa chừng không làm
mất các câu đã hoàn thành; chạy lại cùng YAML sẽ resume.

Nếu Hugging Face model đã được cache, có thể tắt network check trước khi chạy:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
```

Chỉ tạo dataset mới khi corpus/ground truth thay đổi; xem
[evaluation_dataset.md](evaluation_dataset.md).
