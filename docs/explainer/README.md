# Tổng quan ngắn

NewsQA RAG trả lời câu hỏi bằng cách tìm các đoạn báo liên quan rồi đưa chúng
cho mô hình ngôn ngữ tạo câu trả lời có citation.

```text
Bài báo → làm sạch → chia chunk → Chroma/BM25
                                  ↓
Câu hỏi → dense/BM25/hybrid → rerank → LLM → câu trả lời [n]
                                  ↓
                       metrics + benchmark report
```

Phần đang hoạt động: ingestion, ba kiểu retrieval, reranker, chat RAG một lượt,
dataset evaluation, benchmark có resume, RAGAS và dashboard đọc report thật.
Phần chưa có: agent nhiều bước để tự tìm nhiều lần và tổng hợp nhiều nguồn.

Đọc tiếp:

- [README](../../README.md): cài đặt, chạy app và trạng thái dự án.
- [Evaluation pipeline](../evaluation.md): kiến trúc, artifact và metrics.
- [Benchmarking](../benchmarking.md): lệnh chạy benchmark.
- [Evaluation dataset](../evaluation_dataset.md): cách tạo và review dataset.
- [Database](../database.md): chunk ID, metadata và index.
