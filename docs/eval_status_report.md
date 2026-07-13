# Báo cáo Evaluation — Duy Le

Tóm tắt những gì đã làm cho 3 task, mức hoàn thiện, ghi chú phần chưa làm / cần
chú ý, và phân tích kết quả so với deliverable. Dựa trên đúng những gì đã được
**chạy thật và verify** trong quá trình làm.

---

## Task 1 — Select evaluation dataset
**Deliverable:** 1,000-article evaluation dataset · **Assigned:** Lam-Thang

**Đã làm:** Viết `scripts/build_mini_testset.py` + hàm `build_article_testset()` trong
`src/evaluation/testset.py`: stream NewsQA, **gom theo bài** (không theo `key` per-question),
chọn N bài + toàn bộ câu hỏi của bài, và (tùy chọn) ingest luôn thành collection khớp. Script
cũ `prepare_testset.py` (1000 bài) vẫn giữ.

**Mức hoàn thiện: ~60% — hạ tầng xong, artifact 1000 bài chưa sinh/kiểm chứng.**
- ✅ Verified thật ở quy mô 12–25 bài (map 100%, benchmark ra số thật).
- ⚠️ Chưa chạy/kiểm bản **1000 bài** + collection khớp cho nó. Chỉ cần `--n-articles 1000 --max-scan` lớn, nhưng chưa thực thi.
- ⚠️ "1,000 **latest**" chưa xử lý: NewsQA không có mốc thời gian → "latest" mơ hồ; hiện là lấy N bài đầu khi stream.
- *(Task này assign cho Lam-Thang; phần tôi làm là tooling hỗ trợ, trùng một phần.)*

## Task 2 — Map evidence to chunks
**Deliverable:** Evidence-to-chunk mapping · **Assigned:** Lam-Thang

**Đã làm:** 3 hàm thuần trong `testset.py` — `evidence_to_span` (char-offset),
`chunk_char_ranges` (định vị chunk trong bài), `map_evidence_to_chunks` (giao khoảng + fallback
substring). Kèm notebook `03_newsqa_mini_dataset.ipynb` giải thích trực quan + self-check offline.

**Mức hoàn thiện: ~95% — xong và verified.**
- ✅ Chạy thật trên NewsQA: **map 100% (110/110 câu, 0 range lỗi)**, chính xác bằng char-offset (chuẩn hơn word-overlap).
- ⚠️ Chưa xử lý các ca biên: câu **unanswerable**, evidence **nhiều span**, bài mà chunker trôi whitespace mạnh (có fallback nhưng chưa test kỹ).

## Task 3 — Build evaluation notebook/UI  *(task của tôi)*
**Deliverable:** Working evaluation notebook or UI · **Assigned:** Duy Le

**Đã làm:**
- `notebooks/02_evaluation.ipynb`: driver đầy đủ — preflight (báo lỗi thiếu package / sai path / sai format), chunking + indexing diagnostics, benchmark dense-vs-hybrid, plot.
- Nối **dashboard UI vào dữ liệu thật**: `eval_service.py` bỏ mock, đọc `reports/*/report.json`; `run_benchmark.py` thêm capture failures + cảnh báo zero-overlap.
- Docs: `docs/evaluation.md` mục 6 (data contracts) + mục 7 (flow, swap points, UI); explainer Chương V (hiểu metrics).

**Mức hoàn thiện: ~90% — notebook chạy được, UI đọc số thật.**
- ✅ Notebook 02 chạy end-to-end, ra metric thật (dense hit@5=0.85, hybrid mrr@5=0.73).
- ✅ Dashboard cards / bar-chart / failure-table đọc report thật.
- ⚠️ Metric **RAGAS** (Faithfulness…) cần `OPENAI_API_KEY` + 1 lần chạy `--run-ragas` mới điền được thẻ; chưa chạy.
- ⚠️ Nút "Trigger Crawler" và Pipeline Logs vẫn là stub (ngoài phạm vi task này).

---

## Phân tích: kết quả vs deliverable / description / độ match project

**Khớp deliverable:**

| Task | Deliverable | Đạt? |
|---|---|---|
| 1 | 1,000-article dataset | ⚠️ Một phần — có script sinh được, **chưa có artifact 1000 bài thực tế** |
| 2 | Evidence-to-chunk mapping | ✅ Đạt — có cơ chế chạy thật, verified |
| 3 | Working eval notebook **or** UI | ✅ Đạt cả hai — notebook chạy + UI nối số thật |

**Khớp description:**
- Task 2 & 3 khớp sát mô tả.
- Task 1 lệch 2 điểm: (a) "1,000 latest" — NewsQA không sắp theo thời gian nên "latest" khó thực hiện đúng nghĩa; (b) note của Lam-Thang nói đã test 20 bài với metric retrieve+rerank — phần tôi làm **chồng lấn**, cần thống nhất ai giữ script nào để khỏi trùng.

**Độ match của các task với project: Cao.** Đây là đồ án RAG lấy *đo lường chất lượng* làm
trọng tâm (có hẳn Evaluation Dashboard cho Admin). Cả 3 task đúng lõi: có dataset chuẩn (T1) →
map ground-truth (T2) → đo & hiển thị (T3).

**Điểm cần lưu ý quan trọng nhất (ràng buộc T1↔T2↔T3):** testset và collection **phải cùng nguồn
bài + cùng chunker** thì metric mới đúng. Collection production `newsqa_cnn` (ingest từ CNN HTML
thô) **không** khớp testset NewsQA → chấm lên nó thì mọi metric = 0. Vì vậy eval phải dùng
collection riêng `newsqa_eval` do `build_mini_testset --build-collection` sinh ra. Cả nhóm cần nắm
ràng buộc này.

---

### Phụ lục — file liên quan
| Mục đích | File |
|---|---|
| Build dataset + collection khớp (1 lệnh) | `scripts/build_mini_testset.py` |
| Logic gom bài + evidence→chunk mapping | `src/evaluation/testset.py` |
| Metric math | `src/evaluation/metrics.py` |
| Chạy benchmark → report | `scripts/run_benchmark.py` |
| Report → dashboard | `src/services/eval_service.py` → `api/routers/admin.py` |
| Notebook driver | `notebooks/02_evaluation.ipynb` |
| Notebook giải thích mapping | `notebooks/03_newsqa_mini_dataset.ipynb` |
| Tài liệu flow + contract | `docs/evaluation.md` (mục 6–7) |
| Giải thích metric cho người mới | `docs/explainer/05-hieu-metrics.md` |
