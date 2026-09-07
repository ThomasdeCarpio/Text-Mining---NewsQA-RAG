# 03 — Sổ đăng ký phát biểu

Mỗi phát biểu báo cáo đưa ra, kèm **artifact chống lưng** và **mức mạnh được phép
nói**. Viết mục nào thì tra mục đó.

Mức mạnh — *cách phân loại này chờ quyết, xem [`06_decisions.md`](06_decisions.md) D1:*

- **A** — vượt cả hai điều kiện: khoảng cách > 7,0% *và* CI95 không chứa 0.
- **B** — CI95 không chứa 0 nhưng khoảng cách < 7,0%. Nói "đo được", không nói "chắc chắn".
- **C** — đúng theo dữ liệu nhưng chưa kiểm định được: không có CI, hoặc n quá nhỏ, hoặc là lập luận cơ chế.
- **D** — CI95 chứa 0. Viết "không phân định được", **không** viết "tương đương".

---

## Phase 1

| # | Phát biểu | Bằng chứng | Mức |
| ---: | :--- | :--- | :-: |
| 1.1 | Sparse (BGE-M3) hơn dense (e5-base-v2) +0,1634 nDCG@5, CI95 [+0,1231; +0,2080] | `phase1/paired_significance.json` | **A** |
| 1.2 | Cơ chế: câu hỏi neo vào tên riêng / ngày / số nên khớp từ vựng có lợi | EDA §5 (từ hiếm/câu 0,34 → 0,93) | **C** — lập luận cơ chế |
| 1.3 | Không phân định được mô hình dense nào tốt nhất | CI95 chứa 0 ở mọi metric | **D** |
| 1.4 | Không phân định được BGE-M3 với BM25-stemmed trên `resolved` | CI95 [−0,0125; +0,0508] | **D** |
| 1.5 | Reranker thêm +0,0659 nDCG@5 | `round2.csv` | **A** |
| 1.6 | Hybrid không chứng minh được có lợi | `round2.csv` | **D** — không viết thành "hybrid có hại" |
| 1.7 | Kích thước chunk không phân định được (spread 0,0469, ba CI chồng lấn) | `round3.csv` | **D** |
| 1.8 | Contextual chunking giúp dense +0,0293, hại sparse −0,0208 | `contextual_chunking_ablation.json` | **B** — CI bốc theo câu, không gom cụm |
| 1.9 | Sparse thắng dense trên cả hai kho: 0,1143 → 0,0641 | như trên | **A** |
| 1.10 | Dense chưa tái lập được: trôi 0,0143 > khoảng cách giữa các mô hình 0,0094 | `chroma_store.py:33`, `embeddings.py:148` | **C** — quan sát hai lần chạy |
| 1.11 | Dataset không có metadata thật để index | notebook 16 cell 7, kiểm 200/200 bài | **A** |

## Phase 2 — development, 281 câu

| # | Phát biểu | Bằng chứng | Mức |
| ---: | :--- | :--- | :-: |
| 2.1 | Baseline P0: AC 0,6308 · EM 0,0000 · Faithfulness 0,9761 | `report_baseline_p0_d5.json` | **A** — số đo trực tiếp |
| 2.2 | Câu trả lời bám bằng chứng nhưng không trùng khớp span đáp án | 2.1, hai metric đọc cùng nhau | **C** |
| 2.3 | P1 (siết grounding) không tăng faithfulness mà giảm correctness | vòng sàng lọc prompt | **C** — RAGAS chỉ chấm 20 câu ở vòng này |
| 2.4 | P2-depth5 tăng AC +0,1043 so với P0, CI95 [+0,0844; +0,1240] | `paired_significance.json` | **A** |
| 2.5 | P2-depth5 **không phân biệt được** với P0 ở Faithfulness: +0,0040, CI95 [−0,0127; +0,0189] | như trên | **D** — 🔒 cấm viết "trung thực hơn P0"; qua guardrail không phải là hơn |
| 2.6 | P2-depth3 tụt Faithfulness −0,0200 so với P0, CI95 [−0,0372; −0,0048] | như trên | **C** — công suất 66%, xem [`04_stats.md`](04_stats.md) |
| 2.7 | P2-depth3 tụt Citation Validity −0,0142, CI95 [−0,0282; −0,0034] | như trên | **C** — công suất 56% |
| 2.8 | P2-depth3 trượt **hai** guardrail (2.6 + 2.7) nên bị loại | `phase2b_winner_decision.json` | **A** — áp luật, không phải kiểm định |
| 2.9 | Depth 3 cắt mất bằng chứng ở đúng 7/281 câu, khớp Hit@3 của Phase 1 | Hit@k theo câu trong `scores/*.jsonl` | **A** — đếm trực tiếp |
| 2.10 | Không bài báo nào chi phối: micro ↔ macro lệch < 0,004 | `paired_significance.json` | **A** |
| 2.11 | Trên nhóm truy xuất trượt, đổi prompt nhích AC 0,1504 → 0,1829 | `by_stratum` | **C** — n = 12; Δ (+0,0325) nhỏ hơn nửa MDE₈₀ (0,0847). Nêu kèm n, không nêu như kết luận |
| 2.12 | 23/30 câu điểm thấp thực ra đúng về ngữ nghĩa | `low_correctness_review_annotations.json` | **C** — một người duyệt có AI hỗ trợ, mẫu chọn có chủ đích |
| 2.13 | Giám khảo bất đồng với người duyệt ở 18/30 câu | như trên | **C** — bằng chứng cho "AC là cận dưới" |

### So trực tiếp hai finalist

| # | Phát biểu | Mức |
| ---: | :--- | :-: |
| 2.14 | d3 cao hơn d5 ở AC +0,0360, CI95 [+0,0190; +0,0554] | **A** |
| 2.15 | d3 cao hơn d5 ở Exact Match +0,1708 và Token F1 +0,1358 | **A** |
| 2.16 | d5 cao hơn d3 ở Faithfulness +0,0240, CI95 [+0,0056; +0,0440] | **C** — trị tuyệt đối Δ nhỏ hơn MDE₈₀ 0,0271 |
| 2.17 | d5 **không phân biệt được** với d3 ở Citation Validity | **D** — CI95 [−0,0037; +0,0269] |

> Tái lập 2.14–2.17:
> `python scripts/phase2_paired_comparison.py --scores-dir docs/reports/phase2/scores --baseline p2_d3_development.jsonl`
>
> Đáng chú ý cho §7: khoảng cách **đem đánh đổi** (AC, 2.14) đủ công suất, còn
> khoảng cách **đổi lấy được** (Faithfulness, 2.16) thì không.

## Held-out — 284 câu, chạy một lần

| # | Phát biểu | Bằng chứng | Mức |
| ---: | :--- | :--- | :-: |
| 3.1 | AC 0,7157 (micro) / 0,7230 (macro), coverage 100% | `heldout_final_summary.json` | **A** |
| 3.2 | Quyết định winner đóng băng trước run: ký 13:31:30Z, run 14:23:24Z | `heldout_access.json` | **A** |
| 3.3 | Băm artifact kiểm lại khớp | `heldout_access.json` | **A** — xem [`06_decisions.md`](06_decisions.md) D9 |
| 3.4 | Trên `gold_in_top5`, held-out **không phân biệt được** với dev | `heldout_retrieval_subgroups.csv` | **C** — 🔒 held-out 0,7689 CI95 [0,7391; 0,7978] chứa giá trị dev 0,7597. Viết "không thấp hơn", cấm viết "cao hơn" |
| 3.5 | Tỉ lệ truy xuất trượt: 12/281 → 35/284 | như trên | **A** |
| 3.6 | Citation F1 nhóm trượt = 0,0000 theo định nghĩa | như trên | **A** |
| 3.7 | AC toàn tập bị chặn trên bởi Hit@5 | 3.4 + 3.5 | **C** — lập luận, không phải kiểm định |

> 3.4 và 3.7 là hai phát biểu quan trọng nhất của báo cáo, và cả hai đều là **C**.
> Dev với held-out là hai tập khác nhau nên không ghép cặp được.

## Phase 2C — chờ số

| # | Phát biểu dự kiến |
| ---: | :--- |
| 4.1 | C1/C2/C3 so với C0 về Hit@5, Recall@5 |
| 4.2 | C3 có cân bằng được retrieval precision và context completeness không |
| 4.3 | Chiến lược nào tăng AC mà qua hết guardrail |
| 4.4 | Mức cải thiện có xứng chi phí index / token / latency không |
