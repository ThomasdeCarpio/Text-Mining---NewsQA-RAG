# Reports

Báo cáo thực nghiệm của dự án **NewsQA RAG**.

**Đọc cái này trước:**

- **[Tóm tắt](report.md)** — hệ thống là gì, Phase 1 làm gì, Phase 2 làm gì, chốt được gì, còn thiếu gì. Chia rõ theo từng phase.

**Bản nộp:** [`docs/latex/`](../latex/) — slide beamer 27 trang và báo cáo A4 9 trang. Mọi con số sinh tự động từ artifact bằng `scripts/build_latex_numbers.py`, nên hai bản không lệch nhau được.

**Báo cáo chi tiết từng phase:**

| Báo cáo | Nội dung | Kế hoạch đăng ký trước |
| :--- | :--- | :--- |
| [EDA](../eda/eda_report.md) | Dữ liệu thực sự trông như thế nào | — |
| [Phase 1](phase1/report.md) | Giải đấu 23 cấu hình truy xuất, 3 vòng | [`phase_1_retrieval_test_plan.md`](../Detailed%20Test%20Plans/phase_1_retrieval_test_plan.md) |
| [Phase 2](phase2/report.md) | Baseline end-to-end và tối ưu generation | [`phase_2_baseline_test_plan.md`](../Detailed%20Test%20Plans/phase_2_baseline_test_plan.md), [`phase_2_generation_tuning_plan.md`](../Detailed%20Test%20Plans/phase_2_generation_tuning_plan.md) |
| [Prompt nguyên văn](../prompts.md) | Bốn system prompt, cách ghép request, `context_depth`, hash đối chiếu | — |
| Phase 2C | Ablation chiến lược chia đoạn — 4 chiến lược, winner không đổi | [`phase_2c_chunking_strategy_test_plan.md`](../Detailed%20Test%20Plans/phase_2c_chunking_strategy_test_plan.md) |
| Phase 3 | Abstention — 3 chính sách, winner không đổi | [`phase_3_abstention_test_plan.md`](../Detailed%20Test%20Plans/phase_3_abstention_test_plan.md) |

**Nội dung nằm giữa hai phase:**

- **[Báo cáo gộp Phase 1 + Phase 2](report_detail.md)** — sợi chỉ EDA → Phase 1 → Phase 2, checklist nghiệm thu theo test plan, các chỗ thực tế lệch với kế hoạch, provenance.
- [Phase 1 — bản tiếng Anh, gọn](../phase1_results.md)

**Artifact:**

- `phase1/` — CSV từng vòng, `winner_lock.jsonl`, `paired_significance.json`.
- `phase1/heldout/` — **nghiệm thu Phase 1**: 871 câu / 150 bài chưa từng chạm tới, bản ghi giao thức có ba cờ chống chọn lại, điểm theo từng câu, và khoảng tin cậy gom cụm theo bài.
- `phase2/` — `report.json` của baseline và hai finalist, điểm theo từng câu, `paired_significance.json`, `phase2b_winner_decision.json`.
- `phase2/heldout/` — **bằng chứng của con số công bố**: access record kèm chuỗi hash, summary micro + article macro, phân tầng theo truy xuất, điểm theo từng câu và từng bài.
- `phase2/provenance/` — manifest chuẩn bị, danh sách id từng partition, `retrieval_lock.json`, cấu hình judge.
- `phase2c/` — điểm từng câu của bốn cấu hình chung kết, `paired_significance.json`, kết quả ba vòng sàng lọc, và bản tổng kết của Thắng (`notes_thang.md`).
- `phase3/` — kết quả đầy đủ 200 case ba chính sách, `policy_comparison.csv`, `policy_significance.json`.
