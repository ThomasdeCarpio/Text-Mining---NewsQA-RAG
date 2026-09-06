# Reports

Báo cáo thực nghiệm của dự án **NewsQA RAG**.

**Đọc cái này trước:**

- **[Tóm tắt](report.md)** — hệ thống là gì, Phase 1 làm gì, Phase 2 làm gì, chốt được gì, còn thiếu gì. Chia rõ theo từng phase.

**Báo cáo chi tiết từng phase:**

| Báo cáo | Nội dung | Kế hoạch đăng ký trước |
| :--- | :--- | :--- |
| [EDA](../eda/eda_report.md) | Dữ liệu thực sự trông như thế nào | — |
| [Phase 1](phase1/report.md) | Giải đấu 23 cấu hình truy xuất, 3 vòng | [`phase_1_retrieval_test_plan.md`](../Detailed%20Test%20Plans/phase_1_retrieval_test_plan.md) |
| [Phase 2](phase2/report.md) | Baseline end-to-end và tối ưu generation | [`phase_2_baseline_test_plan.md`](../Detailed%20Test%20Plans/phase_2_baseline_test_plan.md), [`phase_2_generation_tuning_plan.md`](../Detailed%20Test%20Plans/phase_2_generation_tuning_plan.md) |
| Phase 3 | Abstention — chưa chạy | [`phase_3_abstention_test_plan.md`](../Detailed%20Test%20Plans/phase_3_abstention_test_plan.md) |

**Nội dung nằm giữa hai phase:**

- **[Báo cáo gộp Phase 1 + Phase 2](report_detail.md)** — sợi chỉ EDA → Phase 1 → Phase 2, checklist nghiệm thu theo test plan, các chỗ thực tế lệch với kế hoạch, provenance.
- [Phase 1 — bản tiếng Anh, gọn](../phase1_results.md)

**Artifact:**

- `phase1/` — CSV từng vòng, `winner_lock.jsonl`, `paired_significance.json`.
- `phase2/` — `report.json` của baseline và hai finalist, điểm theo từng câu, `paired_significance.json`.
