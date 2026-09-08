# NewsLens — 20 slide thực nghiệm và ứng dụng

[Mở PDF](slides/main.pdf) · [Mã Beamer](slides/main.tex)

Bộ 20 trang 16:9 (320 × 180 mm), nền trắng, xanh `#003F88` lấy từ logo gốc.
Slide dùng bảng kẻ nhẹ, sơ đồ vector, thanh so sánh và ô điểm nổi bật. Mỗi slide
nêu mục đích hoặc kết luận rõ ràng; model/cấu hình/mẫu đánh giá nằm trong nội dung
chính. Chú thích trên slide chỉ giữ điều kiện diễn giải; đường dẫn nguồn nằm ở
bảng dưới đây. Ablation là một slide phụ, không phải nhánh chính của bài trình bày.

## Nội dung và dấu vết nguồn

Slide 1–7 tiếp nối phác thảo đã duyệt và phản hồi thiết kế. Phần 8–20 triển khai
mạch chọn cấu hình → kiểm tra held-out → áp dụng vào NewsLens, đối chiếu từ báo cáo
thực nghiệm và code ứng dụng hiện có. Các đường dẫn ngắn trong bảng tính từ
`docs/reports/`, trừ đường dẫn ghi đầy đủ.

| Trang | Vấn đề trình bày | Nguồn đối chiếu |
| :--- | :--- | :--- |
| 1 | Bìa, môn học, giảng viên, nhóm | Phác thảo do người dùng cung cấp |
| 2 | Corpus, chunk, gold, original/resolved | `docs/latex/outline/01_terms.md`; ví dụ Gabon chỉ là minh họa |
| 3 | Độ đúng, bằng chứng, trích dẫn; model sinh và judge | `phase2/report_baseline_p0_d5.json`; `phase2/provenance/experiment_run.json` |
| 4 | Bootstrap khác với điều kiện chọn | `phase1/paired_significance.json`; `phase2/paired_significance.json`; baseline plan §6, generation plan §7 |
| 5 | Pipeline RAG và chia tập theo bài | `phase2/report.md` §2; `phase2/provenance/subset_manifest.json`, `retrieval_lock.json`; `phase1/contextual_chunking_ablation.json` cho tổng chunk |
| 6 | Các quyết định dẫn tới cấu hình ứng dụng | Các test plan Phase 1/2; `phase2/phase2b_winner_decision.json`; `phase2c/notes_thang.md` cho thời điểm 2C |
| 7 | Baseline P0-depth5 | `phase2/report_baseline_p0_d5.json`; đối chiếu `phase2/provenance/comparison.csv` |
| 8 | So sánh retriever, original và resolved | `phase1/round1.csv`; `phase1/paired_significance.json`, gồm so sánh BGE-M3/BM25 trên original |
| 9 | Reranker: chất lượng và latency | `phase1/round2.csv`, lọc resolved; delta nDCG tính trực tiếp từ hai hàng sparse |
| 10 | Xác nhận retrieval trên bài chưa thấy | `phase1/heldout/heldout_significance.json`; `final_protocol.json` |
| 11 | Giả thuyết P0–P3 và AC sàng lọc | `configs/experiments/phase2_generation_prompts.yaml`; `phase2/report.md` §7.2 (bộ sinh đọc bảng đã commit); subset manifest cho mẫu 80/20 |
| 12 | Context depth và hai finalist | `phase2/scores/p0_d5_development.jsonl` cho Hit@k và số câu mất gold; `phase2/paired_significance.json`; `phase2/report.md` §8 |
| 13 | Guardrail loại P2-depth3, chọn P2-depth5 | `phase2/paired_significance.json`; `phase2/phase2b_winner_decision.json` |
| 14 | P0-depth5 so với P2-depth5 trên dev | Baseline report; `phase2/scores/p2_d5_development.jsonl`; `phase2/paired_significance.json` |
| 15 | Kết quả held-out; chốt trước khi chạy | `phase2/heldout/heldout_final_summary.json`, `heldout_access.json`; winner decision |
| 16 | Phân tầng có/không có gold trong context | `phase2/paired_significance.json` → `by_stratum`; `phase2/heldout/heldout_retrieval_subgroups.csv` |
| 17 | Cấu hình đích từ thực nghiệm | Retrieval lock; prompt P2; winner decision; held-out summary |
| 18 | Luồng ứng dụng và phần cần nối để khớp benchmark | `app/backend/newsqa_app/services/chat_service.py`, `retrieval_service.py`; `common/newsqa_rag/agents/rag_agent.py`, `llm.py`; `app/frontend/src/pages/ChatPage.tsx`, `components/CitationList.tsx` |
| 19 | Ablation phụ: kích thước, contextual, 2C, từ chối | `phase1/contextual_chunking_ablation.json`; retrieval lock → round3 candidates; `phase2c/paired_significance.json`; `phase3/policy_comparison.csv`, `policy_significance.json`; `phase2c/notes_thang.md`, `phase3/README.md` |
| 20 | Kết quả, giới hạn và bước đưa vào sử dụng | Các artifact trên; `phase2/report.md` cho audit/độ ổn định chưa hoàn tất; code ứng dụng cho phần tích hợp còn thiếu |

### Các ranh giới diễn giải

- AC screening chấm 20 câu; không đồng nhất với AC full development 281 câu.
- Dev và held-out khác bài: không diễn giải bảng dev–held-out như A/B ghép cặp.
- Final-test retrieval 871 câu chứa held-out hỏi đáp 284 câu; hai tập không độc lập.
- Guardrail dùng delta trung bình. Slide 13 dùng sáu chữ số để giữ nguyên việc
  `−0,020029 < −0,02`; màu đạt/loại không phụ thuộc số bị làm tròn.
- CI chứa 0 không chứng minh tương đương hoặc không giảm. Slide 14 chỉ nói chưa
  chứng minh Faithfulness tốt hơn; không tuyên bố P2 “trung thực hơn”.
- Không có gold theo nhãn không đồng nghĩa không có đáp án hợp lý. Không dùng
  Hit@5 làm chặn trên toán học của AC; slide 16 trình bày chẩn đoán phân tầng.
- Slide 4 nêu quy tắc chính; toàn bộ tie-break nằm trong generation plan §7.
- Coverage generation lấy mẫu số là số câu cần chạy; coverage RAGAS lấy mẫu số
  là các câu sinh thành công. Các finalist/baseline trong deck đều được chấm đủ.
- Phase 2C là khảo sát sau held-out. Phase 3 có confound khi đổi cả schema và
  chỉ thị đáp án; deck không suy ra tác động riêng của schema.

### Ứng dụng: cấu hình đích và hiện trạng

Đối chiếu code khi dựng deck: chat có nhánh locked sparse + reranker khi artifact
sẵn sàng; câu trả lời và citation được đưa về giao diện, có nút View Sources.
Tuy nhiên `_run_rag_pipeline` gọi `RAGAgent.run(question)` mà chưa truyền P2;
`OpenAILLM.DEFAULT_SYSTEM_PROMPT` khớp P0. `CHAT_MODEL` mặc định `gpt-4o-mini`
nhưng có thể bị runtime override. Không đọc hay suy đoán giá trị bí mật/runtime
đang dùng. Vì vậy slide 18 nêu phần còn phải nối, không nói app đã được nghiệm thu
với P2/Gemini. Sơ đồ ứng dụng là sơ đồ chức năng từ code, không phải screenshot
hay output một lần chạy thật. Task này chỉ sửa slide và công cụ dựng/kiểm tra.

## Build

Cần Python 3 và XeLaTeX/latexmk (TeX Live có Beamer, TikZ, fontspec,
TeX Gyre, microtype, booktabs, tabularx, colortbl):

```bash
make -C docs/latex slides
```

Hoặc dùng Tectonic (nền XeTeX; tự tải gói ở lần đầu):

```bash
make -C docs/latex slides-tectonic
make -C docs/latex slides-tectonic TECTONIC=/path/to/tectonic
```

Cả hai sinh lại `common/numbers.tex` trước khi build `slides/main.pdf`. Không
cần gọi model/API để dựng slide. Font Windows Times New Roman / Arial / Consolas
được giữ nếu có; Linux tự dùng TeX Gyre Termes / Heros / Cursor. Không dùng pdfLaTeX.

```text
docs/latex/
├── common/preamble.tex       font và macro chung
├── common/numbers.tex        sinh tự động, không sửa tay
├── figures/hcmus-logo.png    logo gốc
├── slides/main.tex           điểm vào Beamer, 20 frame
├── slides/theme.tex          palette, bảng, thẻ số, sơ đồ và chân trang
├── slides/sections/01..20*   nội dung từng slide
├── slides/main.pdf           bản trình chiếu
└── Makefile
```

## Số liệu và biểu đồ

Mọi số thực nghiệm trong slide dùng `\Num...` từ
[`scripts/build_latex_numbers.py`](../../scripts/build_latex_numbers.py).
Thanh biểu đồ và nhãn điểm dùng cùng bộ sinh: macro `...Plot` giữ số thập phân
thô cho hình học; macro hiển thị dùng dấu phẩy và bốn chữ số. Corpus/split,
ngưỡng coverage và AC screening được đọc từ bảng báo cáo/test plan đã commit
khi không có artifact JSON tương ứng trong checkout; bảng nguồn phía trên ghi rõ.

Thêm số vào `collect()` rồi sinh lại, không gõ tay vào `.tex`. Tên model, phase,
metric, nhãn cấu hình và hằng số kiểm định không phải kết quả thực nghiệm.
Phần số cũ phục vụ báo cáo vẫn được giữ; task không dựng lại báo cáo LaTeX.

## Kiểm tra

```bash
python3 -m pytest -q tests/test_latex_numbers.py
uv run --with pymupdf python scripts/check_latex_slides.py
# Xuất ảnh từng trang để duyệt hình học và dấu tiếng Việt:
uv run --with pymupdf python scripts/check_latex_slides.py --render-dir /tmp/newslens-review
```

Bộ kiểm tra PDF yêu cầu build log bên cạnh PDF, kiểm đúng 20 trang 16:9, nội dung
quan trọng, số liệu, logo, không có đường dẫn kỹ thuật, chữ không ra ngoài trang,
khoảng cách nội dung–chú thích, và log không có overfull/missing glyph. Cần xem đủ
20 trang: kiểm tra tự động không chứng minh mọi căn hàng/cột hoặc mức độ đọc tốt.

Bản này đã build bằng Tectonic 0.17.0, kiểm tra bằng PyMuPDF 1.28.2 và duyệt ảnh.
Bốn test macro đạt. Không chạy test ứng dụng/API vì không thay code ứng dụng.

## Logo

`figures/hcmus-logo.png` là bản sao nguyên byte của **Logo Đơn.png** do người dùng
cung cấp, hiển thị giữ tỷ lệ. SHA-256:
`4fb92754801610cb32ffcc2f2c3a05ed15118168784d5b3936913c097803115f`.
