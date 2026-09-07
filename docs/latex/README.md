# Tài liệu LaTeX — slide và báo cáo

Hai tài liệu nộp, dùng chung một preamble và **một nguồn số duy nhất**.

```
docs/latex/
├── common/
│   ├── preamble.tex     font, màu, macro dùng chung cho cả hai
│   └── numbers.tex      SINH TỰ ĐỘNG — đừng sửa tay
├── slides/main.tex      beamer 16:9, sections/01..07
├── report/main.tex      article A4, sections/01..05
├── figures/             hình chèn tay (hiện chưa có)
└── Makefile
```

## Build

```bash
cd docs/latex
make            # sinh lại số, build cả hai
make slides     # chỉ slide  →  slides/main.pdf
make report     # chỉ báo cáo →  report/main.pdf
```

Không có `make` thì chạy tay:

```bash
python scripts/build_latex_numbers.py      # từ gốc repo
cd docs/latex/slides && latexmk -xelatex main.tex
cd ../report        && latexmk -xelatex main.tex
```

**Phải dùng XeLaTeX**, không phải pdfLaTeX — preamble dùng `fontspec` để lấy
font có đủ dấu tiếng Việt.

## Nguyên tắc: không gõ số vào `.tex`

Mọi con số trong cả hai tài liệu đến từ `common/numbers.tex`, do
[`scripts/build_latex_numbers.py`](../../scripts/build_latex_numbers.py) đọc
thẳng từ artifact đã commit:

| Nguồn | Cho gì |
| :--- | :--- |
| `docs/reports/phase1/paired_significance.json` | khoảng cách và CI95 vòng 1 |
| `docs/reports/phase1/contextual_chunking_ablation.json` | ablation contextual chunking |
| `docs/reports/phase2/provenance/retrieval_lock.json` | cấu hình khóa, số vòng 3 |
| `docs/reports/phase2/paired_significance.json` | guardrail, phân tầng, macro |
| `docs/reports/phase2/phase2b_winner_decision.json` | winner và mốc ký duyệt |
| `docs/reports/phase2/heldout/` | toàn bộ số held-out |
| `docs/reports/phase2/scores/*.jsonl` | Hit@k theo câu, trung bình dev |

Thí nghiệm chạy lại → chạy lại script → hai PDF tự khớp. **Sửa số trực tiếp
trong `.tex` là cách chắc chắn nhất để hai tài liệu lệch nhau.**

Thêm một con số mới: thêm vào `collect()` trong script, rồi dùng
`\NumTênMacro` trong `.tex`. Tên macro chỉ được có chữ cái, nên chữ số viết
thành chữ (`\NumPtwodFiveAC` = P2-depth5 Answer Correctness).

## Font

Preamble dùng font hệ thống Windows (Times New Roman / Arial / Consolas) vì
chúng luôn có sẵn trên máy build và phủ đủ dấu tiếng Việt. Build trên Overleaf
hay Linux thì đổi trong `common/preamble.tex`:

```tex
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}[Scale=MatchLowercase]
```

## Quan hệ với báo cáo Markdown

Bản Markdown trong `docs/reports/` là tài liệu làm việc — dài, đầy đủ dấu vết.
Bản LaTeX ở đây là bản nộp — gọn, chỉ giữ lập luận và số chốt. Khi hai bên lệch
nhau thì **artifact JSON/CSV mới là trọng tài**, không phải bên nào viết trước.
