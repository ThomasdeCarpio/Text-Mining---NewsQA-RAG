#!/usr/bin/env python3
"""Validate the standalone draft PDF; optionally render all pages for review.

Run: uv run --no-project --with pymupdf python scripts/check_latex_report.py
No benchmark data, application imports, models, or API calls are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import unicodedata

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/latex/report"


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text)).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=REPORT / "main.pdf")
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    log_path = args.pdf.with_suffix(".log")
    if not log_path.exists():
        errors.append("Build log missing beside PDF")
    else:
        log = log_path.read_text(errors="replace")
        for pattern in (r"Overfull \\[hv]box", r"Missing character:",
                        r"undefined references", r"Citation .+ undefined",
                        r"Reference .+ undefined", r"^!", r"Label .+ multiply defined"):
            if re.search(pattern, log, re.M):
                errors.append(f"Build log matches {pattern}")
    with pymupdf.open(args.pdf) as document:
        texts = [page.get_text() for page in document]
        all_text = normalized("\n".join(texts))
        required = ["Lê Thanh Tùng", "Nhóm 9", "Khai thác dữ liệu văn bản và ứng dụng",
                    "Thông tin nhóm và phân công", "Tóm tắt và từ viết tắt",
                    "Danh mục hình và bảng", "Tài liệu tham khảo",
                    "Chương 1.", "Chương 2.", "Chương 3.", "Chương 4.",
                    "Chương 5.", "Chương 6.", "Giai đoạn 1", "Giai đoạn 2", "Khảo sát phụ",
                    "Nguyễn Hiệp Thắng",
                    "Giao thức đánh giá chung", "Cấu hình hệ thống và nhánh áp dụng",
                    "Tóm tắt thiết lập các thực nghiệm chính", "Prompt và cấu hình quan trọng",
                    "Cài đặt và tái lập", "EDA", "0,3559", "0,3239",
                    "0,8112", "0,8015", "0,8164", "0,7157", "0,7350"]
        for token in required:
            if normalized(token) not in all_text:
                errors.append(f"Missing required content: {token}")
        for token in ("P1-BOSUNG", "P2C-KQ", "P3-KQ"):
            if normalized(token) in all_text:
                errors.append(f"Out-of-scope experimental placeholder: {token}")
        if normalized("Khai thác dữ liệu và văn bản") in all_text:
            errors.append("Outdated course name in PDF")
        if len(document) > 30:
            errors.append(f"Draft exceeds final page budget: {len(document)} > 30")
        if len(document) < 10:
            errors.append("Implausibly short report; check missing inputs")
        if "??" in all_text or "\ufffd" in all_text:
            errors.append("Unresolved reference or replacement glyph in PDF")
        # Cover + four frontmatter pages, then main text at Arabic 1.
        if len(document) > 5 and "chương1." not in normalized(texts[5]):
            errors.append("Main content must start on physical page 6")
        if not document[0].get_image_info():
            errors.append("Cover logo missing")
        toc = document.get_toc()
        chapter_titles = ["Giới thiệu đề tài", "Cơ sở lý thuyết và công trình liên quan",
                          "Dữ liệu và quy trình chuẩn bị", "Thiết kế và hiện thực hệ thống",
                          "Thiết kế thực nghiệm và kết quả", "Thảo luận, kết luận và hướng phát triển"]
        for expected in chapter_titles:
            if not any(level == 1 and normalized(title) == normalized(expected) for level, title, _ in toc):
                errors.append(f"Chapter PDF outline missing: {expected}")
        fonts: set[str] = set()
        checked_fonts: set[int] = set()
        for index, page in enumerate(document):
            for font in page.get_fonts():
                if font[0] not in checked_fonts:
                    checked_fonts.add(font[0])
                    if not document.extract_font(font[0])[3]:
                        errors.append(f"Font is not embedded: {font[3]}")
            if abs(page.rect.width - 595.276) > 1 or abs(page.rect.height - 841.89) > 1:
                errors.append(f"Page {index+1}: not A4")
            if len(texts[index].strip()) < 30:
                errors.append(f"Page {index+1}: blank or near blank")
            spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                     for line in block["lines"] for span in line["spans"]]
            for span in spans:
                if not span["text"].strip():
                    continue
                fonts.add(span["font"])
                box = pymupdf.Rect(span["bbox"])
                if not page.rect.contains(box):
                    errors.append(f"Page {index+1}: text outside paper: {span['text'][:40]}")
                # The header sits above the text block and the footer starts below 790 pt.
                # Allow glyph overhang, but catch body/caption text pushed into the bottom margin.
                if index > 0 and 60 <= box.y0 < 790 and box.y1 > 774:
                    errors.append(f"Page {index+1}: body text below 25 mm margin: {span['text'][:40]}")
                # Allow 3 PDF pt for glyph overhang and microtype protrusion at 25 mm margins.
                # Footers are intentionally below the body, but not outside paper.
                if box.y0 < 775 and (box.x0 < 68 or box.x1 > 528):
                    errors.append(f"Page {index+1}: text outside horizontal margins: {span['text'][:40]}")
            for link in page.get_links():
                if link["kind"] == pymupdf.LINK_GOTO and not 0 <= link.get("page", -1) < len(document):
                    errors.append(f"Page {index+1}: broken internal link")
        if not any("Termes" in f or "TimesNewRoman" in f for f in fonts):
            errors.append("Times/Termes font not found")
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(document):
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
                    args.render_dir / f"page-{index+1:02d}.png")
            # Two-page sheets preserve legibility for a full visual pass.
            for start in range(0, len(document), 2):
                sheet = pymupdf.open()
                canvas = sheet.new_page(width=1190.552, height=861.89)
                for slot, index in enumerate(range(start, min(start+2, len(document)))):
                    rect = pymupdf.Rect(slot*595.276, 20, (slot+1)*595.276, 861.89)
                    canvas.show_pdf_page(rect, document, index)
                    canvas.insert_text((slot*595.276+15, 14), f"PDF page {index+1}", fontsize=10)
                canvas.get_pixmap(matrix=pymupdf.Matrix(1.25, 1.25), alpha=False).save(
                    args.render_dir / f"spread-{start+1:02d}-{min(start+2,len(document)):02d}.png")
                sheet.close()
        print(f"PDF: {len(document)} pages; fonts: {', '.join(sorted(fonts))}")
    if errors:
        raise SystemExit("\n".join(dict.fromkeys(errors)))
    print("PASS: A4, page budget, required sections, results and remaining placeholders, embedded fonts, margins, links, logo and build log")


if __name__ == "__main__":
    main()
