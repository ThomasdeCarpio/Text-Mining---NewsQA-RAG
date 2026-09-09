#!/usr/bin/env python3
"""Check the rendered NewsLens deck, including common silent Beamer failures.

Run after building with `uv run --with pymupdf python scripts/check_latex_slides.py`.
Use --render-dir to export every page for visual inspection.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import unicodedata

import pymupdf

from build_latex_numbers import ROOT, collect


def normalized(text: str) -> str:
    # PDF extractors sometimes omit inter-word spaces beside Vietnamese marks.
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "docs/latex/slides/main.pdf")
    parser.add_argument("--render-dir", type=Path)
    args = parser.parse_args()
    numbers = collect()

    def number(key: str) -> str:
        return numbers[key].replace("{,}", ",").replace(r"\%", "%").replace("$-$", "−")

    # Content checks catch a table silently parsed as a hidden frame subtitle.
    expected = [
        ["NEWSLENS", "Lê Thanh Tùng", "Khai thác dữ liệu và văn bản"],
        ["Corpus / distractor", "Evidence span", "Gold chunk", "Gabon"],
        ["Hit Rate@5", "Recall@5", "MRR@5", "nDCG@5", "Answer Correctness",
         "Exact Match", "Token F1", "Faithfulness", "Citation F1",
         "Citation Validity", number("GeneratorModel"), number("JudgeModel")],
        ["Bootstrap", "Guardrail", number("CoverageThreshold"), number("BootstrapSamples")],
        [number(k) for k in ("CorpusArticles", "CorpusChunks", "NResolved", "NDev", "NHeldout", "NReserve")],
        ["Henrik Stenson", "Generator", "Câu trả lời", "64646a6038ad_chunk_0"],
        ["GĐ1", "GĐ2", "Ứng dụng", "P0", "Baseline"],
        [number(k) for k in ("PzeroAC", "PzeroFaith", "PzeroEM", "PzeroFone", "PzeroCitFone", "PzeroCitVal")],
        ["MRR@5", "nDCG@5", "Hit@5", "BGE-M3", number("RoneSparseBgeMrr"),
         number("RoneSparseBge"), number("RoneDenseEfiveMrr"), number("RoneDenseEfive")],
        ["Hybrid", "MiniLM", "MRR@5", "nDCG@5", number("RtwoSparseLargeMrr"),
         number("RtwoSparseLarge"), number("RtwoSparseLargeLatency")],
        [number(k) for k in ("RetrievalFinalQuestions", "RetrievalFinalHitFive", "RetrievalFinalNdcgFive")],
        ["BGE-M3 sparse", "top 20", "BGE reranker", "top 5", "64646a..._0"],
        ["P0", "P1", "P2", "P3", number("ScreenPtwoAC"), number("NJudgeScreening")],
        [number(k) for k in ("LostAtDepthThree", "PtwodThreeAC", "PtwodFiveAC")],
        ["Loại", "Chọn", number("PtwodThreeGuardFaithExact"), number("PtwodFiveGuardFaithExact")],
        [number(k) for k in ("DevAC", "DevEM", "DevFone", "PtwodFiveDeltaAC")],
        [number(k) for k in ("HoAC", "HoFaith", "HoCitFone", "HoCitVal", "WinnerApprovedTime", "HoStartedTime")],
        ["Không có gold", number("HoNMiss"), number("HoHitAC"), number("HoMissAC")],
        ["P2", number("GeneratorModel"), "cấu hình đích"],
        ["View Sources", "mặc định P0", "Nối prompt P2", "gpt-4o-mini", "fallback"],
        ["Khảo sát phụ", "Contextual chunking", "Phase 2C", "B0", number("CtxNSamples")],
        [number("HoAC"), number("RetrievalFinalHitFive"), "Đồng bộ P2", "Giới hạn"],
    ]
    errors: list[str] = []
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as document:
        if len(document) != len(expected):
            errors.append(f"Expected {len(expected)} pages, got {len(document)}")
        for index, page in enumerate(document):
            label = f"Page {index + 1}"
            text = unicodedata.normalize("NFC", page.get_text())
            if abs(page.rect.width / page.rect.height - 16 / 9) > .001:
                errors.append(f"{label}: not 16:9")
            if re.search(r"docs/|provenance/|\.(jsonl?|csv|md|yaml|tex|py)\b", text):
                errors.append(f"{label}: visible technical source path")
            if "\ufffd" in text:
                errors.append(f"{label}: replacement glyph")
            if normalized("NewsLens · Nhóm 9") not in normalized(text):
                errors.append(f"{label}: missing footer")
            if index < len(expected):
                for token in expected[index]:
                    if normalized(token) not in normalized(text):
                        errors.append(f"{label}: missing {token!r}")
            images = page.get_image_info()
            if len(images) != 1 or not page.rect.contains(pymupdf.Rect(images[0]["bbox"])):
                errors.append(f"{label}: missing or clipped logo")
            spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                     for line in block["lines"] for span in line["spans"]]
            for span in spans:
                if not page.rect.contains(pymupdf.Rect(span["bbox"])):
                    errors.append(f"{label}: clipped text {span['text']!r}")
            body = [span["bbox"][3] for span in spans if span["size"] >= 13.4]
            notes = [span["bbox"][1] for span in spans
                     if 11.6 < span["size"] < 12.4 and span["bbox"][1] > 400]
            limit = min(notes) - 6 if notes else 479
            if body and max(body) > limit:
                errors.append(f"{label}: body approaches note/footer ({max(body):.1f} > {limit:.1f} pt)")
            if args.render_dir:
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(args.render_dir / f"{index + 1:02}.png")
    log = args.pdf.with_suffix(".log")
    if not log.exists():
        errors.append("Missing build log; build with --keep-logs before validation")
    else:
        errors.extend(line for line in log.read_text().splitlines()
                      if re.search(r"Overfull|Missing character|^!", line))
    logo = ROOT / "docs/latex/figures/hcmus-logo.png"
    if hashlib.sha256(logo.read_bytes()).hexdigest() != "4fb92754801610cb32ffcc2f2c3a05ed15118168784d5b3936913c097803115f":
        errors.append("The original supplied logo has changed")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(expected)} pages, 16:9, expected text/numbers, original logo, no source paths, "
          "no clipping/overflow or body-note collisions. Visual inspection is still required.")


if __name__ == "__main__":
    main()
