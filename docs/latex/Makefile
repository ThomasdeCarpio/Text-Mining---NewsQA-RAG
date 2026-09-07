# Build both documents. Numbers are regenerated first, so a stale figure in a
# PDF is not a possible failure mode.
#
#   make          -- regenerate numbers, build slides and report
#   make slides   -- slides only
#   make report   -- report only
#   make clean    -- remove build artifacts, keep the PDFs
#   make cleanall -- remove the PDFs too

LATEXMK := latexmk -xelatex -interaction=nonstopmode
NUMBERS := common/numbers.tex

.PHONY: all slides report numbers clean cleanall

all: slides report

numbers:
	cd ../.. && python scripts/build_latex_numbers.py

slides: numbers
	cd slides && $(LATEXMK) main.tex

report: numbers
	cd report && $(LATEXMK) main.tex

clean:
	cd slides && latexmk -c
	cd report && latexmk -c

cleanall:
	cd slides && latexmk -C
	cd report && latexmk -C
