PYTHON ?= python3
LATEXMK ?= latexmk
TECTONIC ?= tectonic

.PHONY: all numbers slides slides-tectonic
all: slides

numbers:
	$(PYTHON) ../../scripts/build_latex_numbers.py

slides: numbers
	cd slides && $(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error main.tex

# Tectonic uses XeTeX and can build without a system TeX Live installation.
slides-tectonic: numbers
	cd slides && $(TECTONIC) --keep-logs main.tex
