# Paper artifact

`main.tex` is the audited first report based on the eight completed manual
protocols: four fixed-budget rank allocations, two task orders, and seed 42.
It intentionally does not claim a completed benchmark comparison for the
spectral allocation.

Regenerate all tables and figures from the repository root:

```bash
uv run python scripts/build_first_paper.py
```

Compile locally when a TeX distribution is available:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

For Overleaf, upload `main.tex`, `references.bib`, the five `.sty`/`.bst`
files, `generated/*.tex`, and `figures/*.pdf`. The main paper is exactly two
pages before references; methodological detail and secondary figures are in
the appendices.
