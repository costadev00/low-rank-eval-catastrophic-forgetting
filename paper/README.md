# Paper artifact

`main.tex` is the audited first report based on the eight completed manual
protocols: three depth-heavy rank allocations plus a uniform reference, two
task orders, and seed 42.

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
