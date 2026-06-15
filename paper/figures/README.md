# figures/

Place figure files here. Required figures referenced in `main.tex`:

| Filename | \label | Description |
|----------|--------|-------------|
| `lab_setup.pdf` (or `.png`) | `fig:lab_setup` | Physical lab layout — 4 seats, 7 cameras, sensor placement |
| `pipeline_overview.pdf` (or `.png`) | `fig:pipeline_overview` | Block diagram of the 3 processing pipelines |

## Guidelines
- Prefer PDF or EPS for vector graphics (lossless scaling in pdflatex)
- PNG is acceptable for screenshots or rendered outputs (≥300 dpi)
- All figures must include a `\Description{}` command in `main.tex` (ACM accessibility requirement)
- Target width: single-column = `\columnwidth`, full-width = `\linewidth`
- Do not commit large rasterised files; keep repo size manageable
