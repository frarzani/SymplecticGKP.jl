# `data/`

Two clearly separated kinds of data:

- **`paper/`** — *frozen, version-controlled* data that produced the figures in the
  paper. Do not overwrite it; it is the reproducibility record. See
  [`paper/README.md`](paper/README.md).

- **`generated/`** — *scratch* output for anyone running the package for their own
  interest. The generation scripts (`scripts/generate_*.jl`,
  `scripts/voronoi_comparison.jl`) write here by default. It is git-ignored (only
  `.gitkeep` is tracked), so nothing you generate here is committed.

The figure notebook `examples/reproduce_paper_figures.ipynb` reads **only** from
`paper/`; the generation and exploration scripts read/write **only** `generated/`.
