# Examples

Four self-contained examples. The three data/plotting notebooks run in the
lightweight `examples` environment (no LatticeDecoder/Oscar); the API tutorial
runs in the package environment.

Instantiate the lightweight environment once:

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

## Reproduce the paper figures

**`reproduce_paper_figures.ipynb`** — regenerates **Figure 1** (both panels) from
the frozen data in [`../data/paper/`](../data/paper): panel (a) the logical
distances of the reduced GKP-LDLC codes, panel (b) their logical error probability
`P_L(σ)` vs. matched surface-code baselines. It extracts `data/paper/instances.zip`
on first run and writes `distance_plot_panel.pdf` / `voronoi_measure_panel.pdf`.
Open it and run top to bottom.

## The other examples

- **`canonical_forms_and_balancing.ipynb`** — API tutorial for the *symplectic
  canonical form / balancing* side of the package: invariant factors, Pfaffian
  divisors, `balanced_gkp_form`, the impact on GKP code distance, and equivalence
  classes. It uses `SymplecticGKP` directly, so it activates the **package
  environment** at the repo root (its first cell does
  `Pkg.activate(dirname(@__DIR__))`).

- **`dimension_reduction_walkthrough.ipynb`** — walks through the
  generate → reduce → distances pipeline and plots distances vs. number of modes.
  It reads instances from `../data/generated/` (produced by the generation
  scripts); point its `OUTDIR` at `../data/paper/instances/` to use the frozen set
  instead.

- **`plot_voronoi_comparison.jl`** — standalone script that plots the *coarse* σ
  comparison CSV (`../data/generated/voronoi_comparison.csv`, written by
  `scripts/voronoi_comparison.jl`). This is exploratory and **not** a paper figure.

  ```bash
  julia --project=examples examples/plot_voronoi_comparison.jl [path/to.csv]
  ```
