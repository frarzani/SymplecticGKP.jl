# Frozen paper data

The exact data behind **Figure 1** of the paper. Regenerate the figures from it
with `examples/reproduce_paper_figures.ipynb`.

## Contents

| File | Feeds | Description |
|------|-------|-------------|
| `instances.zip` | Fig 1(a) | 60 dimension-reduced GKP-LDLC instances (JLD2), 10 each for `n = 15…20`. Unzipped to `instances/` (git-ignored) on first notebook run. |
| `voronoi_sigma_fine.csv` | Fig 1(b) | Monte-Carlo logical-error-probability `P_L(σ)` on a fine σ grid, for the matched LDLC / surface-code pairs. |

## Provenance

**Instances** (`instances.zip`) were produced by the generation pipeline

```
julia --project=. scripts/generate_worker.jl \
    --n 15:20 --d 4 --instances 10 --reduction kz --outdir data/generated
```

i.e. degree-4 classical LDLCs → trivial GKP code → single-qubit dimension
reduction, with distances computed by exact CVP (`kz` pre-reduction). Each JLD2
file stores the reduced generator, the logical representatives, and the three
distances `dX ≤ dZ ≤ dY` (schema v2; see `src/InstanceIO.jl`). Generation is slow
(minutes per accepted generator), which is why the results are frozen here.

**Voronoi CSV** (`voronoi_sigma_fine.csv`) was produced by
`scripts/voronoi_comparison.jl` with the fine noise grid `SIGMAS = 0.18:0.01:0.25`
(the paper's low-noise window), evaluating `P_L` by time-boxed nearest-point (CVP)
Monte-Carlo on the dual lattices. The paper figure plots the matched pairs:

- LDLC `n = 15, 16, 17` (instances `reduced_ldlc_gkp_n_15_5`, `_16_8`, `_17_15`), vs.
- hexagonal-GKP rotated surface codes `SC_3x5` (n=15), `SC_4x4` (n=16), `SC_5x5` (n=25).

The coarse-grid `voronoi_comparison.csv` that the script writes by default (into
`data/generated/`) is *not* part of the paper.
