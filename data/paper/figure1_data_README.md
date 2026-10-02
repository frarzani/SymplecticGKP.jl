# `figure1_data.csv` — data behind Figure 1

One tidy (long-format) table with every plotted value in Figure 1 of the paper
(arXiv:2609.03021). Each row is one plotted datum; the `panel` column says which
part of the figure it belongs to. All columns are described below.

## Columns

| column | meaning |
|---|---|
| `panel` | `a_distances` (per-instance logical distances, Fig 1a markers), `a_reference_line` (surface-code horizontal reference distances, Fig 1a lines), or `b_error_probability` (logical error probability curves, Fig 1b) |
| `series` | which plotted series the row belongs to (see below) |
| `code_type` | `GKP-LDLC`, `surface-square`, or `surface-hex` |
| `n_modes` | number of bosonic modes `n` |
| `instance` | source instance file (`data/paper/instances.zip`); panel (a) instance rows only |
| `sigma_physical` | **x-axis of panel (b)**: physical displacement std, `sigma_physical = sqrt(2*pi) * sigma_internal` (panel b only) |
| `sigma_internal` | σ in the package's plain, unscaled lattice units (square-GKP stabilizer spacing `sqrt(2)`, no `sqrt(2*pi)`), as stored in `voronoi_sigma_fine.csv` (panel b only) |
| `value` | the plotted y-value: a logical distance (panel a) or `P_L` (panel b) |
| `value_type` | `logical_distance`, `reference_distance`, or `logical_error_probability` |
| `value_se` | standard error of `P_L` (panel b only); used for the error bars |
| `nsamples` | Monte-Carlo sample count behind that `P_L` point (panel b only) |

Empty cells mean "not applicable to this row type".

## Panel (a) — logical distances vs. number of modes

Plot `value` against `n_modes`. Each GKP-LDLC instance contributes **three** points
at the same `n_modes`, distinguished by `series`:
`Delta_X` (○), `Delta_Y` (□), `Delta_Z` (◇). Distances are **plain Euclidean
lattice 2-norms** (no `sqrt(2*pi)` scaling). Within a mode-number group the points
are spread horizontally only for legibility (cosmetic; not stored here).

The four `a_reference_line` rows are the horizontal baselines — distance of a
rotated distance-`d` surface code (`n = d^2`) concatenated with a single-mode GKP
code, `Delta_conc = sqrt(d) * Delta_loc`:

| series | code | formula | value |
|---|---|---|---|
| `Sq-SC_d3_n9`   | square, d=3  | `sqrt(3/2)`     | 1.22474 |
| `Sq-SC_d5_n25`  | square, d=5  | `sqrt(5/2)`     | 1.58114 |
| `Hex-SC_d3_n9`  | hex, d=3     | `3^(1/4)`       | 1.31607 |
| `Hex-SC_d5_n25` | hex, d=5     | `sqrt(5/sqrt(3))` | 1.69904 |

## Panel (b) — logical error probability vs. noise

Plot `value` (`P_L`, log scale) against `sigma_physical`, with error bars
`±value_se`. GKP-LDLC series are drawn solid/filled, surface-code series
dashed/open; matched mode numbers share a colour:

| series | code_type | n_modes |
|---|---|---|
| `LDLC_n15` / `surface_3x5_n15` | GKP-LDLC / surface-hex | 15 |
| `LDLC_n16` / `surface_4x4_n16` | GKP-LDLC / surface-hex | 16 |
| `LDLC_n17` / `surface_5x5_n25` | GKP-LDLC / surface-hex | 17 / 25 |

The rows already have the figure's selection applied: `sigma_internal <= 0.25`,
`nsamples >= 20`, and at least 5 logical-error events per point. The full
(unfiltered) Monte-Carlo table is `voronoi_sigma_fine.csv` in this folder.

## Noise-normalization note

`sigma_internal` is the std applied to the plain (unscaled) lattice; the physical
displacement std is `sigma_physical = sqrt(2*pi) * sigma_internal`. Panel (b) is
plotted in `sigma_physical`, matching the concatenated-GKP decoder figures
(e.g. Fig. 6). Panel (a) distances carry no noise axis.
