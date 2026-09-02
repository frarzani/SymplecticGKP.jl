# SymplecticGKP

Pure-Julia tools for multi-mode Gottesman–Kitaev–Preskill (GKP) codes: integer
symplectic canonical forms, invariant-factor (min–max) balancing, and a
**dimension-reduction pipeline** that turns random classical low-density lattice
codes (LDLCs) into single-qubit GKP codes and computes their distances and
logical error probabilities.

Conventions are **qqpp** throughout: `x = (q₁…qₙ, p₁…pₙ)`,
`J = [0 1; -1 0] ⊗ Iₙ`. A GKP generator `M` is `2n×2n` with **rows as stabilizer
generators**; the lattice is the integer row-combinations `aᵀM`; the symplectic
Gram matrix is `A = M J Mᵀ` and the logical dimension is `|det M|`.

This code accompanies the paper by T. Hillmann, J. Eisert, and F. Arzani
([arXiv:2609.03021](https://arxiv.org/abs/2609.03021); see [Citation](#citation));
a longer derivation is in [`docs/technical-note.md`](docs/technical-note.md).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/frarzani/SymplecticGKP.jl")
Pkg.instantiate()
```

The package `[sources]`-depends on
**[LatticeDecoder](https://github.com/timohillmann/LatticeDecoder.jl)** (used only
for classical-LDLC *generation*), which transitively pulls **Oscar/Nemo** — the
**first precompile is heavy** (many minutes). Instantiate once up front; on a
cluster do it on a login node with a shared `JULIA_DEPOT_PATH`:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); using SymplecticGKP'
```

Requires Julia ≥ 1.11 (the `[sources]` blocks require it).

## Reproduce the paper figures

Figure reproduction needs **only the frozen data** in [`data/paper/`](data/paper)
and the **lightweight `examples` environment** (no LatticeDecoder / Oscar):

```bash
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

then open **`examples/reproduce_paper_figures.ipynb`** and run it top to bottom. It
extracts `data/paper/instances.zip`, and regenerates both panels of Figure 1
(distances and logical error probability `P_L`) as vector PDFs. See
[`examples/README.md`](examples/README.md) for all examples.

## Repository layout

```
src/          the package (symplectic canonical forms, balancing, reduction,
              distances, JLD2 I/O, surface-code baselines, Voronoi measure)
examples/     notebooks + a plot script (see examples/README.md); its own light env
scripts/      instance generation (local / SLURM) and the Voronoi comparison
data/paper/   FROZEN data behind the paper figures (committed)
data/generated/  scratch output for your own runs (git-ignored)
docs/         technical note
test/         test suite
```

## Pipeline

1. `trivial_gkp_from_ldlc(d, n)` — `M = √d·H` from `LatticeDecoder.classical_ldlc`
   (`H` a `d`-regular magic-square LDLC), returning an integer `2n×2n` `M`.
2. `gkp_gram(M)` → `A`; `is_single_qubit_reducible(A)` returns the canonical
   congruence `C` and invariant factors `d = (1,…,1,2ℓ)` (or `nothing`).
3. `reduce_to_qubit(M; C, d)` → a `QubitReduction` with a determinant-`±2`
   `qubit_generator` and logical representatives `XL, YL, ZL`.
4. `logical_distances(red)` → distances `dX, dY, dZ` (shortest coset
   representatives via CVP). The three logicals are **labelled by distance**
   (`dX ≤ dZ ≤ dY`; the X/Y/Z assignment is a free Klein-four choice), and the
   reduced generators are reported to match: `mu_tilde = 2·XL`, `nu_tilde = 2·ZL`.
5. `save_instance` / `load_instance` — JLD2 schema-v2 persistence (reads legacy v1 too).

```julia
using SymplecticGKP, Random
Random.seed!(1)
M   = trivial_gkp_from_ldlc(4, 13)           # 26×26 integer GKP generator (slow, see below)
r   = is_single_qubit_reducible(gkp_gram(M)) # nothing unless factors (1,…,1,2ℓ)
red = reduce_to_qubit(M; C=r.C, d=r.d)
ld  = logical_distances(red)                 # (; dX, dY, dZ, short_XL, …)
```

> **Generation cost.** `trivial_gkp_from_ldlc` accepts only **4-cycle-free** LDLCs,
> and LatticeDecoder's loop-removal converges rarely (≈0.03% per attempt at n=13,
> ≈0 for n ≲ 10), so producing one generator takes on the order of **minutes** and
> small `n` is impractical. This is a property of the LDLC construction, not the
> reduction/distance code (which is fast). The frozen `data/paper/` instances save
> you this cost when reproducing the figures.

## Distances / CVP

Distances are **plain lattice norms** (no `√(2π)` scaling). The CVP is solved via
`LatticeAlgorithms.closest_point` (`kz`/`lll` pre-reduction) after exact
`Rational{BigInt}` integer scaling; large canonical entries are optionally
pre-reduced with an exact `BigInt` LLL (`LLLplus`) so the `Float64` step stays
exact. Do **not** use `LatticeAlgorithms.distances` (interleaved convention +
`√(2π)`).

CVP is exponential and not safely interruptible; the generation scripts save each
instance *before* computing distances and bound the CVP phase with `timeout`, so a
killed run leaves valid distance-less instances (finishable with `--distances-only`).

## Surface-code baselines & Voronoi measure

Beyond the minimum distance, a code is characterized by the **Gaussian measure of
the Voronoi cell of its dual lattice** — the *logical error probability* `P_L`
under a Gaussian displacement channel with a nearest-point (CVP) decoder
(`VoronoiMeasure`):

- `fine_lattice(qubit_generator, mu_tilde//2, nu_tilde//2; exact=true)` builds the
  exact dual for the skewed **LDLC** codes; `fine_lattice(M, XL, ZL)` builds the
  `Float64` dual (`Λ = J A⁻¹ M`) for well-conditioned generators.
- `logical_error_rate(fl, σ; nsamples, tcap)` is a time-boxed Monte-Carlo of `P_L`;
  `union_bound_logical_error` is the leading-term (`σ→0`) asymptote.

`SurfaceCodes` builds the baselines: `gkp_surface_generator(dx, dz;
lattice=:square|:hexagonal)` is the rotated `dx×dz` surface code with square or
hexagonal single-mode GKP at each mode. Note the **CVP cost is dramatically higher
on the LDLC dual lattices** (huge orthogonality defect) than on the surface-code
duals: feasible to `n≈16–17` for LDLC but fast to `n=25` for surface codes;
reaching `n~50` for LDLC needs a faster decoder (e.g. message passing), not exact CVP.

## Generating new data

New instances and comparison data are written to `data/generated/` (git-ignored):

- `scripts/generate_local.jl` — small local run / smoke test.
- `scripts/generate_worker.jl` — shared CLI core (`--n a:b --d 4 --attempts …
  --instances … --seed … --outdir … [--reduction kz|lll] [--skip-distances]
  [--distances-only]`).
- `scripts/generate_slurm.sh` — SLURM array job (one `n` per task), CVP guarded by `timeout`.
- `scripts/voronoi_comparison.jl` — logical-error-probability comparison vs. the
  surface-code baselines (writes `data/generated/voronoi_comparison.csv`; plotted by
  `examples/plot_voronoi_comparison.jl`). The paper's fine-σ CSV lives in `data/paper/`.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Covers the BigInt genericization, the reduction layout/validity, the balanced
divisor split, distances vs. brute force, JLD2 round-trips (v1 + v2), and a smoke run.

## License

GPL-3.0 — see [`LICENSE`](LICENSE). © 2026 Francesco Arzani, Timo Hillmann.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the software and the
accompanying paper, [arXiv:2609.03021](https://arxiv.org/abs/2609.03021).
