# SymplecticGKP — Technical Note

**A pure-Julia toolkit for constructing, dimension-reducing, and characterizing
multi-mode GKP codes derived from classical low-density lattice codes (LDLCs).**

Package version 0.2.0 · qqpp convention throughout · exact arithmetic where it matters.

---

## 1. Motivation and research context

This package is the *construction-and-characterization* half of a research program
that ports techniques from **classical low-density lattice codes (LDLCs)** into
**bosonic quantum error correction**, specifically **Gottesman–Kitaev–Preskill (GKP)
codes**. The program has two long-term goals:

1. **Message-passing decoding of GKP codes.** Classical LDLCs come with efficient
   belief-propagation-style decoders that operate directly on real-valued lattice
   coordinates. The aim is to carry those decoders over to concatenated
   GKP–stabilizer codes, where exact decoding (a closest-vector problem, CVP) is
   intractable at scale.

2. **New high-distance quantum codes.** The randomized construction of classical
   LDLCs, combined with a **dimensionality-reduction** step that collapses the
   logical dimension, yields families of quantum codes that are *not* obviously
   concatenated codes, but can have high distance and may be amenable to
   message-passing decoding.

A GKP code is, geometrically, a **symplectic lattice**. A classical LDLC is a
lattice with a sparse generator and a randomized, loop-free construction. The
bridge is that a classical generator `H` can be promoted to a GKP generator
`M = √d · H`; the resulting code generally encodes a large logical dimension
`K = |det M|`, which this package then **reduces to a single logical qubit** while
preserving the underlying dense, high-distance lattice. The scientific question is
whether such **dimension-reduced LDLC-GKP codes** are competitive with standard
baselines (surface-code GKP) — measured both by minimum distance and by the more
refined **Voronoi / logical-error-probability** figure of merit — and whether they
scale to mode counts (`n ~ 50`) beyond the reach of exact CVP.

`SymplecticGKP` provides everything needed to build these codes, compute their
distances, characterize their logical error probability under a Gaussian
displacement channel, and compare them head-to-head with surface-code baselines —
all in pure Julia, replacing an earlier Julia+Sage+Python prototype.

---

## 2. Conventions and mathematical background

### 2.1 Phase-space convention

All phase-space vectors use the **qqpp** ordering
`x = (q₁ … qₙ, p₁ … pₙ) ∈ ℝ^{2n}`, with symplectic form

```
J = [ 0   Iₙ ]  =  [0 1; -1 0] ⊗ Iₙ .
    [-Iₙ  0  ]
```

### 2.2 GKP codes as symplectic lattices

A GKP code on `n` modes is a full-rank lattice in `ℝ^{2n}` with **generator matrix
`M` (2n×2n), whose rows are the stabilizer generators**; the stabilizer lattice is
the set of integer row-combinations `aᵀM`. Two derived objects govern everything:

- **Symplectic Gram matrix** `A = M J Mᵀ` — antisymmetric and (for a valid GKP
  code) **integer**; it encodes the commutation structure of the stabilizer group.
- **Logical dimension** `K = |det M|` (the number of encoded qudit levels).

Two GKP codes are related by a **Gaussian unitary** (a symplectic transformation)
**iff their Gram matrices are congruent under a unimodular `U`:** `A' = U A Uᵀ`.
This is why integer symplectic *canonical forms* are the central computational
primitive.

### 2.3 Dual (fine) lattice and logicals

The **symplectic dual** lattice `Λ = dual(S)` has generator
`M⊥ = J A⁻¹ M` (`dual_generator`). It contains all logical + stabilizer
displacements: `L(M⊥) ⊇ L(M)`, and the quotient `Λ/S` is the group of logical
operators. For a single encoded qubit, `|Λ/S| = 4` (the identity plus the three
Paulis `X, Y, Z`). The fine lattice is what a nearest-point decoder decodes onto,
and its Voronoi cell is what the logical-error probability integrates over.

---

## 3. The dimension-reduction pipeline

The core workflow turns a random classical LDLC into a single-qubit GKP code and
measures it. Each step maps to an exported function.

| Step | Function | Produces |
|------|----------|----------|
| 1. Trivial GKP from LDLC | `trivial_gkp_from_ldlc(d, n)` | integer `M = √d·H`, `2n×2n` |
| 2. Gram + reducibility test | `gkp_gram(M)`, `is_single_qubit_reducible(A)` | `A`; congruence `C`, invariant factors `d=(1,…,1,2ℓ)` |
| 3. Reduce to one qubit | `reduce_to_qubit(M; C, d)` | `QubitReduction`: det-`±2` generator, logicals `XL,YL,ZL` |
| 4. Distances (CVP) | `logical_distances(red)` | `dX ≤ dZ ≤ dY`, shortest reps, reduced generator |
| 5. Persist | `save_instance` / `load_instance` | JLD2 schema-v2 record |
| 6. Characterize | `fine_lattice` + `logical_error_rate` | logical error probability `P_L(σ)` |

### 3.1 Trivial construction — `GKPLattices`

`trivial_gkp_from_ldlc(d, n)` builds `M = √d·H` where `H` is a `d`-regular
"magic-square" LDLC check matrix from `LatticeDecoder`. For the default weights the
entries of `√d·H` are exactly `±1` and `±√d` (asserted before rounding to `Int`).
It calls `LatticeDecoder`'s building blocks (`init_p_mat`, `loop_removal!`,
`build_H_mat`) directly rather than `classical_ldlc`, so that **each attempt is a
single loop-removal pass with a caller-controlled retry budget** — only
**4-cycle-free** draws are accepted. Loop-removal convergence is stochastic and
strongly `n`-dependent: essentially zero for very small `n` (≲10), rising with `n`,
so the construction is practical for `n ≳ 13`. It draws from the **global** RNG
(`Random.seed!` for reproducibility).

### 3.2 Integer symplectic canonical form — `SymplecticBasisZZ`

`symplectic_basis_over_ZZ(A)` returns the **Smith-type symplectic canonical form**
`F = C A Cᵀ`, block-diagonal with invariant factors `d₁ | d₂ | ⋯ | dₙ` on the
symplectic pairs, together with the unimodular congruence `C`. This is a pure-Julia
reimplementation of SageMath's `symplectic_basis_over_ZZ`;
`extract_invariant_factors(F)` reads off `(d₁,…,dₙ)`, and `verify_with_smith`
offers an optional independent check.

### 3.3 Min–max balancing — `BalancedGKP`

Within a fixed congruence class (fixed Pfaffian divisors), the invariant-factor
diagonal is not unique. `balanced_diagonal` / `balanced_gkp_form` find an
alternative decomposition that **minimizes `max(dⱼ)`** while preserving the
Pfaffian divisor sequence (`pfaffian_divisors`, `verify_equivalence`). Because the
per-mode logical dimension is bounded by the largest factor, this **reduces the
maximum mode-wise logical dimension** of a GKP code obtained trivially from an LDLC
— the "dimensionality reduction" in the paper's sense — and improves the normal-form
distance. Prime factorization uses `Primes.jl`; exact optimization over the (tiny)
divisor set uses `Combinatorics.jl`.

### 3.4 Single-qubit reduction — `DimensionReduction`

When the invariant factors are `(1, …, 1, 2ℓ)`, the code reduces to **one logical
qubit**. In this package's canonical layout the non-unit factor `2ℓ` sits on the
last symplectic pair — rows `(2n-1, 2n)` of `Mcan = C M`. The reduction:

1. Split `ℓ = k₁·k₂` as balanced as possible (`balanced_divisor_split`: the divisor
   of `ℓ` closest to `√ℓ`).
2. Form `μ̃ = Mcan[2n-1]/k₁`, `ν̃ = Mcan[2n]/k₂`, giving `μ̃ᵀJν̃ = 2`.
3. Replace those two rows to get a `qubit_generator` of **determinant `±2`** (one
   qubit), with logicals `XL = μ̃/2`, `ZL = ν̃/2`, `YL = XL + ZL`.

Every algebraic identity is guarded by runtime assertions in exact
`Rational{BigInt}` arithmetic — stabilizer commutation `Mcan J μ̃ ∈ ℤ`, logical
anticommutation `XLᵀJZL = ±1/2`, and `|det(qubit_generator)| = 2` — so a returned
`QubitReduction` is provably valid. `is_single_qubit_reducible` is the pure-Julia
replacement for Sage's `check_matrix_reducible`.

### 3.5 Distances via CVP — `Distances`

The three code distances are the Euclidean norms of the **shortest coset
representatives** of `XL, YL, ZL` modulo the stabilizer lattice — each a **closest
vector problem**. The raw `qubit_generator` is highly skewed (two rows of magnitude
`~1/k` alongside integer canonical rows), so the pipeline:

1. **Exactly** LLL-reduces the generator over `BigInt` (Nemo/FLINT `lll`), after
   clearing denominators by `lcm(k₁,k₂)` — a small, well-conditioned basis of the
   *same* lattice (`_conditioned_basis`).
2. Babai-reduces each huge logical target into the fundamental domain **exactly**,
   so the `Float64` sphere decoder sees an `O(1)` target.
3. Runs `LatticeAlgorithms.closest_point` (`:kz` or `:lll` pre-reduction) on the
   `Float64` image, recovers **integer** lattice coordinates, and reconstructs the
   coset representative **exactly** in `Rational{BigInt}`. The float step only picks
   integer coordinates, so it is exact.

`logical_distances` canonically labels the logicals by distance (`dX ≤ dZ ≤ dY`;
the X/Y/Z assignment is a free Klein-four choice) and reports matched
reduced-dimension generators `mu_tilde = 2·XL`, `nu_tilde = 2·ZL`.
`babai_upper_bounds` gives cheap distance upper bounds (no sphere decoding) as a
fallback when exact CVP is skipped or times out.

> **Two exactness caveats baked into the code.**
> (a) The pre-reduction **must span the same lattice** or the coset representatives
> are invalid — hence Nemo/FLINT `lll` (exact at any magnitude), *not* `LLLplus.lll`,
> which multiplies `BigInt` entries by a floating `mu` and silently corrupts the
> lattice once entries exceed `~10^77` (dense canonical generators at `n ≳ 17`).
> (b) Distances are **plain lattice norms (no `√(2π)`)**, matching the legacy data;
> do **not** use `LatticeAlgorithms.distances`, which assumes the interleaved
> `(q₁p₁q₂p₂…)` convention and rescales by `√(2π)`.

---

## 4. Code characterization

Distance is a worst-case scalar; two further modules give a **direction-averaged,
probability-weighted** figure of merit and a fair set of baselines.

### 4.1 Voronoi / logical-error measure — `VoronoiMeasure`

The **Gaussian measure of the Voronoi cell of the dual (fine) lattice** equals the
**logical error probability `P_L`** under an isotropic Gaussian displacement channel
decoded by a nearest-point (CVP) decoder. For a Gaussian sample
`ξ ~ N(0, σ²I₂ₙ)`, the decoder returns `p = closest_point(ξ, Λ)`; the logical class
is read off `p mod S` by symplectic pairing with `XL, ZL`. Monte-Carlo
(`logical_error_rate`) estimates:

- the **single-cell measure** `P(p == 0)` (measure of the fundamental Voronoi cell),
- the **logical error** `P_L = P(p ∉ S)`, split into `P_X, P_Y, P_Z`,

with binomial standard errors and a time budget (`tcap`), since the LDLC dual CVP
slows sharply with `n` and `σ`. `union_bound_logical_error` gives the leading-term
(`σ → 0`) asymptote `Σ_c Q(d_c / 2σ)`.

`fine_lattice` provides two builders:

- an **exact** path for the skewed **LDLC** codes (from `qubit_generator, mu_tilde//2,
  nu_tilde//2`; `exact=true`), reconstructing the closest point in `Rational{BigInt}`;
- a **Float64** path for well-conditioned generators (from `M, XL, ZL`), using the
  general integer-Gram dual `Λ = J A⁻¹ M`, which works for **square and hexagonal**
  single-mode GKP even though hexagonal entries are irrational.

Because the estimates are Bernoulli fractions `K/N`, results **pool losslessly**:
independent Monte-Carlo batches combine as `K_total/N_total`, so error bars shrink
as `1/√N` and any run can be resumed to tighten a point further.

### 4.2 Surface-code baselines — `SurfaceCodes`

`gkp_surface_generator(dx, dz; lattice = :square | :hexagonal)` builds the **rotated
rectangular** `dx × dz` surface code as a GKP lattice, with a square or hexagonal
single-mode GKP code at each of the `N = dx·dz` modes. It matches
`LatticeDecoder.GKP_Surface_Code` exactly for odd squares and generalizes to any
rectangle (bulk weight-4 plaquettes colored by `(i+j)` parity, weight-2 boundaries
derived from the bulk checkerboard so the code closes for any `dx, dz`). Distances
are `dX = √(dx/2)`, `dZ = √(dz/2)` for square GKP; **hexagonal** GKP applies a
per-mode area-preserving (det-1) symplectic deformation `hexagonal_deformation` that
keeps the integer Gram matrix but rescales each mode's distance by `(2/√3)^{1/2}`
(giving the well-known `3^{1/4}` distance for the distance-3 code).
`validate_surface_css` checks any patch is a valid `[[dx·dz, 1, min(dx,dz)]]` CSS
code. These are the codes the LDLC-GKP families are benchmarked against.

### 4.3 Persistence — `InstanceIO`

`save_instance` / `load_instance` implement a JLD2 **schema v2** record (each field a
top-level key), storing the generator, congruence, invariant factors, balanced split,
logicals, distances (and Babai upper bounds), a `distances_status`, timings, and
package versions. `load_instance` transparently also reads **legacy v1** files,
returning the same `NamedTuple` shape with `missing` for absent fields. Filenames
follow `reduced_ldlc_gkp_n_<n>_<id>.jld2`.

---

## 5. Batch generation and scripts

- `scripts/generate_local.jl` — small local run / smoke test.
- `scripts/generate_worker.jl` — shared CLI core
  (`--n a:b --d 4 --attempts … --instances … --seed … --outdir …
  [--reduction kz|lll] [--skip-distances] [--distances-only]`).
- `scripts/generate_slurm.sh` — SLURM array job (one `n` per task), with the
  exponential, non-interruptible CVP phase bounded by `timeout`. Because each
  instance is **saved before** its distances are computed, a killed CVP leaves valid
  distance-less instances, finishable later with `--distances-only`.
- `scripts/voronoi_comparison.jl` — logical-error-probability comparison of the LDLC
  codes vs. square/hexagonal, square/rectangular surface-code baselines
  (writes `voronoi_comparison.csv`).
- `examples/plot_voronoi_comparison.jl`, `examples/plot_voronoi_showcase.jl` — figures;
  the *showcase* plot colors matched mode-counts with a shared hue (LDLC solid/filled,
  surface dashed/open), draws capped error bars in the series color, filters
  Poisson-dead points, and restricts to a low-noise window.

---

## 6. Dependencies and implementation notes

- **`LatticeDecoder`** (local path dependency, branch off `origin/submission`) —
  the classical-LDLC construction (`classical_ldlc` and its building blocks) and the
  reference `GKP_Surface_Code`. It transitively pulls **Oscar/Nemo**, so the **first
  precompile is heavy** (many minutes); instantiate once up front (on a login node
  with a shared depot for cluster runs).
- **`LatticeAlgorithms`** (git source, pinned commit) — `closest_point`, `kz`, `lll`.
  Pinned because the current `main` has a duplicate-`include` bug that breaks
  precompilation, and because a path dependency's own `[sources]` are not resolved
  transitively (so the git source is re-declared in this package's `Project.toml`).
- **`Nemo`** (FLINT) — exact `BigInt` LLL and determinants.
- **`JLD2`**, **`Primes`**, **`Combinatorics`**, standard libs.
- Requires **Julia ≥ 1.11** (the `[sources]` block).

**Design principle: exact where correctness depends on it, `Float64` only where it
cannot go wrong.** Canonical forms, reductions, coset reconstruction, and validity
checks are exact (`BigInt` / `Rational{BigInt}`); `Float64` appears only inside the
sphere decoder, where it merely selects integer lattice coordinates that are then
lifted back to exact rationals.

---

## 7. Feasibility, findings, and outlook

**The CVP wall.** The decisive practical constraint is that the **LDLC dual lattice
has an enormous orthogonality defect** (empirically `~10^{11}`), which makes the
sphere-decoder CVP catastrophically slow: throughput drops from tens of samples/s at
`n = 15` to well below `0.1`/s by `n ≈ 19–20`. Exact distances and Voronoi-measure
sampling are therefore feasible only to `n ≈ 16–17` for the LDLC codes, whereas the
better-conditioned surface-code duals stay fast to `n = 25`. Reaching `n ~ 50`
requires a **faster/approximate decoder — message passing** — not more CVP time; this
is precisely goal (1) of the research program and the reason the classical
message-passing machinery is worth porting.

**What the characterization shows.** In the feasible window, the dimension-reduced
LDLC-GKP codes are **competitive with or better than matched surface-code GKP
baselines**, especially at low noise. Benchmarked mode-for-mode against hexagonal-GKP
rotated surface codes, the LDLC codes sit **below** their matched partners (lower
logical error probability), with the advantage widening as `σ` decreases; at the
lowest noise studied, a 15–16 mode LDLC code matches or beats even a 25-mode surface
code. This is driven by the LDLC codes' higher lattice distance and is exactly the
kind of evidence the program seeks that randomized-LDLC + dimension-reduction is a
viable route to strong bosonic codes.

**Outlook.** The package delivers the construction-and-characterization half:
build dimension-reduced LDLC-GKP codes, certify them exactly, and measure both their
distance and their Voronoi/logical-error probability against principled baselines.
The natural next step — decoding these codes at `n ~ 50` via message passing — sits
outside exact CVP and is the subject of ongoing work.

---

## 8. Quick reference

```julia
using SymplecticGKP, Random
Random.seed!(1)

M   = trivial_gkp_from_ldlc(4, 13)            # 26×26 integer GKP generator
r   = is_single_qubit_reducible(gkp_gram(M))  # nothing unless factors (1,…,1,2ℓ)
red = reduce_to_qubit(M; C=r.C, d=r.d)         # single-qubit reduction
ld  = logical_distances(red)                   # (; dX, dY, dZ, short_XL, …)

# logical error probability under a Gaussian displacement channel
fl  = fine_lattice(red.qubit_generator, red.mu_tilde .// 2, red.nu_tilde .// 2; exact=true)
m   = logical_error_rate(fl, 0.20; nsamples=3000, tcap=45.0)   # (; P_L, P_L_se, …)

# surface-code baseline (hexagonal single-mode GKP), same interface
Ms, XL, ZL, _ = gkp_surface_generator(3, 5; lattice=:hexagonal)
fls = fine_lattice(Ms, XL, ZL)                 # Float64 dual path
```

*Generated as a standalone note; not a substitute for the module docstrings, which
carry the authoritative per-function contracts.*
