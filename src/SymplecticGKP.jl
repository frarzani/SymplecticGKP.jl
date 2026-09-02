"""
    SymplecticGKP

A Julia package for computing canonical forms of symplectic Gram matrices
over the integers, with applications to multi-mode Gottesman-Kitaev-Preskill
(GKP) quantum error-correcting codes.

# Overview

A GKP code on `n` bosonic modes is specified by a full-rank lattice in
`R^{2n}` whose generator matrix `M` defines the displacement stabilizers.
The symplectic Gram matrix `A = M * J * M'` (anti-symmetric, integer) encodes
the commutation structure of the stabilizer group. Two GKP codes are related
by a Gaussian unitary (symplectic transformation) if and only if their
symplectic Gram matrices are congruent under a unimodular transformation
`A' = U * A * U'`.

This package provides three submodules:

- **`SymplecticBasisZZ`**: Computes the unique Smith-type symplectic canonical
  form `F = C * A * C'` with invariant factors `d₁ | d₂ | ⋯ | dₙ`.
  This is a pure-Julia reimplementation of SageMath's
  `symplectic_basis_over_ZZ`.

- **`BalancedGKP`**: Given invariant factors, finds an alternative diagonal
  decomposition `(d₁', …, dₙ')` that minimizes `max(dⱼ)` while preserving
  the Pfaffian divisors (i.e., remaining in the same congruence class).
  This improves the normal-form distance of the GKP code.

- **`SymplecticGKPIntegration`**: Combines both submodules to compute the
  balanced canonical form and the unimodular transformation matrix in one
  call.

# Quick start

```julia
using SymplecticGKP

# A symplectic Gram matrix with invariant factors (1, 10)
A = [0 0 1 0; 0 0 0 10; -1 0 0 0; 0 -10 0 0]

# Compute the balanced form: (1, 10) → (2, 5), reducing max from 10 to 5
d, U, A_bal = balanced_gkp_form(A)

# Verify: U is unimodular and conjugates A to A_bal
@assert U * A * U' == A_bal
@assert abs(det(U)) == 1

Exported functions

From SymplecticBasisZZ:

    symplectic_basis_over_ZZ(A): canonical form with divisibility.
    extract_invariant_factors(F): read off (d₁, …, dₙ) from F.
    verify_with_smith(A, F): optional check via SmithNormalForm.jl.

From BalancedGKP:

    balanced_diagonal(e): min-max diagonal for given invariant factors.
    pfaffian_divisors(d): compute Pfaffian divisor sequence.
    verify_equivalence(d_old, d_new): check same Pfaffian divisors.

From SymplecticGKPIntegration:

    balanced_gkp_form(A): end-to-end balanced canonical form.
    symplectic_gram(d): build J₂ ⊗ diag(d) from diagonal entries.

Dependencies

    Primes.jl: prime factorization for balanced_diagonal.
    Combinatorics.jl: permutation enumeration for exact optimization.
    SmithNormalForm.jl (optional): independent verification only.

References

    Gottesman, Kitaev, Preskill, "Encoding a qubit in an oscillator", Phys. Rev. A 64, 012310 (2001).
    Conrad, Eisert, Arzani, "Gottesman-Kitaev-Preskill codes: A lattice perspective", Quantum 6, 648 (2022).
    Burchards, Flammia, Conrad, "Fiber bundle fault tolerance of GKP codes", Quantum 9, 1899 (2025).
    Blömer, Xiao, Raissi, Soltan, "Symplectic lattices and GKP codes", arXiv:2509.10183 (2025). """ 


module SymplecticGKP

# Core symplectic canonical form + balancing (self-contained).
include("SymplecticBasisZZ.jl")
include("BalancedGKP.jl")
include("SymplecticGKPIntegration.jl")

# GKP-LDLC dimension-reduction pipeline (depends on LatticeDecoder,
# LatticeAlgorithms, Nemo, JLD2).
include("GKPLattices.jl")
include("DimensionReduction.jl")
include("Distances.jl")
include("InstanceIO.jl")

# GKP code characterization: surface-code baselines + Voronoi/logical-error measure.
include("SurfaceCodes.jl")
include("VoronoiMeasure.jl")

using .SymplecticBasisZZ
using .BalancedGKP
using .SymplecticGKPIntegration
using .GKPLattices
using .DimensionReduction
using .Distances
using .InstanceIO
using .SurfaceCodes
using .VoronoiMeasure

export # SymplecticBasisZZ
       symplectic_basis_over_ZZ,
       extract_invariant_factors,
       verify_with_smith,
       # BalancedGKP
       balanced_diagonal,
       pfaffian_divisors,
       verify_equivalence,
       # SymplecticGKPIntegration
       balanced_gkp_form,
       symplectic_gram,
       # GKPLattices
       symplectic_form,
       gkp_gram,
       dual_generator,
       det_exact,
       is_integer_matrix,
       trivial_gkp_from_ldlc,
       # DimensionReduction
       QubitReduction,
       is_single_qubit_reducible,
       balanced_divisor_split,
       reduce_to_qubit,
       # Distances
       shortest_logical_representatives,
       logical_distances,
       babai_upper_bounds,
       # InstanceIO
       SCHEMA_VERSION,
       build_instance,
       save_instance,
       load_instance,
       list_instances,
       instance_filename,
       # SurfaceCodes
       rotated_surface_stabilizers,
       rotated_surface_logicals,
       gkp_surface_generator,
       hexagonal_deformation,
       validate_surface_css,
       # VoronoiMeasure
       fine_lattice,
       membership,
       logical_error_rate,
       union_bound_logical_error

end