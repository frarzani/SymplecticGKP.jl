# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║   MODULE 3: SymplecticGKPIntegration                             ║
# ║                                                                  ║
# ║   Combines modules 1 and 2 to compute the balanced GKP form     ║
# ║   of a symplectic Gram matrix.                                   ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

module SymplecticGKPIntegration

export balanced_gkp_form, symplectic_gram

using LinearAlgebra: diagm, det, I
using ..SymplecticBasisZZ: symplectic_basis_over_ZZ, extract_invariant_factors
using ..BalancedGKP: balanced_diagonal, verify_equivalence


"""
    symplectic_gram(d::Vector{Int}) -> Matrix{Int}

Build the 2n x 2n symplectic Gram matrix J2 x D
for diagonal entries d = (d1, ..., dn):

    A = [ 0   D ]
        [-D   0 ]
"""
function symplectic_gram(d::Vector{Int})
    n = length(d)
    D = diagm(d)
    Z = zeros(Int, n, n)
    return [Z D; -D Z]
end


"""
    balanced_gkp_form(A::AbstractMatrix{<:Integer})
        -> (d_balanced::Vector{Int}, U::Matrix{Int}, A_balanced::Matrix{Int})

Given a symplectic Gram matrix A (anti-symmetric, integer, 2n x 2n):

1. Compute the Smith-type canonical form (with divisibility).
2. Find the balanced diagonal minimizing max(dj).
3. Compute the unimodular transform U such that U*A*U' = A_balanced.

Returns:
- `d_balanced`: the balanced diagonal entries
- `U`:          unimodular integer matrix (|det| = 1)
- `A_balanced`: the balanced symplectic Gram matrix = J2 x diag(d_balanced)
"""
function balanced_gkp_form(A::AbstractMatrix{<:Integer})
    n_full = size(A, 1)
    @assert n_full % 2 == 0 "Matrix size must be even (got $n_full)"
    n = n_full ÷ 2

    # Step 1: canonical form with divisibility
    F, C1 = symplectic_basis_over_ZZ(A)
    inv_factors = extract_invariant_factors(F)

    # Step 2: balanced diagonal
    d_bal = balanced_diagonal(inv_factors)

    # If already the same, no further work needed
    if d_bal == inv_factors
        return d_bal, C1, F
    end

    # Step 3: build target form and get its canonical decomposition
    A_prime = symplectic_gram(d_bal)
    F2, C2 = symplectic_basis_over_ZZ(A_prime)

    # Sanity check: both must reduce to same canonical form
    @assert F == F2 "Bug: invariant factors don't match after balancing"

    # Step 4: compose transforms using exact integer arithmetic
    # C2 is unimodular integer, so inv(C2) is the adjugate divided by det.
    # Since |det(C2)| = 1, inv(C2) = det(C2) * adjugate(C2) = +/- adjugate.
    d2 = round(Int, det(C2))
    C2_inv = round.(Int, d2 .* inv(C2))
    U = C2_inv * C1

    # Verify
    @assert U * A * U' == A_prime "Bug: U*A*U' != A_balanced"
    @assert abs(round(Int, det(U))) == 1 "Bug: U is not unimodular"

    return d_bal, U, A_prime
end


end # module SymplecticGKPIntegration