# ╔══════════════════════════════════════════════════════════════════╗
# ║   MODULE: DimensionReduction                                     ║
# ║                                                                  ║
# ║   Single-qubit dimensionality reduction of a trivial GKP-LDLC    ║
# ║   code, following paper_quantum_ldlc/main_new.tex §"Quantum      ║
# ║   almost-low-density lattice codes".                             ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
    DimensionReduction

Reduce a trivial GKP code `M` (from a classical LDLC) whose symplectic Gram
matrix `A = M J Mᵀ` has invariant factors `(1,…,1,2ℓ)` down to a **single
logical qubit**.

The canonical form `F = C A Cᵀ` (via [`symplectic_basis_over_ZZ`](@ref)) is
block-diagonal with the non-unit factor `2ℓ` on the last symplectic pair —
rows `(2n-1, 2n)` of the canonical generator `Mcan = C M` in this package's
adjacent-pair layout. Dividing that pair by a balanced factorisation `k₁k₂=ℓ`
yields `μ̃, ν̃` with `μ̃ᵀJν̃ = 2`; replacing rows `(2n-1,2n)` of `Mcan` gives a
generator of determinant `±2` (one qubit). The logicals are `XL = μ̃/2`,
`ZL = ν̃/2`, `YL = XL + ZL`.
"""
module DimensionReduction

using LinearAlgebra: det
import Primes
using ..SymplecticBasisZZ: symplectic_basis_over_ZZ, extract_invariant_factors
using ..GKPLattices: symplectic_form

export QubitReduction, is_single_qubit_reducible, balanced_divisor_split,
       reduce_to_qubit

"""
    QubitReduction

Result of [`reduce_to_qubit`](@ref). Fields (all exact):

- `M::Matrix{BigInt}`                 — the original GKP generator (`2n×2n`)
- `Mcan::Matrix{BigInt}`              — canonical generator `C M`
- `C::Matrix{BigInt}`                 — unimodular congruence (`C A Cᵀ = F`)
- `invariant_factors::Vector{BigInt}` — `(1,…,1,2ℓ)`
- `ell::BigInt`                       — `ℓ = dₙ ÷ 2`
- `k1::BigInt`, `k2::BigInt`          — balanced split, `k1·k2 = ℓ`, `k1 ≤ k2`
- `qubit_generator::Matrix{Rational{BigInt}}` — determinant `±2` generator
- `mu_tilde`, `nu_tilde`              — reduced pair, `μ̃ᵀJν̃ = 2`
- `XL`, `YL`, `ZL::Vector{Rational{BigInt}}` — logical representatives
"""
struct QubitReduction
    M::Matrix{BigInt}
    Mcan::Matrix{BigInt}
    C::Matrix{BigInt}
    invariant_factors::Vector{BigInt}
    ell::BigInt
    k1::BigInt
    k2::BigInt
    qubit_generator::Matrix{Rational{BigInt}}
    mu_tilde::Vector{Rational{BigInt}}
    nu_tilde::Vector{Rational{BigInt}}
    XL::Vector{Rational{BigInt}}
    YL::Vector{Rational{BigInt}}
    ZL::Vector{Rational{BigInt}}
end

"""
    is_single_qubit_reducible(A) -> nothing | (; C, d, F)

Test whether the GKP code with symplectic Gram matrix `A` reduces to a single
qubit. Computes the integer symplectic canonical form `F = C A Cᵀ` and the
invariant factors `d`. Returns `nothing` unless `d = (1,…,1,2ℓ)` (all-but-last
unit, last even), else the (`BigInt`) congruence `C`, factors `d`, and `F`.

Pure-Julia replacement for the Sage `check_matrix_reducible`.
"""
function is_single_qubit_reducible(A::AbstractMatrix{<:Integer})
    F, C = symplectic_basis_over_ZZ(BigInt.(A))
    d = extract_invariant_factors(F)
    (all(isone, @view d[1:end-1]) && iseven(d[end])) || return nothing
    return (C=Matrix{BigInt}(C), d=Vector{BigInt}(d), F=Matrix{BigInt}(F))
end

# All positive divisors of `m`, sorted ascending.
function _divisors(m::Integer)
    M = BigInt(m)
    M >= 1 || throw(ArgumentError("divisors require m ≥ 1 (got $m)"))
    divs = BigInt[1]
    for (p, e) in Primes.factor(M)
        divs = BigInt[dv * BigInt(p)^k for dv in divs for k in 0:e]
    end
    return sort!(divs)
end

"""
    balanced_divisor_split(ℓ::Integer) -> (k1::BigInt, k2::BigInt)

Split `ℓ = k1·k2` as balanced as possible: `k1` is the divisor of `ℓ` closest
to `√ℓ` (minimising `|log k1 − ½ log ℓ|`), and `k2 = ℓ ÷ k1`, returned with
`k1 ≤ k2`. Exact replacement for the mwe's JuMP/GLPK `factor_partition` (which
optimised the same log-balance objective over divisors of `ℓ`); exhaustive
enumeration is exact and divisor counts here are tiny.
"""
function balanced_divisor_split(ℓ::Integer)
    L = BigInt(ℓ)
    L >= 1 || throw(ArgumentError("ℓ must be ≥ 1 (got $ℓ)"))
    L == 1 && return (BigInt(1), BigInt(1))
    divs = _divisors(L)
    target = log(L) / 2
    k1 = divs[argmin(abs.(log.(Float64.(divs)) .- target))]
    k2 = L ÷ k1
    return k1 <= k2 ? (k1, k2) : (k2, k1)
end

"""
    reduce_to_qubit(M; C, d[, F]) -> QubitReduction

Core single-qubit reduction. `M` is the trivial GKP generator (`2n×2n`
integer); `C`, `d` (and optionally `F`) come from
[`is_single_qubit_reducible`](@ref).

Layout note (verified against Sage's `reduce`): in this package's canonical
form the non-unit invariant factor `2ℓ` sits on rows `(2n-1, 2n)` of
`Mcan = C M` (Sage used rows `(n, 2n)`). Only that one symplectic pair is
touched. Runtime assertions guard every algebraic identity (exact arithmetic).
"""
function reduce_to_qubit(M::AbstractMatrix{<:Integer};
                         C::AbstractMatrix{<:Integer},
                         d::AbstractVector{<:Integer})
    r, cc = size(M)
    @assert r == cc && iseven(r) "M must be square of even size (got $(size(M)))"
    n = r ÷ 2
    @assert all(isone, @view d[1:end-1]) && iseven(d[end]) "invariant factors must be (1,…,1,2ℓ)"

    Mb   = Matrix{BigInt}(M)
    Cb   = Matrix{BigInt}(C)
    Mcan = Cb * Mb
    J    = symplectic_form(n; T=BigInt)

    # --- assertions on the canonical pair structure --------------------
    # Mcan Gram = C A Cᵀ = F: each pivot pair (2k-1,2k) carries d[k]; the
    # non-unit factor is on the last pair (2n-1, 2n).
    for (k, pair) in enumerate(1:2:(2n - 1))
        prod = transpose(Mcan[pair, :]) * J * Mcan[pair + 1, :]
        @assert prod == d[k] "canonical pivot pair $pair has product $prod ≠ d[$k]=$(d[k])"
    end
    @assert transpose(Mcan[2n - 1, :]) * J * Mcan[2n, :] == d[end] "last pivot pair ≠ dₙ"

    # --- balanced split of ℓ and the reduced pair ----------------------
    ℓ = BigInt(d[end]) ÷ 2
    k1, k2 = balanced_divisor_split(ℓ)
    @assert k1 * k2 == ℓ "balanced split invalid: $k1 * $k2 ≠ $ℓ"

    μ̃ = Rational{BigInt}.(Mcan[2n - 1, :]) .// k1
    ν̃ = Rational{BigInt}.(Mcan[2n, :])     .// k2

    qubit_generator = Rational{BigInt}.(Mcan)
    qubit_generator[2n - 1, :] = μ̃
    qubit_generator[2n, :]     = ν̃

    XL = μ̃ .// 2
    ZL = ν̃ .// 2
    YL = XL .+ ZL

    # --- exact validity assertions (mwe VERBOSE prints → asserts) ------
    Jr = symplectic_form(n; T=Rational{BigInt})
    @assert all(isinteger, Mcan * Jr * μ̃) "μ̃ does not commute with all stabilizers"
    @assert all(isinteger, Mcan * Jr * ν̃) "ν̃ does not commute with all stabilizers"
    @assert transpose(μ̃) * Jr * ν̃ == 2 "μ̃ᵀJν̃ ≠ 2"
    @assert transpose(XL) * Jr * ZL == 1 // 2 "XLᵀJZL ≠ 1/2 (logicals must anticommute)"

    Gq = qubit_generator * Jr * transpose(qubit_generator)
    @assert all(isinteger, Gq) "qubit_generator Gram is non-integer"
    dq = extract_invariant_factors(symplectic_basis_over_ZZ(BigInt.(numerator.(Gq)))[1])
    @assert all(isone, @view dq[1:end-1]) && dq[end] == 2 "qubit Gram factors ≠ (1,…,1,2): $dq"
    @assert abs(det(qubit_generator)) == 2 "|det(qubit_generator)| ≠ 2"

    return QubitReduction(Mb, Mcan, Cb, Vector{BigInt}(d), ℓ, k1, k2,
                          qubit_generator, μ̃, ν̃, XL, YL, ZL)
end

end # module DimensionReduction
