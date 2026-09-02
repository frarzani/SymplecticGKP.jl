# ╔══════════════════════════════════════════════════════════════════╗
# ║   MODULE: VoronoiMeasure                                          ║
# ║                                                                    ║
# ║   Logical error probability of a GKP code under a Gaussian        ║
# ║   displacement channel = Gaussian measure of the dual-lattice     ║
# ║   Voronoi cell, via a nearest-point (CVP) decoder.                ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
    VoronoiMeasure

Characterize a single-logical-qubit GKP code by the **Gaussian measure of the
Voronoi cell of its dual (fine) lattice** — equivalently, the *logical error
probability* `P_L` under an isotropic Gaussian displacement channel with a
nearest-point (closest-vector) decoder. A richer figure of merit than the minimum
distance: it integrates over all directions, weighted by probability.

For a code with stabilizer lattice `S` and logicals `XL, ZL` (`XLᵀJZL=±1/2`), the
fine lattice `Λ = dual(S)` carries all logical+stabilizer displacements, with
`|Λ/S| = 4`. For a Gaussian sample `ξ ~ N(0, σ²I₂ₙ)` the decoder returns
`p = closest_point(ξ, Λ)`; the logical class is read off `p mod S` by symplectic
pairing with `XL, ZL`. Then

* **single-cell measure** `P(p == 0)` (measure of `V_Λ(0)`), and
* **logical error** `P_L = P(p ∉ S)`, split into `P_X, P_Y, P_Z`,

are estimated by Monte-Carlo ([`logical_error_rate`](@ref)). A leading-term
[`union_bound_logical_error`](@ref) from the coset minima gives the `σ→0` asymptote.

Two fine-lattice builders are provided by [`fine_lattice`](@ref):

* an **exact** path for the skewed LDLC codes (from `qubit_generator, mu_tilde,
  nu_tilde`), reconstructing the closest point in `Rational{BigInt}`;
* a **Float64** path for well-conditioned surface-code generators (from a GKP
  generator `M` and logicals `XL, ZL`), using the general integer-Gram dual
  `Λ = J A⁻¹ M` so it works for square *and* hexagonal single-mode GKP.

The decoder is nearest-point (hard-decision) CVP: standard and well-defined, but
maximum-likelihood only approximately (true ML sums Gaussian weight over each
coset).
"""
module VoronoiMeasure

import LatticeAlgorithms
import Nemo
using LinearAlgebra: inv, dot, norm, det
using Random: AbstractRNG, default_rng, randn
using ..GKPLattices: dual_generator, symplectic_form

export fine_lattice, membership, logical_error_rate, union_bound_logical_error

# ─── fine-lattice representations ─────────────────────────────────────────────

"""
    ExactFineLattice

Fine (dual) lattice of a skewed LDLC-GKP code, with an exact `Rational{BigInt}`
reduced basis for exact closest-point reconstruction and coset classification.
"""
struct ExactFineLattice
    n::Int
    Bexact::Matrix{Rational{BigInt}}          # exact reduced row-basis of Λ
    basis                                     # LatticeAlgorithms reduced basis (TLQ)
    invBf::Matrix{Float64}
    XL::Vector{Rational{BigInt}}
    ZL::Vector{Rational{BigInt}}
    J::Matrix{Rational{BigInt}}
    q::Rational{BigInt}                       # XLᵀ J ZL = ±1/2
end

"""
    FloatFineLattice

Fine (dual) lattice of a well-conditioned (surface-code) GKP generator, in
`Float64`. Classification uses `Float64` symplectic pairings (values are clean
half-integers, well separated).
"""
struct FloatFineLattice
    n::Int
    basis                                     # LatticeAlgorithms reduced basis (TLQ)
    XL::Vector{Float64}
    ZL::Vector{Float64}
    J::Matrix{Float64}
    q::Float64
end

# Exact `Rational{BigInt}` reduced basis of the lattice spanned by the rows of
# rational `G`: clear denominators, reduce exactly with Nemo/FLINT LLL (exact at
# any magnitude, unlike LLLplus/float LLL), restore the scale.
function _conditioned_basis(G::AbstractMatrix{<:Union{Integer,Rational}})
    Gr  = Rational{BigInt}.(G)
    den = reduce(lcm, denominator.(Gr); init = BigInt(1))
    Gi  = BigInt.(den .* Gr)
    Zr  = Nemo.lll(Nemo.matrix(Nemo.ZZ, Gi))
    @assert abs(Nemo.det(Zr)) == abs(Nemo.det(Nemo.matrix(Nemo.ZZ, Gi))) "LLL changed the lattice"
    Lb  = BigInt[BigInt(Zr[i, j]) for i in axes(Gi, 1), j in axes(Gi, 2)]
    return Lb .// den
end

"""
    fine_lattice(M, XL, ZL; exact=false, reduction=:kz)

Fine (dual) lattice of a GKP stabilizer generator `M` (2n×2n, qqpp) with logical
column-vectors `XL, ZL` (`XLᵀJZL = ±1/2`), for use with [`membership`](@ref) /
[`logical_error_rate`](@ref).

* `exact=false` (default) → `FloatFineLattice`: well-conditioned generators
  (surface codes). Uses the integer-Gram dual `Λ = J A⁻¹ M` (`A = M J Mᵀ`), exact
  for any `M` with integer symplectic Gram — including **hexagonal** single-mode
  GKP (irrational entries, integer Gram).
* `exact=true` → `ExactFineLattice`: the highly skewed LDLC-GKP duals, where the
  `Float64` sphere decoder needs an exactly-`BigInt`-LLL-reduced basis and exact
  `Rational{BigInt}` closest-point reconstruction. For LDLC instances pass the
  reduced stabilizer generator as `M` and the logicals as `XL = mu_tilde//2`,
  `ZL = nu_tilde//2`.
"""
function fine_lattice(M::AbstractMatrix, XL::AbstractVector, ZL::AbstractVector;
                      exact::Bool = false, reduction::Symbol = :kz)
    n = size(M, 1) ÷ 2
    if exact
        Λ = dual_generator(Matrix{Rational{BigInt}}(M))
        Bexact = _conditioned_basis(Λ)
        Bf = Float64.(Bexact)
        basis = reduction === :kz ? LatticeAlgorithms.kz(Bf) : LatticeAlgorithms.lll(Bf)
        XLr = Vector{Rational{BigInt}}(XL); ZLr = Vector{Rational{BigInt}}(ZL)
        J = symplectic_form(n; T = Rational{BigInt})
        q = transpose(XLr) * J * ZLr
        @assert abs(q) == 1 // 2 "XLᵀJZL must be ±1/2 (got $q)"
        return ExactFineLattice(n, Bexact, basis, inv(Bf), XLr, ZLr, J, q)
    else
        Mf = Matrix{Float64}(M)
        Jf = Float64.(symplectic_form(n))
        A  = Mf * Jf * transpose(Mf)
        Aint = round.(BigInt, A)
        @assert maximum(abs.(A .- Float64.(Aint))) < 1e-6 "symplectic Gram must be integer"
        Λ = Jf * Float64.(inv(Rational{BigInt}.(Aint))) * Mf     # J A⁻¹ M
        @assert abs(det(Λ)) > 1e-9 "fine generator is singular"
        basis = reduction === :kz ? LatticeAlgorithms.kz(Λ) : LatticeAlgorithms.lll(Λ)
        XLf = Vector{Float64}(XL); ZLf = Vector{Float64}(ZL)
        q = dot(XLf, Jf * ZLf)
        @assert abs(abs(q) - 0.5) < 1e-8 "XLᵀJZL must be ±1/2 (got $q)"
        return FloatFineLattice(n, basis, XLf, ZLf, Jf, q)
    end
end

# ─── membership: one CVP + coset classification ──────────────────────────────

_class(a, b) = a == 0 ? (b == 0 ? :I : :Z) : (b == 0 ? :X : :Y)

"""
    membership(fl, ξ) -> (is_zero::Bool, class::Symbol)

Decode `ξ` on the fine lattice (one CVP) and classify the closest point modulo the
stabilizer lattice: `is_zero` marks `p == 0` (single-cell event); `class ∈
{:I,:X,:Y,:Z}` is the logical class (`:I` = no logical error).
"""
function membership(fl::ExactFineLattice, ξ::Vector{Float64})
    y      = LatticeAlgorithms.closest_point(ξ, fl.basis)
    coeffs = round.(BigInt, vec(transpose(y) * fl.invBf))     # integer lattice coords
    p      = vec(transpose(coeffs) * fl.Bexact)               # exact closest point ∈ Λ
    is_zero = all(iszero, p)
    a = mod(round(BigInt, (transpose(fl.ZL) * fl.J * p) / fl.q), 2)   # X-bit
    b = mod(round(BigInt, (transpose(fl.XL) * fl.J * p) / fl.q), 2)   # Z-bit
    return is_zero, _class(a, b)
end

function membership(fl::FloatFineLattice, ξ::Vector{Float64})
    y = LatticeAlgorithms.closest_point(ξ, fl.basis)
    is_zero = norm(y) < 1e-6
    a = mod(round(Int, dot(fl.ZL, fl.J * y) / fl.q), 2)
    b = mod(round(Int, dot(fl.XL, fl.J * y) / fl.q), 2)
    return is_zero, _class(a, b)
end

# ─── Monte-Carlo estimator (time-boxed) ──────────────────────────────────────

"""
    logical_error_rate(fl, σ; nsamples=3000, tcap=Inf, rng=default_rng())
        -> (; P_single, P_single_se, P_L, P_L_se, P_X, P_Y, P_Z, cvp_per_s, secs, nsamples)

Monte-Carlo estimate at noise `σ`: draws up to `nsamples` isotropic Gaussian
displacements (stopping after `tcap` seconds — the LDLC dual-lattice CVP slows
sharply with `n`/`σ`), and returns the single-cell measure `P_single = P(p==0)`,
the logical error probability `P_L = P(p∉S)` with per-Pauli split, binomial
standard errors, and timing. `nsamples` in the result is the actual count taken.
"""
function logical_error_rate(fl, σ::Real; nsamples::Integer = 3000,
                            tcap::Real = Inf, rng::AbstractRNG = default_rng())
    n = fl.n
    zero_hits = 0; err = 0; eX = 0; eY = 0; eZ = 0; done = 0
    t0 = time()
    for i in 1:nsamples
        ξ = randn(rng, 2n) .* σ
        is_zero, class = membership(fl, ξ)
        is_zero && (zero_hits += 1)
        if class !== :I
            err += 1
            class === :X && (eX += 1)
            class === :Y && (eY += 1)
            class === :Z && (eZ += 1)
        end
        done = i
        (time() - t0) > tcap && break
    end
    t = time() - t0
    se(k) = sqrt(max(k, 1) / done * (1 - max(k, 1) / done) / done)
    return (P_single = zero_hits / done, P_single_se = se(zero_hits),
            P_L = err / done, P_L_se = se(err),
            P_X = eX / done, P_Y = eY / done, P_Z = eZ / done,
            cvp_per_s = done / t, secs = t, nsamples = done)
end

"""
    union_bound_logical_error(σ, dX, dZ, dY) -> Float64

Leading-term union bound `Σ_c Q(d_c/2σ)` (multiplicity 1) on `P_L`, with
`Q(x)=½·erfc(x/√2)`. The `σ→0` asymptote; it undercounts once secondary Voronoi
shells matter (moderate σ). `erfc` is the Numerical-Recipes rational approximation
(|error| ≲ 1.2e-7), so no `SpecialFunctions` dependency is needed.
"""
function union_bound_logical_error(σ::Real, dX::Real, dZ::Real, dY::Real)
    Q(x) = 0.5 * _erfc(x / sqrt(2))
    return Q(dX / (2σ)) + Q(dZ / (2σ)) + Q(dY / (2σ))
end

function _erfc(x::Real)
    z = abs(float(x)); t = 1.0 / (1.0 + 0.5z)
    ans = t * exp(-z * z - 1.26551223 +
          t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
          t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
          t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? ans : 2.0 - ans
end

end # module VoronoiMeasure
