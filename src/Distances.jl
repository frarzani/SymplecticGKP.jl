# ╔══════════════════════════════════════════════════════════════════╗
# ║   MODULE: Distances                                              ║
# ║                                                                  ║
# ║   Shortest logical-operator representatives (code distances) of  ║
# ║   a single-qubit-reduced GKP-LDLC code, via CVP.                 ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
    Distances

Compute the GKP code distances `dX, dY, dZ` of a [`QubitReduction`](@ref) as
the Euclidean norms of the shortest coset representatives of the logicals
`XL, YL, ZL` modulo the stabilizer lattice `L(qubit_generator)`.

Each distance is a closest-vector problem (CVP). The raw `qubit_generator` is
highly skewed, so we first LLL-reduce it **exactly** over `BigInt` (Nemo/FLINT
`lll`) to a small, well-conditioned basis of the same lattice, then feed that
basis's `Float64` image to `LatticeAlgorithms.closest_point` (`kz`/`lll`
pre-reduction), recover the **integer** lattice coordinates, and reconstruct the
coset representative **exactly** in `Rational{BigInt}`. The `Float64` step only
picks integer coordinates, so its rounding is exact.

Exactness of the pre-reduction is essential: it must span the *same* lattice, or
the reconstructed representatives are not valid cosets. `LLLplus.lll` is **not**
usable here — it multiplies `BigInt` basis entries by a floating `mu`, silently
corrupting the lattice once entries exceed ~`10^77` (which happens for dense
canonical generators at n ≳ 17). Nemo/FLINT `lll` is exact at any magnitude.

**Norm convention.** These are *plain* lattice norms (no `√(2π)` scaling),
matching the mwe / legacy data. This deliberately differs from
`LatticeAlgorithms.distances`, which assumes the interleaved `(q₁p₁q₂p₂…)`
convention and multiplies by `√(2π)`; do **not** use those utilities here.
"""
module Distances

using LinearAlgebra: inv
import LatticeAlgorithms
import Nemo
using ..GKPLattices: symplectic_form
using ..DimensionReduction: QubitReduction

export shortest_logical_representatives, logical_distances, babai_upper_bounds

# Return an exact `Rational{BigInt}` basis `Lbasis` of the qubit stabilizer
# lattice `L(qubit_generator)` that is small and well-conditioned as `Float64`.
#
# The raw `qubit_generator` is highly skewed — two reduced rows of magnitude
# `~1/k` alongside integer canonical rows — so its `Float64` image makes
# `LatticeAlgorithms.{lll,kz}` need enormous unimodular coefficients that
# overflow `Int64`. We therefore always LLL-reduce **exactly** over `BigInt`
# first, yielding a balanced basis of the same lattice with small entries.
# Denominators (`k1` on row `2n-1`, `k2` on row `2n`) are cleared by
# `den = lcm(k1, k2)` before the exact reduction and restored afterwards.
#
# Exactness matters: the reduced basis must span exactly `L(qubit_generator)`,
# else the reconstructed coset representatives are invalid. We use Nemo/FLINT
# `lll` (exact at any magnitude); `LLLplus.lll` cannot be used — it corrupts the
# lattice for `BigInt` entries beyond ~`10^77` (dense canonical generators, n ≳ 17).
function _conditioned_basis(red::QubitReduction)
    G = red.qubit_generator
    den = lcm(red.k1, red.k2)
    Gi = BigInt.(den .* G)                           # rows span den·L(G)
    Zi = Nemo.matrix(Nemo.ZZ, Gi)
    Zr = Nemo.lll(Zi)                                # exact row LLL, same lattice
    Lb = BigInt[BigInt(Zr[i, j]) for i in axes(Gi, 1), j in axes(Gi, 2)]
    @assert abs(Nemo.det(Zr)) == abs(Nemo.det(Zi)) "lattice not preserved by LLL"
    return Lb .// den                                # rational reduced basis of L(G)
end

"""
    shortest_logical_representatives(red::QubitReduction; reduction=:kz)
        -> (; short, distances, timings, qubit_generator_lll)

For each logical `P ∈ {XL, YL, ZL}` of `red`, return the shortest coset
representative (`short[:X]`, `short[:Y]`, `short[:Z]` as `Rational{BigInt}`
vectors), its distance (`distances[:X]…`, `Float64` plain norms), per-CVP wall
times (`timings`), and a `Float64` LLL/KZ-reduced square generator of the qubit
stabilizer lattice (`qubit_generator_lll`) for downstream decoding.

`reduction` selects the `LatticeAlgorithms` pre-processing (`:kz` or `:lll`);
the reduced basis is computed **once** and reused for all three CVPs.
"""
function shortest_logical_representatives(red::QubitReduction;
                                          reduction::Symbol=:kz)
    n = size(red.qubit_generator, 1) ÷ 2
    Lbasis = _conditioned_basis(red)

    Bf = Float64.(Lbasis)
    basis = reduction === :kz ? LatticeAlgorithms.kz(Bf) : LatticeAlgorithms.lll(Bf)
    invBf = inv(Bf)                                    # Float64: small-residual recovery
    invL = inv(Rational{BigInt}.(Lbasis))             # exact: Babai pre-reduction
    Jr = symplectic_form(n; T=Rational{BigInt})

    shorts = Dict{Symbol,Vector{Rational{BigInt}}}()
    dists  = Dict{Symbol,Float64}()
    timings = Dict{String,Float64}()
    for (name, P) in ((:X, red.XL), (:Y, red.YL), (:Z, red.ZL))
        # The canonical generator (hence P) can have astronomically large exact
        # entries. Babai-reduce P into the fundamental domain *exactly* so the
        # Float64 sphere decoder sees an O(1) target and never overflows Int64.
        Pc = round.(BigInt, vec(transpose(P) * invL))
        Pr = P .- vec(transpose(Pc) * Lbasis)             # exact small residual
        t = @elapsed y = LatticeAlgorithms.closest_point(Float64.(Pr), basis)
        coeffs = round.(BigInt, vec(transpose(y) * invBf))    # O(1) integer coords
        shortP = Pr .- vec(transpose(coeffs) * Lbasis)        # exact coset rep of P
        # Validity: a coset representative must commute with all stabilizers.
        @assert all(isinteger, red.M * Jr * shortP) "short $name is not a valid coset representative"
        shorts[name] = shortP
        dists[name] = sqrt(Float64(sum(abs2, shortP)))
        timings[string(name)] = t
    end
    @assert transpose(shorts[:X]) * Jr * shorts[:Z] in (1 // 2, -1 // 2) "short X/Z do not anticommute"

    # Exact reduced generator of the qubit lattice, for downstream decoding.
    Tinv = inv(Rational{BigInt}.(basis.T))                    # unimodular ⇒ integer inverse
    qubit_generator_lll = Float64.(Tinv * Lbasis)

    return (short=shorts, distances=dists, timings=timings,
            qubit_generator_lll=qubit_generator_lll)
end

"""
    logical_distances(red::QubitReduction; kwargs...)
        -> (; dX, dY, dZ, short_XL, short_YL, short_ZL, mu_tilde, nu_tilde,
              timings, qubit_generator_lll)

Convenience wrapper over [`shortest_logical_representatives`](@ref) that
**canonically labels** the three logicals by distance. The X/Y/Z assignment is a
free choice (the three non-identity logicals form a Klein four-group, so any two
multiply to the third), so we relabel them as `X = smallest, Z = median,
Y = largest` ⇒ `dX ≤ dZ ≤ dY`. The reduced-dimension generators are set to match
the new labels: `mu_tilde = 2·XL`, `nu_tilde = 2·ZL` (with the sign fixed so
`mu_tildeᵀ J nu_tilde = 2`), so `XL = mu_tilde/2`, `ZL = nu_tilde/2`,
`YL = XL + ZL` still hold. `qubit_generator`/`qubit_generator_lll` (the stabilizer
lattice) are unaffected by the relabeling.
"""
function logical_distances(red::QubitReduction; kwargs...)
    r = shortest_logical_representatives(red; kwargs...)
    # (reduction rep = P/1 as logical, short rep, distance) for each logical.
    cand = [(red.XL, r.short[:X], r.distances[:X]),
            (red.ZL, r.short[:Z], r.distances[:Z]),
            (red.YL, r.short[:Y], r.distances[:Y])]
    sort!(cand; by = t -> t[3])                 # ascending distance
    (rX, sX, dX) = cand[1]                       # min  → X
    (rZ, sZ, dZ) = cand[2]                       # med  → Z
    (rY, sY, dY) = cand[3]                        # max  → Y  ⇒ dX ≤ dZ ≤ dY

    n = size(red.qubit_generator, 1) ÷ 2
    Jr = symplectic_form(n; T=Rational{BigInt})
    mu = Rational{BigInt}.(2 .* rX)
    nu = Rational{BigInt}.(2 .* rZ)
    # rX, rZ are two of the three Klein-group logicals ⇒ mu'Jnu = ±2; negate nu
    # (same Z coset, since 2·rZ ∈ L(qubit_generator)) to enforce +2.
    if transpose(mu) * Jr * nu == -2
        nu = -nu
    end
    return (dX=dX, dY=dY, dZ=dZ,
            short_XL=sX, short_YL=sY, short_ZL=sZ,
            mu_tilde=mu, nu_tilde=nu,
            timings=r.timings, qubit_generator_lll=r.qubit_generator_lll)
end

"""
    babai_upper_bounds(red::QubitReduction; reduction=:lll)
        -> Dict(:X=>…, :Y=>…, :Z=>…)

Cheap **upper bounds** on the distances: Babai rounding of each logical against
an LLL/KZ-reduced basis (no sphere decoding). Recorded when exact distances are
skipped/timed-out, so partial results still carry a bound.
"""
function babai_upper_bounds(red::QubitReduction; reduction::Symbol=:lll)
    Lbasis = _conditioned_basis(red)
    Bf = Float64.(Lbasis)
    basis = reduction === :kz ? LatticeAlgorithms.kz(Bf) : LatticeAlgorithms.lll(Bf)
    Tinv = inv(Rational{BigInt}.(basis.T))
    Rbasis = Tinv * Lbasis                                    # exact reduced basis
    invR = inv(Rational{BigInt}.(Rbasis))                    # exact (P may be huge)
    ubs = Dict{Symbol,Float64}()
    for (name, P) in ((:X, red.XL), (:Y, red.YL), (:Z, red.ZL))
        coeffs = round.(BigInt, vec(transpose(P) * invR))    # exact Babai rounding
        shortP = P .- vec(transpose(coeffs) * Rbasis)
        ubs[name] = sqrt(Float64(sum(abs2, shortP)))
    end
    return ubs
end

end # module Distances
