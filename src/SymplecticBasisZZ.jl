
# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║   MODULE 1: SymplecticBasisZZ                                    ║
# ║                                                                  ║
# ║   Integer symplectic canonical form via congruence reduction.    ║
# ║   Equivalent to SageMath's symplectic_basis_over_ZZ.             ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

module SymplecticBasisZZ

export symplectic_basis_over_ZZ, extract_invariant_factors, verify_with_smith

using LinearAlgebra: I, det


"""
symplectic_basis_over_ZZ(M::AbstractMatrix{T}) where {T<:Integer}
                            -> (F::Matrix{T}, C::Matrix{T})

Compute the integer symplectic canonical form of an anti-symmetric
alternating integer matrix `M`.

Returns `(F, C)` where:
- `F = C * M * C'` is block-diagonal with 2x2 blocks `[0 dk; -dk 0]`
  and invariant factors `d1 | d2 | ... | dm`
- `C` is unimodular (`|det(C)| = 1`)

# Algorithm

The algorithm performs congruence reduction D -> T*D*T' via three
phases per pivot pair:

1. **Place pivot**: find minimum nonzero entry, move to (k, k+1).
2. **Reduce pivot**: use extended GCD operations so that D[k,k+1]
   divides every entry in the remaining submatrix D[k:n, k:n].
3. **Eliminate**: zero out all entries in rows/cols k and k+1
   beyond the 2x2 pivot block.

This is repeated for k = 1, 3, 5, ... yielding the block-diagonal
form with divisibility d1 | d2 | ... | dm.
"""
function symplectic_basis_over_ZZ(M::AbstractMatrix{T}) where {T<:Integer}
    n = size(M, 1)
    @assert size(M, 2) == n          "Matrix must be square (got $(size(M)))"
    @assert iszero(M + M')           "Matrix must be anti-symmetric"
    @assert all(M[i,i] == 0 for i in 1:n) "Diagonal must be zero"

    # Work in the element type of the input (e.g. `BigInt`). The congruence
    # intermediates can be much larger than the entries of `M`, so callers that
    # risk `Int64` overflow (2n ≳ 100) should pass `BigInt.(M)`.
    D = Matrix{T}(M)
    C = Matrix{T}(I, n, n)

    k = 1
    while k <= n - 1
        _place_pivot!(D, C, k, n) || break
        _reduce_pivot!(D, C, k, n)
        _eliminate!(D, C, k, n)
        k += 2
    end

    return D, C
end


"""
extract_invariant_factors(F::AbstractMatrix{<:Integer}) -> Vector{Int}

Given the canonical form `F` returned by `symplectic_basis_over_ZZ`,
extract the invariant factor sequence `(d1, d2, ...)` with `d1 | d2 | ...`.
"""
function extract_invariant_factors(F::AbstractMatrix{T}) where {T<:Integer}
    n = size(F, 1)
    d = T[]
    for k in 1:2:n
        k + 1 <= n || break
        F[k, k+1] > 0 && push!(d, F[k, k+1])
    end
    return d
end


"""
verify_with_smith(M, F) -> Bool

Optional verification using SmithNormalForm.jl.
For a skew-symmetric matrix, the Smith form has each invariant factor
appearing exactly twice: diag(S) = (d1, d1, d2, d2, ..., 0, ...).
"""
function verify_with_smith(M::AbstractMatrix{<:Integer},
                           F::AbstractMatrix{<:Integer})
    SNF = Base.require(Main, :SmithNormalForm)
    result = SNF.smith(M)
    S = result.S

    n = size(M, 1)
    smith_diag = sort!([abs(S[i, i]) for i in 1:n], rev=true)
    filter!(>(0), smith_diag)

    our_factors = extract_invariant_factors(F)
    our_paired = sort!(vcat(our_factors, our_factors), rev=true)

    return smith_diag == our_paired
end


# ──────────────────────────────────────────────────────────────────
#  Phase 1: Find and place pivot
# ──────────────────────────────────────────────────────────────────

function _place_pivot!(D, C, k, n)
    min_val = 0
    mi, mj = 0, 0
    for i in k:n, j in (i+1):n
        v = abs(D[i, j])
        if v > 0 && (min_val == 0 || v < min_val)
            min_val = v
            mi, mj = i, j
        end
    end
    min_val == 0 && return false

    if mi != k
        _swap!(D, C, mi, k, n)
        mj == k && (mj = mi)
    end
    mj != k + 1 && _swap!(D, C, mj, k + 1, n)

    D[k, k+1] < 0 && _negate!(D, C, k, n)
    return true
end


# ──────────────────────────────────────────────────────────────────
#  Phase 2: Reduce pivot to divide everything in remaining block
# ──────────────────────────────────────────────────────────────────

function _reduce_pivot!(D, C, k, n)
    while true
        d = D[k, k+1]
        changed = false

        # 2a: entries in row k, columns k+2 ... n
        for j in (k+2):n
            if D[k, j] != 0 && D[k, j] % d != 0
                _gcd_reduce!(D, C, k, k+1, j, n)
                D[k, k+1] < 0 && _negate!(D, C, k, n)
                changed = true
                break
            end
        end
        changed && continue

        # 2b: entries in row k+1, columns k+2 ... n
        for j in (k+2):n
            if D[k+1, j] != 0 && D[k+1, j] % d != 0
                _gcd_reduce!(D, C, k+1, k, j, n)
                D[k, k+1] < 0 && _negate!(D, C, k, n)
                changed = true
                break
            end
        end
        changed && continue

        # 2c: entries in remaining submatrix D[k+2:n, k+2:n]
        for i in (k+2):n, j in (i+1):n
            if D[i, j] != 0 && D[i, j] % d != 0
                _add_multiple!(D, C, i, k+1, 1, n)
                D[k, k+1] < 0 && _negate!(D, C, k, n)
                changed = true
                break
            end
        end
        changed && continue

        break
    end
end


"""
GCD-reduce D[pivot_row, col_a] using D[pivot_row, col_b].
Applies a 2x2 unimodular congruence on cols (col_a, col_b) so that
D[pivot_row, col_a] becomes gcd(D[pivot_row, col_a], D[pivot_row, col_b]).
"""
function _gcd_reduce!(D, C, pivot_row, col_a, col_b, n)
    a = D[pivot_row, col_a]
    b = D[pivot_row, col_b]
    g, u, v = gcdx(a, b)
    _apply_2x2!(D, C, col_a, col_b, u, v, -(b ÷ g), a ÷ g, n)
end


# ──────────────────────────────────────────────────────────────────
#  Phase 3: Eliminate off-pivot entries
# ──────────────────────────────────────────────────────────────────

function _eliminate!(D, C, k, n)
    d = D[k, k+1]

    for j in (k+2):n
        D[k, j] == 0 && continue
        _add_multiple!(D, C, k+1, j, -(D[k, j] ÷ d), n)
    end

    for j in (k+2):n
        D[k+1, j] == 0 && continue
        _add_multiple!(D, C, k, j, D[k+1, j] ÷ d, n)
    end
end


# ──────────────────────────────────────────────────────────────────
#  Elementary congruence operations
#
#  Every operation performs D -> T*D*T'  and  C -> T*C
#  for some unimodular T, preserving skew-symmetry of D.
# ──────────────────────────────────────────────────────────────────

"""Swap rows/cols i <-> j."""
function _swap!(D, C, i, j, n)
    @inbounds for col in 1:n
        D[i, col], D[j, col] = D[j, col], D[i, col]
    end
    @inbounds for row in 1:n
        D[row, i], D[row, j] = D[row, j], D[row, i]
    end
    @inbounds for col in 1:n
        C[i, col], C[j, col] = C[j, col], C[i, col]
    end
end


"""Negate row and column i."""
function _negate!(D, C, i, n)
    @inbounds for j in 1:n
        D[i, j] = -D[i, j]
        D[j, i] = -D[j, i]
    end
    @inbounds for j in 1:n
        C[i, j] = -C[i, j]
    end
end


"""
Add c * row/col `from` to row/col `to`.
Corresponds to shearing E = I + c*e_to*e_from'  (det = 1).
"""
function _add_multiple!(D, C, from, to, c, n)
    @inbounds for i in 1:n
        D[i, to] += c * D[i, from]
    end
    @inbounds for j in 1:n
        D[to, j] += c * D[from, j]
    end
    @inbounds for j in 1:n
        C[to, j] += c * C[from, j]
    end
end


"""
Apply 2x2 unimodular congruence on rows/cols (p, q).
T at (p,q) = [Q11 Q12; Q21 Q22],  det(T) = Q11*Q22 - Q12*Q21 = +/-1
Performs D -> T*D*T'  and  C -> T*C.
"""
function _apply_2x2!(D, C, p, q, Q11, Q12, Q21, Q22, n)
    # Column operation: D <- D * T'
    @inbounds for i in 1:n
        op, oq = D[i, p], D[i, q]
        D[i, p] = Q11 * op + Q12 * oq
        D[i, q] = Q21 * op + Q22 * oq
    end
    # Row operation: D <- T * D
    @inbounds for j in 1:n
        op, oq = D[p, j], D[q, j]
        D[p, j] = Q11 * op + Q12 * oq
        D[q, j] = Q21 * op + Q22 * oq
    end
    # Transform C <- T * C
    @inbounds for j in 1:n
        op, oq = C[p, j], C[q, j]
        C[p, j] = Q11 * op + Q12 * oq
        C[q, j] = Q21 * op + Q22 * oq
    end
end

end # module SymplecticBasisZZ
