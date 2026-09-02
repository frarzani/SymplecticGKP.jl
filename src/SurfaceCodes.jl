# ╔══════════════════════════════════════════════════════════════════╗
# ║   MODULE: SurfaceCodes                                            ║
# ║                                                                    ║
# ║   Rotated rectangular surface-code GKP lattices, with a square or ║
# ║   hexagonal single-mode GKP code at each mode.                    ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
    SurfaceCodes

Rotated **rectangular** surface-code GKP lattices, in the qqpp convention.

Generalizes the square, odd-distance rotated surface code (as built by
`LatticeDecoder.GKP_Surface_Code`) to any `dx × dz` patch, and lets the
single-mode GKP code at each mode be **square** or **hexagonal** (the latter is
the densest single-mode GKP lattice, giving the well-known `3^{1/4}` distance for
the distance-3 code).

Geometry (data qubits row-major on a `dx × dz` grid, `q(i,j)=(i-1)·dz+j`):
bulk weight-4 plaquettes on 2×2 blocks colored by `(i+j)` parity; weight-2
boundary stabilizers (Z on left/right columns, X on top/bottom rows) with parities
derived from the bulk checkerboard so the code closes for any `dx,dz`. The X-logical
is the left column (length `dx ⇒ dX=√(dx/2)` for square GKP), the Z-logical the top
row (length `dz ⇒ dZ=√(dz/2)`). For **odd squares** the stabilizer supports and
logicals reproduce `LatticeAlgorithms.surface_code_*` exactly.

The GKP generator (qqpp, `N=dx·dz` modes) matches `surface_code_M`: `Mq = 2·I_N`
with X-stabilizer rows, `Mp = 2·I_N` with Z-stabilizer rows, both `/√2`, block
`M=[Mq 0; 0 Mp]` (`|det M|=2`, one logical qubit). Concatenating with **hexagonal**
GKP applies a per-mode area-preserving (det-1) symplectic deformation `T` that keeps
the integer symplectic Gram matrix but rescales each mode's distance by
`(2/√3)^{1/2}`.
"""
module SurfaceCodes

using LinearAlgebra: I, det

export rotated_surface_stabilizers, rotated_surface_logicals,
       gkp_surface_generator, hexagonal_deformation, validate_surface_css

qidx(i, j, dz) = (i - 1) * dz + j

"""
    rotated_surface_stabilizers(dx, dz) -> (Xsupports, Zsupports)

Qubit-index supports of the X- and Z-type stabilizers of the rotated `dx × dz`
surface code. Reproduces `LatticeAlgorithms.surface_code_{X,Z}_stabilizers(d)` for
odd `dx=dz=d`.
"""
function rotated_surface_stabilizers(dx::Int, dz::Int)
    Xs = Vector{Vector{Int}}()
    Zs = Vector{Vector{Int}}()
    for i in 1:dx-1, j in 1:dz-1                     # bulk weight-4 plaquettes
        blk = sort([qidx(i, j, dz), qidx(i, j + 1, dz),
                    qidx(i + 1, j, dz), qidx(i + 1, j + 1, dz)])
        push!(iseven(i + j) ? Xs : Zs, blk)
    end
    for i in 1:dx-1                                  # weight-2 Z on left/right columns
        isodd(i)      && push!(Zs, sort([qidx(i, 1, dz),  qidx(i + 1, 1, dz)]))
        isodd(i + dz) && push!(Zs, sort([qidx(i, dz, dz), qidx(i + 1, dz, dz)]))
    end
    for j in 1:dz-1                                  # weight-2 X on top/bottom rows
        iseven(j)      && push!(Xs, sort([qidx(1, j, dz),  qidx(1, j + 1, dz)]))
        iseven(j + dx) && push!(Xs, sort([qidx(dx, j, dz), qidx(dx, j + 1, dz)]))
    end
    return Xs, Zs
end

"""
    rotated_surface_logicals(dx, dz) -> (Xlog, Zlog)

X-logical = left column (length `dx`), Z-logical = top row (length `dz`).
"""
rotated_surface_logicals(dx::Int, dz::Int) =
    ([qidx(i, 1, dz) for i in 1:dx], [qidx(1, j, dz) for j in 1:dz])

# N×N subspace generator: 2·I with each stabilizer support folded onto its min row.
function _subspace_generator(N::Int, supports::Vector{Vector{Int}})
    M = 2 * Matrix{Float64}(I, N, N)
    for S in supports
        m = minimum(S)
        for j in S
            M[m, j] = 1.0
        end
    end
    return M
end

"""
    hexagonal_deformation(N) -> Matrix{Float64}

The `2N×2N` qqpp block matrix that applies, to each mode's `(qᵢ,pᵢ)`, the det-1
symplectic map `H = s·[1 1/2; 0 √3/2]` (`s=(2/√3)^{1/2}`) taking the square
single-mode GKP lattice to the hexagonal one. Being symplectic it preserves the
integer Gram matrix; being non-orthogonal it rescales distances by `s`.
"""
function hexagonal_deformation(N::Int)
    s = sqrt(2 / sqrt(3))
    H = s .* [1.0 0.5; 0.0 sqrt(3) / 2]
    @assert abs(det(H) - 1) < 1e-12 "hexagonal deformation must be symplectic (det 1)"
    Id = Matrix{Float64}(I, N, N)
    return [H[1, 1]*Id H[1, 2]*Id; H[2, 1]*Id H[2, 2]*Id]
end

"""
    gkp_surface_generator(dx, dz; lattice=:square) -> (M, XL, ZL, N)

GKP stabilizer generator `M` (2N×2N, qqpp, `N=dx·dz`) of the rotated `dx × dz`
surface code and its logical column-vectors `XL` (q-displacement) and `ZL`
(p-displacement). `|det M| = 2`.

`lattice=:square` uses square single-mode GKP (distances `√(dx/2)`, `√(dz/2)`);
`lattice=:hexagonal` concatenates with hexagonal single-mode GKP via
[`hexagonal_deformation`](@ref) (distances scaled by `(2/√3)^{1/2}`; the distance-3
square patch then has distance `3^{1/4}`).
"""
function gkp_surface_generator(dx::Int, dz::Int; lattice::Symbol = :square)
    N = dx * dz
    Xs, Zs = rotated_surface_stabilizers(dx, dz)
    Mq = _subspace_generator(N, Xs) ./ sqrt(2)      # q-block ← X-stabilizers
    Mp = _subspace_generator(N, Zs) ./ sqrt(2)      # p-block ← Z-stabilizers
    M = zeros(Float64, 2N, 2N)
    M[1:N, 1:N] .= Mq
    M[N+1:2N, N+1:2N] .= Mp
    Xlog, Zlog = rotated_surface_logicals(dx, dz)
    XL = zeros(Float64, 2N); XL[Xlog] .= 1 / sqrt(2)             # q-displacement
    ZL = zeros(Float64, 2N); ZL[N .+ Zlog] .= 1 / sqrt(2)       # p-displacement
    if lattice === :hexagonal
        T = hexagonal_deformation(N)
        M = M * transpose(T)                        # rows: v → T·v  ⇒  M → M·Tᵀ
        XL = T * XL                                 # column vectors: L → T·L
        ZL = T * ZL
    elseif lattice !== :square
        throw(ArgumentError("lattice must be :square or :hexagonal (got $lattice)"))
    end
    return M, XL, ZL, N
end

# qqpp symplectic form J = [0 I; -I 0].
_J(N) = [zeros(Int, N, N) Matrix{Int}(I, N, N); -Matrix{Int}(I, N, N) zeros(Int, N, N)]

# GF(2) rank via Gaussian elimination.
function _rank_gf2(A::Matrix{Int})
    B = mod.(copy(A), 2); r = 0; nrows, ncols = size(B)
    for c in 1:ncols
        piv = findfirst(i -> i > r && B[i, c] == 1, 1:nrows)
        piv === nothing && continue
        r += 1
        B[[r, piv], :] = B[[piv, r], :]
        for i in 1:nrows
            i != r && B[i, c] == 1 && (B[i, :] .= mod.(B[i, :] .+ B[r, :], 2))
        end
    end
    return r
end

"""
    validate_surface_css(dx, dz) -> NamedTuple

Check the rotated `dx × dz` code is a valid `[[dx·dz, 1, min(dx,dz)]]` CSS code:
correct stabilizer count `dx·dz−1`, all X/Z stabilizers commute (even overlap),
X/Z stabilizers independent over GF(2), logicals commute with all stabilizers and
anticommute with each other, and `|det M| = 2`. `valid` is the conjunction.
"""
function validate_surface_css(dx::Int, dz::Int)
    N = dx * dz
    Xs, Zs = rotated_surface_stabilizers(dx, dz)
    commute = all(iseven(length(intersect(x, z))) for x in Xs, z in Zs)
    nstab = length(Xs) + length(Zs)
    Bx = zeros(Int, length(Xs), N); for (r, S) in enumerate(Xs), j in S; Bx[r, j] = 1; end
    Bz = zeros(Int, length(Zs), N); for (r, S) in enumerate(Zs), j in S; Bz[r, j] = 1; end
    indep = _rank_gf2(Bx) == length(Xs) && _rank_gf2(Bz) == length(Zs)
    M, XL, ZL, _ = gkp_surface_generator(dx, dz)
    J = _J(N)
    comm_XL = all(x -> abs(x - round(x)) < 1e-9, M * J * XL)
    comm_ZL = all(x -> abs(x - round(x)) < 1e-9, M * J * ZL)
    anti = abs(abs(transpose(XL) * J * ZL) - 0.5) < 1e-9
    detok = abs(abs(det(M)) - 2) < 1e-6
    return (commute=commute, nstab=nstab, nstab_expected=N - 1, indep=indep,
            comm_XL=comm_XL, comm_ZL=comm_ZL, anti=anti, detok=detok,
            valid = commute && nstab == N - 1 && indep && comm_XL && comm_ZL && anti && detok)
end

end # module SurfaceCodes
