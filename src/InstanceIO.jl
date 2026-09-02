# ╔══════════════════════════════════════════════════════════════════╗
# ║   MODULE: InstanceIO                                             ║
# ║                                                                  ║
# ║   Persist reduced GKP-LDLC instances (JLD2 schema v2) and load   ║
# ║   both v2 and legacy v1 files transparently.                     ║
# ╚══════════════════════════════════════════════════════════════════╝

"""
    InstanceIO

Save/load reduced GKP-LDLC code instances.

**Schema v2** stores each field as a top-level JLD2 key (no
`single_stored_object` wrapper): `schema_version, n, d, seed, task_id,
classical_generator, C, invariant_factors, ell, k1, k2, mu_tilde, nu_tilde,
qubit_generator, qubit_generator_lll, XL, YL, ZL, dX, dY, dZ, dX_upper, dY_upper,
dZ_upper, distances_status, timings, versions, created`. Logicals are labelled by
distance (`dX ≤ dZ ≤ dY`); `mu_tilde = 2·XL`, `nu_tilde = 2·ZL`.

Distances are **plain lattice norms** (no `√(2π)`). Filenames follow
`reduced_ldlc_gkp_n_<n>_<id>.jld2` (preserving the downstream regex).

[`load_instance`](@ref) also reads **legacy v1** files (`.jls`/`.jld2` written
via `JLD2.save_object`, a `Dict` under `single_stored_object` with keys
`classical_generator, generator, qubit_generator_lll, XL, YL, ZL, dX, dY, dZ`),
returning the same `NamedTuple` shape with `schema_version = 1` and `missing`
for fields absent in v1.
"""
module InstanceIO

import JLD2
import Pkg
import LinearAlgebra
using Dates: now, format

export SCHEMA_VERSION, build_instance, save_instance, load_instance,
       list_instances, instance_filename

const SCHEMA_VERSION = 2

# Some JLD2 versions serialise BigInt inside Rational{BigInt} as base-62
# strings. Teach JLD2 to reconstruct them (mirrors the downstream experiments).
function JLD2.rconvert(
    ::Type{Rational{BigInt}},
    x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}},
)
    return parse(BigInt, x.num, base=62) // parse(BigInt, x.den, base=62)
end

# Reconstruct legacy values written by a newer JLD2 (Rational{BigInt} as base-62
# strings) and read back through this JLD2 as `ReconstructedMutable`. Mirrors the
# downstream experiments' `_qldlc_value`, including Adjoint/Transpose wrappers
# (whose lazy parent is untyped `Any` and cannot be `map`ped directly).
_rebuild(x::JLD2.ReconstructedMutable{Symbol("Rational{BigInt}"),(:num, :den),Tuple{String,String}}) =
    JLD2.rconvert(Rational{BigInt}, x)
_rebuild(x::LinearAlgebra.Adjoint) = Matrix(_rebuild(parent(x))')
_rebuild(x::LinearAlgebra.Transpose) = Matrix(transpose(_rebuild(parent(x))))
_rebuild(x::AbstractMatrix) = Matrix(map(_rebuild, x))
_rebuild(x::AbstractVector) = map(_rebuild, x)
_rebuild(x::AbstractDict) = Dict(string(k) => _rebuild(v) for (k, v) in pairs(x))
_rebuild(x) = x

"""
    instance_filename(n, id) -> String

Canonical filename `reduced_ldlc_gkp_n_<n>_<id>.jld2`.
"""
instance_filename(n::Integer, id) = "reduced_ldlc_gkp_n_$(n)_$(id).jld2"

# Julia + direct-dependency versions (best-effort; never throws).
function _version_info()
    info = Dict{String,String}("julia" => string(VERSION))
    try
        for (_, p) in Pkg.dependencies()
            if p.is_direct_dep && p.version !== nothing
                info[p.name] = string(p.version)
            end
        end
    catch
        # leave whatever we have
    end
    return info
end

"""
    build_instance(; n, d, seed, task_id, classical_generator, red,
                   distances=nothing, upper=nothing, timings=Dict(),
                   status=distances===nothing ? "skipped" : "ok") -> NamedTuple

Assemble a schema-v2 instance `NamedTuple` from a
[`QubitReduction`](@ref)-like `red` (needs fields `C, invariant_factors, ell,
k1, k2, qubit_generator`). `distances`, when given, is the
`logical_distances` result NamedTuple.
"""
function build_instance(; n::Integer, d::Integer, seed, task_id,
                        classical_generator::AbstractMatrix{<:Integer}, red,
                        distances=nothing, upper=nothing,
                        timings::AbstractDict=Dict{String,Float64}(),
                        status::AbstractString = distances === nothing ? "skipped" : "ok")
    getd(f) = distances === nothing ? missing : getproperty(distances, f)
    getu(k) = upper === nothing ? missing : upper[k]
    qgl = distances === nothing ? missing : distances.qubit_generator_lll
    # Reduced-dimension generators μ̃, ν̃ (the "additional generators"): once
    # distances are known they follow the distance-ordered X/Z labels; before
    # then, fall back to the reduction's own pair.
    mu = distances === nothing ? Vector{Rational{BigInt}}(red.mu_tilde) : distances.mu_tilde
    nu = distances === nothing ? Vector{Rational{BigInt}}(red.nu_tilde) : distances.nu_tilde
    return (
        schema_version = SCHEMA_VERSION,
        n = Int(n), d = Int(d), seed = seed, task_id = task_id,
        classical_generator = Matrix{Int}(classical_generator),
        C = Matrix{BigInt}(red.C),
        invariant_factors = Vector{BigInt}(red.invariant_factors),
        ell = BigInt(red.ell), k1 = BigInt(red.k1), k2 = BigInt(red.k2),
        mu_tilde = mu, nu_tilde = nu,
        qubit_generator = Matrix{Rational{BigInt}}(red.qubit_generator),
        qubit_generator_lll = qgl,
        XL = distances === nothing ? missing : distances.short_XL,
        YL = distances === nothing ? missing : distances.short_YL,
        ZL = distances === nothing ? missing : distances.short_ZL,
        dX = getd(:dX), dY = getd(:dY), dZ = getd(:dZ),
        dX_upper = getu(:X), dY_upper = getu(:Y), dZ_upper = getu(:Z),
        distances_status = String(status),
        timings = Dict{String,Float64}(timings),
        versions = _version_info(),
        created = format(now(), "yyyy-mm-ddTHH:MM:SS"),
    )
end

"""
    save_instance(path, inst::NamedTuple)

Write a schema-v2 instance to `path` (JLD2), one top-level key per field.
"""
function save_instance(path::AbstractString, inst::NamedTuple)
    mkpath(dirname(abspath(path)))
    JLD2.jldopen(path, "w") do f
        for k in keys(inst)
            v = getproperty(inst, k)
            f[String(k)] = v === missing ? "__missing__" : v
        end
    end
    return path
end

_unmiss(v) = v == "__missing__" ? missing : v

function _load_v2(f)
    ks = keys(f)
    pairs = Dict{Symbol,Any}(Symbol(k) => _unmiss(f[k]) for k in ks)
    return NamedTuple(pairs)
end

function _load_v1(obj)
    obj isa AbstractDict || error("legacy v1 payload is not a Dict (got $(typeof(obj)))")
    g(k) = haskey(obj, k) ? _rebuild(obj[k]) : missing
    return (
        schema_version = 1,
        n = missing, d = missing, seed = missing, task_id = missing,
        classical_generator = g("classical_generator"),
        C = missing, invariant_factors = missing,
        ell = missing, k1 = missing, k2 = missing,
        mu_tilde = missing, nu_tilde = missing,
        qubit_generator = g("generator"),
        qubit_generator_lll = g("qubit_generator_lll"),
        XL = g("XL"), YL = g("YL"), ZL = g("ZL"),
        dX = g("dX"), dY = g("dY"), dZ = g("dZ"),
        dX_upper = missing, dY_upper = missing, dZ_upper = missing,
        distances_status = "ok",
        timings = missing, versions = missing, created = missing,
    )
end

"""
    load_instance(path) -> NamedTuple

Load an instance, transparently handling schema v2 and legacy v1 files.
"""
function load_instance(path::AbstractString)
    isfile(path) || error("instance file does not exist: $path")
    return JLD2.jldopen(path, "r") do f
        if haskey(f, "schema_version")
            _load_v2(f)
        elseif haskey(f, "single_stored_object")
            _load_v1(f["single_stored_object"])
        else
            error("unrecognised instance file (no schema_version or single_stored_object): $path")
        end
    end
end

"""
    list_instances(dir; n=nothing) -> Vector{String}

Sorted paths of `reduced_ldlc_gkp_n_*.jld2` files in `dir`, optionally filtered
to a given number of modes `n`.
"""
function list_instances(dir::AbstractString; n::Union{Nothing,Integer}=nothing)
    isdir(dir) || return String[]
    pat = n === nothing ? r"^reduced_ldlc_gkp_n_(\d+)_(\d+)\.jld2$" :
                          Regex("^reduced_ldlc_gkp_n_$(n)_(\\d+)\\.jld2\$")
    files = filter(fn -> occursin(pat, fn), readdir(dir))
    return sort!(joinpath.(dir, files))
end

end # module InstanceIO
