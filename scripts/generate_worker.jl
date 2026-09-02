#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# generate_worker.jl — shared core for generating dimension-reduced GKP-LDLC
# instances. Usable directly as a CLI (see flags below) or `include`d by
# generate_local.jl to call `run_worker`/`generate_for_n` in-process.
#
# Two-phase design (CVP is not safely interruptible):
#   Phase A  save the reduced instance immediately (distances_status="skipped",
#            with cheap Babai upper bounds), so an externally-`timeout`ed CVP
#            still leaves a valid, finishable instance on disk.
#   Phase B  compute exact distances and re-save the same file ("ok").
#
# Flags:
#   --n <int|a:b>        number of modes, single or inclusive range (required)
#   --d <int>            LDLC degree d (default 4)
#   --attempts <int>     max LDLC draws per n (default 200)
#   --instances <int>    target reducible instances per n (default 5)
#   --seed <int>         base seed / task id (default 0); also filename id base
#   --outdir <dir>       output directory (default "data/generated")
#   --reduction kz|lll   CVP pre-reduction (default kz)
#   --skip-distances     Phase A only (leave instances for --distances-only)
#   --distances-only     recompute distances for saved instances not "ok"
# ─────────────────────────────────────────────────────────────────────────────

using SymplecticGKP
using Random
using Printf

function parse_n_range(s::AbstractString)
    if occursin(":", s)
        a, b = split(s, ":")
        return parse(Int, a):parse(Int, b)
    end
    v = parse(Int, s)
    return v:v
end

"""
    generate_for_n(n; d, attempts, instances, base_seed, outdir, skip_distances, reduction)

Generate up to `instances` reducible instances for `n` modes. Returns the
number of instances saved.
"""
function generate_for_n(n::Int; d::Int, attempts::Int, instances::Int,
                        base_seed::Int, outdir::AbstractString,
                        skip_distances::Bool, reduction::Symbol,
                        gen::Function = (nn) -> trivial_gkp_from_ldlc(d, nn))
    mkpath(outdir)
    saved = 0
    for attempt in 1:attempts
        saved >= instances && break
        seed_i = hash((base_seed, n, attempt))
        Random.seed!(seed_i)

        local M
        try
            M = gen(n)
        catch e
            @warn "generator failed" n attempt exception = (e, catch_backtrace())
            continue
        end

        A = gkp_gram(M)
        r = is_single_qubit_reducible(A)     # subsumes the even-|det| prefilter
        r === nothing && continue
        red = reduce_to_qubit(M; C = r.C, d = r.d)

        id = attempt                          # numeric ⇒ matches downstream regex
        path = joinpath(outdir, instance_filename(n, id))

        # Phase A: save immediately with Babai upper bounds.
        upper = babai_upper_bounds(red)
        inst0 = build_instance(; n = n, d = d, seed = seed_i,
                               task_id = base_seed, classical_generator = M, red = red,
                               distances = nothing, upper = upper, status = "skipped")
        save_instance(path, inst0)
        saved += 1
        @printf("[n=%d attempt=%d] saved %s  status=skipped  dX≤%.4f dY≤%.4f dZ≤%.4f\n",
                n, attempt, basename(path), upper[:X], upper[:Y], upper[:Z])
        flush(stdout)

        # Phase B: exact distances, re-save.
        if !skip_distances
            distances = logical_distances(red; reduction = reduction)
            inst1 = build_instance(; n = n, d = d, seed = seed_i,
                                   task_id = base_seed, classical_generator = M, red = red,
                                   distances = distances, upper = upper,
                                   timings = distances.timings, status = "ok")
            save_instance(path, inst1)
            @printf("[n=%d attempt=%d] distances  dX=%.4f dY=%.4f dZ=%.4f  (t=%.1fs)\n",
                    n, attempt, distances.dX, distances.dY, distances.dZ,
                    sum(values(distances.timings)))
            flush(stdout)
        end
    end
    return saved
end

"""
    complete_distances!(outdir; reduction, n_filter=nothing)

`--distances-only` mode: recompute exact distances for every saved instance
whose `distances_status != "ok"`, re-saving in place. The reduction is
recomputed deterministically from the stored `classical_generator`.
"""
function complete_distances!(outdir::AbstractString; reduction::Symbol,
                             n_filter::Union{Nothing,Int} = nothing)
    completed = 0
    for path in list_instances(outdir; n = n_filter)
        inst = load_instance(path)
        (inst.distances_status isa AbstractString && inst.distances_status == "ok") && continue
        M = inst.classical_generator
        A = gkp_gram(M)
        r = is_single_qubit_reducible(A)
        if r === nothing
            @warn "instance no longer reducible; skipping" path
            continue
        end
        red = reduce_to_qubit(M; C = r.C, d = r.d)
        distances = logical_distances(red; reduction = reduction)
        upper = Dict(:X => inst.dX_upper, :Y => inst.dY_upper, :Z => inst.dZ_upper)
        newinst = build_instance(; n = inst.n, d = inst.d, seed = inst.seed,
                                 task_id = inst.task_id, classical_generator = M, red = red,
                                 distances = distances, upper = upper,
                                 timings = distances.timings, status = "ok")
        save_instance(path, newinst)
        completed += 1
        @printf("[completed] %s  dX=%.4f dY=%.4f dZ=%.4f\n",
                basename(path), distances.dX, distances.dY, distances.dZ)
        flush(stdout)
    end
    return completed
end

"""
    run_worker(args::Vector{String}) -> nothing

Parse CLI `args` and dispatch. See the header for flags.
"""
function run_worker(args::Vector{<:AbstractString})
    opts = Dict{String,String}()
    flags = Set{String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--skip-distances", "--distances-only")
            push!(flags, a)
            i += 1
        elseif startswith(a, "--")
            i + 1 <= length(args) || error("missing value for $a")
            opts[a] = args[i + 1]
            i += 2
        else
            error("unexpected argument: $a")
        end
    end

    haskey(opts, "--n") || error("--n is required")
    ns          = parse_n_range(opts["--n"])
    d           = parse(Int, get(opts, "--d", "4"))
    attempts    = parse(Int, get(opts, "--attempts", "200"))
    instances   = parse(Int, get(opts, "--instances", "5"))
    base_seed   = parse(Int, get(opts, "--seed", "0"))
    outdir      = get(opts, "--outdir", joinpath("data", "generated"))
    reduction   = Symbol(get(opts, "--reduction", "kz"))
    skip_dist   = "--skip-distances" in flags
    dist_only   = "--distances-only" in flags

    if dist_only
        for n in ns
            c = complete_distances!(outdir; reduction = reduction, n_filter = n)
            @printf("n=%d: completed %d instance(s)\n", n, c)
        end
        return nothing
    end

    for n in ns
        s = generate_for_n(n; d = d, attempts = attempts, instances = instances,
                           base_seed = base_seed, outdir = outdir,
                           skip_distances = skip_dist, reduction = reduction)
        @printf("n=%d: saved %d instance(s) into %s\n", n, s, outdir)
    end
    return nothing
end

# Run as a CLI only when executed directly (not when `include`d).
if abspath(PROGRAM_FILE) == @__FILE__
    run_worker(ARGS)
end
