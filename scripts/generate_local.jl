#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# generate_local.jl — local run of the GKP-LDLC pipeline on a single machine.
#
# Generates reducible instances for the given n (with distances) and writes them
# under --outdir. Any generate_worker.jl flag is accepted and forwarded.
#
# Usage:
#   julia --project=. scripts/generate_local.jl            # defaults below
#   julia --project=. scripts/generate_local.jl --n 13 --instances 1 --outdir /tmp/gkp
# (default --outdir is data/generated)
#
# ⚠ RUNTIME: generation is dominated by LatticeDecoder's LDLC 4-cycle (loop)
# removal, whose per-attempt convergence is extremely low and n-dependent
# (≈0.03% at n=13, essentially 0 for n ≲ 10). Producing one 4-cycle-free LDLC at
# n=13 takes on the order of minutes; small n (≲10) is impractical. For a fast
# end-to-end check of the reduction/distance/IO machinery, run the test suite's
# synthetic-generator testset instead (`Pkg.test()`).
# ─────────────────────────────────────────────────────────────────────────────

include(joinpath(@__DIR__, "generate_worker.jl"))

# Realistic defaults (n=13 is the smallest routinely-viable size); overridable.
const LOCAL_DEFAULTS = ["--n", "13", "--d", "4",
                        "--attempts", "20", "--instances", "1",
                        "--seed", "0", "--outdir", joinpath("data", "generated"),
                        "--reduction", "kz"]

function main(args::Vector{<:AbstractString})
    # User-supplied flags override defaults (last occurrence wins in the parser).
    run_worker(vcat(LOCAL_DEFAULTS, collect(args)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
