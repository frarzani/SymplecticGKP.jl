#!/usr/bin/env julia
# ╔══════════════════════════════════════════════════════════════════╗
# ║   Voronoi-measure / logical-error-probability comparison:         ║
# ║   dimension-reduced LDLC-GKP codes  vs.  surface-code GKP         ║
# ║   (square and hexagonal single-mode GKP; square and rectangular  ║
# ║    patches).                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Uses the SymplecticGKP package: `fine_lattice` + `logical_error_rate`
# (VoronoiMeasure) on the LDLC instances and on `gkp_surface_generator`
# (SurfaceCodes) baselines. Writes one flushed CSV row per (code, σ) so the run
# survives interruption. See `examples/plot_voronoi_comparison.jl` to plot.
#
# Run (package env):   julia --project=. scripts/voronoi_comparison.jl
# Tunables are the `const`s below.

using SymplecticGKP
using Random: seed!
using Printf: @printf, @sprintf

# ─── Tunables ────────────────────────────────────────────────────────────────
const INDIR      = joinpath(@__DIR__, "..", "data", "generated")
const OUTCSV     = joinpath(@__DIR__, "..", "data", "generated", "voronoi_comparison.csv")
# Coarse σ sweep (the P_L transition). The paper's Fig 1(b) uses the FINE grid
# 0.18:0.01:0.25 — set SIGMAS to `collect(0.18:0.01:0.25)` to reproduce
# data/paper/voronoi_sigma_fine.csv (rename the output accordingly).
const SIGMAS     = [0.15, 0.18, 0.21, 0.24]
const N_SAMPLES  = 3000                          # MC samples per point (upper bound)
const TIME_CAP   = 45.0                          # …stop after this many seconds/point
const MAX_CODES  = 6                             # cap LDLC instances (0 = all)
const ONE_PER_N  = true                          # keep the best (largest dX) LDLC per n
# Surface-code patches (dx×dz) and per-mode GKP lattices to sweep. Squares 3×3,5×5
# reproduce the classic d=3,5 codes; rectangles fill the 15–20-mode window.
const SC_PATCHES = [(3, 3), (5, 5), (3, 5), (4, 4), (4, 5)]
const SC_LATTICES = [:square, :hexagonal]        # hexagonal = the stronger baseline
const SC_N       = 1200                          # MC samples for the (fast) SC baselines
const RNG_SEED   = 0xC0FFEE

const CSV_HEADER = "code,lattice,n,dX,dZ,dY,sigma,P_single,P_single_se,P_L,P_L_se," *
                   "P_L_union,P_X,P_Y,P_Z,cvp_per_s,nsamples"

# Select the LDLC instances to characterize (best-per-n if ONE_PER_N).
function select_ldlc()
    insts = Any[]
    for f in list_instances(INDIR)
        inst = load_instance(f)
        get(inst, :distances_status, "ok") == "ok" || continue
        inst.dX === missing && continue
        push!(insts, (file = basename(f), inst = inst))
    end
    if ONE_PER_N
        best = Dict{Int,Any}()
        for e in insts
            n = e.inst.n
            (!haskey(best, n) || e.inst.dX > best[n].inst.dX) && (best[n] = e)
        end
        insts = sort(collect(values(best)); by = e -> e.inst.n)
    end
    MAX_CODES > 0 && length(insts) > MAX_CODES && (insts = insts[1:MAX_CODES])
    return insts
end

row(csv, code, lattice, n, dX, dZ, dY, σ, m, ub) =
    println(csv, @sprintf("%s,%s,%d,%s,%s,%s,%.4f,%.6e,%.6e,%.6e,%.6e,%s,%.6e,%.6e,%.6e,%.1f,%d",
        code, lattice, n,
        dX === nothing ? "" : @sprintf("%.6f", dX),
        dZ === nothing ? "" : @sprintf("%.6f", dZ),
        dY === nothing ? "" : @sprintf("%.6f", dY),
        σ, m.P_single, m.P_single_se, m.P_L, m.P_L_se,
        ub === nothing ? "" : @sprintf("%.6e", ub),
        m.P_X, m.P_Y, m.P_Z, m.cvp_per_s, m.nsamples))

function main()
    seed!(RNG_SEED)
    isdir(INDIR) || error("no instance directory: $INDIR")
    csv = open(OUTCSV, "w"); println(csv, CSV_HEADER); flush(csv)

    println("\n=== LDLC-GKP codes ===")
    for e in select_ldlc()
        inst = e.inst
        fl = fine_lattice(inst.qubit_generator, inst.mu_tilde .// 2, inst.nu_tilde .// 2; exact = true)
        @printf("\n[%s] n=%d  dX=%.4f dZ=%.4f dY=%.4f\n", e.file, inst.n, inst.dX, inst.dZ, inst.dY)
        for σ in SIGMAS
            m  = logical_error_rate(fl, σ; nsamples = N_SAMPLES, tcap = TIME_CAP)
            ub = union_bound_logical_error(σ, inst.dX, inst.dZ, inst.dY)
            @printf("  σ=%.3f  P_L=%.3e ± %.1e  P_single=%.3e  UB=%.2e  cvp/s=%.0f  N=%d\n",
                    σ, m.P_L, m.P_L_se, m.P_single, ub, m.cvp_per_s, m.nsamples)
            row(csv, e.file, "ldlc", inst.n, inst.dX, inst.dZ, inst.dY, σ, m, ub)
            flush(csv)
        end
    end

    println("\n=== Surface-code GKP baselines ===")
    for (dx, dz) in SC_PATCHES, lat in SC_LATTICES
        n = dx * dz
        s = sqrt(2 / sqrt(3))
        dXsc = sqrt(dx / 2) * (lat === :hexagonal ? s : 1)
        dZsc = sqrt(dz / 2) * (lat === :hexagonal ? s : 1)
        code = "SC_$(dx)x$(dz)"
        @printf("\n[%s / %s] n=%d  dX=%.4f dZ=%.4f\n", code, lat, n, dXsc, dZsc)
        try
            M, XL, ZL, _ = gkp_surface_generator(dx, dz; lattice = lat)
            fl = fine_lattice(M, XL, ZL)
            for σ in SIGMAS
                m  = logical_error_rate(fl, σ; nsamples = SC_N, tcap = TIME_CAP)
                ub = union_bound_logical_error(σ, dXsc, min(dXsc, dZsc), dZsc)
                @printf("  σ=%.3f  P_L=%.3e ± %.1e  P_single=%.3e  cvp/s=%.0f  N=%d\n",
                        σ, m.P_L, m.P_L_se, m.P_single, m.cvp_per_s, m.nsamples)
                row(csv, code, String(lat), n, dXsc, dZsc, max(dXsc, dZsc), σ, m, ub)
                flush(csv)
            end
        catch err
            @warn "surface baseline failed" dx dz lat exception = err
        end
    end

    close(csv)
    println("\nSaved results CSV → $OUTCSV")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
