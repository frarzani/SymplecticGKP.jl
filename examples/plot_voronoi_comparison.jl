#!/usr/bin/env julia
# Plot the Voronoi-measure / logical-error-probability comparison.
#
# Reads the CSV written by scripts/voronoi_comparison.jl and draws P_L(σ) for the
# LDLC-GKP codes (circles, cool colors) against the best in-window surface-code
# baselines with HEXAGONAL single-mode GKP (squares, warm colors); the square-GKP
# surface codes are shown faintly for reference. Sample-starved points (the slow
# LDLC dual CVP at large n) are dropped.
#
# Run in the lightweight examples env:
#     julia --project=examples examples/plot_voronoi_comparison.jl [path/to.csv]

import Pkg
Pkg.activate(@__DIR__)
using Plots, Measures, LaTeXStrings
gr()

const CSV_PATH = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(@__DIR__, "..", "data", "generated", "voronoi_comparison.csv")
const OUT_PNG  = joinpath(@__DIR__, "voronoi_measure_plot.png")
const MIN_SAMPLES = 20
const FLOOR = 1e-3
const SHOW_SQUARE = false   # also plot the square single-mode GKP surface codes
                            # (hexagonal is the stronger baseline; off by default)

isfile(CSV_PATH) || error("results CSV not found: $CSV_PATH (run the comparison first)")
lines = readlines(CSV_PATH)
H = split(lines[1], ','); col(name) = findfirst(==(name), H)
ci = (code=col("code"), lat=col("lattice"), n=col("n"), sig=col("sigma"),
      PL=col("P_L"), se=col("P_L_se"), N=col("nsamples"))
pf(s) = s == "" ? NaN : parse(Float64, s)
rows = [split(l, ',') for l in lines[2:end] if !isempty(strip(l))]

curve(pred) = begin
    rs = filter(r -> pred(r) && parse(Int, r[ci.N]) >= MIN_SAMPLES, rows)
    isempty(rs) ? nothing : begin
        o = sortperm(pf.(getindex.(rs, ci.sig)))
        (n=parse(Int, rs[1][ci.n]), σ=pf.(getindex.(rs, ci.sig))[o],
         PL=pf.(getindex.(rs, ci.PL))[o], se=pf.(getindex.(rs, ci.se))[o])
    end
end

Plots.default(size=(1150, 720), margins=8mm)
plt = plot(; yscale=:log10, legend=:bottomright, legendfontsize=8,
           xlabel="Gaussian noise σ", ylabel=L"logical error probability $P_L$",
           title="LDLC-GKP vs. surface-code GKP ($(SHOW_SQUARE ? "hexagonal & square" : "hexagonal") single-mode GKP)\n" *
                 "nearest-point/CVP decoder; <$(MIN_SAMPLES)-sample points dropped",
           titlefontsize=10, left_margin=10mm, bottom_margin=8mm)

# LDLC codes: circles, cool palette by n
ldlc = sort(unique(r[ci.code] for r in rows if r[ci.lat] == "ldlc"))
ns = sort(unique(parse(Int, r[ci.n]) for r in rows if r[ci.lat] == "ldlc"))
pal = cgrad(:cool, max(length(ns), 2); categorical=true)
for c in ldlc
    d = curve(r -> r[ci.code] == c && r[ci.lat] == "ldlc"); d === nothing && continue
    plot!(plt, d.σ, max.(d.PL, FLOOR); yerror=d.se, marker=:circle, ms=6, lw=2.5,
          color=pal[findfirst(==(d.n), ns)], label="LDLC n=$(d.n)")
end

# Hexagonal-GKP surface codes (the strong baseline): warm squares, dashed
hex_codes = sort(unique(r[ci.code] for r in rows if r[ci.lat] == "hexagonal"))
warm = cgrad(:autumn1, max(length(hex_codes), 2); categorical=true)
for (k, c) in enumerate(hex_codes)
    d = curve(r -> r[ci.code] == c && r[ci.lat] == "hexagonal"); d === nothing && continue
    dims = replace(c, "SC_" => "")
    plot!(plt, d.σ, max.(d.PL, FLOOR); yerror=d.se, marker=:square, ms=6, lw=2, linestyle=:dash,
          color=warm[k], label="surface $dims hex-GKP, n=$(d.n)")
end

# Square-GKP surface codes: faint gray diamonds, dotted (reference; off by default)
if SHOW_SQUARE
    for c in sort(unique(r[ci.code] for r in rows if r[ci.lat] == "square"))
        d = curve(r -> r[ci.code] == c && r[ci.lat] == "square"); d === nothing && continue
        dims = replace(c, "SC_" => "")
        plot!(plt, d.σ, max.(d.PL, FLOOR); yerror=d.se, marker=:diamond, ms=4, lw=1, linestyle=:dot,
              color=RGB(0.6,0.6,0.6), label="surface $dims sq-GKP, n=$(d.n)")
    end
end

ylims!(plt, FLOOR * 0.8, 1.0)
savefig(plt, OUT_PNG)
println("saved → ", OUT_PNG)
