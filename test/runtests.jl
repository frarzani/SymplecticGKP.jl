using Test
using LinearAlgebra: I, det, diagm, norm
using Random
import JLD2
import LatticeAlgorithms
using SymplecticGKP

# Build an integer GKP generator M = [I S; 0 D] (qqpp blocks) whose symplectic
# Gram matrix is [[0 D]; [-D 0]] with invariant factors = diag(D); S symmetric.
function build_gkp_generator(S::AbstractMatrix{<:Integer}, dvec::Vector{<:Integer})
    n = length(dvec)
    @assert S == S' "S must be symmetric"
    Idn = Matrix{Int}(I, n, n)
    Z = zeros(Int, n, n)
    return [Idn Matrix{Int}(S); Z diagm(Int.(dvec))]
end

# Brute-force shortest coset representative norm over a coefficient box [-R,R]^{2n}.
function brute_distance(P::Vector{Rational{BigInt}},
                        gen::Matrix{Rational{BigInt}}, R::Int)
    m = length(P)
    best = Inf
    for coeffs in Iterators.product(ntuple(_ -> -R:R, m)...)
        v = P .- vec(collect(Rational{BigInt}, coeffs)' * gen)
        best = min(best, sqrt(Float64(sum(abs2, v))))
    end
    return best
end


@testset "SymplecticGKP" begin

    @testset "symplectic_basis_over_ZZ" begin

        # Test 1: 4x4, invariant factors (1, 2)
        M1 = [ 0  3  0  1;
              -3  0  2  0;
               0 -2  0  0;
              -1  0  0  0]
        F1, C1 = symplectic_basis_over_ZZ(M1)
        @test C1 * M1 * C1' == F1
        @test abs(round(Int, det(C1))) == 1
        d1 = extract_invariant_factors(F1)
        @test d1 == [1, 2]

        # Test 2: already block-diagonal (1, 10)
        M2 = [0 0 1 0; 0 0 0 10; -1 0 0 0; 0 -10 0 0]
        F2, C2 = symplectic_basis_over_ZZ(M2)
        @test C2 * M2 * C2' == F2
        @test abs(round(Int, det(C2))) == 1
        d2 = extract_invariant_factors(F2)
        @test d2 == [1, 10]

        # Test 3: uniform (2, 2)
        M3 = [0 0 2 0; 0 0 0 2; -2 0 0 0; 0 -2 0 0]
        F3, C3 = symplectic_basis_over_ZZ(M3)
        @test C3 * M3 * C3' == F3
        @test abs(round(Int, det(C3))) == 1
        d3 = extract_invariant_factors(F3)
        @test d3 == [2, 2]

        # Test 4: 6x6 with (1, 1, 8)
        M4 = zeros(Int, 6, 6)
        M4[1,2] = 1;  M4[2,1] = -1
        M4[3,4] = 1;  M4[4,3] = -1
        M4[5,6] = 8;  M4[6,5] = -8
        F4, C4 = symplectic_basis_over_ZZ(M4)
        @test C4 * M4 * C4' == F4
        @test abs(round(Int, det(C4))) == 1
        d4 = extract_invariant_factors(F4)
        @test d4 == [1, 1, 8]

        # Test 5: dense 4x4, Pfaffian = 29 (prime)
        M5 = [ 0  5  3  0;
              -5  0  0  2;
              -3  0  0  7;
               0 -2 -7  0]
        F5, C5 = symplectic_basis_over_ZZ(M5)
        @test C5 * M5 * C5' == F5
        @test abs(round(Int, det(C5))) == 1
        d5 = extract_invariant_factors(F5)
        @test d5 == [1, 29]

        # Test 6: zero matrix
        M6 = zeros(Int, 4, 4)
        F6, C6 = symplectic_basis_over_ZZ(M6)
        @test F6 == zeros(Int, 4, 4)
        @test abs(round(Int, det(C6))) == 1

        # Test 7: 2x2
        M7 = [0 6; -6 0]
        F7, C7 = symplectic_basis_over_ZZ(M7)
        @test C7 * M7 * C7' == F7
        d7 = extract_invariant_factors(F7)
        @test d7 == [6]

        # Test 8: SIS lattice, q=5, n=2
        q = 5
        H = [0 1; 1 0]
        J4 = [zeros(Int,2,2) Matrix{Int}(I,2,2);
              -Matrix{Int}(I,2,2) zeros(Int,2,2)]
        M_H = [Matrix{Int}(I,2,2) H;
               zeros(Int,2,2) q*Matrix{Int}(I,2,2)]
        A8 = M_H * J4 * M_H'
        F8, C8 = symplectic_basis_over_ZZ(A8)
        @test C8 * A8 * C8' == F8
        @test abs(round(Int, det(C8))) == 1
        d8 = extract_invariant_factors(F8)
        @test d8 == [5, 5]

        # Test 9: larger 8x8 with off-diagonal coupling
        M9 = zeros(Int, 8, 8)
        M9[1,2] =  3; M9[2,1] = -3
        M9[3,4] =  6; M9[4,3] = -6
        M9[5,6] = 12; M9[6,5] = -12
        M9[7,8] = 12; M9[8,7] = -12
        M9[1,4] =  2; M9[4,1] = -2
        M9[2,3] = -2; M9[3,2] =  2
        F9, C9 = symplectic_basis_over_ZZ(M9)
        @test C9 * M9 * C9' == F9
        @test abs(round(Int, det(C9))) == 1
        d9 = extract_invariant_factors(F9)
        @test all(d9[i] != 0 && d9[i+1] % d9[i] == 0 for i in 1:length(d9)-1)

    end

    @testset "balanced_diagonal" begin

        # Two primes, n=2
        d = balanced_diagonal([1, 10])
        @test d == [2, 5]
        @test verify_equivalence([1, 10], d)

        # Uniform stays uniform
        d = balanced_diagonal([2, 2, 2])
        @test d == [2, 2, 2]

        # Cannot improve (1,1,8) — single prime
        d = balanced_diagonal([1, 1, 8])
        @test d == [1, 1, 8]

        # Two primes: 36 = 4*9
        d = balanced_diagonal([1, 36])
        @test d == [4, 9]
        @test verify_equivalence([1, 36], d)

        # 12 = 4*3
        d = balanced_diagonal([1, 12])
        @test d == [3, 4]
        @test verify_equivalence([1, 12], d)

        # Three modes, two primes
        d = balanced_diagonal([1, 2, 18])
        @test verify_equivalence([1, 2, 18], d)
        @test maximum(d) <= 18

        # Single prime cannot be redistributed
        d = balanced_diagonal([1, 1, 27])
        @test d == [1, 1, 27]

        # Large product, two primes: 900 = 2^2 * 3^2 * 5^2
        d = balanced_diagonal([1, 900])
        @test verify_equivalence([1, 900], d)
        @test maximum(d) < 900

        # Trivial cases
        @test balanced_diagonal([1, 1, 1]) == [1, 1, 1]
        @test balanced_diagonal([7]) == [7]

        # Divisibility validation
        @test_throws ArgumentError balanced_diagonal([3, 5])

    end

    @testset "balanced_gkp_form" begin

        # (1, 10) -> (2, 5)
        A1 = [0 0 1 0; 0 0 0 10; -1 0 0 0; 0 -10 0 0]
        d, U, Ab = balanced_gkp_form(A1)
        @test U * A1 * U' == Ab
        @test abs(round(Int, det(U))) == 1
        @test d == [2, 5]

        # (1, 36) -> (4, 9)
        A2 = [0 0 1 0; 0 0 0 36; -1 0 0 0; 0 -36 0 0]
        d, U, Ab = balanced_gkp_form(A2)
        @test U * A2 * U' == Ab
        @test abs(round(Int, det(U))) == 1
        @test d == [4, 9]

        # Already balanced: (5, 5)
        A3 = [0 0 5 0; 0 0 0 5; -5 0 0 0; 0 -5 0 0]
        d, U, Ab = balanced_gkp_form(A3)
        @test d == [5, 5]

        # Dense input
        A4 = [ 0  5  3  0;
              -5  0  0  2;
              -3  0  0  7;
               0 -2 -7  0]
        d, U, Ab = balanced_gkp_form(A4)
        @test U * A4 * U' == Ab
        @test abs(round(Int, det(U))) == 1

        # 6x6 with balanceable factors
        A5 = zeros(Int, 6, 6)
        A5[1,4] =  1; A5[4,1] = -1
        A5[2,5] =  2; A5[5,2] = -2
        A5[3,6] = 18; A5[6,3] = -18
        d, U, Ab = balanced_gkp_form(A5)
        @test U * A5 * U' == Ab
        @test abs(round(Int, det(U))) == 1
        @test verify_equivalence([1, 2, 18], d)

        # symplectic_gram round-trip
        d_in = [3, 7]
        Ag = symplectic_gram(d_in)
        @test Ag == -Ag'
        @test Ag[1,3] == 3
        @test Ag[2,4] == 7

    end

    @testset "BigInt genericization" begin
        # Re-run a couple of core cases as BigInt.
        M4 = zeros(BigInt, 6, 6)
        M4[1,2]=1; M4[2,1]=-1; M4[3,4]=1; M4[4,3]=-1; M4[5,6]=8; M4[6,5]=-8
        F4, C4 = symplectic_basis_over_ZZ(M4)
        @test eltype(F4) == BigInt && eltype(C4) == BigInt
        @test C4 * M4 * C4' == F4
        @test extract_invariant_factors(F4) == BigInt[1, 1, 8]

        # A case that would overflow Int64 in the invariant factors.
        b = big(3)^30                    # 205_891_132_094_649
        Mo = BigInt[0 b 0 0; -b 0 0 0; 0 0 0 (2b); 0 0 (-2b) 0]
        Fo, Co = symplectic_basis_over_ZZ(Mo)
        @test Co * Mo * Co' == Fo
        @test extract_invariant_factors(Fo) == BigInt[b, 2b]
    end

    @testset "symplectic_form / gkp_gram" begin
        J = symplectic_form(2)
        @test J == [0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
        @test symplectic_form(3; T=BigInt) isa Matrix{BigInt}
        M = build_gkp_generator([0 1; 1 0], [1, 2])
        A = gkp_gram(M)
        @test A == -A'
        @test A == [0 0 1 0; 0 0 0 2; -1 0 0 0; 0 -2 0 0]
        @test det_exact(M) == 2
    end

    @testset "balanced_divisor_split" begin
        for ℓ in (1, 2, 6, 12, 7, 49, 60, 210)
            k1, k2 = balanced_divisor_split(ℓ)
            @test k1 * k2 == ℓ
            @test k1 <= k2
            # optimality vs brute force over divisors
            best = minimum(abs(log(dv) - log(ℓ)/2) for dv in 1:ℓ if ℓ % dv == 0)
            @test isapprox(abs(log(Float64(k1)) - log(ℓ)/2), best; atol=1e-9)
        end
    end

    @testset "reduce_to_qubit (layout + validity)" begin
        cases = ([1, 1, 8], [1, 2], [1, 1, 12], [1, 1, 1, 6])
        for dvec in cases
            n = length(dvec)
            S = zeros(Int, n, n); S[1, n] = 1; S[n, 1] = 1   # symmetric shear
            M = build_gkp_generator(S, dvec)
            A = gkp_gram(M)
            r = is_single_qubit_reducible(A)
            @test r !== nothing
            @test r.d == BigInt.(sort(dvec))
            red = reduce_to_qubit(M; C=r.C, d=r.d)               # asserts run inside
            @test red.ell == BigInt(dvec[end]) ÷ 2
            @test red.k1 * red.k2 == red.ell
            Jr = symplectic_form(n; T=Rational{BigInt})
            @test transpose(red.mu_tilde) * Jr * red.nu_tilde == 2
            @test transpose(red.XL) * Jr * red.ZL == 1//2
            @test abs(det(red.qubit_generator)) == 2
            Gq = red.qubit_generator * Jr * transpose(red.qubit_generator)
            @test all(isinteger, Gq)
            dq = extract_invariant_factors(
                    symplectic_basis_over_ZZ(BigInt.(numerator.(Gq)))[1])
            @test all(isone, dq[1:end-1]) && dq[end] == 2
        end
    end

    @testset "logical_distances vs brute force (tiny n)" begin
        for (dvec, R) in (([1, 2], 4), ([1, 1, 8], 3))
            n = length(dvec)
            S = zeros(Int, n, n); S[1, n] = 1; S[n, 1] = 1
            M = build_gkp_generator(S, dvec)
            r = is_single_qubit_reducible(gkp_gram(M))
            red = reduce_to_qubit(M; C=r.C, d=r.d)
            ld = logical_distances(red; reduction=:kz)
            # distances are labelled by size: dX ≤ dZ ≤ dY
            @test ld.dX <= ld.dZ <= ld.dY
            for (dval, P) in ((ld.dX, ld.short_XL), (ld.dY, ld.short_YL), (ld.dZ, ld.short_ZL))
                @test dval > 0
                @test isapprox(dval, sqrt(Float64(sum(abs2, P))); atol=1e-9)   # dval = |short rep|
                @test dval <= brute_distance(P, red.qubit_generator, R) + 1e-9
            end
            # coset representatives commute with the stabilizers (exact)
            Jr = symplectic_form(n; T=Rational{BigInt})
            @test all(isinteger, M * Jr * ld.short_XL)
            @test all(isinteger, M * Jr * ld.short_ZL)
            # reduced generators match the (relabelled) X/Z logicals:
            @test transpose(ld.mu_tilde) * Jr * ld.nu_tilde == 2       # symplectic pair
            @test transpose(ld.mu_tilde .// 2) * Jr * (ld.nu_tilde .// 2) == 1 // 2
            # mu_tilde/2 lies in the X coset: short_XL − mu_tilde/2 ∈ L(qubit_generator)
            invQ = inv(Rational{BigInt}.(red.qubit_generator))
            @test all(isinteger, transpose(ld.short_XL .- ld.mu_tilde .// 2) * invQ)
        end
    end

    @testset "large-entry lattice reduction (exact LLL)" begin
        # Regression: with LLLplus, dense/large canonical generators (n≈17+) push
        # BigInt entries past ~1e77, silently corrupting the reduced basis so that
        # `short Y is not a valid coset representative` fired. Reproduce huge
        # qubit_generator entries via a large-entry *unimodular* conjugation of a
        # canonical reducible generator (distances are invariant under it), and
        # require the exact (Nemo/FLINT) reduction to stay correct.
        dvec = [1, 1, 1, 1, 1, 1, 12]; n = length(dvec)
        S = zeros(Int, n, n); for k in 1:n-1; S[k, k+1] = 1; S[k+1, k] = 1; end
        M0 = build_gkp_generator(S, dvec)
        m = 2n
        U = Matrix{BigInt}(I, m, m)               # deterministic large unimodular
        for _ in 1:8
            U[1, :]   .+= big(10)^12 .* U[2, :]
            U[2, :]   .+= big(10)^12 .* U[1, :]
            U[m-1, :] .+= big(10)^12 .* U[m, :]
        end
        @test abs(det(U)) == 1
        M = U * BigInt.(M0)
        r = is_single_qubit_reducible(gkp_gram(M))
        @test r !== nothing
        red = reduce_to_qubit(M; C=r.C, d=r.d)
        @test maximum(abs, numerator.(red.qubit_generator)) > big(10)^77   # past LLLplus limit
        Lb = SymplecticGKP.Distances._conditioned_basis(red)
        @test abs(det(Lb)) == 2                    # spans L(qubit_generator) exactly
        ld = logical_distances(red)                # internal validity asserts must hold
        for dval in (ld.dX, ld.dY, ld.dZ)
            @test dval > 0 && isfinite(dval)
        end
        # unimodular conjugation preserves the code ⇒ same distances as the
        # un-scrambled canonical reduction of the same dvec.
        r0 = is_single_qubit_reducible(gkp_gram(BigInt.(M0)))
        red0 = reduce_to_qubit(BigInt.(M0); C=r0.C, d=r0.d)
        ld0 = logical_distances(red0)
        @test isapprox(sort([ld.dX, ld.dY, ld.dZ]), sort([ld0.dX, ld0.dY, ld0.dZ]); atol=1e-9)
    end

    @testset "InstanceIO round-trip (v2)" begin
        dvec = [1, 1, 8]; n = length(dvec)
        S = zeros(Int, n, n); S[1, n] = 1; S[n, 1] = 1
        M = build_gkp_generator(S, dvec)
        r = is_single_qubit_reducible(gkp_gram(M))
        red = reduce_to_qubit(M; C=r.C, d=r.d)
        ld = logical_distances(red)
        inst = build_instance(; n=n, d=4, seed=123, task_id=0,
                              classical_generator=M, red=red, distances=ld,
                              timings=ld.timings, status="ok")
        mktempdir() do dir
            path = joinpath(dir, instance_filename(n, 7))
            save_instance(path, inst)
            @test basename(path) == "reduced_ldlc_gkp_n_3_7.jld2"
            loaded = load_instance(path)
            @test loaded.schema_version == 2
            @test loaded.classical_generator == M
            @test loaded.qubit_generator == red.qubit_generator
            @test loaded.mu_tilde == ld.mu_tilde
            @test loaded.nu_tilde == ld.nu_tilde
            @test isapprox(loaded.dX, ld.dX; atol=1e-12)
            @test loaded.distances_status == "ok"
            @test list_instances(dir; n=3) == [path]
        end
    end

    @testset "InstanceIO legacy v1 loader" begin
        # Emulate a legacy .jls file: JLD2.save_object of a Dict.
        M = build_gkp_generator([0 1; 1 0], [1, 2])
        legacy = Dict(
            "classical_generator" => M,
            "generator" => Rational{BigInt}.(M),
            "qubit_generator_lll" => Float64.(M),
            "XL" => Rational{BigInt}[1//2, 0, 0, 0],
            "YL" => Rational{BigInt}[1//2, 0, 1//2, 0],
            "ZL" => Rational{BigInt}[0, 0, 1//2, 0],
            "dX" => 0.5, "dY" => sqrt(0.5), "dZ" => 0.5,
        )
        mktempdir() do dir
            path = joinpath(dir, "reduced_ldlc_gkp_n_2_1.jls")
            JLD2.save_object(path, legacy)
            loaded = load_instance(path)
            @test loaded.schema_version == 1
            @test loaded.classical_generator == M
            @test loaded.dX == 0.5
            @test loaded.C === missing
        end
    end

    @testset "trivial_gkp_from_ldlc" begin
        # Argument validation (deterministic, fast).
        @test_throws ArgumentError trivial_gkp_from_ldlc(1, 8)     # d < 2
        @test_throws ArgumentError trivial_gkp_from_ldlc(4, 0)     # n < 1
        # Real generation is slow (LDLC loop-removal convergence ≈0.03% at n=13,
        # ~minutes per instance). Opt in with SGKP_TEST_LDLC_GEN=1.
        if get(ENV, "SGKP_TEST_LDLC_GEN", "0") == "1"
            Random.seed!(1)
            M = trivial_gkp_from_ldlc(4, 13)
            @test size(M) == (26, 26) && eltype(M) == Int
            @test all(x -> abs(x) in (0, 1, 2), M)          # d=4 ⇒ entries ±1, ±2
            A = gkp_gram(M); @test A == -A'
        else
            @test_skip "set SGKP_TEST_LDLC_GEN=1 to test real LDLC generation (slow)"
        end
    end

    @testset "worker pipeline (synthetic generator)" begin
        # Exercise the full worker orchestration (Phase-A save → distances →
        # Phase-B save → load → --distances-only) with a fast, deterministic
        # synthetic reducible generator, so it does not depend on the (slow,
        # stochastic) LDLC loop-removal convergence. `trivial_gkp_from_ldlc`
        # itself is covered separately below.
        include(joinpath(@__DIR__, "..", "scripts", "generate_worker.jl"))
        S = zeros(Int, 3, 3); S[1, 3] = 1; S[3, 1] = 1
        synth(_) = build_gkp_generator(S, [1, 1, 8])          # 3 modes, reducible
        mktempdir() do dir
            saved = generate_for_n(3; d=4, attempts=3, instances=1, base_seed=0,
                                   outdir=dir, skip_distances=false, reduction=:kz,
                                   gen=synth)
            @test saved == 1
            files = list_instances(dir)
            @test length(files) == 1
            inst = load_instance(first(files))
            @test inst.schema_version == 2
            @test inst.distances_status == "ok"
            @test inst.dX > 0
            @test inst.classical_generator == build_gkp_generator(S, [1, 1, 8])
            # --distances-only is a no-op here (already "ok")
            @test complete_distances!(dir; reduction=:kz) == 0
        end
        # Phase-A-only save, then complete via --distances-only.
        mktempdir() do dir
            generate_for_n(3; d=4, attempts=1, instances=1, base_seed=1, outdir=dir,
                           skip_distances=true, reduction=:kz, gen=synth)
            f = first(list_instances(dir))
            @test load_instance(f).distances_status == "skipped"
            @test complete_distances!(dir; reduction=:kz) == 1
            @test load_instance(f).distances_status == "ok"
        end
    end

    # Regression vs. legacy mwe data (only if the mwe repo is present). The
    # loadable legacy files are the JLD2 `reconverted/*.jld2` (the sibling `.jls`
    # files are Julia `serialize` format, not JLD2).
    LEGACY_DIR = joinpath(@__DIR__, "..", "..", "gkp_ldlc_mwe", "examples",
                          "reduced_ldlc_gens", "reconverted")
    @testset "legacy regression (reconverted)" begin
        if !isdir(LEGACY_DIR)
            @test_skip "legacy directory not present: $LEGACY_DIR"
        else
            jls = sort(filter(f -> endswith(f, ".jld2"), readdir(LEGACY_DIR; join=true)))
            tested = 0
            for path in jls
                tested >= 3 && break
                loaded = try
                    load_instance(path)
                catch
                    continue
                end
                M = loaded.classical_generator
                (M isa AbstractMatrix && eltype(M) <: Integer) || continue
                A = gkp_gram(Matrix{Int}(M))
                r = is_single_qubit_reducible(A)
                r === nothing && continue
                red = reduce_to_qubit(Matrix{Int}(M); C=r.C, d=r.d)
                ld = logical_distances(red; reduction=:kz)
                # NOTE: the reduction is choice-dependent (our congruence `C`,
                # the reduced pair (2n-1,2n), and the balanced split differ from
                # Sage's), so our single-qubit code — and hence its distances —
                # need not equal the legacy one. We check the pipeline produces a
                # valid code with distances in the same ballpark. (CVP optimality
                # itself is verified exactly by the brute-force testset above.)
                for dval in (ld.dX, ld.dY, ld.dZ)
                    @test dval > 0 && isfinite(dval)
                end
                for (ours, stored) in ((ld.dX, loaded.dX), (ld.dY, loaded.dY), (ld.dZ, loaded.dZ))
                    if stored isa Real && isfinite(stored) && stored > 0
                        @test 0.3 * stored <= ours <= 3.0 * stored
                    end
                end
                tested += 1
            end
            @test tested >= 1
        end
    end

    @testset "SurfaceCodes: CSS validity + reference match" begin
        # odd squares reproduce the reference rotated-surface-code supports exactly
        setof(vv) = Set(sort.(vv))
        refset(dict) = Set(sort(v) for v in values(dict))
        for d in (3, 5)
            Xs, Zs = rotated_surface_stabilizers(d, d)
            @test setof(Xs) == refset(LatticeAlgorithms.surface_code_X_stabilizers(d))
            @test setof(Zs) == refset(LatticeAlgorithms.surface_code_Z_stabilizers(d))
            Xl, Zl = rotated_surface_logicals(d, d)
            @test sort(Xl) == sort(LatticeAlgorithms.surface_code_X_logicals(d)[1])
            @test sort(Zl) == sort(LatticeAlgorithms.surface_code_Z_logicals(d)[1])
        end
        # every patch (incl. even + rectangular) is a valid [[dx·dz,1,min]] CSS code
        for (dx, dz) in [(3, 3), (5, 5), (4, 4), (3, 5), (4, 5), (3, 6)]
            v = validate_surface_css(dx, dz)
            @test v.valid
            @test v.nstab == dx * dz - 1
        end
    end

    @testset "SurfaceCodes: square & hexagonal distances" begin
        ld(M, L) = norm(L - LatticeAlgorithms.closest_point(L, LatticeAlgorithms.kz(M)))
        s = sqrt(2 / sqrt(3))
        for (dx, dz) in [(3, 3), (4, 4), (3, 5)]
            Msq, XLsq, ZLsq, _ = gkp_surface_generator(dx, dz; lattice = :square)
            @test isapprox(ld(Msq, XLsq), sqrt(dx / 2); atol = 1e-6)   # dX = √(dx/2)
            @test isapprox(ld(Msq, ZLsq), sqrt(dz / 2); atol = 1e-6)   # dZ = √(dz/2)
            Mhx, XLhx, ZLhx, _ = gkp_surface_generator(dx, dz; lattice = :hexagonal)
            @test isapprox(ld(Mhx, XLhx), sqrt(dx / 2) * s; atol = 1e-3)  # hex: ×(2/√3)^½
            @test isapprox(ld(Mhx, ZLhx), sqrt(dz / 2) * s; atol = 1e-3)
        end
        # the classic distance-3 hexagonal surface-GKP distance is 3^{1/4}
        M, XL, _, _ = gkp_surface_generator(3, 3; lattice = :hexagonal)
        @test isapprox(ld(M, XL), 3^(1 / 4); atol = 1e-3)
        @test_throws ArgumentError gkp_surface_generator(3, 3; lattice = :triangular)
    end

    @testset "VoronoiMeasure: surface codes (float path)" begin
        Msq, XLsq, ZLsq, _ = gkp_surface_generator(3, 3; lattice = :square)
        Mhx, XLhx, ZLhx, _ = gkp_surface_generator(3, 3; lattice = :hexagonal)
        flsq = fine_lattice(Msq, XLsq, ZLsq)
        flhx = fine_lattice(Mhx, XLhx, ZLhx)
        # low σ: origin Voronoi cell captures ~all the Gaussian mass, no logical error
        r0 = logical_error_rate(flsq, 0.10; nsamples = 500, rng = MersenneTwister(1))
        @test r0.P_L < 0.02 && r0.P_single > 0.95
        # per-Pauli split sums to P_L; probabilities in [0,1]
        @test r0.P_L ≈ r0.P_X + r0.P_Y + r0.P_Z atol = 1e-12
        @test 0 <= r0.P_single <= 1
        # moderate σ: hexagonal GKP (larger distance) → lower logical error
        rsq = logical_error_rate(flsq, 0.24; nsamples = 3000, rng = MersenneTwister(7))
        rhx = logical_error_rate(flhx, 0.24; nsamples = 3000, rng = MersenneTwister(7))
        @test rhx.P_L < rsq.P_L
        # union bound: positive, decreasing in distance
        @test union_bound_logical_error(0.2, 1.2, 1.3, 1.4) >
              union_bound_logical_error(0.2, 1.6, 1.7, 1.8) > 0
    end

    @testset "VoronoiMeasure: exact (LDLC) path" begin
        # small single-qubit-reducible GKP code → exact fine lattice + MC
        dvec = [1, 2]; n = length(dvec)
        S = zeros(Int, n, n); S[1, n] = 1; S[n, 1] = 1
        M = build_gkp_generator(S, dvec)
        r = is_single_qubit_reducible(gkp_gram(M))
        red = reduce_to_qubit(M; C = r.C, d = r.d)
        ld = logical_distances(red; reduction = :kz)
        fl = fine_lattice(red.qubit_generator, ld.mu_tilde .// 2, ld.nu_tilde .// 2; exact = true)
        r = logical_error_rate(fl, 0.10; nsamples = 400, rng = MersenneTwister(3))
        @test r.P_single > 0.9 && r.P_L < 0.1      # low σ ⇒ mostly correct
        @test r.P_L ≈ r.P_X + r.P_Y + r.P_Z atol = 1e-12
    end

end