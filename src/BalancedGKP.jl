# ╔══════════════════════════════════════════════════════════════════╗
# ║                                                                  ║
# ║   MODULE 2: BalancedGKP                                          ║
# ║                                                                  ║
# ║   Given Smith invariant factors (d1 | d2 | ... | dn),           ║
# ║   find diagonal entries minimizing max(dj) subject to having     ║
# ║   the same Pfaffian divisors (same lattice congruence class).    ║
# ║                                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

module BalancedGKP

export balanced_diagonal, pfaffian_divisors, verify_equivalence

using Combinatorics: permutations, combinations
using Primes: factor


"""
    prime_valuation(n::Integer, p::Integer) -> Int

Compute the p-adic valuation v_p(n).
"""
function prime_valuation(n::T, p::T) where T<:Integer
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    p > 1 || throw(ArgumentError("p must be > 1, got $p"))
    v = 0
    while n % p == 0
        v += 1
        n = div(n, p)
    end
    return v
end


"""
    prime_factors(n::Integer) -> Vector{Int}

Return sorted list of distinct prime factors of n.
"""
function prime_factors(n::Integer)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    n == 1 && return Int[]
    return sort!(collect(keys(factor(n))))
end


"""
    pfaffian_divisors(d::AbstractVector{<:Integer}) -> Vector{Int}

Compute Pfaffian divisor sequence (pi_1, ..., pi_n) for diagonal entries d.
pi_k = gcd( prod(d[S]) for all S in combinations(1:n, k) )
"""
function pfaffian_divisors(d::AbstractVector{<:Integer})
    n = length(d)
    pi_vec = Vector{Int}(undef, n)
    for k in 1:n
        g = 0
        for combo in combinations(1:n, k)
            g = gcd(g, prod(d[i] for i in combo))
            g == 1 && break
        end
        pi_vec[k] = g
    end
    return pi_vec
end


"""
verify_equivalence(d_old, d_new) -> Bool

Verify that two diagonal vectors have the same Pfaffian divisors.
"""
function verify_equivalence(d_old::AbstractVector{<:Integer},
                            d_new::AbstractVector{<:Integer})
    length(d_old) == length(d_new) || throw(DimensionMismatch("Vectors must have same length"))
    return pfaffian_divisors(d_old) == pfaffian_divisors(d_new)
end


"""
    balanced_diagonal(inv_factors::Vector{Int}; max_enumerate::Int=8)
                     -> Vector{Int}

Given invariant factors (e1, ..., en) with e1 | e2 | ... | en,
find diagonal entries (d1, ..., dn) that:

1. Have the same Pfaffian divisors (so a unimodular U exists
   with U*(J2 x D_old)*U' = J2 x D_new), and
2. Minimize max(dj).

Returns the sorted diagonal vector.

# Keyword arguments
- `max_enumerate::Int=8`: If n > max_enumerate, use greedy heuristic.

# Examples
```julia
julia> balanced_diagonal([1, 10])
[2, 5]

julia> balanced_diagonal([2, 2, 2])
[2, 2, 2]

julia> balanced_diagonal([1, 1, 8])
[1, 1, 8]

julia> balanced_diagonal([1, 12])
[3, 4]

"""
function balanced_diagonal(inv_factors::Vector{Int}; max_enumerate::Int=8)
    n = length(inv_factors)
    product = prod(inv_factors)
    product == 0 && throw(ArgumentError("Invariant factors must be positive"))
    product == 1 && return ones(Int, n)
    n == 1 && return copy(inv_factors)

    for k in 1:n-1
        inv_factors[k+1] % inv_factors[k] == 0 ||
            throw(ArgumentError(
                "Invariant factors must satisfy divisibility: " *
                "e[$k]=$(inv_factors[k]) does not divide e[$(k+1)]=$(inv_factors[k+1])"
            ))
    end

    all_primes, table = _build_exponent_table(inv_factors)
    m = length(all_primes)

    if m == 0
        return ones(Int, n)
    elseif m == 1
        p = all_primes[1]
        return sort!([p^a for a in table[p]])
    elseif m == 2
        return _two_prime_optimal(n, all_primes, table)
    elseif n <= max_enumerate
        return _exact_enumerate(n, all_primes, table)
    else
        return _greedy_balance(n, all_primes, table)
    end

end
# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

function _build_exponent_table(inv_factors::Vector{Int})
    product = prod(inv_factors)
    all_primes = prime_factors(product)
    n = length(inv_factors)

    table = Dict{Int, Vector{Int}}()
    for p in all_primes
        exps = sort!([prime_valuation(e, p) for e in inv_factors])
        table[p] = exps
    end
    return all_primes, table
end

function _two_prime_optimal(n::Int, all_primes::Vector{Int},
    table::Dict{Int, Vector{Int}})
    p1, p2 = all_primes
    exp1 = sort(table[p1])
    exp2 = sort(table[p2], rev=true)
    d = [p1^exp1[i] * p2^exp2[i] for i in 1:n]
    return sort!(d)
end






#########################################################################
function _exact_enumerate(n::Int, all_primes::Vector{Int},
    table::Dict{Int, Vector{Int}})
    identity_perm = collect(1:n)
    all_perms = collect(permutations(identity_perm))
    best_max = typemax(Int)
    best_d = ones(Int, n)

    p0 = all_primes[1]
    base = [p0^table[p0][i] for i in 1:n]
    remaining = all_primes[2:end]
#########################################################################

















function recurse!(depth::Int, d_accum::Vector{Int}, current_max::Int)
    if depth > length(remaining)
        if current_max < best_max
            best_max = current_max
            best_d .= d_accum
        end
        return
    end

    p = remaining[depth]
    exps = table[p]

    for perm in all_perms
        new_max = current_max
        d_next = copy(d_accum)
        skip = false
        for i in 1:n
            d_next[i] *= p^exps[perm[i]]
            if d_next[i] > new_max
                new_max = d_next[i]
            end
            if new_max >= best_max
                skip = true
                break
            end
        end
        skip && continue
        recurse!(depth + 1, d_next, new_max)
    end
end

recurse!(1, copy(base), maximum(base))
return sort!(best_d)

end

function _greedy_balance(n::Int, all_primes::Vector{Int},
    table::Dict{Int, Vector{Int}})
    spreads = [(maximum(table[p]) - minimum(table[p]), p) for p in all_primes]
    sort!(spreads, rev=true)
    sorted_primes = [s[2] for s in spreads]
    p0 = sorted_primes[1]
    exps0 = sort(table[p0])
    d = [p0^exps0[i] for i in 1:n]

    for p in sorted_primes[2:end]
        exps = sort(table[p])
        order = sortperm(d, rev=true)
        perm = zeros(Int, n)
        for (rank, idx) in enumerate(order)
            perm[idx] = rank
        end
        for i in 1:n
            d[i] *= p^exps[perm[i]]
        end
    end

    return sort!(d)
end 



end # module BalancedGKP
