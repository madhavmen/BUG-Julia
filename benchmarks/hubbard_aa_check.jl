# VALIDATION OF THE SPINFUL AUBRY-ANDRÉ-HUBBARD REFERENCE. Run before trusting any number that
# depends on it, and before the MPO for this model is written -- the MPO will be graded against
# this file, so a defect here would be inherited rather than caught.
#
#   julia --startup-file=no benchmarks/hubbard_aa_check.jl
#
# ⛔ NO `BUGJulia`, NO Telum, ONLY stdlib. That is what makes this runnable in seconds instead of
# behind the package's ~25-minute 3-target precompile, and it is also why agreement between this
# file and the MPO will be evidence: the two share no code.
#
# THE LOAD-BEARING TEST IS #2. At `U = 0` the two species are independent, so the MANY-BODY
# dynamics must reproduce the SINGLE-PARTICLE result exactly. Every hopping sign enters that
# comparison, and a wrong Jordan-Wigner-style sign cannot survive it -- whereas it survives
# hermiticity, particle-number conservation and energy conservation untouched.

using LinearAlgebra, SparseArrays
include(joinpath(@__DIR__, "hubbard_aa_exact.jl"))

const FAILS = String[]
say(x) = (println(x); flush(stdout))

say("── 1. hermiticity at a generic point (Delta and U both non-zero, phi off-lattice) ──")
for (L, N) in ((4, 2), (6, 3), (6, 4))
    H, st, _ = hubbard_aa_sparse(L, N; J = 1.0, Delta = 1.7, U = 2.3, phi = 0.4)
    d = maximum(abs.(Matrix(H) - Matrix(H)'))
    say("   L=$L N=$N dim=$(length(st)): max|H-H'| = $d")
    d < 1e-13 || push!(FAILS, "H not Hermitian at L=$L N=$N: $d")
end

say("── 2. THE SIGN TEST: U=0 many-body dynamics == free fermions ──")
function sp_hopping(L; J, Delta, beta, phi)
    h = zeros(Float64, L, L)
    for i in 1:(L - 1)
        h[i, i + 1] = -J
        h[i + 1, i] = -J
    end
    for i in 1:L
        h[i, i] = Delta * cos(2π * beta * i + phi)
    end
    return h
end
for (L, Delta, phi) in ((4, 0.0, 0.0), (4, 1.3, 0.7), (6, 2.1, 0.3), (6, 0.0, 1.1), (8, 1.9, 2.2))
    beta = 0.721
    evens = collect(2:2:L)
    # BOTH species present, so the test is not secretly a one-species problem.
    species = [isodd(k) ? :up : :down for k in eachindex(evens)]
    H, st, idx = hubbard_aa_sparse(L, length(evens); J = 1.0, Delta = Delta, U = 0.0,
                                   beta = beta, phi = phi)
    v0 = hubbard_cdw_vector(L, st, idx; species = species)
    h = sp_hopping(L; J = 1.0, Delta = Delta, beta = beta, phi = phi)
    occ_up   = [evens[k] for k in eachindex(evens) if species[k] === :up]
    occ_down = [evens[k] for k in eachindex(evens) if species[k] === :down]
    Hd = Matrix(H)
    worst = 0.0
    worstI = 0.0
    for t in (0.3, 1.0, 2.7)
        v = exp(-im * t * Hd) * v0
        nmb = hubbard_density(v, L, st)
        Ut = exp(-im * t * h)
        nsp = [sum(abs2, @view Ut[j, occ_up]) + sum(abs2, @view Ut[j, occ_down]) for j in 1:L]
        worst = max(worst, maximum(abs.(nmb .- nsp)))
        worstI = max(worstI, abs(hubbard_imbalance(nmb) - hubbard_imbalance(nsp)))
    end
    say("   L=$L D=$Delta phi=$phi: max|dn| = $worst   max|dI| = $worstI")
    max(worst, worstI) < 1e-11 ||
        push!(FAILS, "U=0 many-body != free fermions at L=$L D=$Delta: dn=$worst dI=$worstI")
end

say("── 3. U acts on doublons and ONLY on doublons ──")
# ⛔ `Delta` IS NON-ZERO HERE ON PURPOSE. With `Delta = 0` a CDW basis state has zero potential
# energy and zero hopping expectation, so `⟨H⟩ = 0` for every `U` and the single-species check
# compares 0 against 0 -- it passes whatever `U` does. That is exactly the vacuous-gate trap this
# repository documents elsewhere, and it is what the first version of this check did.
let L = 4, beta = 0.721, Delta = 1.5, phi = 0.35
    evens = collect(2:2:L)
    Hu0, st, idx = hubbard_aa_sparse(L, length(evens); J = 1.0, Delta = Delta, U = 0.0,
                                     beta = beta, phi = phi)
    Hu5, _, _ = hubbard_aa_sparse(L, length(evens); J = 1.0, Delta = Delta, U = 5.0,
                                  beta = beta, phi = phi)
    vup = hubbard_cdw_vector(L, st, idx; species = fill(:up, length(evens)))
    e0 = real(dot(vup, Hu0 * vup))
    e5 = real(dot(vup, Hu5 * vup))
    say("   single species: <H>|U=0 = $e0,  <H>|U=5 = $e5  (equal, and NON-ZERO)")
    abs(e0) > 1e-6 || push!(FAILS, "single-species check is VACUOUS: <H> = $e0")
    abs(e0 - e5) < 1e-13 || push!(FAILS, "U acts on a single-species state: $e0 vs $e5")

    Hd0, st2, idx2 = hubbard_aa_sparse(L, 2 * length(evens); J = 1.0, Delta = Delta, U = 0.0,
                                       beta = beta, phi = phi)
    Hd7, _, _ = hubbard_aa_sparse(L, 2 * length(evens); J = 1.0, Delta = Delta, U = 7.0,
                                  beta = beta, phi = phi)
    vd = hubbard_cdw_vector(L, st2, idx2; species = fill(:both, length(evens)))
    d0 = real(dot(vd, Hd0 * vd))
    d7 = real(dot(vd, Hd7 * vd))
    want = 7.0 * length(evens)
    say("   doublon CDW: <H> shift = $(d7 - d0)  (want exactly $want = U per doublon)")
    abs((d7 - d0) - want) < 1e-12 || push!(FAILS, "U miscounted on doublons: $(d7 - d0) != $want")
end

say("── 4. energy conservation under the INTERACTING dynamics ──")
let L = 6, beta = 0.721
    evens = collect(2:2:L)
    species = [isodd(k) ? :up : :down for k in eachindex(evens)]
    H, st, idx = hubbard_aa_sparse(L, length(evens); J = 1.0, Delta = 2.0, U = 3.0,
                                   beta = beta, phi = 0.9)
    v0 = hubbard_cdw_vector(L, st, idx; species = species)
    Hd = Matrix(H)
    e0 = real(dot(v0, H * v0))
    v = exp(-im * 3.5 * Hd) * v0
    et = real(dot(v, H * v)) / real(dot(v, v))
    say("   E(0) = $e0  ->  drift over t=3.5: $(abs(et - e0))")
    abs(et - e0) < 1e-10 || push!(FAILS, "energy not conserved: $e0 -> $et")
end

say("── 5. the imbalance starts at +1 on a CDW and U=0/U≠0 share it at t=0 ──")
let L = 8, beta = 0.721
    evens = collect(2:2:L)
    species = [isodd(k) ? :up : :down for k in eachindex(evens)]
    _, st, idx = hubbard_aa_sparse(L, length(evens); J = 1.0, Delta = 1.0, beta = beta)
    v0 = hubbard_cdw_vector(L, st, idx; species = species)
    I0 = hubbard_imbalance(hubbard_density(v0, L, st))
    say("   I(0) = $I0  (want exactly +1: every atom on an even site)")
    abs(I0 - 1.0) < 1e-14 || push!(FAILS, "CDW initial imbalance is $I0, not 1")
end

println()
if isempty(FAILS)
    println("ALL HUBBARD REFERENCE CHECKS PASSED")
else
    for f in FAILS
        println("FAIL: ", f)
    end
    println("$(length(FAILS)) FAILURES")
    exit(1)
end
