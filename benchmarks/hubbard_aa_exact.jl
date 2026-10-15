# THE SPINFUL AUBRY-ANDRÉ-HUBBARD CHAIN, EXACTLY, IN THE FERMION OCCUPATION BASIS.
#
#     H = -J Σ_{i,σ} (c†_{i+1,σ} c_{i,σ} + h.c.)
#         + Δ Σ_i cos(2πβi + φ) (n_{i↑} + n_{i↓})
#         + U Σ_i n_{i↑} n_{i↓}
#
# This is the Hamiltonian of arXiv:1501.05661 -- the ACTUAL experiment, spinful, with an ON-SITE
# `U`. It is what Figs. 2, 4 and 5 of that paper are measurements of; the spinless XXZ proxy in
# `models/mbl_scars.jl` reaches only Fig. 3 (`U = 0`, "only a single spin component was used").
#
# ⛔⛔ WHY THIS FILE EXISTS AT ALL, AND WHY IT COMES FIRST. The MPS side must reach this model
# through Jordan-Wigner: two species on `L` sites become one spin chain on `2L` sites, and with
# the INTERLEAVED ordering (↑₁ ↓₁ ↑₂ ↓₂ …) a hop ↑ᵢ → ↑ᵢ₊₁ jumps over ↓ᵢ, so it becomes a
# THREE-SITE term `S⁺ σ^z S⁻` carrying a JW STRING. Get that string's sign wrong and the result is
# not an error bar -- it is the wrong statistics, and it looks completely plausible: still
# Hermitian, still particle-number conserving, still a smooth imbalance decay. That is the same
# failure class as the anti-Hermitian `X` in `pxp_operators`, which cost two cluster jobs to find.
#
# So the reference is built HERE, in the occupation basis, where the sign is not inferred from a
# string identity but COMPUTED by counting occupied modes -- the definition of the fermionic
# anticommutator, with nothing to get backwards. The MPO is then validated against it.
#
# ⛔ AND THIS FILE DEPENDS ON NOTHING. Only `LinearAlgebra` and `SparseArrays`, no `BUGJulia`, no
# Telum. That is deliberate: it can be tested in seconds without paying the package's ~25-minute
# precompile, and it shares no code whatsoever with the object it will be used to grade.
#
# MODE ORDERING is interleaved, matching the intended spin-chain ordering exactly:
#
#     mode(i, σ) = 2(i-1) + σ,     σ = 0 for ↑, 1 for ↓,     i = 1 … L
#
# so mode indices `0 … 2L-1` run ↑₁ ↓₁ ↑₂ ↓₂ …. A basis state is an integer whose bit `m` is the
# occupation of mode `m`, and the state is `∏_{m occupied, ASCENDING} c†_m |0⟩` -- fixing the
# order in the product is what fixes every sign below.

using LinearAlgebra, SparseArrays

"Mode index (0-based) of spin `σ` (0 = up, 1 = down) on physical site `i` (1-based)."
hub_mode(i::Int, σ::Int) = 2 * (i - 1) + σ

"""
    hubbard_basis(L, N) -> (states, idx)

All `2L`-mode occupation patterns with exactly `N` particles: the `U(1)` particle-number sector.

`N = L/2` at the experiment's filling (one atom per even site). `C(2L, N)` states, so `L = 8`,
`N = 4` is `C(16,4) = 1820` and `L = 8`, `N = 8` is `C(16,8) = 12870` -- both trivial.

⚠ ONLY TOTAL `N` IS FIXED, NOT `N↑` AND `N↓` SEPARATELY. Both are conserved by this Hamiltonian,
so a finer sector would be cheaper -- but the CDW initial states the paper averages over are
superpositions across `(N↑, N↓)` splittings only if prepared that way; keeping the single total-`N`
sector means any such state is representable without a second basis. The cost is small and the
generality is what avoids a silent projection.
"""
function hubbard_basis(L::Int, N::Int)
    M = 2L
    0 <= N <= M || throw(ArgumentError("N=$N outside 0:$M"))
    states = [b for b in 0:(2^M - 1) if count_ones(b) == N]
    return states, Dict(s => k for (k, s) in enumerate(states))
end

"Occupation of mode `m` (0-based) in pattern `b`."
hub_occ(b::Int, m::Int) = (b >> m) & 1

"""
    hub_sign(b, m, mp) -> Int

The fermionic sign of `c†_{mp} c_m |b⟩`, computed from the anticommutator directly:

    c_m  |b⟩  = (-1)^{Σ_{k<m}  n_k}   |b ⊖ m⟩
    c†_mp |b'⟩ = (-1)^{Σ_{k<mp} n'_k}  |b' ⊕ mp⟩

with `n'` the occupations AFTER removing `m`. Both parities are counted over modes STRICTLY BELOW
the acted-on mode, which is what the ascending-order convention in the file header means.

⛔ THE SECOND PARITY MUST USE `n'`, NOT `n`. Using the original occupations gives a sign that is
wrong exactly when `m < mp` and mode `m` was occupied -- i.e. for every forward hop, which is half
of them. The resulting `H` is still Hermitian and still conserves `N`, so nothing downstream
complains; only the dynamics is wrong.
"""
function hub_sign(b::Int, m::Int, mp::Int)
    hub_occ(b, m) == 1 || return 0
    s1 = count_ones(b & ((1 << m) - 1))          # occupied modes below m, before removal
    b2 = b ⊻ (1 << m)
    hub_occ(b2, mp) == 0 || return 0
    s2 = count_ones(b2 & ((1 << mp) - 1))        # ... below mp, AFTER removal
    return iseven(s1 + s2) ? 1 : -1
end

"""
    hubbard_aa_sparse(L, N; J, Delta, U, beta, phi) -> (H, states, idx)

The spinful Aubry-André-Hubbard Hamiltonian as a sparse matrix in the `N`-particle sector.

`Delta` and `U` are in units of `J`, exactly as arXiv:1501.05661 quotes them -- there is no
Jordan-Wigner factor of 2 here, because this is written in the fermion language the paper uses.
That is precisely the discrepancy this reference exists to expose on the MPO side.
"""
function hubbard_aa_sparse(L::Int, N::Int; J::Float64 = 1.0, Delta::Float64 = 0.0,
                           U::Float64 = 0.0, beta::Float64 = 0.721, phi::Float64 = 0.0)
    states, idx = hubbard_basis(L, N)
    n = length(states)
    I, Jd, V = Int[], Int[], Float64[]
    pot = [Delta * cos(2π * beta * i + phi) for i in 1:L]
    for (k, b) in enumerate(states)
        diag = 0.0
        for i in 1:L
            nu = hub_occ(b, hub_mode(i, 0))
            nd = hub_occ(b, hub_mode(i, 1))
            diag += pot[i] * (nu + nd)
            diag += U * nu * nd
        end
        push!(I, k); push!(Jd, k); push!(V, diag)
        # Hopping, BOTH directions written explicitly rather than one plus h.c.: the sign of
        # `c†_{i+1} c_i` and of `c†_i c_{i+1}` are computed independently, so a sign error cannot
        # be symmetrised away into something that merely looks Hermitian.
        for i in 1:(L - 1), σ in 0:1
            m, mp = hub_mode(i, σ), hub_mode(i + 1, σ)
            for (a, c) in ((m, mp), (mp, m))
                s = hub_sign(b, a, c)
                s == 0 && continue
                bb = (b ⊻ (1 << a)) ⊻ (1 << c)
                push!(I, idx[bb]); push!(Jd, k); push!(V, -J * s)
            end
        end
    end
    return sparse(I, Jd, V, n, n), states, idx
end

"""
    hubbard_cdw_vector(L, states, idx; species) -> Vector{ComplexF64}

A charge-density-wave product state with the EVEN physical sites occupied -- the experiment's
initial state -- as a unit column.

`species[k]` says what sits on the `k`-th even site: `:up`, `:down`, or `:both` (a doublon).
arXiv:1501.05661 prepares "zero, one or two atoms on each even site" and its DMRG curves are
"averaged over ~100 initial states per parameter set", so a SINGLE choice here is not the
experiment -- it is one member of the ensemble the experiment averages over.

⛔ AND THE CHOICE IS NOT FREE, because it decides whether `U` does anything at all. All-`:up`
gives a single species: no double occupancy is ever possible, `U` is exactly inert, and the run
silently measures the NON-INTERACTING model however large `U` is set. Any interacting result must
therefore come from a state with both species present.
"""
function hubbard_cdw_vector(L::Int, states::Vector{Int}, idx::Dict{Int, Int};
                            species::Vector{Symbol})
    evens = collect(2:2:L)
    length(species) == length(evens) || throw(DimensionMismatch(
        "got $(length(species)) species for $(length(evens)) even sites"))
    b = 0
    for (k, i) in enumerate(evens)
        if species[k] === :up
            b |= 1 << hub_mode(i, 0)
        elseif species[k] === :down
            b |= 1 << hub_mode(i, 1)
        elseif species[k] === :both
            b |= 1 << hub_mode(i, 0)
            b |= 1 << hub_mode(i, 1)
        else
            error("species must be :up, :down or :both, got $(species[k])")
        end
    end
    haskey(idx, b) || error("CDW pattern has a different particle number than the basis sector")
    v = zeros(ComplexF64, length(states))
    v[idx[b]] = 1.0
    return v
end

"""
    hubbard_density(v, L, states) -> Vector{Float64}

`⟨n_i⟩ = ⟨n_{i↑}⟩ + ⟨n_{i↓}⟩` per PHYSICAL site -- the quantity the imbalance is built from.
"""
function hubbard_density(v::Vector{ComplexF64}, L::Int, states::Vector{Int})
    p = abs2.(v)
    nrm = sum(p)
    out = zeros(Float64, L)
    for i in 1:L
        acc = 0.0
        mu, md = hub_mode(i, 0), hub_mode(i, 1)
        for (k, b) in enumerate(states)
            acc += p[k] * (hub_occ(b, mu) + hub_occ(b, md))
        end
        out[i] = acc / nrm
    end
    return out
end

"""
    hubbard_imbalance(nprof) -> Float64

`I = (N_e - N_o) / (N_e + N_o)` over PHYSICAL sites, summing both spins -- the experiment's own
definition (arXiv:1501.05661), normalised by the total particle number rather than by `L`.

⚠ THIS IS NOT THE SPIN-CHAIN `imbalance_dense`. That one exploits half filling to write the
normalisation as `L/2` and drops the `1/2` offsets; here the filling is a free parameter and both
spins share a site, so the denominator must be the measured total. At the experiment's filling the
two agree; at any other they do not.
"""
function hubbard_imbalance(nprof::Vector{Float64})
    Ne = sum(nprof[i] for i in eachindex(nprof) if iseven(i); init = 0.0)
    No = sum(nprof[i] for i in eachindex(nprof) if isodd(i); init = 0.0)
    tot = Ne + No
    return tot > 0 ? (Ne - No) / tot : 0.0
end
