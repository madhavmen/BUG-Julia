# tests/common/dissipative_fixtures.jl -- the lossy generators and small exact references of the
# dissipative plan's tests (test_dissipative_t1.jl, _t2.jl, _t3_a1.jl, _t3_a2.jl): the Fock-space
# Hubbard ED (full or one sector) and its basis phases, product-basis amplitudes, observables and
# sector ranks. Needs `adaptive_fixtures.jl`, `dense_reference.jl` and `dd_fixtures.jl`.

# Stdlib-only Fock-space sign conventions (`hub_mode`, `hub_occ`, `hub_sign`), shared with the
# Hubbard benchmark so the two references cannot disagree.
include(joinpath(@__DIR__, "..", "..", "benchmarks", "hubbard_aa_exact.jl"))

_loss_rates(L) = [isodd(j) ? 1.0 : 0.2 for j in 1:L]

"Single-site loss `Q = Σ g_j n_j` (`L_j = √g_j S⁻_j`) on the spin-1/2 chain."
function _site_loss_mpo(L; gamma = _loss_rates(L))
    q = local_space()
    n = to_concrete(q.Sz + 0.5 * q.I)                       # S^+ S^- = |up><up|
    return onsite_mpo(L, [to_concrete(gamma[j] * n) for j in 1:L])
end

"`(H, Q, G)` as MPOs and the dense `H - (i/2) Q`, with `delta` the XXZ anisotropy."
function _lossy_xx(L; delta = 0.0, gamma = _loss_rates(L))
    H = xxz_mpo(L; J = 1.0, delta = delta)
    Q = _site_loss_mpo(L; gamma = gamma)
    Qd = sum(gamma[j] * (_embed(_SZ, j, L) + 0.5 * I) for j in 1:L)
    Hd = dense_heisenberg(L; J = 1.0, delta = delta)
    return H, Q, no_jump_mpo(H, Q), Hd - 0.5im * Qd, Qd
end

"""
`_hb_bond`'s warmed Heisenberg bond with the lossy generator `G = H - (i/2) Q` as the MPO, and its
environments. Returns `(G, psi, f, lenv, renv, H)`.
"""
function _lossy_bond(; L = 8, i = 4, steps = 2, maxdim = 4)
    H, psi, f, _, _ = _hb_bond(; L = L, i = i, steps = steps, maxdim = maxdim)
    G = no_jump_mpo(H, _site_loss_mpo(L))
    lenv, renv = _envs(psi, G, i, L)
    return G, psi, f, lenv, renv, H
end
_lossy_wide() = _lossy_bond(; L = 12, i = 6, steps = 4, maxdim = 16)

# ── Hubbard: the d = 4 MPO against the Fock space, in the interleaved mode order ↑₁ ↓₁ ↑₂ … ──

"`c_m |b⟩ = (-1)^{Σ_{k<m} n_k} |b ⊖ m⟩` as `(sign, b')`; sign 0 if `m` is empty."
_annihilate(b, m) = hub_occ(b, m) == 0 ? (0, b) :
    (iseven(count_ones(b & ((1 << m) - 1))) ? 1 : -1, b ⊻ (1 << m))

"""
`(H, Q, jumps)` of the Hubbard chain with pair loss on the Fock space (bit `m` = mode `m`),
`H = -Σ_b J_b (c'c + h.c.) + U Σ n↑n↓ + Σ (Vup n↑ + Vdn n↓)` (`J` one hopping or one per bond),
`L_j = √κ_j c_j↓ c_j↑`, on the patterns `basis` (default all
`4^L`; a sector from [`_hub_sector`](@ref) for ED beyond L = 4). On the full space `Q = Σ L'L` from
the jumps; on a sector, where the jumps leave the basis, `jumps` is `nothing` and `Q` sums
`|<b'|L_j|b>|²` directly. `wrong = true` is the CONTROL: the up-spin hop between `j` and `j+1`
takes an extra `(-1)^{n_j↓}`, a Hermitian sign error that no basis phase removes.
"""
function _hub_fock(L; J = 1.0, U = 0.0, kappa = zeros(L), wrong = false, basis = 0:(4^L - 1),
                   Vup = zeros(L), Vdn = zeros(L))
    Jb = J isa Number ? fill(J, L - 1) : J
    D = length(basis)
    full = D == 4^L
    pos = Dict(b => k for (k, b) in enumerate(basis))
    H = zeros(ComplexF64, D, D)
    Qd = zeros(ComplexF64, D, D)
    jumps = full ? [zeros(ComplexF64, D, D) for _ in 1:L] : nothing
    for (k, b) in enumerate(basis)
        for j in 1:L
            nu, nd = hub_occ(b, hub_mode(j, 0)), hub_occ(b, hub_mode(j, 1))
            H[k, k] += U * nu * nd + Vup[j] * nu + Vdn[j] * nd
            s1, b1 = _annihilate(b, hub_mode(j, 0))
            s2, b2 = s1 == 0 ? (0, b1) : _annihilate(b1, hub_mode(j, 1))
            s2 == 0 && continue
            Qd[k, k] += kappa[j] * (s1 * s2)^2
            full && (jumps[j][pos[b2], k] = sqrt(kappa[j]) * s1 * s2)
        end
        for j in 1:(L - 1), σ in 0:1
            Jb[j] == 0 && continue          # a cut bond (decoupled leads) leaves the basis closed
            m, mp = hub_mode(j, σ), hub_mode(j + 1, σ)
            for (a, c) in ((m, mp), (mp, m))
                s = hub_sign(b, a, c)
                s == 0 && continue
                wrong && σ == 0 && hub_occ(b, hub_mode(j, 1)) == 1 && (s = -s)
                H[pos[b ⊻ (1 << a) ⊻ (1 << c)], k] += -Jb[j] * s  # hops conserve (N↑, N↓)
            end
        end
    end
    return H, full ? sum(Lj' * Lj for Lj in jumps) : Qd, jumps
end

"The Fock patterns of the `(N↑, N↓)` sector, ascending."
_hub_sector(L, nu, nd) = [b for b in 0:(4^L - 1) if _hub_charges(b, L) == (nu, nd)]

"The Fock pattern of a d = 4 occupation list (`hub4_product_state`'s argument)."
_hub_pattern(occ) = sum((o in (:up, :both) ? 1 << hub_mode(j, 0) : 0) +
                        (o in (:down, :both) ? 1 << hub_mode(j, 1) : 0)
                        for (j, o) in enumerate(occ))

"The d = 4 product state of Fock pattern `b`."
function _hub4_state(b, L)
    occ = map(1:L) do j
        u, d = hub_occ(b, hub_mode(j, 0)), hub_occ(b, hub_mode(j, 1))
        u == 1 ? (d == 1 ? :both : :up) : (d == 1 ? :down : :empty)
    end
    return hub4_product_state(occ)
end

"`(N↑, N↓)` of Fock pattern `b`: the d = 4 MPO conserves both."
_hub_charges(b, L) = (sum(hub_occ(b, hub_mode(j, 0)) for j in 1:L),
                      sum(hub_occ(b, hub_mode(j, 1)) for j in 1:L))

"`⟨b'|mpo|b⟩` on product states `states[k]`; pairs `skip(k', k)` says are charge-forbidden are 0."
function _mpo_matrix(mpo, states; skip = (a, b) -> false)
    D = length(states)
    M = zeros(ComplexF64, D, D)
    for k in 1:D, kp in 1:D
        skip(kp, k) || (M[kp, k] = mpo_matrix_element(states[kp], mpo, states[k]))
    end
    return M
end

"""
`||M - S A S'|| / ||A||` for the unit-modulus diagonal `S` that the nonzero entries of `A` fix
(breadth-first from one state per connected block): `M` and `A` are the same operator in bases
that differ by state phases only, iff this is ~0. Gauge-invariant content: every closed loop of
hops, so a fermionic sign error that is not a basis convention cannot pass.
"""
function _gauge_residual(M, A; tol = 1e-12)
    S = Diagonal(_gauge_phases(M, A; tol = tol))
    return norm(M - S * A * S') / norm(A)
end

"The state phases `s` of [`_gauge_residual`](@ref), `M[b2,b] = s[b2] A[b2,b] conj(s[b])`."
function _gauge_phases(M, A; tol = 1e-12)
    n = size(A, 1)
    s = zeros(ComplexF64, n)
    seen = falses(n)
    for r in 1:n
        seen[r] && continue
        s[r], seen[r] = 1, true
        queue = [r]
        while !isempty(queue)
            b = popfirst!(queue)
            for b2 in 1:n
                (!seen[b2] && abs(A[b2, b]) > tol) || continue
                s[b2] = M[b2, b] * s[b] / A[b2, b]          # M[b2,b] = s[b2] A[b2,b] conj(s[b])
                seen[b2] = true
                push!(queue, b2)
            end
        end
    end
    return s
end

"""
`(s, residual)`: [`_gauge_phases`](@ref) of the MPO `mpo` on product `states` against the Fock
matrix `A` (same order), evaluating the MPO only on `A`'s nonzero entries (the sparse ED graph),
so a 400-state sector costs a few thousand matrix elements. The residual is over those entries;
U02 checks that the MPO has no others.
"""
function _edge_gauge(mpo, states, A; tol = 1e-12)
    n = size(A, 1)
    M = zeros(ComplexF64, n, n)
    for k in 1:n, kp in 1:n
        abs(A[kp, k]) > tol && (M[kp, k] = mpo_matrix_element(states[kp], mpo, states[k]))
    end
    s = _gauge_phases(M, A; tol = tol)
    return s, norm(M - Diagonal(s) * A * Diagonal(s)') / norm(A)
end

"`[<states[k]|psi>]`: the MPS in a product basis."
_amplitudes(psi, states) = ComplexF64[overlap(st, psi) for st in states]

# ── observables and sector ranks, MPS side and ED side ──

"`P0 = ||ψ||²` by overlap (no canonical form assumed)."
_p0(psi) = real(overlap(psi, psi))

"Keep the worst value of error `k` in `d`."
_upd!(d, k, x) = (d[k] = max(get(d, k, 0.0), x))

"`n↑ n↓` on one d = 4 site, for `site_expval`."
function _hub4_nupdn()
    q = hub4_space()
    nu = to_concrete(lock(q.Fu', 2) * q.Fu)
    nd = to_concrete(lock(q.Fd', 2) * q.Fd)
    return to_concrete(lock(nu, 1) * nd)
end

"""
Multiply each site-leg sector `s` of `psi[j]` by `f(s)`, in place: an onsite operator constant on
each charge sector, by sector projection as `apply_sz!` does (contracting an operator onto the
site leg re-sorts its sectors). Needs a charge-resolved site leg (`:U1`).
"""
function _scale_site_sectors!(psi, j, f)
    canonical!(psi, j)
    A = psi[j]
    terms = Any[]
    for (s, _) in A.spaces[2]
        B = to_concrete(getsub(A, 2, t -> t == s ? Colon() : nothing; preserve_space = true))
        isempty(B.qlabels) || push!(terms, to_concrete(f(s) * B))
    end
    psi[j] = reduce((a, b) -> to_concrete(a + b), terms)
    return psi
end

"`S^z_j = (n↑ - n↓)/2` on a d = 4 site, by sector scaling."
_hub4_sz!(psi, j) = _scale_site_sectors!(psi, j, s -> s == HUB4_UP ? 0.5 : s == HUB4_DOWN ? -0.5 : 0.0)

"`<ψ|S^z_i S^z_j|ψ>` (unnormalised) with `apply!(psi, j)` the in-place `S^z_j`."
_szsz(psi, i, j, apply!) = real(overlap(apply!(copy(psi), i), apply!(copy(psi), j)))

"""
Schmidt values of the amplitude vector `v` over basis patterns `bs` at a cut, per left-block
charge: `split(b) -> (left, right, charge)`. Returns `Dict(charge => singular values)`.
"""
function _sector_schmidt(v, bs, split)
    blocks = Dict{Any, Vector{Tuple{Int, Int, ComplexF64}}}()
    for (k, b) in enumerate(bs)
        l, r, c = split(b)
        push!(get!(blocks, c, Tuple{Int, Int, ComplexF64}[]), (l, r, v[k]))
    end
    out = Dict{Any, Vector{Float64}}()
    for (c, es) in blocks
        ls, rs = sort(unique(first.(es))), sort(unique(getindex.(es, 2)))
        li, ri = Dict(x => n for (n, x) in enumerate(ls)), Dict(x => n for (n, x) in enumerate(rs))
        M = zeros(ComplexF64, length(ls), length(rs))
        for (l, r, a) in es
            M[li[l], ri[r]] = a
        end
        out[c] = svdvals(M)
    end
    return out
end

"d = 4 cut after site `i`: left modes `0 .. 2i-1`, charge `(N↑, N↓)` of the left block."
_hub_split(i) = b -> (b & (4^i - 1), b >> (2i),
                      (count(j -> hub_occ(b, hub_mode(j, 0)) == 1, 1:i),
                       count(j -> hub_occ(b, hub_mode(j, 1)) == 1, 1:i)))

"Spin cut after site `i` in `dense_state`'s order (site 1 most significant, 0 = up); charge `N↑`."
_spin_split(i, L) = idx -> (idx >> (L - i), idx & (2^(L - i) - 1),
                            i - count_ones(idx >> (L - i)))

"""
`Dict(charge => dim)` of the MPS bond after site `i`, with `decode(label, i)` the left-block
charge of a bond label. The decoders below are checked on product states, where the charge of
each bond is known, before any rank is compared.
"""
_mps_sector_dims(psi, i, decode) = Dict(decode(lab, i) => d for (lab, d) in psi[i].spaces[3])
# Telum labels: d = 4 `((n - 1,), (n↑ - n↓,))` per site (`HUB4_*`), spin `((2 S^z,),)`. The bond
# leg (leg 3, outgoing) carries the NEGATED left-block charge (dt3a: a doublon on site 1 reads
# `((-1,), (0,))` on bond 1).
_hub_decode(lab, i) = (N = i - only(lab[1]); s = -only(lab[2]); ((N + s) ÷ 2, (N - s) ÷ 2))
_spin_decode(lab, i) = (i - only(only(lab))) ÷ 2

"""
Grade the MPS sector ranks at every bond against ED Schmidt values `sv` (from
[`_sector_schmidt`](@ref)): every MPS sector is one ED has (charge conservation), holds at most
the sector's full rank, and at least its count of Schmidt values above `rtol * ||ψ||`.
Returns the worst `(missing, excess)` counts, both 0 when it passes.
"""
function _rank_check(dims::Dict, sv::Dict, nrm; rtol = 1e-6)
    miss = excess = 0
    for (c, d) in dims
        haskey(sv, c) || (excess += d; continue)
        excess += max(0, d - length(sv[c]))
    end
    for (c, s) in sv
        need = count(>(rtol * nrm), s)
        miss += max(0, need - get(dims, c, 0))
    end
    return miss, excess
end

# ── collective emission on the spin-1/2 chain ──

"Dense `(L₊, L₋)` with `L_± = √Γ_± Σ_j e^{±ikj} S⁻_j`, in `dense_state`'s basis."
_emission_jumps(L; gp, gm, k) =
    (sqrt(gp) * sum(cis(k * j) * _embed(_SM, j, L) for j in 1:L),
     sqrt(gm) * sum(cis(-k * j) * _embed(_SM, j, L) for j in 1:L))

"All `2^L` spin product states in `dense_state`'s order (site 1 most significant, 0 = up)."
_spin_states(L) = [product_state([((idx >> (L - j)) & 1) == 0 ? :up : :down for j in 1:L])
                   for idx in 0:(2^L - 1)]
_spin_charge(idx, L) = count(j -> ((idx >> (L - j)) & 1) == 0, 1:L)

# ── quantum-jump trajectories (Task 4) ──

"""
Collective emission (A2) on `L` spins: XXZ in a staggered field, `L_± = √Γ_± Σ_j e^{±ikj} S⁻_j`.
`start` is `:warm` (Neel after two TDVP2 steps of `H`) or a product-state occupation list.
Returns the MPOs `H, Q, G`, their dense `Hd, Qd`, `Gd = Hd - (i/2)Qd`, the dense jumps `Ld`, the
channels `ch` and `psi0`.
"""
function _emission_model(L; Jz = 1.0, gp = 1.0, gm = gp, k = π / 3, start = :warm)
    set_symmetry!(:U1)
    h = [0.2 * (-1)^j for j in 1:L]
    H = field_mpo_from_terms(xxz_chain(L; J = 1.0, delta = Jz), h)
    Q = collective_emission_mpo(L; gamma_plus = gp, gamma_minus = gm, k = k)
    Lp, Lm = _emission_jumps(L; gp = gp, gm = gm, k = k)
    Ld = [Matrix(Lp), Matrix(Lm)][[gp > 0, gm > 0]]
    Qd = sum(X' * X for X in Ld)
    Hd = Matrix(dense_heisenberg(L; J = 1.0, delta = Jz) +
                sum(h[j] * _embed(_SZ, j, L) for j in 1:L))
    psi0 = start === :warm ? neel_state(L) : product_state(start)
    if start === :warm
        for _ in 1:2
            tdvp2_step!(psi0, H, ComplexF64(-0.2im); maxdim = 64, trunc_thresh = 1e-14)
        end
    end
    ch = collective_emission_channels(L; gamma_plus = gp, gamma_minus = gm, k = k)
    return (; H, Q, G = no_jump_mpo(H, Q), Hd, Qd, Gd = Hd - 0.5im * Qd, Ld, ch, psi0)
end

"The excitation numbers a dense spin vector occupies."
_spin_support(v, L; tol = 1e-12) = Set(_spin_charge(b - 1, L) for b in eachindex(v) if abs(v[b]) > tol)

"`t -> exp(t A) v0` for a small diagonalisable `A`, from one eigendecomposition (the jump-time
bisection reads the flow hundreds of times)."
function _ed_flow(A, v0)
    F = eigen(Matrix(A))
    c = F.vectors \ v0
    return t -> F.vectors * (exp.(F.values .* t) .* c)
end

"The first jump time of threshold `u`: `||v(t)||² = u ||v0||²` on the exact no-jump `flow`, by
bisection on `[0, tmax]`; `Inf` if the survival stays above `u`."
function _ed_jump_time(flow, v0, u; tmax, tol = 1e-13)
    f(t) = norm(flow(t))^2 / norm(v0)^2 - u
    f(tmax) > 0 && return Inf
    a, b = 0.0, Float64(tmax)
    while b - a > tol
        m = (a + b) / 2
        f(m) > 0 ? (a = m) : (b = m)
    end
    return (a + b) / 2
end

"""
`(rhos, counts)` on the uniform grid `ts` (from 0): the Lindblad states `exp(t S) rho0` and the mean
jump count `∫_0^t Tr(Q rho) dt'`, Simpson on `nsub` (even) substeps per interval.
"""
function _lindblad_counts(S, Q, rho0, ts; nsub = 8, tol = 1e-12)
    h = (ts[2] - ts[1]) / nsub
    all(abs(ts[k + 1] - ts[k] - nsub * h) < 1e-12 for k in 1:(length(ts) - 1)) ||
        throw(ArgumentError("_lindblad_counts needs a uniform grid"))
    fine = [ts[1] + m * h for m in 1:(nsub * (length(ts) - 1))]
    rf = vcat([Matrix{ComplexF64}(rho0)], sparse_lindblad_trajectory(S, rho0, fine; tol = tol))
    q = [real(tr(Q * r)) for r in rf]
    counts = [0.0]
    for k in 1:(length(ts) - 1)
        y = q[((k - 1) * nsub + 1):(k * nsub + 1)]
        push!(counts, counts[end] + h / 3 * (y[1] + 4 * sum(y[2:2:end - 1]) +
                                               2 * sum(y[3:2:end - 2]; init = 0.0) + y[end]))
    end
    return rf[1:nsub:end], counts
end

"`(ok, worst)`: every ensemble mean within `z` standard errors plus `atol` of its exact value, and the
largest `|mean - exact| / (z stderr + atol)`."
function _within_ci(means, errs, exact; z = 4.0, atol = 2e-3)
    dev = maximum(abs(m - e) / (z * s + atol) for (m, s, e) in zip(means, errs, exact))
    return dev <= 1, dev
end

# ── the B1 junction and B2 counting statistics (Task 4) ──

"The Fock annihilator of mode `m` on the patterns `basis` (to the patterns `target`), dense."
function _fock_c(m, basis, target = basis)
    pos = Dict(b => k for (k, b) in enumerate(target))
    C = zeros(ComplexF64, length(target), length(basis))
    for (k, b) in enumerate(basis)
        sg, b2 = _annihilate(b, m)
        sg == 0 || haskey(pos, b2) && (C[pos[b2], k] = sg)
    end
    return C
end

"The bond current `j = iJ (c'_{b+1,s} c_{b,s} - h.c.)` (flow `b -> b+1` under the hopping `-J`) and
`n_{j,s}` on the full Fock space."
function _fock_current(L, b, s, J)
    bs = 0:(4^L - 1)
    c1, c2 = _fock_c(hub_mode(b, s), bs), _fock_c(hub_mode(b + 1, s), bs)
    return 1im * J * (c2' * c1 - c1' * c2)
end
_fock_n(L, j, s) = Diagonal(ComplexF64[hub_occ(b, hub_mode(j, s)) for b in 0:(4^L - 1)])

"The projector onto the dark states `ker L₊ ∩ ker L₋` of the dense jumps `Ld`."
function _dark_projector(Ld; tol = 1e-10)
    F = svd(reduce(vcat, Ld); full = true)
    r = count(>(tol), F.S)
    K = F.V[:, (r + 1):end]
    return K * K'
end

"""
The photon-count distribution `P(n, t) = Tr ρ_n(t)`, `n = 0..nmax`, of the jumps `Ld` from `rho0`:
the count-resolved master equation `dρ_n/dt = L0 ρ_n + J ρ_{n-1}`, `J ρ = Σ X ρ X'`, `L0 = L - J`,
with `ρ_nmax` absorbing every later count (so `Σ_n P = 1`). Row-major `vec` as `sparse_lindblad.jl`.
"""
function _count_distribution(Hd, Ld, rho0, t; nmax)
    D = size(Hd, 1)
    S = lindbladian_from(Hd, Ld)
    Jsup = sum(kron(sparse(X), sparse(conj(X))) for X in Ld)
    L0 = S - Jsup
    n = nmax + 1
    blocks = spzeros(ComplexF64, n * D^2, n * D^2)
    for k in 1:n
        r = ((k - 1) * D^2 + 1):(k * D^2)
        blocks[r, r] = k < n ? L0 : S                     # the last block keeps every later jump
        k > 1 && (blocks[r, r .- D^2] = Jsup)
    end
    v = zeros(ComplexF64, n * D^2)
    v[1:(D^2)] = vec_rm(ComplexF64.(rho0))
    w = arnoldi_expv_sparse(blocks, v, t; tol = 1e-12)
    return [real(tr(unvec_rm(w[((k - 1) * D^2 + 1):(k * D^2)], D))) for k in 1:n]
end
