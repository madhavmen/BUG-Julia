# tests/common/dissipative_fixtures.jl -- the lossy generators and small exact references of the
# dissipative plan's tests (test_dissipative_t1.jl, test_dissipative_t2.jl). Needs
# `adaptive_fixtures.jl`, `dense_reference.jl` and `dd_fixtures.jl`.

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
`(H, Q, jumps)` of the Hubbard chain with pair loss on the full Fock space (`4^L`, bit `m` = mode
`m`), `H = -J Σ (c'c + h.c.) + U Σ n↑n↓`, `L_j = √κ_j c_j↓ c_j↑`. `wrong = true` is the CONTROL:
the up-spin hop between `j` and `j+1` takes an extra `(-1)^{n_j↓}`, a Hermitian sign error that no
basis phase removes.
"""
function _hub_fock(L; J = 1.0, U = 0.0, kappa = zeros(L), wrong = false)
    D = 4^L
    H = zeros(ComplexF64, D, D)
    jumps = [zeros(ComplexF64, D, D) for _ in 1:L]
    for b in 0:(D - 1)
        for j in 1:L
            nu, nd = hub_occ(b, hub_mode(j, 0)), hub_occ(b, hub_mode(j, 1))
            H[b + 1, b + 1] += U * nu * nd
            s1, b1 = _annihilate(b, hub_mode(j, 0))
            s2, b2 = s1 == 0 ? (0, b1) : _annihilate(b1, hub_mode(j, 1))
            s2 == 0 || (jumps[j][b2 + 1, b + 1] = sqrt(kappa[j]) * s1 * s2)
        end
        for j in 1:(L - 1), σ in 0:1
            m, mp = hub_mode(j, σ), hub_mode(j + 1, σ)
            for (a, c) in ((m, mp), (mp, m))
                s = hub_sign(b, a, c)
                s == 0 && continue
                wrong && σ == 0 && hub_occ(b, hub_mode(j, 1)) == 1 && (s = -s)
                H[(b ⊻ (1 << a) ⊻ (1 << c)) + 1, b + 1] += -J * s
            end
        end
    end
    return H, sum(Lj' * Lj for Lj in jumps), jumps
end

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
    S = Diagonal(s)
    return norm(M - S * A * S') / norm(A)
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
