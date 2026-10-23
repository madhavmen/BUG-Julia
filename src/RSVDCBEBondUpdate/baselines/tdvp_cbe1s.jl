# ── one-site TDVP with CBE on the MPO path: a line-for-line port of TDVPSweepCBE1Si.m ──
# The backward 0-site step (-tau) removes the double count of the bond between two forward updates.
# MPO path: needed for long-range H (Haldane-Shastry), which tdvp1_cbe.jl cannot carry.

"""
    TDVPCBEInfo

What one [`tdvp_cbe1s_step!`](@ref) did.

  - `max_bond`, `max_expanded`: final rank, and the widest CBE proposal before the split. Compare
    them to see whether the expansion or the truncation is limiting.
  - `n_new`, `err_pre`, `err_fnl`, `discarded`: directions admitted and the largest discarded weights.
  - `krylov_dims`: operator applications over both solves; not comparable with a 0-site-only scheme.
"""
struct TDVPCBEInfo
    max_bond::Int
    max_expanded::Int
    n_new::Vector{Int}
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dims::Int
    peak_elements::Int
end

"""
    tdvp_cbe1s_step!(psi, mpo, tau; kwargs...) -> TDVPCBEInfo

One symmetric second-order one-site TDVP-with-CBE step of `tau`, in place. `tau = -im*dt` is real
time and `tau = -dt` imaginary time. The caller renormalises; for imaginary time that is load-bearing.

The expansion is [`cbe_expand`](@ref) with its fixed-count settings. `exact = false` (default) is the
RSVD selection; `exact = true` the full fused basis. `maxdim` and `trunc_thresh` bound the splits.

`reorth = true` is a correctness fix: without it the Lanczos silently loses norm.
"""
function tdvp_cbe1s_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                          exact::Bool = false,
                          # ── fold_omega: fold the sketch before the environment contractions ──
                          # Without it every bond builds the full H*Theta for basis selection alone. Folding saves chi/g and so
                          # grows with chi, but it can lose at small chi or a wide sketch; the default stays off.
                          fold_omega::Bool = false,
                          maxdim::Int = 200,
                          trunc_thresh::Float64 = 1e-12,
                          maxiter::Int = 30,
                          tol::Float64 = 1e-15,
                          reorth::Bool = true,
                          # ── conv_tol: early exit of the Krylov solves, a fairness fix ──
                          # `lanczos_expv` exits on breakdown only, so every solve burned `maxiter`. CBE-BUG already stops
                          # adaptively, so timing against `conv_tol = 0` would credit that difference to the algorithm.
                          conv_tol::Float64 = 0.0,
                          substeps::Int = 1,
                          # `hermitian = false` -> Arnoldi, for a non-Hermitian generator. See
                          # the note in `cbe_bug_step!`: Lanczos does not fail on one, it lies.
                          hermitian::Bool = true,
                          rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("tdvp_cbe1s_step! needs at least two sites"))

    cut = max(trunc_thresh, 1e-14)
    half = tau / 2
    n_new = zeros(Int, L - 1)
    maxexp = 0; epre = 0.0; efnl = 0.0; disc = 0.0; nmv = 0

    # `share_ht` is deliberately NOT in here: the MPO `cbe_expand` REFUSES the pair, and its
    # default (`nothing`) already means "share when project-first" -- so leaving it out is what
    # lets one `exkw` serve both paths.
    exkw = (rmax = maximum(bond_dims(psi); init = 0), exact = exact, fold_omega = fold_omega,
            rng = rng)

    # Working set at this instant: the state as it stands plus the transients still alive. SAME
    # DEFINITION as `CBEBugSweepInfo.peak_elements` and `TDVP2Info.peak_elements`, which is the
    # only reason the three are comparable -- a memory claim measured three different ways is not
    # a claim. It was missing here, so the campaign's `peak` column read 0 for this sweep.
    peak = 0
    peak!(ts...) = (peak = max(peak, _state_stored(psi) +
                               sum(tensor_elements(t) for t in ts; init = 0)))

    function record!(info)
        maxexp = max(maxexp, info.dim_out)
        n_new[info.bond] += info.n_new
        epre = max(epre, info.err_pre)
        efnl = max(efnl, info.err_fnl)
    end

    "One-site propagation at site `j`, no split -- the reference's `'vv'` closing update."
    function onesite!(j, lch, rch)
        H1 = one_site_h(mpo, j, lch, rch)
        psi[j] = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[j];
                      hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
                      conv_tol = conv_tol, substeps = substeps)
        psi.center = j
    end

    # ======== left to right ========
    canonical!(psi, 1)
    rstack = right_env_stack(psi, mpo; downto = 3)
    lch = boundary_channels(mpo)
    for i in 1:(L - 1)
        rch1, info = expand_bond!(psi, mpo, i, :right, lch,
                                  right_channels(rstack, i + 2); exkw...)
        record!(info)

        # --- forward one-site update at site i, +tau/2 ---
        H1 = one_site_h(mpo, i, lch, rch1)
        M = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[i];
                 hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
                 conv_tol = conv_tol, substeps = substeps)

        peak!(M)
        res = svd(M, (1, 2); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i] = res.U
        C = to_concrete(res.S * res.Vd)

        # --- backward bond (0-site) update, -tau/2 ---
        # The environments are pushed through the tensors AS THEY NOW STAND: `psi[i]` is the
        # freshly written left isometry and `psi[i+1]` the expanded right frame, so this is
        # exactly the reference's `Update0Si(OrthC, VL, VR)` with its updated `VL` and its
        # expanded `VR`.
        H0 = zero_site_h(mpo, i, lch, right_channels(rstack, i + 2), psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -half, C;
                 hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
                 conv_tol = conv_tol, substeps = substeps)

        # Retag BOTH ends of the bond in one breath: each result is rank-3 with only one leg
        # wanting `tag`, so neither construction ever sees a duplicate index.
        psi[i + 1] = to_concrete(setitag(
            to_concrete!(contract(C, (2,), psi[i + 1], (1,))), 1, tag))
        psi[i] = to_concrete(setitag(psi[i], 3, tag))
        psi.center = i + 1

        lch = push_left_channels(lch, mpo, psi[i], i)
    end
    onesite!(L, lch, boundary_channels(mpo))

    # ======== right to left ========
    lstack = left_env_stack(psi, mpo; upto = L - 2)
    rch = boundary_channels(mpo)
    for i in (L - 1):-1:1
        lch1, info = expand_bond!(psi, mpo, i, :left,
                                  left_channels(lstack, i), rch; exkw...)
        record!(info)

        H1 = one_site_h(mpo, i + 1, lch1, rch)
        M = expv(x -> (nmv += 1; apply_one_site(H1, x)), half, psi[i + 1];
                 hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
                 conv_tol = conv_tol, substeps = substeps)

        res = svd(M, (1,); cutoff = cut, Nkeep = maxdim, get_lists = true)
        disc = max(disc, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i + 1] = res.Vd
        C = to_concrete(res.U * res.S)

        H0 = zero_site_h(mpo, i, left_channels(lstack, i), rch, psi[i], psi[i + 1])
        C = expv(x -> (nmv += 1; apply_zero_site(H0, x)), -half, C;
                 hermitian = hermitian, maxiter = maxiter, tol = tol, reorth = reorth,
                 conv_tol = conv_tol, substeps = substeps)

        psi[i] = to_concrete(setitag(
            to_concrete!(contract(psi[i], (3,), C, (1,))), 3, tag))
        psi[i + 1] = to_concrete(setitag(psi[i + 1], 1, tag))
        psi.center = i

        rch = push_right_channels(rch, mpo, psi[i + 1], i + 1)
    end
    onesite!(1, boundary_channels(mpo), rch)

    return TDVPCBEInfo(maximum(bond_dims(psi); init = 0), maxexp, n_new,
                       epre, efnl, disc, nmv, peak)
end

tdvp_cbe1s_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    tdvp_cbe1s_step!(psi, mpo, ComplexF64(tau); kwargs...)

tdvp_cbe1s_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    tdvp_cbe1s_step!(psi, mpo_from_terms(h), tau; kwargs...)
