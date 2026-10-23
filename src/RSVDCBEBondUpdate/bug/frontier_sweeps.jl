# frontier_sweeps.jl: the frontier half-sweeps and their measurements (from `krylov_closure.jl`, included right after it).

"""
    _frontier_left(Fm, mpo, fr, tau, alloc, ckw; enrich, gen, workcap, amp, rng, nsample)

The left half-sweep of one frontier MPS `Fm` (centre at the root, built from `fr`): bond by bond
the current frame `fr.W[i]` is embedded into the grown basis of bond `i-1` (`U0 = Pm W[i]`, an
overlap, not an assumption), the exact outside-space action of the frontier's two-site block is
measured, and -- if it exceeds the bond's share and `enrich` -- the step's CBE
([`_frontier_expand`](@ref)) and `_krylov_enrich` append validated directions to `U0`, which is
kept verbatim. Returns the new frames (re-tagged to generation `gen`), the root environment, the
bond ledger and the per-bond misses.

A MEASURING sweep (`enrich = false`) grows nothing, so it runs in the frames themselves: `U0 =
fr.W[i]` with no overlap `Pm` (identity there) and the generation's cached frame chain for `lch`
([`_side_cache!`](@ref)). `chain` returns the `lch` an enriching sweep built from the new frames,
which is the next generation's chain. With `bonds` its bonds are measured concurrently
([`_bond_measurements`](@ref)) and the loop only logs them.
"""
function _frontier_left(Fm::SymMPS, mpo, fr::KLFrames, tau::ComplexF64,
                        alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool, gen::Int,
                        workcap::Int, amp::Float64, rng::AbstractRNG, nsample::Int = 0,
                        fold::Bool = false, bonds::Bool = false)
    c, L = fr.c, length(Fm)
    canonical!(Fm, 1)
    chain, zst = _side_cache!(fr, Fm, mpo, :left)
    E = copy(zst)                                # links >= c+1: shared by every frontier
    env = E[c + 1]
    for k in c:-1:3                              # links 3..c: this frontier's own gauge
        env = _mpo_right_step(env, Fm[k], mpo[k])
        E[k] = env
    end
    rstack = MPORightEnvStack(E)
    pre = bonds && !enrich ? _bond_measurements(Fm, mpo, fr, chain, rstack, :left; fold = fold,
                                                nsample = nsample, rng = rng) : nothing
    lch = boundary_channels(mpo)
    lenv = lch
    lchs = Vector{Any}(undef, L - 1)
    C, Pm = nothing, nothing
    Wn = Vector{Any}(undef, c)
    lg = AdaptiveLedger()
    leak, sleak = zeros(c), zeros(c)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for i in 1:c
        enrich || (lch = chain[i])
        U0 = _check_frame(Pm === nothing ? fr.W[i] : to_concrete!(contract(Pm, (2,), fr.W[i], (1,))),
                          :left, "frontier U0 = Pm W[$i] (gen $(fr.gen))")
        if pre === nothing
            K = C === nothing ? Fm[i] : to_concrete!(contract(C, (2,), Fm[i], (1,)))
            f = _frame_onto_left(U0, K, Fm[i + 1])
            rch = right_channels(rstack, i + 2)
            skl, skr, leak[i], k2 = _bond_action(f, mpo, i, lch, rch, U0, nothing, :left, fold,
                                                 rng)
            nsample > 0 && (sleak[i] = _sampled_leak(f, skl, skr, :left, nsample, rng))
        else
            K, leak[i], k2, sleak[i] = pre[i]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[i]) <= alloc.per_bond
        Wi = if enrich && !quiet
            ex = _frontier_expand(f, mpo, i, lch, rch, :left, ckw, bl, rng)
            M = to_concrete!(contract(Fm[i + 1], (2, 3), ex.V_ex', (2, 3)))
            Kex = to_concrete!(contract(K, (3,), M, (1,)))
            H1 = shift_h(one_site_h(mpo, i, lch, push_right_channels(rch, mpo, ex.V_ex, i + 1)),
                         fr.eshift)
            reuse = get(ckw, :krylov_reuse, true)
            Q, st = _krylov_enrich(Kex, x -> (nh1[] += 1; apply_one_site(H1, x)),
                                   reuse ? ex.U_ex : U0, :left, tau, alloc; workcap = workcap,
                                   amp = amp, state_seed = false)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.U_ex, :left)
        else
            nskip += 1
            U0
        end
        Wn[i] = enrich ? to_concrete(setitag(Wi, 3, _tagWg(i, gen))) : U0
        _check_frame(Wn[i], :left, "frontier Wn[$i] (gen $gen)")
        nadd = leg_dim(Wn[i], 3) - leg_dim(U0, 3)
        nsec = _new_sectors(U0, Wn[i], 3)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, i, :left, leg_dim(Wn[i], 3))
        _tag_last!(lg, (leak = indicator(alloc, leak[i]), skipped = quiet || !enrich,
                        added = nadd, new_sectors = nsec, width_before = leg_dim(U0, 3)))
        enrich && (Pm = to_concrete!(contract(Wn[i]', (1, 2), U0, (1, 2))))  # (new_i, old_i)
        pre === nothing && (C = to_concrete!(contract(Wn[i]', (1, 2), K, (1, 2))))
        lchs[i] = lch
        i == c && (lenv = lch)
        enrich && i < c && (lch = push_left_channels(lch, mpo, Wn[i], i))
    end
    return (frames = Wn, env = lenv, chain = lchs, ledger = lg, leak = leak, sleak = sleak,
            added = added, new_sectors = newsec, skipped = nskip, n_h2 = nh2, n_h1 = nh1[])
end

"Row mirror of [`_frontier_left`](@ref): grows `fr.Z[L-1..c]`."
function _frontier_right(Fm::SymMPS, mpo, fr::KLFrames, tau::ComplexF64,
                         alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool, gen::Int,
                         workcap::Int, amp::Float64, rng::AbstractRNG, nsample::Int = 0,
                         fold::Bool = false, bonds::Bool = false)
    c, L = fr.c, length(Fm)
    canonical!(Fm, L)
    chain, wst = _side_cache!(fr, Fm, mpo, :right)
    E = copy(wst)                                # links <= c: shared by every frontier
    env = E[c]
    for i in c:(L - 2)                           # links c+1..L-1: this frontier's own gauge
        env = _mpo_left_step(env, Fm[i], mpo[i])
        E[i + 1] = env
    end
    lstack = MPOLeftEnvStack(E)
    pre = bonds && !enrich ? _bond_measurements(Fm, mpo, fr, chain, lstack, :right; fold = fold,
                                                nsample = nsample, rng = rng) : nothing
    rch = boundary_channels(mpo)
    renv = rch
    rchs = Vector{Any}(undef, L - 1)
    D, Pm = nothing, nothing
    Zn = Vector{Any}(undef, L - 1)
    lg = AdaptiveLedger()
    leak, sleak = zeros(L - 1), zeros(L - 1)
    added, newsec, nskip, nh2 = 0, 0, 0, 0
    nh1 = Ref(0)
    for j in (L - 1):-1:c
        enrich || (rch = chain[j])
        V0 = _check_frame(Pm === nothing ? fr.Z[j] : to_concrete!(contract(fr.Z[j], (3,), Pm, (1,))),
                          :right, "frontier V0 = Z[$j] Pm (gen $(fr.gen))")
        if pre === nothing
            B = D === nothing ? Fm[j + 1] : to_concrete!(contract(Fm[j + 1], (3,), D, (1,)))
            f = _frame_onto_right(Fm[j], B, V0)
            lch = left_channels(lstack, j)
            skl, skr, leak[j], k2 = _bond_action(f, mpo, j, lch, rch, nothing, V0, :right, fold,
                                                 rng)
            nsample > 0 && (sleak[j] = _sampled_leak(f, skl, skr, :right, nsample, rng))
        else
            B, leak[j], k2, sleak[j] = pre[j]
        end
        nh2 += k2
        bl = AdaptiveLedger()
        quiet = indicator(alloc, leak[j]) <= alloc.per_bond
        Zj = if enrich && !quiet
            ex = _frontier_expand(f, mpo, j, lch, rch, :right, ckw, bl, rng)
            M = to_concrete!(contract(ex.U_ex', (1, 2), Fm[j], (1, 2)))
            Bex = to_concrete!(contract(M, (2,), B, (1,)))
            H1 = shift_h(one_site_h(mpo, j + 1, push_left_channels(lch, mpo, ex.U_ex, j), rch),
                         fr.eshift)
            reuse = get(ckw, :krylov_reuse, true)
            Q, st = _krylov_enrich(Bex, x -> (nh1[] += 1; apply_one_site(H1, x)),
                                   reuse ? ex.V_ex : V0, :right, tau, alloc; workcap = workcap,
                                   amp = amp, state_seed = false)
            _record_krylov!(bl, st, alloc)
            reuse ? Q : _merge_frame(Q, ex.V_ex, :right)
        else
            nskip += 1
            V0
        end
        Zn[j] = enrich ? to_concrete(setitag(Zj, 1, _tagZg(j, gen))) : V0
        _check_frame(Zn[j], :right, "frontier Zn[$j] (gen $gen)")
        nadd = leg_dim(Zn[j], 1) - leg_dim(V0, 1)
        nsec = _new_sectors(V0, Zn[j], 1)
        added += nadd
        newsec += nsec
        _log_bond!(lg, bl, j, :right, leg_dim(Zn[j], 1))
        _tag_last!(lg, (leak = indicator(alloc, leak[j]), skipped = quiet || !enrich,
                        added = nadd, new_sectors = nsec, width_before = leg_dim(V0, 1)))
        enrich && (Pm = to_concrete!(contract(V0, (2, 3), Zn[j]', (2, 3))))  # (old_j, new_j)
        pre === nothing && (D = to_concrete!(contract(B, (2, 3), Zn[j]', (2, 3))))
        rchs[j] = rch
        j == c && (renv = rch)
        enrich && j > c && (rch = push_right_channels(rch, mpo, Zn[j], j + 1))
    end
    return (frames = Zn, env = renv, chain = rchs, ledger = lg, leak = leak, sleak = sleak,
            added = added,
            new_sectors = newsec, skipped = nskip, n_h2 = nh2, n_h1 = nh1[])
end

_frame_width(fr::KLFrames, L::Int) =
    max(maximum(leg_dim(fr.W[i], 3) for i in 1:fr.c),
        maximum(leg_dim(fr.Z[j], 1) for j in fr.c:(L - 1)))

"""
    _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; enrich, workcap, nrm, parallel, rng,
                     pass, nsample) -> NamedTuple

Test one frontier's outside-space action on both half-sweeps and, when `enrich`, grow `fr` by the
validated directions (plan: `enrich_KL_from_frontiers!`). The frames are replaced -- and their
generation advanced -- only if something was added, so frontiers built in them stay valid
otherwise. `leak` is the time-weighted outside-space action `|tau| sqrt(left^2 + right^2) /
||psi||` of the frontier BEFORE this sweep enriched; `met` is whether every bond ended within
its share.
"""
function _frontier_sweep!(fr::KLFrames, psi::SymMPS, mpo, F::KrylovFrontier, tau::ComplexF64,
                          alloc::BasisAllocation, ckw::NamedTuple; enrich::Bool = true,
                          workcap::Int = 0, nrm::Float64 = 1.0, parallel::Bool = false,
                          rng::AbstractRNG = MersenneTwister(0x5EED), pass::Int = 0,
                          nsample::Int = 0, fold::Bool = false)
    F.gen == fr.gen || throw(ArgumentError(
        "stale frontier: built in frames generation $(F.gen), the frames are at $(fr.gen) -- " *
        "the root operator and frontiers must be rebuilt whenever the frames change"))
    t0 = time_ns()
    L, c = length(psi), fr.c
    nF = norm(F.core)
    zero_leaks = (zeros(c), zeros(L - 1))
    if nF <= _FRONTIER_FLOOR * max(nrm, eps())
        lg = AdaptiveLedger()
        push!(lg.closure_log, (pass = pass, frontier = F.degree, enrich = enrich,
                               norm = nF / max(nrm, eps()), leak = 0.0, added = 0,
                               zero = true, t = 0.0))
        return (added = 0, leak = 0.0, leak_left = 0.0, leak_right = 0.0, met = true,
                capped = false, ledger = lg, leaks_left = zero_leaks[1], leaks_right = zero_leaks[2],
                sleaks_left = zero_leaks[1], sleaks_right = zero_leaks[2])
    end
    Fm = _frontier_mps(psi, fr, F.core)
    FmR = copy(Fm)
    gen = fr.gen + 1
    amp = nF / max(nrm, eps())
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    kw(r) = (enrich = enrich, gen = gen, workcap = workcap, amp = amp, rng = r,
             nsample = nsample, fold = fold,
             bonds = !enrich && _BOND_TASKS[] && _frontier_parallel(parallel))
    rl, rr = if parallel
        tR = Threads.@spawn _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...)
        l = _frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...)
        (l, fetch(tR))
    else
        (_frontier_left(Fm, mpo, fr, tau, alloc, ckw; kw(rng)...),
         _frontier_right(FmR, mpo, fr, tau, alloc, ckw; kw(rngR)...))
    end
    lg = AdaptiveLedger()
    for half in (rl.ledger, rr.ledger)
        half.bond_log = Any[merge(e, (pass = pass, frontier = F.degree)) for e in half.bond_log]
        merge_ledger!(lg, half)
    end
    lL, lR = sqrt(sum(abs2, rl.leak)), sqrt(sum(abs2, rr.leak))
    leak = indicator(alloc, sqrt(lL^2 + lR^2))
    met = enrich ? lg.tolerance_met :
          all(indicator(alloc, x) <= alloc.per_bond for x in vcat(rl.leak, rr.leak))
    added = rl.added + rr.added
    wbefore = _frame_width(fr, L)
    if enrich && added > 0
        for i in 1:c
            fr.W[i] = rl.frames[i]
        end
        for j in c:(L - 1)
            fr.Z[j] = rr.frames[j]
        end
        fr.lenv, fr.renv, fr.gen = rl.env, rr.env, gen
        # the channels this sweep carried through the new frames ARE the new generation's chain
        fr.lcache = (gen = gen, chain = rl.chain, stack = nothing)
        fr.rcache = (gen = gen, chain = rr.chain, stack = nothing)
    end
    lg.n_closure_sweeps = 1
    lg.n_h2_leak = rl.n_h2 + rr.n_h2
    lg.n_h1_closure = rl.n_h1 + rr.n_h1
    push!(lg.closure_log,
          (pass = pass, frontier = F.degree, enrich = enrich, norm = amp, leak = leak,
           leak_left = indicator(alloc, lL), leak_right = indicator(alloc, lR), met = met,
           added = added, added_left = rl.added, added_right = rr.added,
           new_sectors = rl.new_sectors + rr.new_sectors,
           skipped = rl.skipped + rr.skipped, bonds = c + (L - c),
           width_before = wbefore, width_after = _frame_width(fr, L),
           n_h2 = lg.n_h2_leak, n_h1 = lg.n_h1_closure, n_probes = lg.n_probes,
           n_selected = lg.n_selected, n_krylov = lg.n_krylov, zero = false,
           capped = enrich && lg.workcap_reached, t = (time_ns() - t0) / 1e9))
    # `capped`: an enrichment of this sweep stopped at the work ceiling (`_closure_loop!`).
    return (added = added, leak = leak, leak_left = indicator(alloc, lL),
            leak_right = indicator(alloc, lR), met = met,
            capped = enrich && lg.workcap_reached, ledger = lg,
            leaks_left = rl.leak, leaks_right = rr.leak,
            sleaks_left = rl.sleak, sleaks_right = rr.sleak)
end

"""
    estimate_global_complement_action(psi, mpo, fr, core; nsample = 0, rng) -> NamedTuple

The outside-space action `(I - P) H v` of the Galerkin vector `v = (W ⊗ Z) core`, matrix-free,
from the per-bond two-site misses of a non-enriching frontier sweep (no time weight).

Writing `P^(i)` for the projector onto the left frame chain up to bond `i`, `I - P_L =
sum_i (P^(i-1) - P^(i))` is a sum of ORTHOGONAL pieces, and bond `i`'s miss
`||(I - U0 U0') H Theta||` is exactly the norm of piece `i` whenever every term of `H` that acts
left of the cut also stops at site `i+1` (nearest neighbour): the right environment then spans
everything the term leaves on the right. So, for a nearest-neighbour `H`,

    left  = ||(I - P_L ⊗ 1) H v||,     right = ||(1 ⊗ (I - P_R)) H v||   (exactly)
    max(left, right) <= ||(I - P) H v|| <= bound = sqrt(left^2 + right^2)

For longer-range terms the per-bond numbers miss what a term leaves outside the right
environment's span, i.e. they become an estimate. `nsample > 0` adds the randomised per-bond
estimates (`sampled_*`). Returns `left`, `right`, `bound`, `per_left[1..c]`, `per_right[c..L-1]`
and, if sampled, `sampled_left`, `sampled_right`, `per_sampled_left`, `per_sampled_right`.
"""
function estimate_global_complement_action(psi::SymMPS, mpo, fr::KLFrames, core;
                                           nsample::Int = 0,
                                           rng::AbstractRNG = MersenneTwister(0x5EED))
    L, c = length(psi), fr.c
    alloc = basis_allocation(AdaptiveBudget(1.0, 0.0, 1.0), L - 1)    # scale 1: raw norms
    r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(core, 0, fr.gen), ComplexF64(0), alloc,
                         NamedTuple(); enrich = false, nrm = max(norm(core), eps()),
                         nsample = nsample, rng = rng)
    pl, pr = r.leaks_left, r.leaks_right[c:(L - 1)]
    sl, sr = r.sleaks_left, r.sleaks_right[c:(L - 1)]
    left, right = sqrt(sum(abs2, pl)), sqrt(sum(abs2, pr))
    return (left = left, right = right, bound = sqrt(left^2 + right^2),
            per_left = pl, per_right = pr,
            sampled_left = sqrt(sum(abs2, sl)), sampled_right = sqrt(sum(abs2, sr)),
            per_sampled_left = sl, per_sampled_right = sr)
end

"""
    _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, m; nrm, parallel, rng, pass = 0)
        -> NamedTuple

Measure -- never enrich -- the frontiers `j = 0..m-1` of `S0` on the frames `fr` as they stand:
what the frames already capture of the global Krylov sequence. On the half-sweep frames this is
the ordinary step's own closure, `rho_j = ||(I - P) A F_j|| / ||F_j||` (the bound
`sqrt(left^2 + right^2)`), with no closure sweep involved. `fr` is untouched. Returns the
time-weighted `leak[j+1]`, the scale-free `rho[j+1]`, `met` (every bond of every frontier
`j >= 1` within its share -- `j = 0` is the half-sweeps' own test), the merged `ledger` and
`n_root` zero-site applications; `met0` is the `j = 0` verdict.

The frontiers are independent measurements on one set of frames, so with `parallel` and at
least [`_FRONTIER_THREADS`](@ref) threads they run CONCURRENTLY (each with its own RNG stream
drawn from `rng`, after the shared environments are built once by [`_prime_caches!`](@ref));
otherwise in order, exactly as before. The results are merged in `j` order either way.
"""
function _measure_frontiers(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                            alloc::BasisAllocation, ckw::NamedTuple, m::Int;
                            nrm::Float64 = 1.0, parallel::Bool = false,
                            rng::AbstractRNG = MersenneTwister(0x5EED), pass::Int = 0,
                            workcap::Int = 0, fold::Bool = false)
    H0 = _root_generator(mpo, fr)
    nroot = Ref(0)
    F = m > 1 ? _projected_frontiers(x -> (nroot[] += 1; apply_zero_site(H0, x)), S0, tau,
                                     m - 1) : Any[]
    lg = AdaptiveLedger()
    leak, rho, met, met0 = zeros(m), zeros(m), true, true
    cores = Any[j == 0 ? S0 : F[j] for j in 0:(m - 1)]
    measure(j, r) = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(cores[j + 1], j, fr.gen), tau,
                                     alloc, ckw; enrich = false, workcap = workcap, nrm = nrm,
                                     parallel = parallel, rng = r, pass = pass, fold = fold)
    rs = if _frontier_parallel(parallel) && m > 1
        _prime_caches!(fr, psi, mpo, S0)
        rngs = [MersenneTwister(rand(rng, UInt64)) for _ in 1:m]
        fetch.([Threads.@spawn(measure(j, rngs[j + 1])) for j in 0:(m - 1)])
    else
        [measure(j, rng) for j in 0:(m - 1)]
    end
    for j in 0:(m - 1)
        core, r = cores[j + 1], rs[j + 1]
        merge_ledger!(lg, r.ledger)
        nF = norm(core)
        raw = sqrt(sum(abs2, r.leaks_left) + sum(abs2, r.leaks_right))
        leak[j + 1] = r.leak
        rho[j + 1] = nF > 0 ? raw / nF : 0.0
        j >= 1 ? (met &= r.met) : (met0 = r.met)
    end
    lg.leak_initial, lg.rho_initial = leak, rho
    lg.n_root_frontier = nroot[]
    return (leak = leak, rho = rho, met = met, met0 = met0, ledger = lg, n_root = nroot[])
end

"""
    _closure_loop!(frontier, sweep!, order; maxpass, include_zero = true) -> NamedTuple

The termination logic of the order-`p` closure, independent of how a space is stored.
`frontier(j)` builds frontier `j` in the CURRENT space; `sweep!(F, j, pass)` tests it and
enriches, returning at least `(added, leak, met)`.

Pass 1 sweeps `j = 0..p-1` (`j = 1..p-1` when the caller already closed `j = 0`); later passes
sweep `j = 1..p-1` only -- the spaces are nested, so a closed `j = 0` (whose frontier is `psi`
itself and does not depend on `B`) stays closed. Each frontier is rebuilt immediately before its
sweep, so it reflects every enrichment before it. The loop is CONVERGED after a pass in which
nothing was added and every sweep met its share: every frontier was then built from the final
`B` and tested against the final frames. A pass that adds nothing but fails a share stops too
(not converged), so every non-final pass grows the space by at least one direction: termination
is guaranteed by the work ceiling, and `maxpass` is a safety bound whose hit is reported.

THE WORKING-RANK LIMIT ENDS THE LOOP TOO (plan step 3: "until every ... estimate fits the
internally allocated `tol_basis` budget OR A DERIVED WORKING-RANK LIMIT IS REACHED"). A pass in
which some enriching sweep hit the work ceiling (`r.capped`) and a share was still unmet stops
the loop, not converged (`capped = true`): the bonds that leak cannot widen, so another pass
re-measures the same leak (10x4 D=256/512 order 2 and 4, jobs 17284922/17285201: every pass
after the first added nothing and left the leak unchanged). The final `j = 0` measurement still
runs; the `j >= 1` leaks are the last measured, before that pass's enrichment.
"""
function _closure_loop!(frontier, sweep!, order::Int; maxpass::Int,
                        include_zero::Bool = true)
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    log = Any[]
    later = 1:(order - 1)
    initial = include_zero ? (0:(order - 1)) : later
    isempty(initial) &&
        return (passes = 0, converged = true, exhausted = false, capped = false, log = log)
    ok0 = true
    for pass in 1:maxpass
        grew, allmet, capped = false, true, false
        for j in (pass == 1 ? initial : later)
            r = sweep!(frontier(j), j, pass)
            push!(log, (pass = pass, frontier = j, added = r.added, leak = r.leak, met = r.met))
            grew |= r.added > 0
            allmet &= r.met
            capped |= get(r, :capped, false)
            j == 0 && (ok0 = r.met)
        end
        (!grew || isempty(later)) &&
            return (passes = pass, converged = allmet && ok0, exhausted = false,
                    capped = false, log = log)
        capped && !allmet &&
            return (passes = pass, converged = false, exhausted = false, capped = true,
                    log = log)
    end
    return (passes = maxpass, converged = false, exhausted = true, capped = false, log = log)
end
