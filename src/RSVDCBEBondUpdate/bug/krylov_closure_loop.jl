# krylov_closure_loop.jl: the closure loop of `order = p` steps (from `krylov_closure.jl`, included
# right after it).

"""
    _krylov_closure!(fr, psi, mpo, S0, tau, alloc, ckw; order, workcap, nrm, parallel, rng,
                     ledger, include_zero = false, maxpass, checks = false) -> NamedTuple

Close the frontiers `j = 0..order-1` of the frames `fr` in place, for `psi` with root coefficients
`S0`. `checks = false` (default) runs one enriching sweep per frontier, each rebuilt in the frames
the previous one grew, and each sweep measures its own leaks. `checks = true` first measures every
frontier on the incoming frames ([`_measure_frontiers`](@ref)) and returns `fr` untouched if all
close; otherwise it enriches and then measures `j = 0` on the final frames. Non-convergence is
`:closure`, never a silent pass; `capped` marks a stop at the working-rank limit
([`_closure_loop!`](@ref)).
"""
function _krylov_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                          alloc::BasisAllocation, ckw::NamedTuple; order::Int,
                          workcap::Int = 0, nrm::Float64 = 1.0, parallel::Bool = false,
                          rng::AbstractRNG = MersenneTwister(0x5EED),
                          ledger::Union{Nothing, AdaptiveLedger} = nothing,
                          include_zero::Bool = false,
                          maxpass::Int = _closure_maxpass(order), fold::Bool = false,
                          checks::Bool = false)
    t0 = time_ns()
    lg = AdaptiveLedger()
    lg.order = order
    sweepkw = (workcap = workcap, nrm = nrm, parallel = parallel, rng = rng, fold = fold)
    checks || return _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0;
                                           order = order, include_zero = include_zero,
                                           ledger = ledger, sweepkw...)
    # CHECK FIRST: what the incoming frames (the ordinary CBE + local-Krylov half-sweeps)
    # already capture of the global Krylov sequence. If they close every frontier, no closure
    # sweep runs and the frames stay exactly as the half-sweeps built them.
    pre = _measure_frontiers(fr, psi, mpo, S0, tau, alloc, ckw, order; sweepkw...)
    merge_ledger!(lg, pre.ledger)
    nroot = Ref(pre.n_root)
    S = Ref{Any}(S0)
    final_leak = Dict{Int, Float64}(j - 1 => pre.leak[j] for j in 1:order)
    gen0 = fr.gen
    res = if pre.met && (pre.met0 || !include_zero)
        (passes = 0, converged = true, exhausted = false, capped = false, log = Any[])
    else
        H0 = Ref{Any}(_root_generator(mpo, fr))
        applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
        frontier = j -> KrylovFrontier(j == 0 ? S[] :
                                       _projected_frontiers(applyB, S[], tau, j)[j], j, fr.gen)
        # ⚠ SPECULATIVE FINAL MEASUREMENT (>= _FRONTIER_THREADS threads). A later pass usually
        # adds nothing and is then the last; its last sweep cannot change the frames unless it
        # adds. So `j = 0` is measured CONCURRENTLY on a snapshot of the frames, and kept only
        # if the loop ends in that generation -- otherwise it is discarded and redone below.
        spec = Ref{Any}(nothing)
        r0kw = (workcap = workcap, nrm = nrm, parallel = parallel, fold = fold, enrich = false)
        sweep! = (F, j, pass) -> begin
            if _frontier_parallel(parallel) && pass >= 2 && j == order - 1
                spec[] === nothing || fetch(spec[].task)
                snap = KLFrames(copy(fr.W), copy(fr.Z), fr.c, fr.lenv, fr.renv, fr.gen,
                                fr.lcache, fr.rcache, fr.eshift)
                F0 = KrylovFrontier(S[], 0, snap.gen)
                rs0 = MersenneTwister(rand(rng, UInt64))
                spec[] = (gen = snap.gen, task = Threads.@spawn(
                    _frontier_sweep!(snap, psi, mpo, F0, tau, alloc, ckw; r0kw...,
                                     rng = rs0, pass = pass + 1)))
            end
            r = _frontier_sweep!(fr, psi, mpo, F, tau, alloc, ckw; sweepkw..., pass = pass)
            merge_ledger!(lg, r.ledger)
            final_leak[j] = r.leak
            if r.added > 0
                H0[] = _root_generator(mpo, fr)
                S[] = _project_root(psi, fr)
            end
            r
        end
        loop = _closure_loop!(frontier, sweep!, order; maxpass = maxpass,
                              include_zero = include_zero)
        sr = spec[] === nothing ? nothing : (gen = spec[].gen, r = fetch(spec[].task))
        if fr.gen == gen0
            # Nothing grew: the frames and `S0` are the ones `pre` measured `j = 0` on, so its
            # measurement IS the final one (`final_leak[0]` already holds it).
        elseif sr !== nothing && sr.gen == fr.gen
            merge_ledger!(lg, sr.r.ledger)
            final_leak[0] = sr.r.leak
        else
            r0 = _frontier_sweep!(fr, psi, mpo, frontier(0), tau, alloc, ckw; r0kw...,
                                  rng = rng, pass = loop.passes + 1)
            merge_ledger!(lg, r0.ledger)
            final_leak[0] = r0.leak
        end
        loop
    end
    lg.n_passes = res.passes
    lg.leak = [final_leak[j] for j in 0:(order - 1)]
    lg.defect = sum(lg.leak[j + 1] / (j + 1) for j in 0:(order - 1))
    lg.n_root_frontier = nroot[]
    res.converged || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = res.passes, converged = res.converged,
            exhausted = res.exhausted, capped = res.capped, log = res.log, ledger = lg,
            n_root = nroot[],
            n_h1 = lg.n_h1_closure)
end

"""
    _single_pass_closure!(fr, psi, mpo, S0, tau, alloc, ckw, lg, t0; order, include_zero,
                          ledger, sweepkw...) -> NamedTuple

[`_krylov_closure!`](@ref) with `checks = false`: one enriching sweep per frontier, each rebuilt
in the frames the previous sweep grew, and nothing measured before or after. Same return fields.
"""
function _single_pass_closure!(fr::KLFrames, psi::SymMPS, mpo, S0, tau::ComplexF64,
                               alloc::BasisAllocation, ckw::NamedTuple, lg::AdaptiveLedger,
                               t0::UInt64; order::Int, include_zero::Bool,
                               ledger::Union{Nothing, AdaptiveLedger}, sweepkw...)
    nroot = Ref(0)
    S = Ref{Any}(S0)
    H0 = Ref{Any}(_root_generator(mpo, fr))
    applyB = x -> (nroot[] += 1; apply_zero_site(H0[], x))
    leak = fill(NaN, order)                      # NaN: not measured (j = 0 unless include_zero)
    log = Any[]
    met, capped = true, false
    for j in (include_zero ? 0 : 1):(order - 1)
        core = j == 0 ? S[] : _projected_frontiers(applyB, S[], tau, j)[j]
        r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(core, j, fr.gen), tau, alloc, ckw;
                             sweepkw..., enrich = true, pass = 1)
        merge_ledger!(lg, r.ledger)
        push!(log, (pass = 1, frontier = j, added = r.added, leak = r.leak, met = r.met))
        leak[j + 1] = r.leak                     # before this sweep's enrichment
        met &= r.met
        capped |= r.capped
        if r.added > 0
            H0[] = _root_generator(mpo, fr)
            S[] = _project_root(psi, fr)
        end
    end
    lg.n_passes = 1
    lg.leak = leak
    lg.defect = sum((leak[j + 1] / (j + 1) for j in 0:(order - 1) if !isnan(leak[j + 1]));
                    init = 0.0)
    lg.n_root_frontier = nroot[]
    met || _unmet!(lg, :closure)
    lg.t_closure = (time_ns() - t0) / 1e9
    ledger === nothing || merge_ledger!(ledger, lg)
    return (S0 = S[], passes = 1, converged = met, exhausted = false, capped = capped && !met,
            log = log, ledger = lg, n_root = nroot[], n_h1 = lg.n_h1_closure)
end
