"""
    cbe_bug_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One step of the CBE-BUG sweep, in place. `tau = -im*dt` is real time, `tau = -dt` imaginary; the
caller renormalises. Only the root bond evolves; both half-sweeps just generate bases.

Keywords: `tol_basis` (its local Krylov share separately: `tol_krylov`), `tol_trunc`, `tol_expv` (alias `tol_root`) and `maxdim` (the resource limit,
`adaptive_budget.jl`); `order` (temporal order `p`, see `krylov_closure.jl`); `root` (the Galerkin
bond, default `L/2`); `reorth` (re-orthogonalisation in the root solve).
"""
function cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64;
                       # ── THE API: three tolerances and one resource limit ──────────────
                       # `nothing` takes the default of `TOL_DEFAULTS` (`adaptive_budget.jl`).
                       tol_basis::Union{Nothing, Float64} = nothing,
                       tol_trunc::Union{Nothing, Float64} = nothing,
                       tol_expv::Union{Nothing, Float64} = nothing,
                       # The plan's name for `tol_expv` (cbe_bug_high_order_rsvd_parallel_tasks.md
                       # invariant 1); naming both is refused.
                       tol_root::Union{Nothing, Float64} = nothing,
                       # The share of `tol_basis` the recycled local Krylov stage (Tests C/D, half-
                       # and frontier sweeps) is held to, set on its own (`basis_allocation`);
                       # `nothing` is `tol_basis`, the plan's one-knob default.
                       tol_krylov::Union{Nothing, Float64} = nothing,
                       # The temporal order `p` -- a METHOD choice, not a fourth tolerance.
                       # `p > 1` closes the global Krylov frontiers `j = 1..p-1` of the root
                       # generator before the one root solve (`krylov_closure.jl`); `1` is the
                       # ordinary step, unchanged.
                       order::Int = 1,
                       # ⚠ INSTRUMENTATION, NOT A METHOD CHOICE. `m > 0` on an `order = 1`
                       # step MEASURES (never enriches) the frontiers `j = 0..m-1` on the frames
                       # the ordinary step solves in -- `ledger.leak_initial`/`rho_initial`,
                       # what CBE + local Krylov already capture of the global Krylov sequence.
                       # The evolved state is unchanged. `order > 1` always records them.
                       closure_probe::Int = 0,
                       # ⚠ VALIDATION, NOT A METHOD CHOICE (USER 2026-10-17: off by default).
                       # `false`: an `order = p` step is the two half-sweeps plus ONE enriching
                       # sweep per frontier j = 1..p-1. `true`: the closure as it was validated --
                       # measure every frontier first, loop until a pass adds nothing, measure
                       # j = 0 on the final frames (`_krylov_closure!(...; checks)`).
                       closure_checks::Bool = false,
                       # ⚠ ABLATION, NOT A METHOD CHOICE (plan T4.1 / P07). `true`: every local
                       # Krylov stage (half-sweeps and frontier sweeps) grows from the CBE output
                       # `[U0 | U_CBE]`, so it never regenerates a CBE direction. `false`: it grows
                       # from `U0` alone and the CBE directions are merged afterwards
                       # (`_merge_frame`) -- the reuse-off arm of P07.
                       krylov_reuse::Bool = true,
                       # `order > 1` (dissipative plan §1.2: "transport/reorthogonalize old useful
                       # directions into the new gauge, and extend only from active frontiers"):
                       # each closure round's local Krylov chain is augmented by the previous
                       # round's weighted vectors at that bond (`recycled_krylov.jl`); when the
                       # joint basis already holds the evolution, the chain stops extending.
                       # Real-time and no-jump steps only. On by default (USER 2026-10-29: no
                       # unnecessary bond growth in the closure sweeps); `false` is the ablation.
                       krylov_recycle::Bool = true,
                       # ── IMAGINARY TIME (plan: cbe_bug_imaginary_open_benchmark, Task 1) ──
                       # The root integrator (`ROOT_METHODS`): `:exp` is the Saad-converged
                       # exponential (the reference); `:trapezoid`/`:radau2`/`:radau3` apply that
                       # implicit method's stability function in one stage; `:radau2_gmres` solves
                       # Radau IIA(2)'s coupled stages by GMRES to `tol_root`. A METHOD choice: the
                       # CBE, local Krylov and frontier enrichment before the root are the same.
                       root_method::Symbol = :exp,
                       # The step integrates `H - eshift` (`shift_h`): the root and every Krylov
                       # generator, so the basis weights measure the change of the NORMALISED
                       # state rather than its `exp(-dbeta*E)` growth. `0.0` is bit-identical.
                       eshift::Real = 0.0,
                       # ⚠ INSTRUMENTATION. A `Ref` receives the root's account: energies of the
                       # state before and after the root solve (`<S|H|S>/<S|S>`, exact in the
                       # Galerkin space), the solve's report, and with `outside_residual = true`
                       # the outside-Galerkin action `||(I - P) H psi_1|| / ||psi_1||` of the
                       # evolved state (`estimate_global_complement_action`, one measuring sweep).
                       root_report::Union{Nothing, Base.RefValue} = nothing,
                       outside_residual::Bool = false,
                       # ⚠ INSTRUMENTATION. A `Ref` receives the root's final Galerkin frames
                       # (`KLFrames`), for small-system diagnostics of the product space the step
                       # integrated in (plan T3.5 Taylor leak). Never read by the step.
                       frames_report::Union{Nothing, Base.RefValue} = nothing,
                       # ── TRIAL SPACE CONTAINS `keep_states` (plan Task 2) ──
                       # Each half-sweep carries these MPS through the new frames, so the root space
                       # contains each one exactly. Keeping `|I>>` makes an MPDO step
                       # trace-preserving before the compression. The states must share `psi`'s site
                       # legs; `[]` is the ordinary step.
                       keep_states::AbstractVector = Any[],
                       # ⚠ DEBUG REFERENCE (plan: exact DD references debug-only). `true` draws the
                       # complete DD-projected probe instead of the randomised one.
                       exact::Bool = false,
                       root::Union{Nothing, Int} = nothing,
                       truncate::Bool = true,
                       maxdim::Union{Nothing, Int} = nothing,
                       reorth::Bool = true,
                       # Build the bases from THIS state instead of from `psi`, while still
                       # evolving `psi` over the full `tau`. `nothing` is the ordinary step.
                       # Exists for [`cbe_bug_midpoint_step!`](@ref); see the comment at `psiL`.
                       frame_state = nothing,
                       # ⛔ `hermitian = false` SWITCHES THE ROOT SOLVE TO ARNOLDI. Lanczos on a
                       # non-Hermitian generator -- an open system's Lindbladian or a
                       # non-Hermitian H_eff -- returns a plausible vector and no error at all.
                       # The half-sweep basis needs no flag: it already builds the full
                       # upper-Hessenberg projection, which is the tridiagonal when H = H'.
                       hermitian::Bool = true,
                       # ── `parallel`: the two half-sweeps run concurrently ──
                       # They share no data; only the root solve needs both. Needs `julia -t 2` or
                       # more, off by default. Roundoff-level differences from serial. `:full`
                       # selects the K/L/S integrator.
                       parallel::Union{Bool, Symbol} = false,
                       # ⚠ A DIFFERENT INTEGRATOR, set by `parallel_kls_step!` (plan Task 6): the
                       # parallel K/L/S step in the preliminary frames, with this K/L/S schedule
                       # (`KLS_SCHEDULES`). `nothing` is the standard augmented root step.
                       kls::Union{Nothing, Symbol} = nothing,
                       # The closure's leak measurement (`krylov_closure.jl`): fold the probe into
                       # the two-site contraction instead of building `H*Theta`. `nothing` folds
                       # for `order > 1`, so the closure forms no dense two-site action (dissipative
                       # plan §1.2), except under `exact`: the folded leak is sampled, and the debug
                       # reference draws no random numbers. `false` keeps the exact `H*Theta` leak.
                       fold_omega::Union{Nothing, Bool} = nothing,
                       # Which DD-QR CBE directions survive when the work ceiling binds:
                       # `:weight` the heaviest across all sectors, `:sector` the first sectors in
                       # storage order (`dd_cbe_expand`). Below the ceiling both are identical.
                       cap_order::Symbol = :sector,
                       rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(mpo) == L || throw(DimensionMismatch(
        "MPO has $(length(mpo)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_bug_step! needs at least two sites"))
    c = root === nothing ? _bug_root_link(L) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))
    parallel_sym = parallel isa Symbol ? parallel : (parallel ? :half_sweeps : :serial)
    parallel = _resolve_parallel(parallel)
    parallel_sym === :full && (kls = something(kls, :concurrent))
    (tol_root !== nothing && tol_expv !== nothing) && throw(ArgumentError(
        "tol_root and tol_expv name the same tolerance; pass one"))
    tol_root === nothing || (tol_expv = tol_root)

    budget = AdaptiveBudget(something(tol_basis, TOL_DEFAULTS.basis),
                            something(tol_trunc, TOL_DEFAULTS.trunc),
                            something(tol_expv,  TOL_DEFAULTS.expv))
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    fold_omega = something(fold_omega, order > 1 && !exact)
    order == 1 || frame_state === nothing || throw(ArgumentError(
        "order > 1 is a single-step method and does not combine with the midpoint predictor"))
    closure_probe >= 0 || throw(ArgumentError(
        "closure_probe must be a non-negative count, got $closure_probe"))
    if kls !== nothing
        kls in KLS_SCHEDULES || throw(ArgumentError(
            "the K/L/S schedule must be one of $(KLS_SCHEDULES), got $kls"))
        frame_state === nothing && !outside_residual || throw(ArgumentError(
            "the parallel K/L/S step does not combine with the midpoint predictor or the " *
            "outside-residual re-enrichment"))
    end
    # Plan §1 signature: `maxdim = 128`.
    maxdim = something(maxdim, 128)

    # The per-bond, per-test allowances `tol_basis` buys, and the `|tau|/||psi||` scale every
    # residual is measured in. The norm comes from the centre tensor, which carries the whole norm
    # only while the state is canonical (the sweeps gauge before reading). An out-of-range `center`
    # falls back to 1.0 rather than erroring in a diagnostic.
    _nrm = (1 <= psi.center <= L) ? norm(psi[psi.center]) : 1.0
    # Plan T3.2: DD-QR spends `tol_basis` as a linear sum of shares.
    alloc  = basis_allocation(budget, L - 1; tau = tau, nrm = _nrm, nfront = order,
                              rule = :linear, krylov = tol_krylov)
    ledger = AdaptiveLedger()
    ledger.order = order
    bd0 = bond_dims(psi)                 # per-bond `r_old` for the bond log (plan §8)
    ledger.r_old = maximum(bd0; init = 0)
    bwts = zeros(L - 1)                  # per-bond discarded norm of the compression

    expanded = zeros(Int, L - 1)
    # ⚠ ONE ARRAY PER HALF-SWEEP: bond `c` is written by BOTH (`n_new_l` from the left sweep,
    # `n_new_r` from the right), so a single shared array is a genuine write conflict there --
    # benign when the sweeps ran in sequence and last-writer-wins, undefined when they run
    # concurrently. Merged after the join with the RIGHT sweep winning at `c`, which is exactly
    # what the sequential order used to produce.
    n_newL   = zeros(Int, L - 1)
    n_newR   = zeros(Int, L - 1)
    accL = _HalfAcc()
    accR = _HalfAcc()

    # The ONE work ceiling, shared by the CBE sketch and the Krylov enrichment (plan §4/§5), read
    # ONCE, globally, before anything widens.
    wcap_step = work_ceiling(maxdim, maximum(bond_dims(psi); init = 0), local_dim(psi))
    # The expansion settings every DD-QR CBE of the step takes, half-sweeps and frontier sweeps.
    exkw = (growth = ADAPTIVE_GROWTH, exact = exact, alloc = alloc, workcap = wcap_step,
            cap_order = cap_order)

    # ⛔ THE LEDGER AND THE RNG ARE PER HALF-SWEEP, because under `parallel = true` the two run
    # concurrently: a shared `AdaptiveLedger` loses `+=` updates, and a shared `MersenneTwister`
    # is not thread-safe at all. The right sweep gets a stream SEEDED FROM `rng` only when
    # parallel, so the serial path draws exactly the sequence it always did. The two ledgers are
    # folded into the step's after the join.
    ledgerL, ledgerR = AdaptiveLedger(), AdaptiveLedger()
    rngR = parallel ? MersenneTwister(rand(rng, UInt64)) : rng
    rk = krylov_recycle && order > 1 ? RecycledKrylov() : nothing
    cx = _HalfSweepCtx(mpo, tau, eshift, c, alloc, exkw, wcap_step, krylov_reuse, keep_states, rk)

    troot = Ref(0.0); ttrunc = Ref(0.0)
    _t0step = time_ns()
    _b0step = Base.gc_bytes()
    W = Vector{Any}(undef, c)          # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)      # augmented RIGHT frames, Z[j] absorbs site j+1

    # ── `frame_state`: bases from one state, evolution from another ──
    # The midpoint BUG integrator (arXiv:2402.08607) takes its bases from `Y_{1/2}` but integrates
    # the Galerkin ODE over the full step from `Y_0`. The start vector `S0 = <frames|Y_0>` must not
    # come from the half-step state, or the second half would be integrated twice. `psi` stays
    # L-canonical so that all assembly reads one gauge; with no `frame_state`, `psiR` aliases `psi`.
    psiL = copy(frame_state === nothing ? psi : frame_state)
    psiR = frame_state === nothing ? psi : copy(frame_state)

    # ── RUN THEM ── (`parallel` needs `julia -t 2`; why it is safe: `half_sweeps.jl`)
    # `lenv_root` / `renv_root`: the root environments at links `c` / `c+2`, in the NEW basis;
    # the right sweep's final carry IS `AR` (`_half_right!`).
    run_left() = _half_left!(psiL, W, expanded, n_newL, accL, ledgerL, rng, cx)
    run_right() = _half_right!(psiR, Z, expanded, n_newR, accR, ledgerR, rngR, cx)
    if parallel
        tR = Threads.@spawn run_right()
        lenv_root = run_left()
        renv_root, AR = fetch(tR)
    else
        lenv_root = run_left()
        renv_root, AR = run_right()
    end
    merge_ledger!(ledger, ledgerL)
    merge_ledger!(ledger, ledgerR)

    # `n_new` at bond `c` is written by BOTH sweeps; the right one wins, which is what the old
    # sequential order produced.
    n_new = vcat(n_newL[1:(c - 1)], n_newR[c:(L - 1)])
    epre = max(accL.epre, accR.epre)
    efnl = max(accL.efnl, accR.efnl)

    # ── with a separate `frame_state`, the right sweep no longer touches `psi` ──
    # Gauge: `AL` and `AR` must be contracted in one gauge, since mixing them is silently wrong
    # rather than an error. `AR` must be `<Z|psi>`, not the right carry `<Z|frame_state>`: the
    # midpoint's Galerkin step starts from `Y_0`, and reusing the carry integrates the first
    # half-step twice, quietly making the method first order.
    if frame_state !== nothing
        canonical!(psi, L)
        AR = _overlap_right(psi, Z, c)           # mirrors `_half_right!`'s `D` recursion on `psi`
    end

    AL = _overlap_left(psi, W, c)                                     # (b_L, link_{c+1})
    S0 = to_concrete!(contract(AL, (2,), AR, (1,)))                   # (b_L, b_R)

    # ── GENERAL ORDER (plan: krylov_closure_cbe_bug_realtime_L16_plan.md) ───────────────────
    # The half-sweeps closed the frontier j = 0 (psi itself). For `order > 1` the frontiers
    # j = 1..order-1 of the ROOT generator are closed against the GLOBAL operator, growing `W`
    # and `Z` in place; the root environments and `S0` come back rebuilt in the final frames,
    # and the single root solve below runs in them.
    nclose = 0
    if order > 1
        fr = KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift))
        # Plan T4.1: the frontier sweeps expand by the same DD-QR CBE as the half-sweeps.
        ckw = merge(exkw, (krylov_reuse = krylov_reuse, recycle = rk))
        cl = _krylov_closure!(fr, psi, mpo, S0, tau, alloc, ckw; order = order,
                              workcap = wcap_step, nrm = _nrm, parallel = parallel, rng = rng,
                              ledger = ledger, fold = fold_omega, checks = closure_checks)
        lenv_root, renv_root, S0 = fr.lenv, fr.renv, cl.S0
        for i in 1:(c - 1)
            expanded[i] = leg_dim(W[i], 3)
        end
        for j in (c + 1):(L - 1)
            expanded[j] = leg_dim(Z[j], 1)
        end
        nclose = cl.n_h1 + cl.n_root
    elseif closure_probe > 0
        # A copied `rng` and untouched frames: the step itself stays bit-identical.
        pr = _measure_frontiers(KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing,
                                         Float64(eshift)), psi, mpo, S0, tau,
                                alloc, NamedTuple(), closure_probe; nrm = _nrm,
                                parallel = parallel, rng = copy(rng), fold = fold_omega)
        merge_ledger!(ledger, pr.ledger)
    end
    # PLAN §5 `state_projection_error`: how much of `psi` the frames miss. On the ordinary step the
    # frames contain `psi`, so this is roundoff; on the midpoint's second stage it is the genuine
    # error of projecting `Y_0` onto frames built from `Y_{1/2}`, charged against `tol_basis`. The
    # norm is taken from the state, not from `sqrt(||psi||^2 - ||S0||^2)`, which is pure
    # cancellation at ~1e-8 and would flag steps spuriously.
    npsi = norm(psi[psi.center])
    ledger.r_aug = max(maximum(expanded; init = 0), leg_dim(Z[c], 1))
    ledger.proj_err = npsi > 0 ? _projection_miss(psi, W, Z, c) / npsi : 0.0
    ledger.proj_err > budget.basis && _unmet!(ledger, :projection)
    peak = accL.peak + accR.peak +
           sum(tensor_elements(t) for t in (W[c], Z[c], S0); init = 0)

    # Plan Task 6: the K and L stages of the parallel K/L/S step start from `psi` in these frames
    # and run beside the root solve (the S stage), and `_kls_finish` merges the three. `tol_expv`
    # drives the root solve's substep split (`substeps = 0`), so the iteration count follows from
    # it; the basis cap and breakdown threshold are numerical settings only.
    kls_stage(apply, x, rep) = expv(apply, tau, x; hermitian = hermitian,
                                    maxiter = _ROOT_MAXITER, tol = 1e-15, reorth = reorth,
                                    conv_tol = budget.expv, substeps = 0, report = rep,
                                    root_method = root_method)
    klsst = kls === nothing ? nothing :
        _kls_start(psi, mpo, W, Z, c, lenv_root, renv_root, kls_stage, eshift, kls)

    kroot = KernelCounters()
    H0 = RootOp(shift_h(zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c]), eshift), c, kroot)
    nroot = Ref(0)
    rootrep = Ref{Any}(nothing)
    _t0root = time_ns()
    solve_root(S) = kls_stage(x -> (nroot[] += 1; apply_zero_site(H0, x)), S, rootrep)
    S1 = solve_root(S0)
    troot[] = (time_ns() - _t0root) / 1e9
    if root_report !== nothing
        # Rayleigh quotients in the Galerkin space: `S0` and `S1` ARE the state before the root
        # and after it (before the compression), and `<S|H_0|S> = <psi|P H P|psi> = <psi|H|psi>`
        # for `psi` in the frames' span, so these are exact energies, not projections of them.
        rq(S) = (n2 = norm(S)^2; n2 > 0 ? real(tensor_inner(S, apply_zero_site(H0, S))) / n2 +
                                          Float64(eshift) : NaN)
        e_pre = rq(S0)
        outside_of(S) = estimate_global_complement_action(psi, mpo,
            KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift)),
            S; rng = copy(rng))
        out = outside_residual ? outside_of(S1) : nothing
        resid_of(o, S) = o === nothing ? NaN : abs(tau) * o.bound / max(norm(S), eps())
        resid0 = resid_of(out, S1)
        # ── re-enrich and re-solve on the failing residual (plan Task 1) ──
        # The closure can meet its shares while the outside action fails: grow frames by S1's own
        # sweep, re-solve the root until the defect fits, nothing is added, or the ceiling binds.
        npass, nadded = 0, 0
        if outside_residual
            while !(resid_of(out, S1) <= budget.basis) && npass < _closure_maxpass(order)
                fr = KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing, Float64(eshift))
                npass += 1
                r = _frontier_sweep!(fr, psi, mpo, KrylovFrontier(S1, 0, fr.gen), tau, alloc, exkw;
                                     enrich = true, workcap = wcap_step, nrm = _nrm,
                                     parallel = parallel, rng = rng, pass = npass,
                                     fold = fold_omega)
                merge_ledger!(ledger, r.ledger)
                r.added == 0 && break
                nadded += r.added
                lenv_root, renv_root = fr.lenv, fr.renv
                S0 = _project_root(psi, fr)
                H0 = shift_h(zero_site_h(mpo, c, lenv_root, renv_root, W[c], Z[c]), eshift)
                _t0root = time_ns()
                S1 = solve_root(S0)
                troot[] += (time_ns() - _t0root) / 1e9
                out = outside_of(S1)
                r.capped && break
            end
            if nadded > 0
                for i in 1:(c - 1)
                    expanded[i] = leg_dim(W[i], 3)
                end
                for j in (c + 1):(L - 1)
                    expanded[j] = leg_dim(Z[j], 1)
                end
                ledger.r_aug = max(maximum(expanded; init = 0), leg_dim(Z[c], 1))
            end
        end
        e_root = rq(S1)
        n1 = max(norm(S1), eps())
        root_report[] = (root_method = root_method, eshift = Float64(eshift), e_pre = e_pre,
                         e_root = e_root, solve = rootrep[], n_root = nroot[],
                         # the defect before the re-enrichment above, its passes and directions
                         resid_initial = resid0, outside_passes = npass, outside_added = nadded,
                         outside = out === nothing ? NaN : out.bound / n1,
                         outside_left = out === nothing ? NaN : out.left / n1,
                         outside_right = out === nothing ? NaN : out.right / n1,
                         # per bond (left frames 1..c, right frames c..L-1), scale-free like the
                         # above
                         per_left = out === nothing ? Float64[] : out.per_left ./ n1,
                         per_right = out === nothing ? Float64[] : out.per_right ./ n1,
                         # the step's outside-Galerkin defect, `|tau| ||(I-P) H psi_1|| / ||psi_1||`
                         resid = out === nothing ? NaN : abs(tau) * out.bound / n1)
    end
    nkls = 0
    if klsst !== nothing
        # The four-block core in the grown root frames, or -- when the omitted new-new block is
        # above `tol_basis` -- the standard augmented root solve in them (plan T6.3).
        solve_aug(h, S) = (op = RootOp(shift_h(h, eshift), c, kroot);
                           kls_stage(x -> (nroot[] += 1; apply_zero_site(op, x)), S, rootrep))
        kr = _kls_finish(klsst, mpo, W, Z, c, lenv_root, renv_root, S0, S1, tau, solve_aug;
                         nrm = _nrm, tol_missing = budget.basis)
        W[c], Z[c], S1 = kr.W, kr.Z, kr.S1
        merge_counters!(kroot, kr.kc)
        nkls = kr.account.n_K + kr.account.n_L
        ledger.kls = kr.account
        ledger.r_aug = max(ledger.r_aug, leg_dim(W[c], 3), leg_dim(Z[c], 1))
        max(kr.account.err_K, kr.account.err_L) > budget.expv * (1 + 1e-8) &&
            _unmet!(ledger, :expv)
    end
    if rootrep[] !== nothing
        ledger.expv_err   = rootrep[].err
        ledger.expv_iters = rootrep[].iterations
        # Relative slack: each substep is accepted at `tol_expv`, so only a sum over several
        # substeps (or a step that could not advance) can exceed it.
        rootrep[].err > budget.expv * (1 + 1e-8) && _unmet!(ledger, :expv)
    end

    # ⛔ THE ROOT CORE SPLIT IS LOSSLESS (plan §5). This SVD only splits `S1` into the two site
    # tensors below: `cutoff = 0.0` keeps every singular value with `sv > 0` (Telum's selector is
    # relative to `sv_max`) and there is no `Nkeep`, so the closing pass is the step's one
    # compression -- and, with `truncate = false`, there is no rank control at the root at all.
    frames_report === nothing ||
        (frames_report[] = KLFrames(W, Z, c, lenv_root, renv_root, 0, nothing, nothing,
                                    Float64(eshift)))
    res = svd(S1, (1,); cutoff = 0.0, get_lists = true)
    expanded[c] = leg_dim(W[c], 3)
    # bytes by hand, not `_measure`: a closure here would box the reassigned `maxdim`
    bcomp = Base.gc_bytes()
    root_rank, discarded, ttrunc[] = _assemble_and_compress!(psi, W, Z, c, res, ledger, bwts, bd0;
                                                            maxdim = maxdim,
                                                            tol_trunc = budget.trunc,
                                                            truncate = truncate)
    bcomp = Base.gc_bytes() - bcomp

    # ⚠ THE PHASE TIMERS ARE SUMS ACROSS BOTH HALF-SWEEPS, i.e. CPU time, while `t_step` is WALL
    # clock. Under `parallel = false` they are the same clock and the shares sum to ~1 as before.
    # Under `parallel = true` they DELIBERATELY DO NOT: two sweeps running at once spend more CPU
    # seconds than the step takes, and a share above 1 is the speedup showing up, not a bug. Keep
    # comparing phases arm-to-arm in seconds; only the serial arm's shares are fractions of a step.
    t_step = (time_ns() - _t0step) / 1e9
    kc = merge_counters!(merge_counters!(merge_counters!(KernelCounters(), accL.kc), accR.kc),
                         kroot)
    kc.b_compress += bcomp
    rec = StepRecord(:dd_qr_streamed, _qr_backend(), order > 1 ? :dd_qr_streamed : :none,
                     order, kls === nothing ? parallel_sym : :full, kc,
                     maximum(bd0; init = 0), max(maximum(expanded; init = 0), leg_dim(Z[c], 1)),
                     maximum(bond_dims(psi); init = 0), maxdim, wcap_step, nroot[], t_step,
                     Base.gc_bytes() - _b0step, Int(Sys.maxrss()),
                     _step_limits(ledger), alloc.rule)
    return CBEBugSweepInfo(expanded, n_new, c, root_rank, epre, efnl,
                           discarded, accL.nmv + accR.nmv + nroot[] + nclose + nkls, peak,
                           accL.ngrow + accR.ngrow, leg_dim(Z[c], 1),
                           accL.tcbe + accR.tcbe, accL.tkry + accR.tkry, troot[], ttrunc[],
                           accL.tsplit + accR.tsplit, accL.tframe + accR.tframe,
                           accL.tenv + accR.tenv, t_step,
                           ledger, rec)
end

cbe_bug_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_step!(psi, mpo_from_terms(h), tau; kwargs...)
