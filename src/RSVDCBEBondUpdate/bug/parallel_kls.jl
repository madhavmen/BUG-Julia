# bug/parallel_kls.jl -- the fully parallel K/L/S integrator after the p preliminary CBE
# half-sweeps (updated_cbe_high_order_parallel.md, Task 6).
#
# A DIFFERENT INTEGRATOR, not a schedule of the standard step. On the root bipartition at bond `c`,
# with the preliminary frames `U_p = W[1..c]`, `V_p = Z[c..L-1]`, three projected equations run
# from `Y_n` at once:
#
#     K(0) = Y_n V_p          K' = F(K V_p') V_p          one site tensor at `c`
#     L(0) = U_p' Y_n         L' = (U_p' F(U_p L'))       one site tensor at `c+1`
#     S(0) = U_p' Y_n V_p     S' = U_p' F(U_p S V_p') V_p the root core: the standard solve
#
# K lives in `W[1..c-1] ⊗ site c ⊗ V_p` and L in `U_p ⊗ site c+1 ⊗ Z[c+1..]`: the one-site spaces
# the half-sweeps' local Krylov already works in, so every action is a local tensor contraction
# against MPO environments. The new directions are the parts of K(h), L(h) outside the root frames,
#
#     U_+ = orth[(1 - W[c] W[c]') K(h)],   V_+ = orth[L(h) (1 - Z[c]' Z[c])],
#
# and the step returns the four-block core in `[W[c], U_+] ⊗ [Z[c]; V_+]`
#
#     S_par = [ S(h)       L(h) V_+' ]
#             [ U_+' K(h)  0         ]
#
# ⛔ THE ZERO BLOCK IS AN APPROXIMATION: `U_+' Y(t+h) V_+ = h U_+' F(Y_n) V_+ + O(h^2)` (plan
# T6.3). Its leading term is measured on every step (one thin streamed action at the root bond) and
# a block above `tol_basis` is never accepted: the step falls back to the standard augmented root
# solve in `[W[c], U_+] ⊗ [Z[c]; V_+]`, recorded as `fallback`.

const KLS_SCHEDULES = (:sequential, :kl_then_s, :concurrent)

_join(x) = x isa Task ? fetch(x) : x

"""
    _kls_start(psi, mpo, W, Z, c, lenv, renv, stage, eshift, schedule) -> NamedTuple

The K and L stages. `lenv`/`renv` are the root environments (links `c` and `c+2`, frame basis),
`stage(apply, x, report)` the step's root integrator (`expv` at `tol_root`, its `root_method`).
`:sequential` and `:kl_then_s` return the solved stages; `:concurrent` returns them as running
tasks, which `_kls_finish` joins after the root solve. Nothing here writes `psi`, `W` or `Z`, and
each stage owns its operator, environments and counters.
"""
function _kls_start(psi::SymMPS, mpo, W, Z, c::Int, lenv, renv, stage, eshift, schedule::Symbol)
    schedule in KLS_SCHEDULES || throw(ArgumentError(
        "the K/L/S schedule must be one of $(KLS_SCHEDULES), got $schedule"))
    L = length(psi)
    # The frames' carries up to the root's neighbours, in any gauge (full contractions).
    CL = c == 1 ? nothing : _overlap_left(psi, W, c - 1)                   # (b_{c-1}, link_c)
    DR = c == L - 1 ? nothing : _overlap_right(psi, Z, c + 1)              # (link_{c+2}, b_{c+1})
    Kc = CL === nothing ? psi[c] : to_concrete!(contract(CL, (2,), psi[c], (1,)))
    Lc = DR === nothing ? psi[c + 1] : to_concrete!(contract(psi[c + 1], (3,), DR, (1,)))
    AR = to_concrete!(contract(Lc, (2, 3), Z[c]', (2, 3)))                 # (link_{c+1}, b_R)
    AL = to_concrete!(contract(W[c]', (1, 2), Kc, (1, 2)))                 # (b_L, link_{c+1})
    K0 = to_concrete!(contract(Kc, (3,), AR, (1,)))                        # (b_{c-1}, s_c, b_R)
    L0 = to_concrete!(contract(AL, (2,), Lc, (1,)))                        # (b_L, s_{c+1}, b_{c+1})
    kK, kL = KernelCounters(), KernelCounters()
    HK = OneSiteOp(shift_h(one_site_h(mpo, c, lenv, push_right_channels(renv, mpo, Z[c], c + 1)),
                           eshift), c, kK)
    HL = OneSiteOp(shift_h(one_site_h(mpo, c + 1, push_left_channels(lenv, mpo, W[c], c), renv),
                           eshift), c, kL)
    rK, rL = Ref{Any}(nothing), Ref{Any}(nothing)
    solveK() = _measure(() -> stage(x -> apply_one_site(HK, x), K0, rK))
    solveL() = _measure(() -> stage(x -> apply_one_site(HL, x), L0, rL))
    K, Lh = if schedule === :sequential
        (solveK(), solveL())
    elseif schedule === :kl_then_s
        t = Threads.@spawn solveK()
        l = solveL()
        (fetch(t), l)
    else
        (Threads.@spawn(solveK()), Threads.@spawn(solveL()))
    end
    return (; schedule, K0, L0, HK, HL, K, L = Lh, rK, rL, kK, kL)
end

"""
    _kls_finish(st, mpo, W, Z, c, lenv, renv, S0, Sh, tau, solve0; nrm, tol_missing)
        -> (W, Z, S1, Up, Vp, K, L, M11, account, kc)

Join the K/L stages of `st`, build `U_+`/`V_+` and the four-block core from the root solution
`Sh = S(h)`, and measure the omitted block's leading term `M11 = tau U_+' H Theta_n V_+'`
(`Theta_n = W[c] S0 Z[c]`, streamed against the thin `V_+`, never formed as `H Theta`). When
`||M11|| / nrm > tol_missing` the core is instead `solve0(H0, S0)` in the enlarged frames, the
standard augmented root step. `W`/`Z` are the root frames to use from here on (`W[c]`, `Z[c]`
grown); the caller's vectors are not written.
"""
function _kls_finish(st, mpo, W, Z, c::Int, lenv, renv, S0, Sh, tau::ComplexF64, solve0;
                     nrm::Real = 1.0, tol_missing::Real = Inf)
    (Kh, tK, _), (Lh, tL, _) = _join(st.K), _join(st.L)
    Up, _ = _preselect_left(W[c], Kh, 0.0)                 # numerical rank only: no physical cut
    Vp, _ = _preselect_right(Z[c], Lh, 0.0)
    Up === nothing || (Up = to_concrete(setitag(Up, 3, W[c].inds[3].itags)))
    Vp === nothing || (Vp = to_concrete(setitag(Vp, 1, Z[c].inds[1].itags)))
    Wn = Up === nothing ? W[c] : to_concrete(oplus([W[c], Up], (3,)))
    Zn = Vp === nothing ? Z[c] : to_concrete(oplus([Z[c], Vp], (1,)))
    EL = to_concrete!(contract(Wn', (1, 2), W[c], (1, 2)))                 # (b_L', b_L)
    ER = to_concrete!(contract(Z[c], (2, 3), Zn', (2, 3)))                 # (b_R, b_R')
    embed(S) = to_concrete!(contract(to_concrete!(contract(EL, (2,), S, (1,))), (2,), ER, (1,)))
    kc = merge_counters!(merge_counters!(KernelCounters(), st.kK), st.kL)
    M11, missing = nothing, 0.0
    if Up !== nothing && Vp !== nothing                    # one side alone leaves no omitted block
        HV, dt, db = _measure(() -> _stream_left(to_concrete(W[c] * S0), Z[c], mpo, c, lenv,
                                                 renv, Vp))
        kc.n_stream += 1
        kc.n_stream_cols += leg_dim(Vp, 1)
        kc.t_stream += dt
        kc.b_stream += db
        M11 = _scaled(tau, to_concrete!(contract(Up', (1, 2), HV, (1, 2))))  # (b_+L, b_+R)
        missing = norm(M11) / max(nrm, eps())
    end
    fallback = missing > tol_missing
    S1 = if fallback
        solve0(zero_site_h(mpo, c, lenv, renv, Wn, Zn), embed(S0))
    else
        acc = embed(Sh)
        Up === nothing || (acc = acc + to_concrete!(contract(
            to_concrete!(contract(Wn', (1, 2), perp_component(W[c], Kh), (1, 2))), (2,), ER, (1,))))
        Vp === nothing || (acc = acc + to_concrete!(contract(EL, (2,), to_concrete!(contract(
            perp_component_right(Z[c], Lh), (2, 3), Zn', (2, 3))), (1,))))
        to_concrete(acc)
    end
    account = (schedule = st.schedule,
               n_new_l = Up === nothing ? 0 : leg_dim(Up, 3),
               n_new_r = Vp === nothing ? 0 : leg_dim(Vp, 1),
               missing = missing, fallback = fallback,
               n_K = st.kK.n_one_site, n_L = st.kL.n_one_site,
               err_K = st.rK[].err, err_L = st.rL[].err, t_K = tK, t_L = tL)
    return (W = Wn, Z = Zn, S1 = S1, Up = Up, Vp = Vp, K = Kh, L = Lh, M11 = M11,
            account = account, kc = kc)
end

"""
    _parallel_kls(psi, mpo, fr, S0, tau, stage; schedule = :sequential, nrm, tol_missing)

The whole K/L/S kernel on frames `fr` (`KLFrames`): `_kls_start`, the root solve of `S0` and
`_kls_finish`. `cbe_bug_step!` interleaves the three itself; this is the kernel the tests drive.
"""
function _parallel_kls(psi::SymMPS, mpo, fr::KLFrames, S0, tau::ComplexF64, stage;
                       schedule::Symbol = :sequential, nrm::Real = 1.0, tol_missing::Real = Inf)
    c = fr.c
    st = _kls_start(psi, mpo, fr.W, fr.Z, c, fr.lenv, fr.renv, stage, fr.eshift, schedule)
    root(h, S) = stage(x -> apply_zero_site(shift_h(h, fr.eshift), x), S, Ref{Any}(nothing))
    Sh = root(zero_site_h(mpo, c, fr.lenv, fr.renv, fr.W[c], fr.Z[c]), S0)
    return _kls_finish(st, mpo, fr.W, fr.Z, c, fr.lenv, fr.renv, S0, Sh, tau, root;
                       nrm = nrm, tol_missing = tol_missing)
end

"""
    parallel_kls_step!(psi, mpo, tau; schedule = :concurrent, kwargs...) -> CBEBugSweepInfo

ONE step of the fully parallel K/L/S integrator (plan Task 6): the preliminary half-sweeps (and,
for `order = p > 1`, the frontier closure) of [`cbe_bug_step!`](@ref), then the K, L and root S
equations from `psi` in those frames, the four-block reconstruction and the one compression.
Takes `cbe_bug_step!`'s normal-API keywords. `schedule` is `:sequential` (K, L, then S),
`:kl_then_s` (K and L concurrently, then S) or `:concurrent` (all three); the three give the
same state up to the order of floating-point reductions. `ledger.kls` reports the new directions,
the omitted block `missing` and whether the step fell back to the standard augmented root solve.
"""
parallel_kls_step!(psi::SymMPS, mpo, tau; schedule::Symbol = :concurrent, kwargs...) =
    cbe_bug_step!(psi, mpo, tau; kwargs..., kls = schedule)
