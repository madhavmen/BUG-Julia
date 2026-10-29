# ── GP ground states: imaginary-time BUG, QTT-TDVP and QTT gradient descent (QTT plan §3.1) ──
# The normalised gradient flow d_tau psi = -(H0 + g|psi|^2 - mu[psi]) psi with the SAME implicit BUG
# p2/3/4 under two schedules -- uniform and geometric -- and two baselines, compared on the energy
# and stationarity residual reached per WALL second, never per step count. Every method renormalises
# in the physical quadrature after each accepted step and rejects (halves) a step whose stage solve
# fails or whose energy rises.

"`n` steps of `h` (the uniform policy)."
uniform_schedule(h::Real, n::Int) = fill(Float64(h), n)

"""
    geometric_schedule(h0, q, hmax, n) -> Vector{Float64}

`dtau_k = min(hmax, h0 q^k)`, `k = 0..n-1` -- the logarithmically expanding policy. `q` is an
experiment-level schedule parameter, not an accuracy tolerance.
"""
geometric_schedule(h0::Real, q::Real, hmax::Real, n::Int) =
    [min(Float64(hmax), h0 * q^k) for k in 0:(n - 1)]

"""
    gp_ground_state!(psi, m, schedule; method = :bug, order = 2, maxdim = 128, tol_trunc = 1e-10,
                     tol_basis = 1e-8, max_reject = 6, kwargs...) -> Vector{NamedTuple}

Imaginary-time relaxation along `schedule` (`:bug` = `qtt_bug_step!` of the normalised GP flow,
`:tdvp` = `qtt_tdvp2_step!` lagged). A step is REJECTED and retried at half the step when its stage
solve does not converge or the normalised energy increases beyond roundoff; the ledger records
every accepted and rejected attempt with energy, mu, R_mu, chi and wall time.
"""
function gp_ground_state!(psi::SymMPS, m::GPModel, schedule::AbstractVector;
                          method::Symbol = :bug, order::Int = 2, maxdim::Int = 128,
                          tol_trunc::Real = 1e-10, tol_basis::Real = 1e-8, max_reject::Int = 6,
                          kwargs...)
    method in (:bug, :tdvp) || throw(ArgumentError("method must be :bug or :tdvp, got :$method"))
    fl = GPFlow(m; imaginary = true)
    g = m.grid
    qtt_normalize!(psi, g)
    obs = gp_observables(m, qtt_cores(psi), 0.0)
    hist = NamedTuple[(tau = 0.0, h = 0.0, accepted = true, reason = :start, E = obs.E,
                      mu = obs.mu, R_mu = obs.R_mu, chi = obs.chi, wall = 0.0)]
    tau = 0.0
    wall = 0.0
    for h0 in schedule
        h = Float64(h0)
        done = 0.0
        while done < h0 - 1e-14
            h = min(h, h0 - done)
            trial = copy(psi)
            t1 = time_ns()
            ok = true
            if method === :bug
                r = qtt_bug_step!(trial, fl, tau, h; order = order, maxdim = maxdim,
                                  tol_trunc = tol_trunc, tol_basis = tol_basis, kwargs...)
                ok = r.converged
            else
                qtt_tdvp2_step!(trial, fl, tau, h; density = :lagged, maxdim = maxdim,
                                tol_trunc = tol_trunc)
            end
            qtt_normalize!(trial, g)
            o = gp_observables(m, qtt_cores(trial), 0.0)
            wall += (time_ns() - t1) / 1e9
            rise = o.E > hist[end].E + 1e-10 * abs(hist[end].E)
            if (!ok || rise) && count(e -> !e.accepted, hist) < max_reject * length(schedule)
                push!(hist, (tau = tau, h = h, accepted = false, reason = ok ? :energy : :stage,
                            E = o.E, mu = o.mu, R_mu = o.R_mu, chi = o.chi, wall = wall))
                h /= 2
                continue
            end
            _replace_state!(psi, trial)
            tau += h
            done += h
            push!(hist, (tau = tau, h = h, accepted = true, reason = :ok, E = o.E, mu = o.mu,
                        R_mu = o.R_mu, chi = o.chi, wall = wall))
        end
    end
    return hist
end

"""
    gp_gradient_descent!(psi, m; eta = 0.05, maxiter = 500, tol = 1e-8, maxdim = 128,
                         tol_trunc = 1e-10) -> Vector{NamedTuple}

QTT gradient descent on the normalised energy (the faster ground-state baseline of Chen et al.,
arXiv:2507.04279v5): `psi <- normalise(psi - eta (H[psi] - mu[psi]) psi)`, compressed, with
backtracking (`eta` halved while the energy rises, grown by 1.2 after an accepted step). Stops at
`R_mu < tol`.
"""
function gp_gradient_descent!(psi::SymMPS, m::GPModel; eta::Real = 0.05, maxiter::Int = 500,
                              tol::Real = 1e-8, maxdim::Int = 128, tol_trunc::Real = 1e-10)
    g = m.grid
    qtt_normalize!(psi, g)
    A = qtt_cores(psi)
    o = gp_observables(m, A, 0.0)
    hist = NamedTuple[(it = 0, eta = 0.0, E = o.E, mu = o.mu, R_mu = o.R_mu, chi = o.chi, wall = 0.0)]
    wall = 0.0
    et = Float64(eta)
    for it in 1:maxiter
        o.R_mu < tol && break
        t1 = time_ns()
        H, _ = gp_hamiltonian_op(m, A, 0.0)
        HA = op_apply(H, A)
        mu = o.mu - m.mu
        local B, ob
        while true
            B = tt_lincomb([(1.0, A), (-et, HA), (et * mu, A)]; tol = tol_trunc, maxdim = maxdim)
            B = tt_scale(B, 1 / (sqrt(qtt_cell(g)) * tt_norm(B)))
            ob = gp_observables(m, B, 0.0)
            (ob.E <= o.E + 1e-12 * abs(o.E) || et < 1e-8) && break
            et /= 2
        end
        A, o = B, ob
        et *= 1.2
        wall += (time_ns() - t1) / 1e9
        push!(hist, (it = it, eta = et, E = o.E, mu = o.mu, R_mu = o.R_mu, chi = o.chi, wall = wall))
    end
    _replace_state!(psi, qtt_mps(A))
    return hist
end
