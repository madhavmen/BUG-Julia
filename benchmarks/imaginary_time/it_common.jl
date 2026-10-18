# Shared machinery of the imaginary-time MODEL drivers (plan Tasks 3-4: IT-1 chain, IT-2 cylinder):
# the knobs, the grids, one arm on one grid with its per-step CSV row, and the JIT warm-up. A driver
# supplies the model (MPO, Neel start, reference, final observables) and includes this file.
#
# Arms: bugP = ite_step! at order P (Eshift = the state's energy, outside defect vs tol_basis,
# re-enrich / split / FLAG); tdvp2 = tdvp2_step! at -dbeta with the same Eshift, renormalised, the
# SAME per-cut discard rule (`weight_tol = tol_trunc`) and Saad-stopped exponentials at tol_root --
# the order-2 partner of bug2 on the same grid; it has no tol_basis.
#
# Env: IT_BMAX (16) IT_GRIDS ("panel:6:4,uniform:64") IT_ARMS ("bug2,bug3,bug4,tdvp2")
#      IT_TB ("1e-3,1e-5,1e-7") IT_TT (1e-6) IT_TR (1e-6) IT_MAXDIM (256) IT_ROOT (exp)
#      IT_SEEDS ("1") IT_RETRIES (2) IT_OUTSIDE (1) IT_OUT
#      IT_BACKAMP (1): tdvp2's backward-substep amplification per step (one extra H1 action per
#                      bond, so 0 for timing jobs)
#      IT_VAR_EVERY (0): the energy variance every k-th step, untimed (0 = final state only)
#      IT_FRAME_CHECKS (0): assert at every frame-growth site that the frame fits its local space
#                      (`_FRAME_CHECKS`, debug only: a fusion basis per check)

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const IT = (bmax     = parse(Float64, get(ENV, "IT_BMAX", "16")),
            grids    = split(get(ENV, "IT_GRIDS", "panel:6:4,uniform:64"), ","),
            arms     = split(get(ENV, "IT_ARMS", "bug2,bug3,bug4,tdvp2"), ","),
            tbs      = parse.(Float64, split(get(ENV, "IT_TB", "1e-3,1e-5,1e-7"), ",")),
            tt       = parse(Float64, get(ENV, "IT_TT", "1e-6")),
            tr       = parse(Float64, get(ENV, "IT_TR", "1e-6")),
            maxdim   = parse(Int, get(ENV, "IT_MAXDIM", "256")),
            root     = Symbol(get(ENV, "IT_ROOT", "exp")),
            seeds    = parse.(Int, split(get(ENV, "IT_SEEDS", "1"), ",")),
            retries  = parse(Int, get(ENV, "IT_RETRIES", "2")),
            outside  = get(ENV, "IT_OUTSIDE", "1") == "1",
            backamp  = get(ENV, "IT_BACKAMP", "1") == "1",
            var_every = parse(Int, get(ENV, "IT_VAR_EVERY", "0")),
            out      = get(ENV, "IT_OUT", joinpath(@__DIR__, "results")),
            parallel = Threads.nthreads() >= 2)

BUGJulia.RSVDCBEBondUpdate._FRAME_CHECKS[] = get(ENV, "IT_FRAME_CHECKS", "0") == "1"

say(msg) = (println(msg); flush(stdout))

function grid_steps(g)
    a = split(g, ":")
    a[1] == "panel" && return log_panel_steps(IT.bmax, parse(Int, a[2]); per_panel = parse(Int, a[3]))
    a[1] == "uniform" && return uniform_beta_steps(IT.bmax, parse(Int, a[2]))
    error("grid must be panel:P:k or uniform:n, got $g")
end

energy(psi, W) = real(mpo_energy(psi, W)) / real(overlap(psi, psi))
renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

"The per-step CSV header; `key` names the first column (the model size)."
it_step_header(key) = "$key,arm,grid,tol_basis,seed,step,beta,dbeta,E,dE0,dEbeta,chi,wall,status,unmet," *
                      "e_pre_root,e_root,de_root,de_compression,resid,discarded,root_err,nsub,backamp,var"

"""
One arm on one grid from `psi0`, one CSV row per step into `io`; `ref.E0` and, when not `nothing`,
`ref.traj[g]` (the exact E at every grid point) grade it. Returns the final state and the counts.
"""
function it_run_arm(io, key, W, psi0, ref, arm, g, tb, seed)
    steps = grid_steps(g)
    psi = copy(psi0)
    rng = MersenneTwister(seed)
    beta, wall = 0.0, 0.0
    nfail = nretry = 0
    Ecur = NaN
    for (k, db) in enumerate(steps)
        amp = NaN
        if startswith(arm, "bug")
            p = parse(Int, arm[4:end])
            r = ite_step!(psi, W, db; order = p, root_method = IT.root, tol_basis = tb,
                          tol_trunc = IT.tt, tol_root = IT.tr, maxdim = IT.maxdim,
                          retries = IT.retries, outside_residual = IT.outside,
                          parallel = IT.parallel, rng = rng)
            wall += r.wall
            E = r.e_after
            st, um = r.status, join(string.(r.unmet), "+")
            row = (r.e_pre_root, r.e_root, r.de_root, r.de_compression, r.resid, r.discarded,
                   r.root_err, r.nsub)
            nfail += r.status === :failed; nretry += r.status === :retried
        else                                      # tdvp2: the fair order-2 partner
            t0 = time_ns()
            # Eshift = the state's energy, as ite_step! uses: the previous step's readout, so
            # only the start pays for it (and pays inside the wall, as ite_step! does)
            k == 1 && (Ecur = energy(psi, W))
            rep = IT.backamp ? Ref{Any}(NaN) : nothing
            info = tdvp2_step!(psi, W, ComplexF64(-db); maxdim = IT.maxdim, maxiter = 30,
                               conv_tol = IT.tr, substeps = 0, weight_tol = IT.tt, eshift = Ecur,
                               backward_report = rep)
            renorm!(psi)
            # the energy readout counts in the wall, as ite_step!'s own energy logging does
            E = energy(psi, W)
            wall += (time_ns() - t0) / 1e9
            Ecur = E
            amp = rep === nothing ? NaN : Float64(rep[])
            st, um = :tdvp2, ""
            row = (NaN, NaN, NaN, NaN, NaN, info.cumulative, NaN, 1)
        end
        beta += db
        Ex = ref.traj === nothing ? NaN : ref.traj[g][k]
        var = (IT.var_every > 0 && (k % IT.var_every == 0 || k == length(steps))) ?
              mpo_variance(psi, W) : NaN
        @printf(io, "%s,%s,%s,%.0e,%d,%d,%.10g,%.10g,%.14f,%.6e,%.6e,%d,%.3f,%s,%s,%.14f,%.14f,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e,%.6e\n",
                key, arm, g, tb, seed, k, beta, db, E, E - ref.E0, E - Ex,
                maximum(bond_dims(psi)), wall, st, um, row[1], row[2], row[3], row[4],
                row[5], row[6], row[7], row[8], amp, var)
        flush(io)
    end
    var = mpo_variance(psi, W)
    Ef = energy(psi, W)
    say(@sprintf("  %-6s %-12s tb %.0e seed %d: E-E0 %.3e  E-E(beta) %.3e  var %.2e  chi %d  wall %.1fs  failed %d retried %d",
                 arm, g, tb, seed, Ef - ref.E0,
                 ref.traj === nothing ? NaN : Ef - ref.traj[g][end], var,
                 maximum(bond_dims(psi)), wall, nfail, nretry))
    return (psi = psi, var = var, wall = wall, nfail = nfail)
end

"""JIT warm-up, untimed and discarded: two steps of every arm on a small model (`W6`, `psi6`), so
no arm's first measured step carries compilation (the smoke run's first BUG step showed ~17 s)."""
function it_warmup(W6, psi6)
    for arm in IT.arms, _ in 1:2
        p = copy(psi6)
        if startswith(arm, "bug")
            ite_step!(p, W6, 0.1; order = parse(Int, arm[4:end]), root_method = IT.root,
                      tol_basis = IT.tbs[1], tol_trunc = IT.tt, tol_root = IT.tr, maxdim = 16,
                      retries = 0, outside_residual = IT.outside, parallel = IT.parallel)
        else
            tdvp2_step!(p, W6, ComplexF64(-0.1); maxdim = 16, maxiter = 30, conv_tol = IT.tr,
                        substeps = 0, weight_tol = IT.tt, eshift = energy(p, W6),
                        backward_report = IT.backamp ? Ref{Any}(NaN) : nothing)
        end
        IT.var_every > 0 && mpo_variance(p, W6)
    end
end

"Every (arm, tol_basis, seed) combination, in the drivers' order: tdvp2 has one, with no tol_basis."
it_runs(arm) = [(tb, s) for tb in (startswith(arm, "bug") ? IT.tbs : [NaN])
                for s in (startswith(arm, "bug") ? IT.seeds : IT.seeds[1:1])]
