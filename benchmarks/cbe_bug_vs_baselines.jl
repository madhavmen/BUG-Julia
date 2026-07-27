# CBE-BUG against its baselines: accuracy, rank, and the memory of a bond expansion.
#
# Run with: sbatch --job-name=cbe_bench scripts/run_julia.sbatch benchmarks/cbe_bug_vs_baselines.jl
#
# PARKED FOR THE LARGE-SYSTEM CAMPAIGN. The accuracy, order-in-dt and rank comparisons
# here are NOT meaningful at the sizes below: both CBE-BUG and 2-site TDVP are exact once
# the manifold is the whole space (MEASURED ~1e-13 at L=6/maxdim=32, job 94971), and at
# L=8 the rank cap has to be pushed so low to create an error at all that the truncation
# floor dominates whatever is being measured. The comparison belongs to a large-system
# run where the rank cap binds for physical reasons rather than artificial ones.
#
# Kept here, unrun, as the harness for that campaign. What it measures:
#
#   (1) ACCURACY vs RANK. At matched error against dense, what bond dimension does each
#       scheme need? CBE-BUG expands to `r + Dex` with the directions ranked by weight in
#       HTheta; 2-site TDVP goes to the full `d*chi` before truncating; the K/L
#       augmentation goes to `2r` with no knob; `pad=true` fills that budget with random
#       columns.
#
#   (2) MEMORY OF THE EXPANSION. The largest object each scheme builds per bond update.
#       CBE-BUG's is rank 3 by construction (verified structurally in
#       tests/RSVDCBEBondUpdate/test_cbe_bug.jl, which is where that claim is actually
#       pinned); both baselines materialise a rank-4 two-site block, and 2-site TDVP
#       evolves one inside its Krylov loop.
#
# WALL TIME IS NOT THE METHOD'S COST. `cbe_bug_bond_update` and `tdvp2_step!` both rebuild
# their environment stacks at every bond, O(L^2) per step -- a deliberate placeholder
# (docs/cbe_bug.md section 4 stage 3). Timings compare the two implementations to each
# other, nothing more, and that placeholder must go before any campaign run.

using LinearAlgebra, Printf
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
include("../tests/common/dense_reference.jl")

const L       = 8
const DELTA   = 0.0            # XX: the domain-wall rank-growth benchmark
const DT      = 0.05
const NSTEPS  = 20

"Fresh domain-wall start plus its dense reference at the end of the run."
function setup()
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    v0 = dense_state(psi)
    H = dense_heisenberg(L; delta = DELTA)
    return psi, dense_exact_propagate(H, v0, DT * NSTEPS)
end

err_vs(psi, want) = norm(dense_state(psi) - want)

# ── the runs ─────────────────────────────────────────────────────────────────

function run_cbe(maxdim, dex)
    psi, want = setup()
    h = xxz_chain(L; delta = DELTA)
    maxbond, t = 0, 0.0
    t = @elapsed for _ in 1:NSTEPS
        cbe_bug_bond_update(psi, h, -im * DT; dex = dex, maxdim = maxdim,
                            trunc_thresh = 1e-12)
        maxbond = max(maxbond, maximum(bond_dims(psi)))
    end
    return (; scheme = "cbe dex=$(dex == 0 ? "r" : dex)", maxdim, err = err_vs(psi, want),
              maxbond, theta = 0, time = t)
end

function run_tdvp2(maxdim)
    psi, want = setup()
    h = xxz_chain(L; delta = DELTA)
    maxbond, theta, t = 0, 0, 0.0
    t = @elapsed for _ in 1:NSTEPS
        info = tdvp2_step!(psi, h, -im * DT; maxdim = maxdim, trunc_thresh = 1e-12)
        maxbond = max(maxbond, info.max_bond)
        theta = max(theta, info.theta_elements)
    end
    return (; scheme = "tdvp2", maxdim, err = err_vs(psi, want), maxbond, theta, time = t)
end

function run_bug(maxdim; pad = false)
    psi, want = setup()
    gates = bond_gates(psi; delta = DELTA)
    opts = BondUpdateOptions(dt = DT, n_steps = NSTEPS, order = :strang, maxdim = maxdim,
                             trunc_thresh = 1e-12, normalize = false, pad = pad)
    t = @elapsed info = bond_update_bug!(psi, gates; opts = opts)
    return (; scheme = pad ? "bug 2r pad" : "bug 2r K/L", maxdim, err = err_vs(psi, want),
              maxbond = maximum(info.max_bond_dims), theta = -1, time = t)
end

# ── order-in-dt helpers (fixed total time, varying step) ─────────────────────

"Run to `t = dt*nsteps` and return the error against the dense propagator."
function _run_dt_cbe(maxdim, dex, dt, nsteps)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    v0 = dense_state(psi)
    h = xxz_chain(L; delta = DELTA)
    for _ in 1:nsteps
        cbe_bug_bond_update(psi, h, -im * dt; dex = dex, maxdim = maxdim,
                            trunc_thresh = 1e-12)
    end
    want = dense_exact_propagate(dense_heisenberg(L; delta = DELTA), v0, dt * nsteps)
    return norm(dense_state(psi) - want)
end

function _run_dt_tdvp2(maxdim, dt, nsteps)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    v0 = dense_state(psi)
    h = xxz_chain(L; delta = DELTA)
    for _ in 1:nsteps
        tdvp2_step!(psi, h, -im * dt; maxdim = maxdim, trunc_thresh = 1e-12)
    end
    want = dense_exact_propagate(dense_heisenberg(L; delta = DELTA), v0, dt * nsteps)
    return norm(dense_state(psi) - want)
end

# ── the memory micro-benchmark ───────────────────────────────────────────────
#
# Isolated from everything else: the SAME bond, the SAME expanded frames, one Krylov
# matvec each way. The 0-site route contracts bond x bond matrices; the two-site route has
# to build and act on the rank-4 block. Anything else in either step (environment rebuilds
# above all) is excluded, so this is the matvec cost and nothing else.

function matvec_memory(maxdim)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    h = xxz_chain(L; delta = DELTA)
    for _ in 1:8
        cbe_bug_bond_update(psi, h, -im * DT; maxdim = maxdim, trunc_thresh = 1e-12)
    end
    i = L ÷ 2
    canonical!(psi, i)
    f = bond_frame(psi, i)
    lenv = left_env_stack(psi, h; upto = i - 1)
    renv = right_env_stack(psi, h; downto = i + 2)
    ex = cbe_expand(f, h, i, lenv, renv)

    S = to_concrete(contract(contract(ex.U_ex', (1, 2), frame_theta(f), (1, 2)),
                             (2, 3), ex.V_ex', (2, 3)))
    H0 = zero_site_h(h, i, lenv, renv, ex.U_ex, ex.V_ex)
    Theta = to_concrete((ex.U_ex * S) * ex.V_ex)

    apply_zero_site(H0, S)                        # warm up, so the numbers are not compile
    apply_h_two_site(Theta, h, i, lenv, renv)
    a0 = @allocated apply_zero_site(H0, S)
    a2 = @allocated apply_h_two_site(Theta, h, i, lenv, renv)

    return (; chi = leg_dim(ex.U_ex, 3), s_elems = tensor_elements(S),
              theta_elems = tensor_elements(Theta), alloc_0site = a0, alloc_2site = a2)
end

# ── report ───────────────────────────────────────────────────────────────────

function main()
    @printf("CBE-BUG vs baselines: L=%d, XX (delta=%g), dt=%g, %d steps\n",
            L, DELTA, DT, NSTEPS)
    println("full rank at the centre would be ", min(2^(L ÷ 2), 2^(L - L ÷ 2)))
    println()

    rows = Any[]
    for maxdim in (4, 6, 8, 12)
        for dex in (1, 2, 0)
            push!(rows, run_cbe(maxdim, dex))
        end
        push!(rows, run_tdvp2(maxdim))
        push!(rows, run_bug(maxdim))
        push!(rows, run_bug(maxdim; pad = true))
    end

    @printf("%-14s %7s %12s %8s %10s %9s\n",
            "scheme", "maxdim", "err vs dense", "maxbond", "theta el.", "time s")
    println(repeat("-", 66))
    for r in rows
        @printf("%-14s %7d %12.3e %8d %10s %9.2f\n", r.scheme, r.maxdim, r.err, r.maxbond,
                r.theta < 0 ? "rank-4" : (r.theta == 0 ? "none" : string(r.theta)), r.time)
    end

    # Order in dt. MUST be measured with the rank capped below full: both CBE-BUG and
    # 2-site TDVP are exact once the manifold is the whole space (MEASURED ~1e-13 at
    # L=6/maxdim=32, job 94971), so a convergence ratio taken there compares two epsilons.
    # The truncation error sets a floor, so read the ratios only while they are above it.
    println("\nOrder in dt (rank capped below full, so a dt error exists to see):")
    @printf("%-14s %7s %8s %12s %8s\n", "scheme", "maxdim", "dt", "err vs dense", "ratio")
    println(repeat("-", 54))
    for (name, run) in (("cbe dex=r", (m, d, n) -> _run_dt_cbe(m, 0, d, n)),
                        ("cbe dex=1", (m, d, n) -> _run_dt_cbe(m, 1, d, n)),
                        ("tdvp2",     (m, d, n) -> _run_dt_tdvp2(m, d, n)))
        prev = NaN
        for dt in (0.1, 0.05, 0.025)
            e = run(10, dt, round(Int, 1.0 / dt))
            @printf("%-14s %7d %8.4f %12.3e %8s\n", name, 10, dt, e,
                    isnan(prev) ? "-" : @sprintf("%.2f", prev / e))
            prev = e
        end
    end

    println("\nKrylov matvec memory, same bond and same expanded frames:")
    @printf("%7s %8s %10s %12s %12s %8s\n",
            "maxdim", "chi_ex", "S elems", "Theta elems", "0-site B", "2-site B")
    println(repeat("-", 64))
    for maxdim in (4, 8, 12, 16)
        m = matvec_memory(maxdim)
        @printf("%7d %8d %10d %12d %12d %8d\n",
                maxdim, m.chi, m.s_elems, m.theta_elems, m.alloc_0site, m.alloc_2site)
    end

    println("\nBENCH_DONE")
end

main()
