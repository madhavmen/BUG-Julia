# THE ORDER OF THE BUG SCHEMES ON A 2D CYLINDER -- simulation_plan_2 U22, "order checks, measured
# not assumed", with spatial errors suppressed.
#
# Why not the S^zz validation runs: at 16 sites cap 256 is FULL RANK at the centre bond, so the BUG
# trial space holds everything, its evolution is exact at any dt, and every scheme gives the same
# number (TASK4_VALIDATION_RESULTS.md, job 17282199). Measuring order needs (a) a system whose rank
# stays far below the cap and (b) tolerances far below the step error. So: an Lx x Ly square
# cylinder (default 6x4, N = 24, sector dim 2.7M) from the Neel product state, short T, tight
# tolerances, and the EXACT state from sparse Krylov in the S^z = 0 sector. This is the 2D twin of
# the chain closure study (KRYLOV_CLOSURE_AUDIT.md IT2).
#
# Error measure: max_j |<S^z_j>(T) - exact|. A t = 0 gate checks the MPS and the sparse basis
# share one site convention before anything is graded.
#
# Run:  julia -t 2 --project=PKG benchmarks/spectral/order_ladder_2d.jl Lx Ly OUTDIR
# Env:  OL_T (0.8)  OL_DTS ("0.2,0.1,0.05,0.025")  OL_D (512)
#       OL_ARMS ("tdvp2,tdvp4,tdvp4y,bug1,bugmid,bug2,bug4"); tdvp4 / tdvp4y = 2-site TDVP composed to
#       order 4 (Suzuki 5 stages / Yoshida 3 stages, `tdvp2_composed_step!`)
#       OL_TB (1e-10) OL_TT (1e-12) OL_TE (1e-13)

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))

const LX   = parse(Int, get(ARGS, 1, "6"))
const LY   = parse(Int, get(ARGS, 2, "4"))
const OUT  = get(ARGS, 3, joinpath(@__DIR__, "..", "results"))
const T    = parse(Float64, get(ENV, "OL_T", "0.8"))
const DTS  = parse.(Float64, split(get(ENV, "OL_DTS", "0.2,0.1,0.05,0.025"), ","))
const DCAP = parse(Int, get(ENV, "OL_D", "512"))
const ARMS = split(get(ENV, "OL_ARMS", "tdvp2,tdvp4,tdvp4y,bug1,bugmid,bug2,bug4"), ",")
const AKW  = (tol_basis = parse(Float64, get(ENV, "OL_TB", "1e-10")),
              tol_trunc = parse(Float64, get(ENV, "OL_TT", "1e-12")),
              tol_expv  = parse(Float64, get(ENV, "OL_TE", "1e-13")),
              maxdim = DCAP, parallel = Threads.nthreads() >= 2)

sz_mps(p) = magnetisation(copy(p)) ./ max(norm(p)^2, eps())

function stepper(arm, W)
    arm == "tdvp2"  && return (p, tau) -> tdvp2_step!(p, W, tau; maxdim = DCAP, trunc_thresh = 1e-14,
                                                       maxiter = 30)
    arm == "tdvp4"  && return (p, tau) -> tdvp2_composed_step!(p, W, tau; order = 4,
                                    composition = :suzuki, maxdim = DCAP, trunc_thresh = 1e-14,
                                    maxiter = 30)
    arm == "tdvp4y" && return (p, tau) -> tdvp2_composed_step!(p, W, tau; order = 4,
                                    composition = :yoshida, maxdim = DCAP, trunc_thresh = 1e-14,
                                    maxiter = 30)
    arm == "bug1"   && return (p, tau) -> cbe_bug_step!(p, W, tau; AKW...)
    arm == "bugmid" && return (p, tau) -> cbe_bug_midpoint_step!(p, W, tau; AKW...)
    arm == "bug2"   && return (p, tau) -> cbe_bug_step!(p, W, tau; order = 2, AKW...)
    arm == "bug4"   && return (p, tau) -> cbe_bug_step!(p, W, tau; order = 4, AKW...)
    error("unknown arm $arm")
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)
    N = LX * LY
    Jm = square_cylinder_couplings(LX, LY)
    W = square_cylinder_mpo(LX, LY)
    t0 = time()
    H, states, idx = pairs_sparse(N, Jm)
    v0 = neel_vector(N, idx)
    vT = expv_sparse(H, ComplexF64(-im * T), v0; m = 40, tol = 1e-13)
    szx = magnetisation_dense(vT, N, states)
    @printf("square %dx%d N=%d  sector dim %d  exact state at T=%.2f in %.1f s  D=%d  tol %s  threads %d\n",
            LX, LY, N, length(states), T, time() - t0, DCAP,
            string((AKW.tol_basis, AKW.tol_trunc, AKW.tol_expv)), Threads.nthreads())

    # t = 0 GATE: the MPS and the sparse basis must agree on which sites are up.
    s0 = sz_mps(neel_state(N))
    g0 = maximum(abs.(s0 .- magnetisation_dense(v0, N, states)))
    g0 < 1e-12 || error("site convention mismatch at t = 0: max |dSz| = $g0")
    @printf("t=0 gate: max|Sz_mps - Sz_exact| = %.1e\n", g0)

    io = open(joinpath(OUT, "order_ladder_$(LX)x$(LY).csv"), "w")
    println(io, "arm,dt,steps,err_sz,chi,wall,unmet_steps")
    errs = Dict{String, Vector{Float64}}()
    for arm in ARMS
        step! = stepper(arm, W)
        let p = neel_state(N); step!(p, ComplexF64(-im * DTS[end])); end      # JIT, untimed
        errs[arm] = Float64[]
        for dt in DTS
            n = round(Int, T / dt)
            p = neel_state(N)
            unmet = 0
            tw = @elapsed for _ in 1:n
                info = step!(p, ComplexF64(-im * dt))
                lg = hasproperty(info, :ledger) ? info.ledger : nothing
                lg isa AdaptiveLedger && budget_status(lg) === :budget_unmet && (unmet += 1)
            end
            e = maximum(abs.(sz_mps(p) .- szx))
            push!(errs[arm], e)
            chi = maximum(bond_dims(p))
            println(io, join([arm, dt, n, @sprintf("%.6e", e), chi, @sprintf("%.2f", tw), unmet], ","))
            flush(io)
            @printf("  %-6s dt=%.4f steps=%3d  err=%.3e  chi=%4d  %7.1fs  unmet=%d\n",
                    arm, dt, n, e, chi, tw, unmet)
            flush(stdout)
        end
        sl = [log(errs[arm][k] / errs[arm][k + 1]) / log(DTS[k] / DTS[k + 1])
              for k in 1:(length(DTS) - 1)]
        println("  ", arm, " slopes: ", join([@sprintf("%.2f", s) for s in sl], " "))
    end
    close(io)
    println("ORDER-LADDER-DONE")
end

main()
