# benchmarks/qtt/vortex_shedding.jl -- QTT plan §3.2: damped nonlinear GP vortex shedding behind a
# moving rounded paddle (Connor et al.'s EQUATION and setup, not their method): homogeneous
# condensate g = 1, gamma = 0.1, mu = 1 in a periodic box, a paddle swept at v_sweep then ramped
# down. The time reference is dense RK4 of the SAME finite-difference system (independent code,
# benchmarks/qtt/common.jl). Arms: BUG p = 2/3/4 at the candidate steps, TDVP2 lagged (and
# consistent with T3V_CONSISTENT=1), the split-dissipation control. Per snapshot: N, energies,
# grand-canonical F, external power, vortex charges (sampled grid), chi profile, error vs reference.
# ⚠ The paddle is an engineering fixture (`GPPaddle`); Appendix B's exact shape must be read off
# the source before quoting a reproduction (plan §3.2).
#
# ENV: T3V_OUT, T3V_N (8 = 256^2), T3V_BOX (64), T3V_T (12), T3V_HS ("0.005,0.01,0.025,0.05,0.1"),
#      T3V_SNAP (1.0), T3V_ORDERS ("2,3,4"), T3V_MAXDIM (128), T3V_REFDT (1e-3), T3V_CONSISTENT (0),
#      T3V_RELAX (2.0, imaginary-time relaxation of the start with the paddle in place)

include(joinpath(@__DIR__, "common.jl"))

const OUT = envs("T3V_OUT", "t3v_out")
const N = envi("T3V_N", 8)
const BOX = envf("T3V_BOX", 64.0)
const TEND = envf("T3V_T", 12.0)
const HS = envlist("T3V_HS", "0.005,0.01,0.025,0.05,0.1")
const SNAP = envf("T3V_SNAP", 1.0)
const ORDERS = round.(Int, envlist("T3V_ORDERS", "2,3,4"))
const MAXD = envi("T3V_MAXDIM", 128)
mkpath(OUT)

g = qtt_grid(N, N; x = (-BOX / 2, BOX / 2), y = (-BOX / 2, BOX / 2))
pad = GPPaddle(10.0, -BOX / 4, 1.5, BOX / 3 / 1.5, 2.0, 1.0, BOX / 6, 0.0, 0.5)
m = gp_model(g; gnl = 1.0, gamma = 0.1, mu = 1.0, omega = 0.0, paddle = pad)
# start: the uniform condensate n = mu / g, relaxed in imaginary time around the paddle at t = 0
psi0 = qtt_mps(qtt_separable(g, x -> 1.0 + 0im, y -> 1.0 + 0im))
mr = gp_model(g; gnl = 1.0, gamma = 0.0, mu = 1.0, omega = 0.0,
              paddle = GPPaddle(pad.height, pad.x_start, 0.0, 1e9, 0.0, pad.width_x, pad.width_y,
                                pad.y_c, pad.smooth))
N0 = qtt_l2norm(psi0, g)^2
relax = envf("T3V_RELAX", 2.0)
relax > 0 && gp_ground_state!(psi0, mr, uniform_schedule(0.1, round(Int, relax / 0.1)); order = 2,
                              maxdim = MAXD)
qtt_normalize!(psi0, g; N = N0)
A0 = qtt_cores(psi0)
snaps = collect(SNAP:SNAP:TEND)

# the reference trajectory, kept at the snapshot times
refdt = envf("T3V_REFDT", 1e-3)
refv = Dict{Float64, Vector{ComplexF64}}()
v = grid_vec(A0, g)
tcur = 0.0
tref = @elapsed for ts in snaps
    global v = gp_dense_rk4(m, v, tcur, ts; dt = refdt)
    global tcur = ts
    refv[ts] = copy(v)
end
println("reference done in $(round(tref; digits = 1)) s")

rows = NamedTuple[]
function observe!(rows, arm, h, t, psi, wall, extra = (;))
    A = qtt_cores(psi)
    o = gp_observables(m, A, t)
    st = max(1, 2^N ÷ 256)
    _, _, M = qtt_sample(A, g; stride = st)
    vp, vm, _ = vortex_count(M)
    rv = refv[t]
    vv = grid_vec(A, g)
    rp, rm, _ = vortex_count(reshape(rv, 2^N, 2^N)[1:st:end, 1:st:end])
    push!(rows, merge((arm = arm, h = h, t = t, N = o.N, E_kin = o.E_kin, E_int = o.E_int,
                       E_pad = o.E_pad, F_gc = o.F_gc, P_ext = o.P_ext, vortices_plus = vp,
                       vortices_minus = vm, ref_vortices_plus = rp, ref_vortices_minus = rm,
                       err_state = norm(vv - rv) / norm(rv),
                       err_density = norm(abs2.(vv) - abs2.(rv)) / norm(abs2.(rv)),
                       chi_max = o.chi, chi_profile = join(bond_dims(psi), ";"), wall = wall),
                      extra))
    println(rows[end]); flush(stdout)
end

arms = Any[]
for p in ORDERS, h in HS
    push!(arms, (Symbol("bug$p"), h, (ps, t, hh) -> qtt_bug_step!(ps, GPFlow(m), t, hh; order = p,
                                                                  maxdim = MAXD, tol_basis = 1e-6,
                                                                  tol_trunc = 1e-8)))
end
for h in HS
    push!(arms, (:tdvp2_lagged, h, (ps, t, hh) -> qtt_tdvp2_step!(ps, GPFlow(m), t, hh;
                                                                  maxdim = MAXD, tol_trunc = 1e-8)))
    push!(arms, (:split, h, (ps, t, hh) -> qtt_split_tdvp_step!(ps, m, t, hh; maxdim = MAXD,
                                                                tol_trunc = 1e-8)))
    envi("T3V_CONSISTENT", 0) == 1 &&
        push!(arms, (:tdvp2_consistent, h, (ps, t, hh) -> qtt_tdvp2_step!(ps, GPFlow(m), t, hh;
                                                     density = :consistent, maxdim = MAXD,
                                                     tol_trunc = 1e-8)))
end
for (arm, h, step!) in arms
    psi = qtt_mps(A0)
    t, wall, k = 0.0, 0.0, 0
    amax = 1.0
    failed = false
    for ts in snaps
        while t < ts - 1e-12
            hh = min(h, ts - t)
            w = @elapsed r = try
                step!(psi, t, hh)
            catch err
                @warn "arm $arm h=$h failed at t=$t" err
                failed = true
                nothing
            end
            failed && break
            r !== nothing && hasproperty(r, :A_max) && (amax = max(amax, r.A_max))
            wall += w
            t += hh
            k += 1
        end
        failed && break
        observe!(rows, arm, h, ts, psi, wall, (steps = k, A_max = amax))
    end
    write_csv(joinpath(OUT, "shedding.csv"), rows)
end
println("T3V-DONE")
