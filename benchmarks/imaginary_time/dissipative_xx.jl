# THE BOUNDARY-DISSIPATED XX CHAIN (arXiv:2104.11479) -- the first OPEN model with an EXACT ref.
#
# ⛔ WHAT THIS ADDS TO THE OPEN-SYSTEM TRACK, WHICH IS OTHERWISE ALREADY BUILT.
# `PLAN-open-systems.md` §0 lists the `exact ref` column as "—" for EVERY open model: XX open,
# Heisenberg square open and the LGT are all validated against published TRENDS (Znidaric's
# ballistic->diffusive crossover, Zeno slowing) and never against a number. §4 says so outright:
# "All three are TREND validations, not number reproductions".
#
# This model is EXACTLY SOLVABLE by third quantization, so it turns that column into a real error
# column -- the same status the closed models get from Bethe and free fermions. The MPDO machinery
# (fused d=4 sites, `|I>>` as a product state, `pair_mpo` Liouvillian, `restore_trace!`) is
# REUSED FROM `fused_common.jl`; the only new pieces are the boundary jump operators `JP`/`JM`,
# which live there too so there is exactly ONE d=4 local space in the stack.
#
# ⛔ TWO SETTINGS THAT ARE NOT DEFAULTS AND ARE NOT OPTIONAL HERE.
#
#   growth = 4.0   §3.7: the shipped `growth = 2.0` CAPS THE ACHIEVABLE ORDER AT 1 on a d=4 chain
#                  and would make this run silently first order. `Tr rho` is 0.99999 in that
#                  regime -- BETTER than several second-order rows -- so the conservation law does
#                  NOT flag it. `budget = ceil(growth*dmax) - r` is sized for a d=2 site; a fused
#                  d=4 site can QUADRUPLE a bond.
#   restore_trace! §3.11: restore ALONG `|I>>`, never rescale. `rho/Tr(rho)` multiplies the
#                  traceless part too and costs a consistent 1.9x in accuracy; the minimal
#                  correction is exact in trace AND 25% better in state.
#
# ⛔ AND TAU IS REAL AND POSITIVE. The MPO carries the `-i` on the Hamiltonian halves, so it IS the
# Liouvillian and the propagator is `exp(+t*L)`. The suite's real-time `tau = -i*dt` would evolve a
# different generator. `hermitian = false` is mandatory: a Liouvillian is never Hermitian, and
# Lanczos on it does not fail, it LIES.
#
# ⛔ THE REFERENCE IS ANALYTIC, AND NO DENSE OBJECT IS BUILT HERE AT ALL.
# An earlier version of this file compared against a dense `4^L x 4^L` Liouvillian. That works, but
# it caps the whole validation at `L <= 5` -- so it can certify WIRING and can never certify a
# BENCHMARK, which is the entire point of the model. Third quantization does better: the Lindbladian
# is quadratic, the two-point function closes, and the exact `<S^z_j(t)>` is an `N x N` Lyapunov
# solve costing `O(N^3)` IN THE CHAIN LENGTH. So the arms are checked at `N = 16` and `N = 32`,
# where they actually have to truncate, against an exact number.
#
# `benchmarks/imaginary_time/certify_xx_reference.jl` is where the dense object still lives, and it is used ONCE:
# MEASURED agreement with the analytic solution to <= 8.9e-16 on both `<S^z_j>` and the current, at
# `L = 2..5` and `t = 0.1 .. 40`. Beyond that it is retired.
#
# TWO REFERENCE LEVELS ARE AVAILABLE HERE, AND THEY TEST DIFFERENT THINGS:
#   xx_boundary_sz(N, t)             the full transient profile, exact at every t
#   xx_boundary_current_published()  the CLOSED FORM NESS current
#                                    j = (mu_L - mu_R) eps J^2 / (4J^2 + eps^2), INDEPENDENT of N
#                                    -- confirmed against the Lyapunov solve at 240 parameter
#                                    points (N x J x eps x bias) to 1.8e-14.
#
# Run:  julia -t 2 --project=. benchmarks/imaginary_time/dissipative_xx.jl <mode>
#
#   all         spectrum + calibrate + order + ness, then the grid study IF validation passed
#   calibrate   is the MPDO generator right? (vs a dense Liouvillian, small L)
#   order       is it right for the right reason? (dt-convergence order)
#   ness        the steady-state current -- CLUSTER work, 2000 steps per arm per L
#   spectrum    the Liouvillian spectrum behind the "no log grid here" figure
#   grids       the log-vs-uniform grid study alone, ungated
#   chigrowth   accuracy + chi(t) + wall clock on ONE trajectory per arm   [XX_L XX_DT XX_TMAX
#               XX_NSAVE XX_D XX_MAXITER XX_ARMS XX_APPEND]
#   dtrobust    the large-dt / strong-dissipation robustness test           [XX_L XX_EPS XX_D
#               XX_TMAX XX_MAXITER XX_DTS XX_ARMS]  -- ⛔ sweep XX_EPS, its default finds nothing

using LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "..", "fused_common.jl"))
include(joinpath(@__DIR__, "..", "..", "tests", "common", "free_fermion.jl"))
# The grid constructors, `run_arm`, `trajectory_error` and the CSV schema, shared with
# `logtime_suite.jl` so the two cannot drift -- see the header of `grids_common.jl`.
include(joinpath(@__DIR__, "grids_common.jl"))

# ⚠ BOTH ARE ENV-OVERRIDABLE BECAUSE THE CLUSTER NEEDS TO RAISE `CAP` WITHOUT EDITING THIS FILE.
# `submit_dissipative_xx.slurm` exports `DISSIP_CAP`, and a hardcoded const there would have been
# silently ignored -- a submit script that LOOKS configured and is not is worse than one that
# obviously hardcodes, because the resulting run is indistinguishable from the one you asked for.
#
# ⛔ AND `CAP` IS THE KNOB THAT DECIDES WHETHER THE CURRENT MEANS ANYTHING. The superket's operator
# entanglement is what carries transport: a state truncated too hard keeps a plausible FLAT
# magnetisation profile and a perfect `Tr rho = 1` while its current is wrong. Raise it before
# concluding anything about an integrator from a missed `ness` tolerance.
const GROWTH = parse(Float64, get(ENV, "DISSIP_GROWTH", "4.0"))  # §3.7 -- NOT the shipped default
const CAP    = parse(Int,     get(ENV, "DISSIP_CAP",    "128"))

# ── the arms, defined ONCE so both checks below drive identical integrators ───────────────────
#
# ⛔ EVERY ARM CARRIES `hermitian = false`. A Liouvillian is never Hermitian and Lanczos on it does
# not fail, it LIES -- and `tdvp2_step!` ACCEPTED-AND-IGNORED this keyword until 2026-08-27, so any
# open-system number taken through that entry point before then ran the wrong solver. That is a
# second reason this file exists: it is the first place `tdvp2` meets a fused generator at all.
#
# ⚠ `growth = 4.0` IS ON THE BUG ARMS ONLY, and is not cosmetic: §3.7's shipped `growth = 2.0` is
# sized for a d = 2 site and CAPS THE ACHIEVABLE ORDER AT 1 on a d = 4 chain. `tdvp2` has no such
# knob -- its rank comes from the two-site SVD -- so the arms are matched on `maxdim`/`maxiter`/
# `trunc_thresh` and on nothing else, which is the honest comparison here.
"""
    open_steppers(W, D, maxiter; arms) -> Vector{Pair{String,Function}}

The arms bound to one generator and one rank cap, as `(psi, dt) -> info` closures.

⚠ A FACTORY, NOT A CONSTANT DICT, so `grids` can vary `D` and `maxiter` while `calibrate`/`order`/
`ness` keep the defaults -- with ONE definition of what each arm is. Two copies would let a knob
drift between the validation and the study that cites it.
"""
function open_steppers(W, D::Int = CAP, maxiter::Int = 30; arms = ("tdvp2", "bug_rsvd", "bug_par"))
    kw = (maxdim = D, trunc_thresh = 1e-12, maxiter = maxiter, hermitian = false)
    rs = (exact = false, dover = 4, growth = GROWTH)
    tbl = Dict{String, Function}(
        "tdvp2"     => (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); kw...),
        "bug_rsvd"  => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); kw..., rs...),
        "bug_par"   => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); kw..., rs...,
                                                parallel = true),
        "bug_exact" => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); kw..., exact = true,
                                                growth = GROWTH))
    for a in arms
        haskey(tbl, a) || throw(ArgumentError("unknown arm $a; have $(sort(collect(keys(tbl))))"))
    end
    return [a => tbl[a] for a in arms]
end

const DEFAULT_ARMS = ("tdvp2", "bug_rsvd", "bug_par")

# ── the reference: ANALYTIC, O(N^3), no dense object ─────────────────────────────────────────

"""
    reference_profiles(L, ts; ...) -> Dict{Float64,Vector{Float64}}

Exact `<S^z_j(t)>` from the Lyapunov solution, computed once and shared by every arm.

⚠ HOISTED OUT OF THE ARM LOOP. Each entry costs a Lyapunov solve plus two `N x N` exponentials;
recomputing them per arm per checkpoint is the same waste that once made a trajectory reference
take 36,000 matrix exponentials.
"""
function reference_profiles(L::Int, ts; J::Float64 = 1.0, eps::Float64 = 1.0,
                            mu_L::Float64 = 1.0, mu_R::Float64 = -1.0)
    return Dict{Float64, Vector{Float64}}(
        Float64(t) => xx_boundary_sz(L, Float64(t); J = J, eps_L = eps, eps_R = eps,
                                     mu_L = mu_L, mu_R = mu_R) for t in ts)
end

"""
    mps_current(Imps, rho, L; J) -> Vector{Float64}

Bond currents read off the superket: `j_k = 2 J Im Tr(sigma^-_k sigma^+_{k+1} rho) / Tr rho`.

⛔ FIXED 2026-08-28: THE SIGN WAS WRONG, AND THE OLD REASONING MISSED HALF THE PROBLEM. It counted
`overlap`'s conjugation but NOT `site_apply!`'s transpose, and the two CANCEL. Working it through:

    site_apply!(o, k, Kp)      applies Kp' = sigma^-_k        (leg-1 contraction = transpose)
    site_apply!(o, k+1, Km)    applies Km' = sigma^+_{k+1}
    => o is the superket of  sigma^-_k sigma^+_{k+1}
    overlap(o, rho)            = Tr( (sigma^- sigma^+)' rho ) = Tr( sigma^+_k sigma^-_{k+1} rho )

so what comes back is the correlator AS NAMED, not its conjugate, and `j_k = -2J Im(...)` applies
directly. The old docstring concluded the opposite and the code carried a `+`.

✅ MEASURED, and this is what caught it: against the analytic solution the MPS current was the exact
NEGATIVE of the reference at every checkpoint -- `-0.000140` vs `+0.000140` at t=0.76, `-0.000206`
vs `+0.000206` at t=0.80 -- while `<S^z_j>` agreed to 5.0e-08. A wrong SIGN on a current leaves the
magnetisation profile untouched, so nothing else in the suite could have flagged it.

⚠ IT HAD NEVER BEEN EXERCISED. `ness()` compares `abs(jb - jxa)` and would have failed instantly
(`dj = 2|j|`), but it is cluster work and had not yet run; every local gate scored `<S^z_j>` only.
⚠ The same cancellation is what makes `trace_op`/`pauli_2site` return the operators they name --
certified against a dense Liouvillian to 1.9e-07 -- so this fix is consistent with that path rather
than a competing convention.
"""
function mps_current(Imps, rho, L::Int; J::Float64 = 1.0)
    trr = real(trace_rho(Imps, rho, L))
    out = Float64[]
    for k in 1:(L - 1)
        o = copy(Imps)
        site_apply!(o, k, Q4.Kp)
        site_apply!(o, k + 1, Q4.Km)
        push!(out, -2 * J * imag(2.0^(L / 2) * overlap(o, rho) / trr))
    end
    return out
end

"""
    evolve_arm(arm, W, shift, Imps, L, ts, dt) -> Vector{(t, profile, trace, current, chi)}

March ONE arm to each checkpoint with a fixed `dt`, reporting the MPS `<S^z_j>` profile, the trace
it carried BEFORE `restore_trace!` touched it, the bond currents and the largest bond dimension.

⛔ THE PRE-RESTORE TRACE IS THE POINT OF LOGGING IT. `Tr rho` is the Lindbladian's conservation law
and the two arms reach it differently: `tdvp2`'s sweep is a product of local exponentials, each of
which annihilates `<<I|` on its own, so it conserves the trace STRUCTURALLY; a Galerkin step evolves
`exp(tau P L P)` and only conserves it if `P|I>> = |I>>`, which nothing in CBE/BUG arranges. Quoting
`Tr = 1` AFTER the restoration as evidence of accuracy would be circular -- so the number reported
is the one measured before the correction.
"""
function evolve_arm(arm::String, W, shift::Float64, Imps, L::Int, ts, dt::Float64;
                    J::Float64 = 1.0)
    step! = open_steppers(W; arms = (arm,))[1].second
    # ⛔ THE MAXIMALLY MIXED STATE IS `|I>> / 2^(L/2)`, NOT `|I>>`. `identity_superket` returns the
    # NORMALISED superket, whose trace is `2^(L/2)`; the state we want is `rho = I/2^L`, i.e.
    # `vec(rho) = vec(I)/2^L = |I>>_normalised / 2^(L/2)`.
    #
    # ⛔ GETTING THIS WRONG DOES NOT LOOK LIKE A WRONG INITIAL STATE -- IT LOOKS LIKE A DEAD
    # INTEGRATOR, AND IT COST A FULL DIAGNOSTIC ROUND. With `rho = copy(Imps)` at L = 8 the trace
    # starts at 16, so `restore_trace!` computes `eps = 1 - 16 = -15` and adds `(-15/16)|I>>`,
    # collapsing the state to `(1/16)|I>>` AT EVERY STEP. The measured symptoms were:
    # `<S^z_j>` EXACTLY zero at every site and time, `chi = 1`, `Tr rho = 1.0000000000` exactly,
    # IDENTICAL for tdvp2 / bug_rsvd / bug_par, and the same deviation at dt = 0.08 .. 0.01
    # (observed order 0.00). Every one of those reads as "the sweep does not apply this generator".
    #
    # ⚠ EVERY OTHER OPEN DRIVER IN THIS REPO IS UNAFFECTED because they all start from
    # `rho0_product`, a PURE state, whose trace is `2^(L/2) * 2^(-L/2) = 1` already. Only a
    # maximally-mixed start meets this.
    rho = copy(Imps)
    apply_shift!(rho, 1.0 / 2.0^(L / 2))
    tr0 = real(trace_rho(Imps, rho, L))
    abs(tr0 - 1) < 1e-10 || error("initial Tr rho = $tr0, expected 1 (arm $arm, L=$L)")
    out = Tuple{Float64, Vector{Float64}, Float64, Vector{Float64}, Int}[]
    # ⛔ `tnow` IS THE ACTUAL ELAPSED TIME AND IT IS WHAT GETS REPORTED -- NOT THE REQUESTED LABEL.
    # This line used to advance `tprev = Float64(t)` (the NOMINAL time) while stepping only
    # `round(Int, (t - tprev)/dt)` times, so whenever `t/dt` was not an integer the state fell
    # BEHIND its own label and was then scored against the exact solution at a time it had never
    # reached.
    #
    # MEASURED, and it fabricated a six-order-of-magnitude error: at `dt = 0.02`, `round(12.5)` is
    # 12 (Julia rounds half to EVEN), so `t = 0.25` was really 0.24 and `t = 0.50` really 0.48. The
    # `order` scan then read
    #     dt 0.0400 -> 3.675e-03      dt 0.0200 -> 3.675e-03  (order 0.00)
    #     dt 0.0100 -> 1.138e-09      dt 0.0050 -> 1.437e-10  (order 2.99)
    # -- the two coarse steps BOTH landing on 0.24 is why they agreed to four digits, and the
    # collapse is simply the first `dt` that divides 0.25 exactly. `CALIBRATION FAILED` came from
    # this, not from the MPO.
    #
    # ⚠ THE TELL WAS `order 0.00`, NOT THE SIZE OF THE ERROR. A residual that does not move when
    # `dt` moves is never a discretisation error, however plausible its magnitude looks.
    #
    # ✅ COMPARING AT `tnow` IS EXACT, NOT A PATCH. The reference is a closed-form Lyapunov solution
    # valid at EVERY `t` (`xx_boundary_sz(L, t)`), so there is nothing to interpolate: score the
    # state at the time it actually reached. Callers MUST use the returned time for the reference.
    tnow = 0.0
    trb = 1.0                                # trace measured BEFORE the last restoration
    for t in ts
        for _ in 1:max(0, round(Int, (Float64(t) - tnow) / dt))
            # ⛔ REAL, POSITIVE tau -- the MPO already carries the -i on the Hamiltonian halves.
            step!(rho, dt)
            apply_shift!(rho, exp(dt * shift))
            trb = restore_trace!(Imps, rho, L; maxdim = CAP)   # §3.11 -- along |I>>, never rescale
            tnow += dt
        end
        trr = real(trace_rho(Imps, rho, L))
        # ⚠ `bond_dims`, not `size(rho[i], 1)`. The latter is the LEFT bond of each site, so it
        # silently omits the last bond and reports 1 for site 1 regardless -- fine for spotting a
        # frozen product state, wrong for reporting a rank.
        chi = maximum(bond_dims(rho))
        push!(out, (tnow, [real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L],
                    trb, mps_current(Imps, rho, L; J = J), chi))
    end
    return out
end

"""
    calibrate(; L = 8, arms = DEFAULT_ARMS) -> Bool

Evolve the SAME parameters two ways -- the fused MPO through EACH integrator, and the ANALYTIC
Lyapunov solution -- and compare `<S^z_j(t)>` SITE BY SITE.

⚠ `L = 8`, NOT 3. The reference costs `O(N^3)` rather than `4^N`, so there is no reason to validate
at a size where nothing truncates. At L = 8 the superket actually grows a bond dimension and the
comparison exercises the truncation path that a benchmark run depends on.

⛔ BOTH FAMILIES ARE VALIDATED HERE, NOT JUST BUG. `tdvp2` is the BASELINE every open-system claim
is measured against, so an unvalidated `tdvp2` on this generator would make the comparison
worthless in the same way an unvalidated MPO would. And it is the arm with the specific reason to
doubt it: `add_mps` only ever worked on `cbe_bug` output (the bond ARROW differs by sweep -- '+'
for tdvp2/cbe1s, '-' for cbe_bug), so `restore_trace!` THREW on the first `tdvp2` trajectory ever
run against a fused generator. That is fixed by gauging both summands, and this is the check that
the fix produces right physics rather than merely a state that no longer throws.

⛔ THE PROFILE, NOT ONE NUMBER, because the two failure modes look different in it: a wrong hopping
RESCALES TIME (every site wrong together, smoothly) while a wrong bra-leg transpose breaks trace
preservation and moves sites DIFFERENTLY. A single scalar cannot separate them, and `Tr rho` is
reported alongside because a Lindbladian that has lost it is not a Lindbladian.

⚠ ASYMMETRIC BOUNDARIES ON PURPOSE (`mu_L = +1` pumps UP at the left, `mu_R = -1` pumps DOWN at the
right). A symmetric choice gives a flat reflection-symmetric profile that hides exactly the
left/right and ket/bra swaps this exists to catch -- the same trap as pinning a site convention
with a Neel state (`dense-vs-sparse` landmine).

⚠ THE TOLERANCE IS A DISCRETISATION BOUND, NOT MACHINE PRECISION. At `dt = 0.005` a second-order
scheme is expected around `1e-5`, so `1e-4` separates "the operator is right" from "the operator is
wrong" without demanding an exactness no finite step can deliver. `order()` is what turns that into
a real statement: agreement must IMPROVE as `dt^2`.
"""
function calibrate(; L::Int = 8, J::Float64 = 1.0, eps::Float64 = 1.0,
                   ts = (0.25, 1.0, 3.0), dt::Float64 = 0.005,
                   arms = DEFAULT_ARMS, tol::Float64 = 1e-4)
    set_symmetry!(:none)
    W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
    ref = reference_profiles(L, ts; J = J, eps = eps)
    Imps = identity_superket(L)
    @printf("\ncalibrate  L=%d J=%g eps=%g  (mu_L=+1, mu_R=-1)  dt=%g  growth=%g  cap=%d\n",
            L, J, eps, dt, GROWTH, CAP)
    @printf("reference: ANALYTIC %d x %d Lyapunov solve -- exact, no dense object\n", L, L)
    for t in sort(collect(Float64.(ts)))
        @printf("  t=%-5.2f exact <S^z_j> = %s\n", t,
                join([@sprintf("%+.6f", x) for x in ref[t]], " "))
    end
    @printf("\n%-10s %-6s %-12s %-14s %-5s %s\n", "arm", "t", "max|dSz|", "Tr rho (pre)",
            "chi", "MPO <S^z_j>")
    flush(stdout)
    ok = true
    # ⛔ THE REFERENCE IS EVALUATED AT THE TIME THE STATE ACTUALLY REACHED, not at the requested
    # label. `evolve_arm` returns `tnow`; when `t/dt` is not an integer those differ, and looking
    # the reference up by the label scores the run against a time it never reached -- which is
    # exactly the bug that produced `order 0.00` and a fake 3.675e-03. The Lyapunov solution is
    # closed-form at EVERY `t`, so this costs one `O(N^3)` solve per row and interpolates nothing.
    ref_at(tt) = xx_boundary_sz(L, tt; J = J, eps_L = eps, eps_R = eps, mu_L = 1.0, mu_R = -1.0)
    for arm in arms
        for (t, prof, trr, _, chi) in evolve_arm(arm, W, shift, Imps, L, ts, dt; J = J)
            dev = maximum(abs.(prof .- ref_at(t)))
            ok &= dev < tol
            @printf("%-10s %-6.2f %-12.3e %-14.10f %-5d %s\n", arm, t, dev, trr, chi,
                    join([@sprintf("%+.6f", x) for x in prof], " "))
            flush(stdout)
        end
    end
    @printf("\n%s\n", ok ? "CALIBRATION PASSED -- every arm reproduces the exact solution" :
                           "CALIBRATION FAILED -- do NOT benchmark against this MPO")
    return ok
end

"""
    ness(; Ls = (8, 16), arms = DEFAULT_ARMS) -> Bool

Run each arm to the STEADY STATE and compare its bulk current against the CLOSED FORM
`j = (mu_L - mu_R) eps J^2 / (4J^2 + eps^2)`.

⛔ THIS IS THE NUMBER THE MODEL EXISTS TO PRODUCE, and it is the one an MPS run can get wrong while
looking healthy. `Tr rho = 1` is restorable by construction and the magnetisation profile is FLAT in
the bulk, so a truncated state that has lost its transport can still show a plausible profile and a
perfect trace. The current cannot hide: it is a single number, fixed by a formula, INDEPENDENT of
`N`, and wrong the moment the superket's operator entanglement is cut too hard.

⚠ THE PROFILE CHECK IS REPORTED ALONGSIDE AND IS NOT REDUNDANT. Bulk flatness is the ballistic
signature; an arm that produced a LINEAR ramp would be reporting diffusion, i.e. the wrong transport
class, and would still be able to hit a single bond's current by accident.

⛔ THE VERDICT IS TAKEN AGAINST THE EXACT CURRENT AT THE SAME FINITE `t`, NOT AGAINST THE `t -> inf`
FORMULA, AND THE DIFFERENCE IS NOT PEDANTRY. The Liouvillian gap CLOSES AS `N^-3`, so reaching the
true steady state at N = 16 needs `t ~ 1300` -- 66,000 steps at `dt = 0.02`, which is a
cluster-scale run, and charging the integrator for the residual transient would report a physical
relaxation as an integrator error. The Lyapunov solution is exact at EVERY `t`, so the honest
integrator test is `|MPS(t) - exact(t)|` at whatever `t` is affordable.

The closed form still does real work: `exact(t)` is printed beside it, so the table shows how far
the finite-`t` current has converged towards the `N`-independent value, and the ballistic claim is
read off the BOND UNIFORMITY, which holds long before the current settles.
"""
function ness(; Ls = (8, 16), J::Float64 = 1.0, eps::Float64 = 1.0, dt::Float64 = 0.02,
              t::Float64 = 40.0, arms = DEFAULT_ARMS, tol::Float64 = 2e-3)
    jref = xx_boundary_current_published(; J = J, eps = eps, mu_L = 1.0, mu_R = -1.0)
    @printf("\nness   J=%g eps=%g  (mu_L=+1, mu_R=-1)  t=%g dt=%g  growth=%g  cap=%d\n",
            J, eps, t, dt, GROWTH, CAP)
    @printf("closed form  j = (mu_L-mu_R) eps J^2 / (4J^2 + eps^2) = %.9f   [independent of N]\n",
            jref)
    @printf("\n%-6s %-10s %-14s %-14s %-12s %-12s %-12s %-5s\n", "L", "arm",
            "j_bulk(MPS)", "j_bulk(exact)", "|dj| exact", "max|dj|bond", "bulk |Sz|", "chi")
    flush(stdout)
    ok = true
    for L in Ls
        set_symmetry!(:none)
        W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                           mu_L = 1.0, mu_R = -1.0)
        Imps = identity_superket(L)
        Cx = xx_boundary_covariance(L, t; J = J, eps_L = eps, eps_R = eps,
                                    mu_L = 1.0, mu_R = -1.0)
        jx = xx_boundary_current(Cx; J = J)[max(1, L ÷ 2)]
        gap = xx_boundary_liouvillian_gap(L; J = J, eps_L = eps, eps_R = eps)
        @printf("  L=%-3d gap=%.3e  t_ness~%.0f  (running to t=%g: %s)\n", L, gap, 6 / gap, t,
                6 / gap <= t ? "STEADY" : @sprintf("transient, %.1f%% of j_inf", 100 * jx / jref))
        for arm in arms
            # ⛔ THE EXACT CURRENT IS TAKEN AT THE TIME THE RUN ACTUALLY REACHED. `evolve_arm` stops
            # SHORT whenever `t/dt` is not an integer, and this used to discard that time and score
            # against `exact(t)` -- the same defect that fabricated a 3.675e-03 profile error in
            # `order` and made two step sizes agree to four digits. The defaults here (t=40, dt=0.02)
            # divide evenly, so nothing was WRONG yet; it was one changed `dt` away from being so,
            # and the failure would have read as an integrator losing transport.
            (t_act, prof, _, j, chi) = evolve_arm(arm, W, shift, Imps, L, (t,), dt; J = J)[1]
            jxa = abs(t_act - t) < 1e-12 ? jx :
                  xx_boundary_current(xx_boundary_covariance(L, t_act; J = J, eps_L = eps,
                                                             eps_R = eps, mu_L = 1.0, mu_R = -1.0);
                                      J = J)[max(1, L ÷ 2)]
            jb = j[max(1, L ÷ 2)]
            dj = abs(jb - jxa)
            uni = length(j) > 1 ? maximum(abs.(diff(j))) : 0.0
            bulk = L > 4 ? maximum(abs.(prof[3:(L - 2)])) : 0.0
            ok &= dj < tol
            @printf("%-6d %-10s %-14.9f %-14.9f %-12.3e %-12.3e %-12.3e %-5d\n",
                    L, arm, jb, jxa, dj, uni, bulk, chi)
            flush(stdout)
        end
    end
    @printf("\n%s\n", ok ? "NESS PASSED -- every arm reproduces the exact current at t" :
                           "NESS FAILED -- an arm's transport is wrong")
    return ok
end

"""
    chi_growth(; L = 10, dt = 0.02, t_max = 20.0, nsave = 41, arms = DEFAULT_ARMS) -> String

The small-system confirmation for the open model, in ONE pass: agreement with the exact solution,
the growth of the bond dimension with time, and wall-clock -- per arm, on the same trajectory.

⛔ THE ACCURACY COLUMN IS AGAINST AN ANALYTIC SOLUTION, NOT AGAINST ANOTHER RUN. `xx_boundary_sz`
is the closed-form Lyapunov result (`O(N^3)`, certified to 7.2e-16 against a dense Liouvillian at
N = 2..5), and it is valid at EVERY `t` -- so each checkpoint is scored against truth at the time
the run ACTUALLY reached, never interpolated and never against our own finest grid. That is what
makes this a confirmation rather than a self-consistency check.

⚠ WALL-CLOCK IS ONLY MEANINGFUL IF NOTHING ELSE IS RUNNING, and it is the one column here that
cannot be repaired after the fact. Run this ALONE, single-threaded. `matvec` is recorded beside it
precisely because it is contention-free: if the two disagree about which arm is cheaper, believe
`matvec`.

⛔ EACH ARM IS WARMED UP BEFORE ITS TIMER STARTS. Julia JITs the first call of every sweep, and
that cost lands entirely on whichever arm runs first -- a fixed several-second penalty that has
nothing to do with the integrator. Timing without a warmup makes arm ordering look like a result.
"""
function chi_growth(; L::Int = 10, J::Float64 = 1.0, eps::Float64 = 1.0, dt::Float64 = 0.02,
                    t_max::Float64 = 20.0, nsave::Int = 41, arms = DEFAULT_ARMS,
                    D::Int = CAP, maxiter::Int = 8, out::String = "", append::Bool = false)
    set_symmetry!(:none)
    out = isempty(out) ? joinpath(@__DIR__, "..", "results",
                                  @sprintf("xx_chigrowth_L%d_dt%g.csv", L, dt)) : out
    mkpath(dirname(out))
    W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
    Imps = identity_superket(L)
    gap = xx_boundary_liouvillian_gap(L; J = J, eps_L = eps, eps_R = eps)
    jref = xx_boundary_current_published(; J = J, eps = eps, mu_L = 1.0, mu_R = -1.0)

    @printf("\n=== XX OPEN: chi growth, accuracy and wall-clock ===\n")
    @printf("  L = %d  dt = %g  t in [0, %g]  cap = %d  growth = %g  maxiter = %d  (mu_L=+1, mu_R=-1)\n",
            L, dt, t_max, D, GROWTH, maxiter)
    @printf("  Liouvillian gap = %.4e  ->  steady state at t ~ %.1f %s\n", gap, 6 / gap,
            6 / gap <= t_max ? "(REACHED)" : "(NOT reached -- transient)")
    @printf("  closed-form NESS current j = %.9f\n", jref)
    @printf("  reference: xx_boundary_sz, analytic at every t\n\n")
    flush(stdout)

    saveat = [t_max * k / (nsave - 1) for k in 0:(nsave - 1)]
    # ⚠ APPEND EXISTS SO ONE ARM CAN BE RUN PER INVOCATION. Each arm is independent -- it starts
    # from its own maximally-mixed state and its own timer -- so splitting them across processes
    # changes nothing except that each pays the load cost again. That is the price of getting a
    # result at all in an environment where a long detached run does not survive.
    # ⛔ IT ALSO MEANS THE `seconds` COLUMN SPANS PROCESSES. Only compare arms that ran under the
    # same conditions, and prefer `matvec`, which is invariant to all of this.
    open(out, append ? "a" : "w") do io
        append || println(io, "arm,L,dt,t,chi,matvec,seconds,trace_pre,err_sz,j_bulk,j_exact")
        for arm in arms
            # ⛔ `maxiter = 8`, NOT THE 30 THE VALIDATION GATES USE, AND IT IS A MEASURED CHOICE.
            # `lanczos_expv` exits on BREAKDOWN only, so every solve burns its full `maxiter`
            # whether or not the exponential has converged -- the depth is a flat cost, not an
            # adaptive one. MEASURED on this stack: `maxiter = 8` ran 3.64x faster at 2e-13, which
            # is four orders below the truncation floor this run sits on. ⚠ It is applied
            # IDENTICALLY to every arm, so it cannot flatter one of them; a per-arm depth would
            # make the wall-clock column meaningless.
            step! = open_steppers(W, D, maxiter; arms = (arm,))[1].second
            # ── warmup on a throwaway copy, OUTSIDE the timer ──────────────────────────────────
            let w = copy(Imps)
                apply_shift!(w, 1.0 / 2.0^(L / 2))
                for _ in 1:2
                    step!(w, dt); apply_shift!(w, exp(dt * shift))
                    restore_trace!(Imps, w, L; maxdim = D)
                end
            end
            rho = copy(Imps); apply_shift!(rho, 1.0 / 2.0^(L / 2))
            let t0 = real(trace_rho(Imps, rho, L))
                abs(t0 - 1) < 1e-10 || error("initial Tr rho = $t0, expected 1 (arm $arm)")
            end
            enable_krylov_log()
            tnow = 0.0; sec = 0.0; trb = 1.0; si = 1
            @printf("  -- %s --\n", arm); flush(stdout)
            while si <= nsave
                if tnow >= saveat[si] - 1e-9
                    trr = real(trace_rho(Imps, rho, L))
                    prof = [real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L]
                    exact = xx_boundary_sz(L, tnow; J = J, eps_L = eps, eps_R = eps,
                                           mu_L = 1.0, mu_R = -1.0)
                    err = maximum(abs.(prof .- exact))
                    jm = mps_current(Imps, rho, L; J = J)
                    jb = jm[max(1, L ÷ 2)]
                    jx = xx_boundary_current(xx_boundary_covariance(L, tnow; J = J, eps_L = eps,
                                                                    eps_R = eps, mu_L = 1.0,
                                                                    mu_R = -1.0);
                                             J = J)[max(1, L ÷ 2)]
                    chi = maximum(bond_dims(rho)); kry = sum(get_krylov_log())
                    @printf(io, "%s,%d,%g,%.6f,%d,%d,%.4f,%.10g,%.6e,%.10g,%.10g\n",
                            arm, L, dt, tnow, chi, kry, sec, trb, err, jb, jx)
                    flush(io)
                    (si <= 2 || si % 8 == 0 || si == nsave) &&
                        (@printf("     t=%6.2f  chi=%-4d mv=%-9d %8.2fs  err=%.3e  j=%+.6f (exact %+.6f)\n",
                                 tnow, chi, kry, sec, err, jb, jx); flush(stdout))
                    si += 1
                    continue
                end
                sec += @elapsed begin
                    step!(rho, dt)
                    apply_shift!(rho, exp(dt * shift))
                    trb = restore_trace!(Imps, rho, L; maxdim = D)
                end
                tnow += dt
            end
            disable_krylov_log()
        end
    end
    @printf("\nwrote %s\n", out)
    return out
end

"""
    dt_robustness(; L = 8, D = 32, t_max = 1.0, dts = (...), eps = 1.0) -> String

⛔ THE TEST FOR THE CLAIM "BUG IS MORE ROBUST THAN 2-SITE TDVP ON AN OPEN SYSTEM", and it is a
LARGE-`dt` test, not a long-time one.

**Why this is the right axis.** TDVP2 removes the double-counting of its two-site sweep with a
BACKWARD substep: having evolved a two-site block forward by `dt`, it evolves the one-site centre
by `-dt`. For a HERMITIAN generator that is harmless -- backward evolution is a phase. For a
LINDBLADIAN it is not: every eigenvalue has `Re lambda <= 0`, so the backward step carries
`exp(+|Re lambda| dt/2)` and AMPLIFIES exactly the modes the dynamics damps. `cbe_bug` has no
backward substep at all -- its Galerkin step is forward-only -- so it has no such term.

⚠ **THE AMPLIFICATION IS `exp(+|Re lambda| dt / 2)`, SO IT IS INVISIBLE UNTIL `|Re lambda| dt` IS
O(1).** MEASURED at the campaign's defaults (`eps = 1`, `L = 8`, `dt = 0.02`) that product is
<= ~0.16, and tdvp2 is not merely stable there but MORE ACCURATE than BUG (err 5.0e-08 against
4.9e-05, an order-3-vs-order-2 gap). A long-time run at small `dt` does NOT probe this: at `D = 32`
and `t = 1.5` tdvp2 still sat at 7.3e-08 with ZERO trace violation. Scanning `t` was the wrong axis;
scanning `dt` (and `eps`) is the right one.

⚠ **THE WITNESS IS THE TRACE, NOT `<S^z_j>`.** `exp(L t)` is trace-preserving exactly, so any
`Tr rho != 1` BEFORE `restore_trace!` is pure integrator error -- and an amplifying backward step
violates it first. `err_sz` is scored after the trace is restored and can look healthy while the
raw state is drifting. ⛔ Both are reported; a claim resting on `err_sz` alone would miss it.

⚠ Report `restore_trace!` THROWING as a RESULT, not an error: it refuses a deficit above 0.5, and
an arm that trips that guard has genuinely lost the state. That is the "TDVP2 dies here" datum.
"""
function dt_robustness(; L::Int = 8, J::Float64 = 1.0, eps::Float64 = 1.0, D::Int = 32,
                       t_max::Float64 = 1.0, maxiter::Int = 8,
                       dts = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
                       arms = ("tdvp2", "bug_rsvd"), out::String = "")
    set_symmetry!(:none)
    out = isempty(out) ? joinpath(@__DIR__, "..", "results",
                                  @sprintf("xx_dtrobust_L%d_D%d_eps%g.csv", L, D, eps)) : out
    mkpath(dirname(out))
    W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
    Imps = identity_superket(L)
    @printf("\n=== XX OPEN: LARGE-dt ROBUSTNESS (the TDVP backward-step test) ===\n")
    @printf("  L = %d  eps = %g  cap = %d  t_max = %g  maxiter = %d\n", L, eps, D, t_max, maxiter)
    @printf("  tdvp2 undoes double-counting with a BACKWARD substep; on a Lindbladian that is\n")
    @printf("  exp(+|Re lambda| dt/2) and AMPLIFIES. cbe_bug is forward-only.\n\n")
    @printf("  %-10s %-7s %-6s %-13s %-13s %-13s %-6s %-9s %s\n",
            "arm", "dt", "steps", "err_sz", "|1-Tr| raw", "Tr final", "chi", "matvec", "status")
    flush(stdout)
    open(out, "w") do io
        # ⚠ `trace_final` IS RECORDED EVEN WHEN THE ARM DIES, and it is the quantitative form of
        # the "TRACE LOST" label: `restore_trace!` only reports that the deficit exceeded 0.5, but
        # the actual `Tr rho` says HOW badly -- an anti-dissipative blow-up should show it large or
        # non-finite, not merely off by 0.6.
        println(io,
                "arm,L,eps,D,dt,nsteps,t_act,err_sz,trace_dev,trace_final,chi,matvec,seconds,status")
        for arm in arms
            step! = open_steppers(W, D, maxiter; arms = (arm,))[1].second
            for dt in dts
                rho = copy(Imps); apply_shift!(rho, 1.0 / 2.0^(L / 2))
                enable_krylov_log()
                n = max(1, round(Int, t_max / dt))
                tnow = 0.0; trb = 1.0; status = "ok"; sec = 0.0
                try
                    sec = @elapsed for _ in 1:n
                        step!(rho, dt)
                        apply_shift!(rho, exp(dt * shift))
                        trb = restore_trace!(Imps, rho, L; maxdim = D)
                        tnow += dt
                    end
                catch err
                    # ⛔ A THROW IS A DATUM. `restore_trace!` refuses a deficit above 0.5, which
                    # means the state is gone -- that is precisely the failure being looked for.
                    status = occursin("max_deficit", sprint(showerror, err)) ? "TRACE LOST" :
                             first(sprint(showerror, err), 40)
                end
                trr = real(trace_rho(Imps, rho, L))
                errsz = NaN
                if status == "ok" && isfinite(trr) && abs(trr) > 1e-300
                    prof = [real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L]
                    ex = xx_boundary_sz(L, tnow; J = J, eps_L = eps, eps_R = eps,
                                        mu_L = 1.0, mu_R = -1.0)
                    errsz = maximum(abs.(prof .- ex))
                    all(isfinite, prof) || (status = "NONFINITE")
                end
                chi = maximum(bond_dims(rho)); kry = sum(get_krylov_log())
                disable_krylov_log()
                @printf(io, "%s,%d,%g,%d,%g,%d,%.6f,%.6e,%.6e,%.6e,%d,%d,%.3f,%s\n",
                        arm, L, eps, D, dt, n, tnow, errsz, abs(1 - trb), trr, chi, kry, sec,
                        status)
                @printf("  %-10s %-7g %-6d %-13.3e %-13.3e %-13.3e %-6d %-9d %s\n",
                        arm, dt, n, errsz, abs(1 - trb), trr, chi, kry, status)
                flush(stdout); flush(io)
            end
        end
    end
    @printf("\nwrote %s\n", out)
    return out
end

"""
    order(; L = 3, t = 1.0, dts = (0.08, 0.04, 0.02, 0.01)) -> Bool

Halve `dt` and watch the deviation from the EXACT profile fall. Reports the observed order
`log2(e(dt) / e(dt/2))` per arm.

⛔ THIS IS THE CHECK THAT CANNOT BE FAKED BY A WRONG OPERATOR. A single small deviation at one `dt`
is consistent with a generator that is slightly wrong in a way that happens to be small; a clean
`dt^2` slope down to the truncation floor is not, because a wrong generator converges to the WRONG
answer and its error PLATEAUS instead of continuing to fall. It is also the direct test of §3.7's
`growth` claim: with the shipped `growth = 2.0` on a d = 4 chain the BUG arms flatten to order 1
while `Tr rho` still reads 0.99999, so the conservation law does not flag it and only this slope
does.

⚠ THE FLOOR IS REAL AND IS NOT A FAILURE. Once the step error drops below what `maxdim`/
`trunc_thresh`/`restore_trace!` leave behind, the ratio collapses towards 1 -- so the verdict is
taken on the FIRST refinement pair, and every ratio is printed so the floor is visible rather than
averaged away.
"""
function order(; L::Int = 8, J::Float64 = 1.0, eps::Float64 = 1.0, t::Float64 = 1.0,
               dts = (0.08, 0.04, 0.02, 0.01), arms = DEFAULT_ARMS)
    set_symmetry!(:none)
    W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
    Imps = identity_superket(L)
    @printf("\norder  L=%d J=%g eps=%g  t=%g  (exact analytic reference)\n", L, J, eps, t)
    # ⛔ `t_act` IS PRINTED BECAUSE A ROW THAT DID NOT REACH `t` IS NOT COMPARABLE WITH ONE THAT DID.
    # Every `dt` that does not divide `t` stops SHORT, and scoring all of them against `exact(t)`
    # made two different step sizes agree to four digits (both landed on 0.24) and report `order
    # 0.00` -- an integrator that looked frozen when it was fine. Each row is now scored at its own
    # `t_act`, and the column makes a short row visible instead of silently wrong.
    @printf("%-10s %-8s %-8s %-12s %-8s\n", "arm", "dt", "t_act", "max|dSz|", "order")
    flush(stdout)
    ok = true
    ref_at(tt) = xx_boundary_sz(L, tt; J = J, eps_L = eps, eps_R = eps, mu_L = 1.0, mu_R = -1.0)
    for arm in arms
        prev = NaN; first_ratio = NaN
        for dt in dts
            res = evolve_arm(arm, W, shift, Imps, L, (t,), Float64(dt))
            t_act = res[1][1]
            dev = maximum(abs.(res[1][2] .- ref_at(t_act)))
            p = isnan(prev) ? NaN : log2(prev / dev)
            isnan(first_ratio) && !isnan(p) && (first_ratio = p)
            @printf("%-10s %-8.4f %-8.4f %-12.3e %-8s\n", arm, dt, t_act, dev,
                    isnan(p) ? "--" : @sprintf("%.2f", p))
            flush(stdout)
            prev = dev
        end
        # ⚠ 1.6 not 2.0: the second-order constant is not asymptotic yet at dt = 0.08, and the
        # trace restoration adds an O(truncation) term. Below 1.6 the arm is FIRST order.
        ok &= !isnan(first_ratio) && first_ratio > 1.6
    end
    @printf("\n%s\n", ok ? "ORDER PASSED -- every arm is second order in dt" :
                           "ORDER FAILED -- an arm is first order (check growth on the BUG arms)")
    return ok
end

"""
    exact_current_fn(L; J, eps, mu_L, mu_R) -> (t -> j_bulk(t))

The exact bulk current as a FUNCTION OF `t`, with the Lyapunov solve hoisted out.

⚠ ONE `lyap` AND ONE `C0 - C_ss`, NOT ONE PER EVALUATION. `trajectory_error` calls this 401 times
per arm; rebuilding the steady state each time is the same waste that once made a trajectory
reference take 36,000 matrix exponentials. What remains per call is a single `L x L` `exp`.
"""
function exact_current_fn(L::Int; J::Float64 = 1.0, eps::Float64 = 1.0,
                          mu_L::Float64 = 1.0, mu_R::Float64 = -1.0)
    X, G = xx_boundary_lyapunov(L; J = J, eps_L = eps, eps_R = eps, mu_L = mu_L, mu_R = mu_R)
    Css = lyap(X, G)
    Dev = Matrix{ComplexF64}(I, L, L) ./ 2 .- Css
    k = max(1, L ÷ 2)
    return function (t)
        E = exp(X .* float(t))
        C = Css .+ E * Dev * E'
        return -2 * J * imag(C[k, k + 1])
    end
end

"""
    grids(io; L = 8, ...) -> nothing

THE COMPLEX-FREQUENCY GRID STUDY: log vs uniform vs panel, on the one model where a logarithmic
grid CANNOT work no matter how many steps it is given.

⛔ WHY THIS MODEL IS THE ONE THAT SETTLES THE GRID QUESTION. The suite's other three models are
either imaginary-time (completely monotone -- geometric spacing is built for exactly that, and the
log grid wins) or a real-time Hermitian quench (real spectrum: oscillation with NO decay). Neither
can produce the combination that breaks a geometric grid structurally. Here every mode is
`lambda = B + 2J cos(theta)` with `theta` COMPLEX -- an oscillation TIMES a decay -- so the
observable keeps oscillating into precisely the late-time region where a geometric grid is
coarsest. The failure does not go away with budget; it gets worse with `t_max`.

⛔ AND THE NYQUIST BOUND IS COMPUTED FROM THE SPECTRUM, NOT ASSUMED. The covariance carries
`exp((mu_a + conj(mu_b)) t)`, so the OBSERVABLE's fastest angular frequency is
`2*max|Im mu| ~ 4J` -- TWICE the fastest single-mode frequency, because a quadratic observable
beats pairs of modes together. Using the single-mode frequency would put the bound at twice the
step it should be and would score half the aliasing arms as resolving.

⚠ THE METRIC IS `trajectory_error`, NOT THE ENDPOINT. `expv` is exact for the local generator, so
one enormous step still lands near the right state AT THAT INSTANT; what a step past Nyquist
destroys is the ability to REPORT THE CURVE. Scoring the endpoint here would rank the aliasing
grids as the winners, which is exactly what it did on the Neel quench before the metric was fixed.
"""
function grids(io; L::Int = 8, D::Int = 128, t_max::Float64 = 20.0, t_0::Float64 = 0.02,
               J::Float64 = 1.0, eps::Float64 = 1.0, maxiter::Int = 40,
               logs = (12, 30, 80, 180), unis = (13, 26, 60), Ps = (1, 2, 4, 8),
               per_panel::Int = 4, arms = DEFAULT_ARMS)
    set_symmetry!(:none)
    W, shift = fused_boundary_lindblad(L; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
    Imps = identity_superket(L)
    psi0 = copy(Imps); apply_shift!(psi0, 1.0 / 2.0^(L / 2))   # rho = I/2^L, see `evolve_arm`
    tr0 = real(trace_rho(Imps, psi0, L))
    abs(tr0 - 1) < 1e-10 || error("initial Tr rho = $tr0, expected 1")

    Xl, _ = xx_boundary_lyapunov(L; J = J, eps_L = eps, eps_R = eps, mu_L = 1.0, mu_R = -1.0)
    fmax = 2 * maximum(abs.(imag.(eigvals(Xl))))       # observable frequency: 2 * fastest mode
    nyq = pi / fmax
    gap = xx_boundary_liouvillian_gap(L; J = J, eps_L = eps, eps_R = eps)
    exact_j = exact_current_fn(L; J = J, eps = eps)
    k = max(1, L ÷ 2)

    println("\n== 4. BOUNDARY-DRIVEN XX CHAIN  L=$L D=$D t_max=$t_max (REAL time, DISSIPATIVE)")
    println("   reference: EXACT Lyapunov (arXiv:2104.11479 / third quantization), no dense object")
    @printf("   closed-form NESS current = %.9f   [independent of N]\n",
            xx_boundary_current_published(; J = J, eps = eps, mu_L = 1.0, mu_R = -1.0))
    @printf("   observable freq 2*max|Im mu| = %.5f -> Nyquist step %.5f;  gap = %.4e\n",
            fmax, nyq, gap)
    for (g, ns) in (("log", logs), ("uniform", unis))
        for n in ns
            dts = g == "log" ? log_steps(t_0, t_max, n) : uniform_steps(t_max, n)
            b = maximum(dts)
            @printf("     %-8s n=%-4d largest step %8.4f   %s\n", g, n, b,
                    b <= nyq ? "resolves" : @sprintf("ALIASES (%.1fx over)", b / nyq))
        end
    end
    for P in Ps
        d = panel_steps(t_max, P; per_panel = per_panel)
        @printf("     %-8s P=%-3d steps=%-4d largest %8.4f   %s\n", "panel", P, length(d),
                maximum(d), maximum(d) <= nyq ? "resolves" : "ALIASES")
    end
    flush(stdout)

    obs(p) = mps_current(Imps, p, L; J = J)[k]
    # ⛔ THE ZERO-BODY SHIFT AND THE TRACE RESTORATION ARE PART OF THE STEP, not post-processing:
    # omitting the shift inflates `Tr rho` (MEASURED 1.1618 at gamma = 0.5) and every current read
    # off it is then divided by the wrong number.
    post!(p, dt) = (apply_shift!(p, exp(dt * shift));
                    restore_trace!(Imps, p, L; maxdim = D); nothing)

    for (nm, st) in open_steppers(W, D, maxiter; arms = arms)
        for (g, dtss) in (("log", [(n, log_steps(t_0, t_max, n)) for n in logs]),
                          ("uniform", [(n, uniform_steps(t_max, n)) for n in unis]),
                          ("panel", [(P * per_panel, panel_steps(t_max, P; per_panel = per_panel))
                                     for P in Ps]))
            for (n, dts) in dtss
                tr = Tuple{Float64, Float64}[]
                o, kv, c, s, mn, fin = run_arm(st, psi0, dts, obs;
                                               imag_time = false, trace = tr, post! = post!)
                emit(io, "xx_dissip", "none", L, g, n, nm, o, exact_j(t_max), kv, c, s, mn, fin;
                     err = trajectory_error(tr, exact_j),
                     tag = maximum(dts) <= nyq ? "" : "  [aliases]")
            end
        end
    end
end

"""
    spectrum_report(; N = 30)

What the exact reference says about grids, BEFORE any MPS run: the oscillation frequency, the
Liouvillian gap against the paper's closed form, and how many steps each grid needs.
"""
function spectrum_report(; N::Int = 30, J::Float64 = 1.0, eps::Float64 = 1.0)
    lam = xx_boundary_rapidities(N; J = J, eps_L = eps, eps_R = eps)
    fmax = maximum(abs.(real.(lam)))          # oscillation
    gap  = xx_boundary_liouvillian_gap(N; J = J, eps_L = eps, eps_R = eps)
    pub  = xx_boundary_gap_published(N; J = J, eps_L = eps, eps_R = eps)
    nyq  = pi / (2 * fmax)
    tmax = 3 / (gap / 2)
    @printf("\nEXACT SPECTRUM  N=%d J=%g eps=%g\n", N, J, eps)
    @printf("  oscillation max|Re l| = %.5f   -> Nyquist step %.5f\n", fmax, nyq)
    @printf("  gap 2*min|Im l|       = %.6e   published %.6e   ratio %.4f\n", gap, pub, gap / pub)
    @printf("  t_max (3 e-foldings)  = %.0f   -> UNIFORM steps needed %d\n",
            tmax, ceil(Int, tmax / nyq))
    for n in (12, 50, 200, 2000)
        r = (tmax / 0.02)^(1 / (n - 1)); ts = [0.02 * r^k for k in 0:(n - 1)]
        big = maximum(vcat(ts[1], diff(ts)))
        @printf("  log n=%-5d largest step %10.3f   %s\n", n, big,
                big <= nyq ? "resolves" : @sprintf("ALIASES (%.0fx over)", big / nyq))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ⚠ ONE PROCESS FOR EVERY CHECK. Loading the package costs ~170 s warm, which dwarfs the
    # arithmetic here, so splitting these into separate launches would multiply the wall time for
    # no reason.
    #
    # ⚠ CHEAPEST FIRST, AND THAT IS AN ORDERING CHOICE NOT AN ACCIDENT. `calibrate` is the answer
    # to "is the MPDO right", `order` confirms it is not right by luck, and `ness` is the most
    # expensive by an order of magnitude (2000 steps per arm per L). Running them in that order
    # means a failure in the cheap check surfaces in minutes instead of after the long one.
    mode = isempty(ARGS) ? "all" : ARGS[1]
    ok = true
    mode in ("all", "spectrum") && spectrum_report()
    mode in ("all", "calibrate") && (ok &= calibrate())
    mode in ("all", "order") && (ok &= order())
    mode in ("all", "ness") && (ok &= ness())
    mode == "chigrowth" && chi_growth(;
        L       = parse(Int,     get(ENV, "XX_L",     "10")),
        dt      = parse(Float64, get(ENV, "XX_DT",    "0.02")),
        t_max   = parse(Float64, get(ENV, "XX_TMAX",  "20.0")),
        nsave   = parse(Int,     get(ENV, "XX_NSAVE", "41")),
        D       = parse(Int,     get(ENV, "XX_D",     string(CAP))),
        maxiter = parse(Int,     get(ENV, "XX_MAXITER", "8")),
        arms    = Tuple(split(get(ENV, "XX_ARMS", join(DEFAULT_ARMS, ",")), ",")),
        append  = get(ENV, "XX_APPEND", "0") == "1")
    # ⛔ `XX_EPS` IS THE AXIS THIS MODE EXISTS FOR, AND ITS DEFAULT OF 1.0 FINDS NOTHING. The
    # backward-step amplification is `exp(+|Re lambda| dt/2)` with `|Re lambda| ~ eps`, so at the
    # campaign's `eps = 1` no reachable `dt` makes the exponent O(1) and every arm comes back
    # stable. The result in `PLAN-open-systems.md` §7.1d was measured by SWEEPING `eps` over
    # 16/24/32/64 — one process per value. Running this mode at its defaults and reporting the
    # null would repeat a mistake this campaign already made once.
    mode == "dtrobust" && dt_robustness(;
        L       = parse(Int,     get(ENV, "XX_L",       "8")),
        eps     = parse(Float64, get(ENV, "XX_EPS",     "1.0")),
        D       = parse(Int,     get(ENV, "XX_D",       "32")),
        t_max   = parse(Float64, get(ENV, "XX_TMAX",    "0.8")),
        maxiter = parse(Int,     get(ENV, "XX_MAXITER", "8")),
        dts     = Tuple(parse.(Float64,
                               split(get(ENV, "XX_DTS", "0.02,0.05,0.1,0.2,0.4,0.8"), ","))),
        arms    = Tuple(split(get(ENV, "XX_ARMS", "tdvp2,bug_rsvd"), ",")))
    # ⛔ THE GRID STUDY IS GATED ON THE VALIDATION PASSING, and that is not politeness. Every row it
    # writes is an error against the analytic reference; if `calibrate` says the generator is wrong
    # then those rows are a table of one wrong number against another, and publishing them is worse
    # than publishing nothing. `grids` alone is still available explicitly for iteration.
    if mode == "grids" || (mode in ("all", "gridstudy") && ok)
        out = get(ENV, "DISSIP_OUT", joinpath(@__DIR__, "..", "results", "logtime_xx_dissip.csv"))
        mkpath(dirname(out))
        open(out, "w") do io
            println(io, join(COLS, ","))
            grids(io)
        end
        println("\nwrote ", out)
    elseif mode in ("all", "gridstudy")
        println("\nGRID STUDY SKIPPED -- validation failed, so its rows would be meaningless.")
    end
    println(ok ? "\nALL CHECKS PASSED" : "\nSOME CHECK FAILED")
    flush(stdout)
end
