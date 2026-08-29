# WHY DOES THE BOUNDARY-DRIVEN SUPERKET NOT MOVE?  A localisation ladder, L = 3, no benchmark.
#
# ⛔ THE OBSERVATION. `dissipative_xx.jl calibrate` at L = 8 returned, for tdvp2, bug_rsvd AND
# bug_par, an `<S^z_j>` profile of EXACTLY ZERO at every site and every time, `chi = 1`, and
# `Tr rho = 1.0000000000`. `order` then gave the SAME deviation 2.451e-01 at dt = 0.08, 0.04, 0.02
# and 0.01 -- observed order 0.00. A quantity independent of BOTH the integrator AND the step size
# is not an integrator error; the state is not moving at all.
#
# ⛔ AND THE GENERATOR IS RIGHT ON PAPER, WHICH IS WHY THIS FILE EXISTS RATHER THAN A REWRITE.
# Working the first-order action out by hand on the fused site vector `u = (|uu> + |dd>)/sqrt2`:
#
#     JP u = e1/sqrt2         (weight 1 at site 1, since mu_L = +1)
#     Kz u = Bz u = (0.5, 0, 0, -0.5)/sqrt2      (weight 0.5 each)
#     => G_1 u = (1.5, 0, 0, -0.5)/sqrt2
#     => d<S^z_1>/dt = <u|Kz|G_1 u> = [0.5*1.5 + (-0.5)(-0.5)]/2 = 0.5
#
# which is EXACTLY the analytic `(g_1 - l_1)/2 = (1 - 0)/2 = 0.5` from the Lyapunov ODE at
# `C = I/2`. The trace also checks: `<u|JP|u> = <u|JM|u> = 1/2` at the two ends and
# `<u|Kz|u> = <u|Bz|u> = 0`, so `d(Tr)/dt = 1/2 + 1/2 + shift = 0` exactly when
# `shift = -(eps_L + eps_R)/2 = -1`, which is what the builder returns.
#
# So a CORRECT integrator must produce `<S^z_1> ~ 0.5*dt` after ONE step, and must do it AT
# `chi = 1` -- the first-order change is a ONE-SITE operator, so the state stays a product and a
# frozen `chi` is NOT itself the bug. The question is only where the response is lost.
#
# ⚠ EACH GATE ISOLATES ONE SUSPECT AND THEY ARE ORDERED CHEAPEST-FIRST:
#   0  the local space          is the physical leg really 4-dimensional, or did `getsub` pin it?
#   1  the observable           does `trace_rho` see a state we MOVED BY HAND?
#   2  one step, each arm       does the integrator move it, against the known 0.5*dt?
#   3  J = 0                    boundary relaxation alone -- removes the Hamiltonian as a suspect
#   4  a non-|I>> start         is the freeze specific to the identity superket?
#
# Run:  julia -t 2 --project=. benchmarks/fused_freeze_probe.jl

using LinearAlgebra, Printf
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const L = 3
const DT = 0.01

set_symmetry!(:none)

say(s) = (println(s); flush(stdout))

sz_profile(Imps, rho) = begin
    trr = real(trace_rho(Imps, rho, L))
    (trr, [real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L])
end
fmt(v) = join([@sprintf("%+.8f", x) for x in v], " ")

W, shift = fused_boundary_lindblad(L; J = 1.0, eps_L = 1.0, eps_R = 1.0, mu_L = 1.0, mu_R = -1.0)
Imps = identity_superket(L)

say("\n=== GATE 0: the local space and the initial state ===")
say("  shift (zero-body)      = $shift        [must be -1.0 for eps_L=eps_R=1]")
for j in 1:L
    t = Imps[j]
    say(@sprintf("  Imps[%d] leg dims = %s   (leg 2 is the PHYSICAL leg; must be 4)",
                 j, string(size(t))))
end
say("  bond_dims(Imps)        = $(bond_dims(Imps))")
# ⛔ THIS GATE IS WHERE THE FREEZE WAS ROOT-CAUSED, AND IT IS THE CHEAPEST LINE IN THE FILE.
# `Tr(identity_superket(L))` is `2^(L/2)`, NOT 1: the maximally mixed state is
# `vec(I/2^L) = |I>>_normalised / 2^(L/2)`. Feeding the unscaled superket in as `rho` made
# `restore_trace!` see a deficit of `1 - 2^(L/2)` and project the state back onto `|I>>` every
# step -- which presents as a dead integrator, not as a bad initial condition.
tr_raw, _ = sz_profile(Imps, Imps)
say(@sprintf("  Tr(identity_superket) = %.12f   [= 2^(L/2) = %.12f, NOT 1 -- must be scaled]",
             tr_raw, 2.0^(L / 2)))
rho_mm() = (q = copy(Imps); apply_shift!(q, 1.0 / 2.0^(L / 2)); q)
tr0, p0 = sz_profile(Imps, rho_mm())
say(@sprintf("  Tr rho0 (scaled)      = %.12f   <S^z_j> = %s   [expect Tr=1, profile all 0]",
             tr0, fmt(p0)))

say("\n=== GATE 1: the OBSERVABLE, on a state moved BY HAND (no integrator) ===")
say("  Apply JP / JM to site 1 of rho0 and read the profile back. These have EXACTLY known")
say("  answers, so a wrong reading here means the fault is `trace_rho`, not the sweep:")
say("    JP u = |uu>/sqrt2  -> site 1 is |up><up|  -> <S^z_1> = +0.5, other sites 0")
say("    JM u = |dd>/sqrt2  -> site 1 is |dn><dn|  -> <S^z_1> = -0.5, other sites 0")
# ⚠ USE AN OPERATOR THAT ALREADY LIVES IN `Q4`. `site_apply!` retags its argument with
# `setitag`, so it needs a TELUM TENSOR, not an `AbstractMatrix` -- a raw 4x4 throws
# `MethodError: no method matching setitag(::Matrix{Float64}, ::Int64, ::Itag)`, and
# `Matrix(Q4.Kz)` throws `CanonicalIndexError` going the other way. Building a bespoke tilt would
# mean editing `fused_space()`, which every open driver shares, to run one diagnostic.
for (nm, op, want) in (("JP", Q4.JP, +0.5), ("JM", Q4.JM, -0.5))
    try
        q = rho_mm()
        site_apply!(q, 1, op)
        trq, pq = sz_profile(Imps, q)
        say(@sprintf("  %-3s  Tr=%.10f  <S^z_j> = %s   [want <S^z_1> = %+.1f]",
                     nm, trq, fmt(pq), want))
    catch e
        say("  $nm THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 1.5: the FIRST-ORDER RESPONSE READ OFF THE MPO, with NO INTEGRATOR ===")
say("  This is the gate that separates 'the MPO is wrong' from 'the sweep does not apply it'.")
say("  ** W EXCLUDES THE ZERO-BODY SHIFT, so <<I|W is NOT zero -- <<I|(W + shift) is. **")
say("  That mattered: an earlier version of this gate used W alone, so BOTH cross terms in")
say("  <p|W|p> survived and it measured their SUM rather than the response -- which is why it")
say("  returned the same number for both jump conventions and looked insensitive to the operator.")
say("  The shift is a pure scalar, so <p|(W+shift)|p> = <p|W|p> + shift*<p|p> corrects it exactly.")
say("  With that, setting p = rho0 + eta*Kz_1*rho0 kills")
say("  three of the four terms in <p|W|p>:")
say("     <rho0|W|rho0> = 0,  <rho0|W|q> = 0,  leaving  eta*<q|W|rho0> + O(eta^2)")
say("  and <q|W|rho0> = <<I|Kz_1 W|I>> / 2^L = d<S^z_1>/dt / 2^L.  So 2^L*<p|W|p>/eta -> 0.5.")
for eta in (1e-3, 1e-4)
    try
        q = rho_mm()
        site_apply!(q, 1, Q4.Kz)
        apply_shift!(q, eta)
        p = add_mps(rho_mm(), q)
        r = 2.0^L * (real(mpo_energy(copy(p), W)) + shift * norm(p)^2) / eta
        say(@sprintf("  eta=%-8.0e  2^L <p|W|p>/eta = %+.8f   [exact d<S^z_1>/dt = +0.50000000]",
                     eta, r))
    catch e
        say("  eta=$eta THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 2: ONE STEP of each arm, against the exact first-order response ===")
say(@sprintf("  dt = %g  =>  EXACT <S^z_1> = %+.8f, <S^z_%d> = %+.8f, bulk 0",
             DT, 0.5 * DT, L, -0.5 * DT))
exact1 = xx_boundary_sz(L, DT; J = 1.0, eps_L = 1.0, eps_R = 1.0, mu_L = 1.0, mu_R = -1.0)
say("  exact profile at t=dt (Lyapunov)      = $(fmt(exact1))")
arms = Dict(
    "tdvp2"    => (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = 128,
                                         trunc_thresh = 1e-12, maxiter = 30, hermitian = false),
    "bug_rsvd" => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = false, dover = 4,
                                           growth = 4.0, maxdim = 128, trunc_thresh = 1e-12,
                                           maxiter = 30, hermitian = false),
    "bug_exact" => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true, growth = 4.0,
                                            maxdim = 128, trunc_thresh = 1e-12,
                                            maxiter = 30, hermitian = false))
for nm in ("tdvp2", "bug_rsvd", "bug_exact")
    q = rho_mm()
    # ⚠ "did the tensors change at all" is measured by an OVERLAP, not by reaching into the
    # tensor's storage: `Array(::Telum tensor)` is not part of the API and a failure there would
    # abort the gate that is supposed to be diagnosing something else.
    # `1 - |<I|q>|^2/(|I|^2 |q|^2)` is 0 iff q is still parallel to |I>>.
    par(a, b) = 1.0 - abs2(overlap(a, b)) / max(abs2(norm(a)) * abs2(norm(b)), 1e-300)
    try
        arms[nm](q, DT)
        say(@sprintf("  %-10s 1-|<I|q>|^2 = %.3e  |q|=%.8f  bond_dims=%s", nm,
                     par(Imps, q), norm(q), string(bond_dims(q))))
        apply_shift!(q, exp(DT * shift))
        trq, pq = sz_profile(Imps, q)
        say(@sprintf("  %-10s after shift: Tr=%.10f  <S^z_j> = %s", nm, trq, fmt(pq)))
    catch e
        say("  $nm THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 3: J = 0 -- boundary relaxation ALONE, no Hamiltonian ===")
say("  Removes the XY term as a suspect. The exact answer is still known (same Lyapunov ODE).")
# ⛔ SMALL J, NOT J = 0 -- AND THE FAILURE WAS IN THE REFERENCE, NOT THE MPS. At exactly `J = 0`
# the Lyapunov generator is `X = -D/2 = diag(-eps/2, 0, ..., 0, -eps/2)`: the BULK sites are
# undamped, so `X` has a zero eigenvalue, the pair `(0,0)` makes the Sylvester equation singular
# and `lyap` throws `LAPACKException(1)`. Physically that is right -- with no hopping the bulk
# never equilibrates and there IS no steady state. A small `J` keeps the solve well-posed while
# leaving the Hamiltonian negligible against the boundary drive, which is all this gate needs.
const J0 = 1e-3
W0, shift0 = fused_boundary_lindblad(L; J = J0, eps_L = 1.0, eps_R = 1.0,
                                     mu_L = 1.0, mu_R = -1.0)
e0 = xx_boundary_sz(L, DT; J = J0, eps_L = 1.0, eps_R = 1.0, mu_L = 1.0, mu_R = -1.0)
say("  exact profile at t=dt = $(fmt(e0))")
for nm in ("tdvp2", "bug_rsvd")
    q = rho_mm()
    st = nm == "tdvp2" ?
        (p, dt) -> tdvp2_step!(p, W0, ComplexF64(dt); maxdim = 128, trunc_thresh = 1e-12,
                               maxiter = 30, hermitian = false) :
        (p, dt) -> cbe_bug_step!(p, W0, ComplexF64(dt); exact = false, dover = 4, growth = 4.0,
                                 maxdim = 128, trunc_thresh = 1e-12, maxiter = 30,
                                 hermitian = false)
    try
        st(q, DT); apply_shift!(q, exp(DT * shift0))
        trq, pq = sz_profile(Imps, q)
        say(@sprintf("  %-10s Tr=%.10f  <S^z_j> = %s", nm, trq, fmt(pq)))
    catch e
        say("  $nm THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 4: a NON-|I>> start -- is the freeze specific to the identity superket? ===")
say("  rho0 = |up down up><up down up| (still a PRODUCT, but not the maximally mixed state).")
for nm in ("tdvp2", "bug_rsvd")
    q = rho0_product([:up, :down, :up])
    st = nm == "tdvp2" ?
        (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = 128, trunc_thresh = 1e-12,
                               maxiter = 30, hermitian = false) :
        (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = false, dover = 4, growth = 4.0,
                                 maxdim = 128, trunc_thresh = 1e-12, maxiter = 30,
                                 hermitian = false)
    tr_b, pb = sz_profile(Imps, q)
    try
        for _ in 1:20
            st(q, DT); apply_shift!(q, exp(DT * shift))
        end
        tr_a, pa = sz_profile(Imps, q)
        say(@sprintf("  %-10s before Tr=%.8f %s", nm, tr_b, fmt(pb)))
        say(@sprintf("  %-10s after  Tr=%.8f %s   bond_dims=%s", nm, tr_a, fmt(pa),
                     string(bond_dims(q))))
    catch e
        say("  $nm THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 5: WHICH JUMP CONVENTION DOES `pair_mpo` USE? ===")
say("  Build the SAME Lindbladian both ways and read the first-order response off each. Exactly")
say("  one of them can be right, and the wrong one gives 0 -- not a small number, ZERO -- because")
say("  the jump term then carries LOSS while the anticommutator still carries GAIN's sign, so the")
say("  two ends balance and |I>> becomes a genuine fixed point.")
say(@sprintf("  %-18s %-22s %s", "jump_transpose", "2^L <p|W|p>/eta", "verdict [want +0.5]"))
for jt in (false, true)
    try
        Wj, _ = fused_boundary_lindblad(L; J = 1.0, eps_L = 1.0, eps_R = 1.0,
                                        mu_L = 1.0, mu_R = -1.0, jump_transpose = jt)
        eta = 1e-4
        q = rho_mm(); site_apply!(q, 1, Q4.Kz); apply_shift!(q, eta)
        p = add_mps(rho_mm(), q)
        Wj_shift = -(1.0 + 1.0) / 2      # -(eps_L + eps_R)/2, same as the builder returns
        r = 2.0^L * (real(mpo_energy(copy(p), Wj)) + Wj_shift * norm(p)^2) / eta
        say(@sprintf("  %-18s %+.10f          %s", string(jt), r,
                     abs(r - 0.5) < 1e-6 ? "*** THIS ONE ***" :
                     abs(r) < 1e-9 ? "frozen (fixed point)" : "neither"))
    catch e
        say("  jump_transpose=$jt THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 6: L = 2 -- ONE two-site block IS the whole Hilbert space ===")
say("  At L=2 the single bond (1,2) spans the ENTIRE 16-dimensional superket space, so a two-site")
say("  update has NOTHING to project onto and NOTHING to truncate: it must reproduce exp(dt*W)")
say("  exactly, to roundoff, for every arm. This is the gate that cannot be argued with --")
say("  if the state still freezes here, the fault is in how the sweep builds its local generator")
say("  from this MPO, not in environments, truncation, rank growth or the initial state.")
let L2 = 2, dt2 = 0.01
    W2, sh2 = fused_boundary_lindblad(L2; J = 1.0, eps_L = 1.0, eps_R = 1.0,
                                      mu_L = 1.0, mu_R = -1.0)
    I2 = identity_superket(L2)
    ex = xx_boundary_sz(L2, dt2; J = 1.0, eps_L = 1.0, eps_R = 1.0, mu_L = 1.0, mu_R = -1.0)
    say("  exact <S^z_j> at t=dt  = " * join([@sprintf("%+.8f", x) for x in ex], " "))
    for (nm, st) in (("tdvp2",
                      (p, dt) -> tdvp2_step!(p, W2, ComplexF64(dt); maxdim = 128,
                                             trunc_thresh = 1e-12, maxiter = 30,
                                             hermitian = false)),
                     ("bug_exact",
                      (p, dt) -> cbe_bug_step!(p, W2, ComplexF64(dt); exact = true, growth = 4.0,
                                               maxdim = 128, trunc_thresh = 1e-12, maxiter = 30,
                                               hermitian = false)))
        try
            q = copy(I2); apply_shift!(q, 1.0 / 2.0^(L2 / 2))
            st(q, dt2); apply_shift!(q, exp(dt2 * sh2))
            t2 = real(trace_rho(I2, q, L2))
            pr = [real(trace_rho(I2, q, L2; at = j)) / t2 for j in 1:L2]
            say(@sprintf("  %-10s Tr=%.10f  <S^z_j> = %s   max|dev| = %.3e", nm, t2,
                         join([@sprintf("%+.8f", x) for x in pr], " "),
                         maximum(abs.(pr .- ex))))
        catch e
            say("  $nm THREW: " * sprint(showerror, e))
        end
    end
end

say("\n=== GATE 7: WHICH one-body TERM does the sweep drop -- diagonal or charge-changing? ===")
say("  GATE 6 proved the sweep misses the boundary terms even when the block IS the whole space,")
say("  while `mpo_energy` sees them (<I|W|I> = 1 can only come from the jumps). So the fault is in")
say("  the LOCAL GENERATOR, and the question is which vertices survive it.")
say("  `fused_lindblad`'s dephasing uses the SAME one-body encoding -- PV(op, I) riding a bond --")
say("  and is validated to order 2.99. Its `op` is JMP = kron(Sz,Sz): DIAGONAL. `JP`/`JM` are")
say("  OFF-DIAGONAL. So build one MPO per operator, each with a SINGLE one-body vertex, and step.")
say("  Both move |I>> in exact arithmetic: Kz u = (0.5,0,0,-0.5)/sqrt2 and JP u = e1/sqrt2, and")
say("  NEITHER is parallel to u = (1,0,0,1)/sqrt2.")
say("")
say("  ⛔ THE SUSPECTED MECHANISM, from `src/RSVDCBEBondUpdate/mpo.jl:154`:")
say("       _side(O, s) = length(O.inds) == 3 ? s : :none")
say("     An operator OPENS or CLOSES a channel only if it has THREE legs (charged, rank-3);")
say("     a two-leg neutral operator gets `:none`. So `PV(JP, I)` opens a CHARGED channel that")
say("     the identity cannot close -- the virtual charge never returns to zero. `PV(Kz, I)` is")
say("     neutral on both sides, rides the trivial identity channel, and works. The leg count is")
say("     printed below so this is MEASURED rather than read off the source.")
let L2 = 2, dt2 = 0.05
    I2 = identity_superket(L2)
    PV = RSVDCBEBondUpdate.PairVertex
    say(@sprintf("  %-24s %-10s %s", "operator", "n_legs", "_side(op, :left)"))
    for (nm, op) in (("I", Q4.I), ("Kz", Q4.Kz), ("JMP", Q4.JMP), ("JP", Q4.JP), ("JM", Q4.JM))
        say(@sprintf("  %-24s %-10d %s", nm, length(op.inds),
                     length(op.inds) == 3 ? ":left  (opens a channel)" : ":none  (rides identity)"))
    end
    say("")
    for (nm, op) in (("Kz  (diagonal)", Q4.Kz), ("JMP (diagonal)", Q4.JMP),
                     ("JP  (charge-changing)", Q4.JP), ("JM  (charge-changing)", Q4.JM))
        try
            A = zeros(ComplexF64, L2, L2); A[1, 2] = 1.0
            Wv = pair_mpo(L2, Vector{AbstractMatrix}(Any[A]),
                          PV[PV(op, Q4.I, 1.0)]; Iloc = Q4.I)
            q = copy(I2); apply_shift!(q, 1.0 / 2.0^(L2 / 2))
            n0 = norm(q)
            tdvp2_step!(q, Wv, ComplexF64(dt2); maxdim = 128, trunc_thresh = 1e-12,
                        maxiter = 30, hermitian = false)
            par = 1.0 - abs2(overlap(I2, q)) / max(abs2(norm(I2)) * abs2(norm(q)), 1e-300)
            say(@sprintf("  %-24s 1-|<I|q>|^2 = %.3e   |q|/|q0| = %.8f   %s",
                         nm, par, norm(q) / n0,
                         par > 1e-10 ? "MOVED" : "*** FROZEN -- term dropped ***"))
        catch e
            say("  $nm THREW: " * sprint(showerror, e))
        end
    end
end

say("\n=== GATE 8: does `add_mps` PRESERVE THE RELATIVE WEIGHT of its two arguments? ===")
say("  GATE 1.5 returned a constant of -15 where trace preservation demands 0, and -15 is what")
say("  you get if |p|^2 = 2 -- i.e. if BOTH summands arrived normalised. `add_mps` calls")
say("  `canonical!` on each argument before `oplus`, so if that normalises, a WEIGHTED sum is")
say("  impossible and `restore_trace!` is not adding a small correction along |I>> -- it is")
say("  replacing the state with an EQUAL mix of itself and the identity superket.")
say("  a = rho0, b = 0.001*rho0  =>  |a + b| must be 1.001*|a|, NOT 2*|a| or sqrt(2)*|a|.")
let
    a = rho_mm()
    b = rho_mm(); apply_shift!(b, 1e-3)
    na, nb = norm(a), norm(b)
    try
        c = add_mps(copy(a), copy(b))
        say(@sprintf("  |a| = %.8f   |b| = %.8f   (|b|/|a| = %.6f, want 0.001)", na, nb, nb / na))
        say(@sprintf("  |a+b| = %.8f   ratio |a+b|/|a| = %.8f   %s", norm(c), norm(c) / na,
                     abs(norm(c) / na - 1.001) < 1e-6 ? "OK -- weights preserved" :
                     "*** WEIGHT DESTROYED -- add_mps normalises its arguments ***"))
    catch e
        say("  add_mps THREW: " * sprint(showerror, e))
    end
end

say("\n=== GATE 9: is it the ZERO-WEIGHT VERTICES? ===")
say("  GATE 7 showed every one-body operator moves |I>> ON ITS OWN, yet the assembled MPO does")
say("  not. The one thing the assembly does that GATE 7 did not is push vertices whose coupling")
say("  matrix is IDENTICALLY ZERO: at mu_L=+1, mu_R=-1 both gp(eps_R,mu_R) and gm(eps_L,mu_L)")
say("  vanish, so 2 of the 12 matrices are all-zero.")
say("  Two independent ways to remove them -- if EITHER moves the state, that is the cause:")
say("    (a) mu = +-0.5, so no weight is exactly zero and all 12 vertices are live;")
say("    (b) mu = +-1 with the zero-weight vertices SKIPPED by the builder (the fix now in place).")
for (tag, mL, mR) in (("(a) mu=+-0.5, all live", 0.5, -0.5),
                      ("(b) mu=+-1,  zeros skipped", 1.0, -1.0))
    try
        L2 = 2; dt2 = 0.05
        Wg, shg = fused_boundary_lindblad(L2; J = 1.0, eps_L = 1.0, eps_R = 1.0,
                                          mu_L = mL, mu_R = mR)
        I2 = identity_superket(L2)
        q = copy(I2); apply_shift!(q, 1.0 / 2.0^(L2 / 2))
        tdvp2_step!(q, Wg, ComplexF64(dt2); maxdim = 128, trunc_thresh = 1e-12,
                    maxiter = 30, hermitian = false)
        apply_shift!(q, exp(dt2 * shg))
        par = 1.0 - abs2(overlap(I2, q)) / max(abs2(norm(I2)) * abs2(norm(q)), 1e-300)
        t2 = real(trace_rho(I2, q, L2))
        pr = [real(trace_rho(I2, q, L2; at = j)) / t2 for j in 1:L2]
        ex = xx_boundary_sz(L2, dt2; J = 1.0, eps_L = 1.0, eps_R = 1.0, mu_L = mL, mu_R = mR)
        say(@sprintf("  %-28s 1-|<I|q>|^2 = %.3e  %s", tag, par,
                     par > 1e-10 ? "MOVED" : "*** STILL FROZEN ***"))
        say(@sprintf("  %-28s <S^z_j> = %s   exact = %s   max|dev| = %.3e", "", fmt(pr),
                     fmt(ex), maximum(abs.(pr .- ex))))
    catch e
        say("  $tag THREW: " * sprint(showerror, e))
    end
end

say("\nREADING THE LADDER:")
say("  gate 1 flat            -> the OBSERVABLE is broken (trace_rho / overlap conjugation)")
say("  gate 1 fine, 2 flat    -> the SWEEP does not apply this generator")
say("  gate 2 flat, 3 moves   -> the Hamiltonian channel is cancelling the dissipative one")
say("  gate 2 flat, 4 moves   -> the freeze is specific to |I>>, i.e. a product-state bootstrap")
