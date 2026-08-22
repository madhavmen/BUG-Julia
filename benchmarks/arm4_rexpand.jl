# ARM 4: expand every bond TWICE per half-sweep (`rexpand`) instead of unioning (`kaug`).
#
# THE QUESTION. `W[i]` is the left basis at bond `i` and it sets site `i+1`'s LEFT bond width.
# The split that produces it is bounded by the evolve, so it comes out narrower than the CBE
# frame just paid for. Two ways to restore that width:
#
#   kaug    = true   W[i] = orth([U_ex | svd(K1).U])      -- merge the frame back in
#   rexpand = true   W[i] = U_ex of a SECOND expansion    -- expand bond i again, no union
#
# `rexpand` is the ordering `tdvp_cbe1s` gets for free, because it expands every bond on BOTH
# its forward and backward passes. Under it every site is evolved with both of its bonds
# widened, and no basis is ever formed by merging two others.
#
# ⛔ OAT IS THE DECISIVE CASE AND THE REASON IS STRUCTURAL, NOT THE ERROR. At L=10, D=6 the cap
# IS the exact Schmidt ceiling, so rank is not the limiter and every arm can reach the right
# answer. OAT's exact rank profile is `min(n, L-n)+1` FOR ALL TIME, so the measured bond profile
# is a pass/fail with no tolerance in it: `[2,3,4,5,6,5,4,3,2]` is right and anything else is a
# scheme filling rank it does not need. The infidelity alone cannot say that.
#
# The references are the repo's own, and neither is a second integrator:
#   OAT -- H is diagonal in the S^z basis, so the exact state is one phase per basis vector.
#   XX  -- free fermions, `exp(-i h t)` on the single-particle Hamiltonian. No 2^L object.
#
# Run:  julia --project=. benchmarks/arm4_rexpand.jl
#       julia --project=. benchmarks/arm4_rexpand.jl model=oat
#
# ⛔ WALL TIME IS NOT RECORDED. Local timing on this machine spreads 2.6-2.8x on bit-identical
# work (see the header of `rsvd_param_scan.jl`), so seconds here would be noise. `krylov` is
# deterministic and is the cost axis.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

_kry(info) = hasproperty(info, :krylov_dims) ? info.krylov_dims : 0

"Evolve `psi0` for `nst` steps of `step!` and return `(final psi, total krylov)`."
function _drive(psi0, nst, step!)
    psi, kry = copy(psi0), 0
    for _ in 1:nst
        kry += _kry(step!(psi))
    end
    return psi, kry
end

# ── the arms, identical in every argument except the one under test ──────────────────────
function _arms(mpo, D, dt, maxiter)
    tau = ComplexF64(-im * dt)
    bug(; kw...) = p -> cbe_bug_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                      maxiter = maxiter, kw...)
    return ["tdvp2"            => p -> tdvp2_step!(p, mpo, tau; maxdim = D,
                                                   trunc_thresh = 1e-14, maxiter = maxiter),
            "tdvp_cbe1s"       => p -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = D,
                                                        trunc_thresh = 1e-14,
                                                        maxiter = maxiter),
            # ⛔ EVERY `kaug` ARM MUST PIN `rexpand = false`. `rexpand` is the default and it
            # overrides `kaug`, so `bug(kaug = true)` alone would run the rexpand arm twice and
            # the table would show two identical rows for two different labels.
            "bug kaug=true"    => bug(kaug = true,  rexpand = false),
            "bug kaug g=2.0"   => bug(kaug = true,  rexpand = false, growth = 2.0),
            "bug rexpand=true" => bug(rexpand = true),
            # `rexpand`'s second expansion runs against a LARGER `U0` (it has already absorbed
            # `K1`), so `budget = ceil(growth*dmax) - r` leaves it starved at the shipped
            # `growth = 1.1`. MEASURED on TFIM L=16 D=32: 3.66e-05 -> 1.51e-05, i.e. parity with
            # `kaug`'s 1.43e-05, while `kaug` itself does NOT want the extra budget (`dex = 8`
            # costs it 12x). This row asks whether that survives on OAT and XX.
            "bug rexpand g=2.0" => bug(rexpand = true, growth = 2.0),
            "bug kaug=false"   => bug(kaug = false, rexpand = false),
            "bug kstep=false"  => bug(kstep = false, rexpand = false)]
end

# ── OAT: infidelity AND the exact rank profile ───────────────────────────────────────────
function run_oat(; L = 10, dt = 0.05, nst = 20, D = 6, maxiter = 30)
    set_symmetry!(:none)
    mpo  = oat_mpo(L)
    psi0 = x_polarized_state(L)

    # H_OAT is diagonal in the S^z basis: the exact state is one phase per basis vector.
    v0    = dense_state(psi0)
    ediag = (diag(dense_total_sz(L)) .^ 2) ./ 2 .- L / 8
    vex   = exp.(-im * (nst * dt) .* ediag) .* v0
    want  = [min(i, L - i) + 1 for i in 1:(L - 1)]

    @printf("\nOAT  L=%d  D=%d (= the exact ceiling, so rank is NOT the limiter)  dt=%g  %d steps\n",
            L, D, dt, nst)
    println("exact rank profile = ", want, "\n")
    @printf("  %-18s %12s  %8s  %s\n", "arm", "infidelity", "krylov", "bond profile")

    for (name, step!) in _arms(mpo, D, dt, maxiter)
        psi, kry = _drive(psi0, nst, step!)
        vn = dense_state(psi)
        f  = 1 - abs2(dot(vex, vn)) / real(dot(vex, vex) * dot(vn, vn))
        bd = bond_dims(psi)
        @printf("  %-18s %12.4e  %8d  %s%s\n", name, f, kry, bd,
                bd == want ? "  <- exact" : "")
    end
end

# ── XX: the rotating-basis case, against free fermions ───────────────────────────────────
function run_xx(; L = 16, dt = 0.05, nst = 120, D = 24, J = 1.0, maxiter = 12)
    set_symmetry!(:U1)
    mpo  = xxz_mpo(L; J = J, delta = 0.0)
    psi0 = domain_wall_state(L)
    ref  = xx_free_fermion_sz(L, nst * dt; J = J)

    @printf("\nXX domain wall  L=%d  D=%d  dt=%g  %d steps (t=%g)\n", L, D, dt, nst, nst * dt)
    @printf("  %-18s %12s  %8s  %s\n", "arm", "max |dSz|", "krylov", "chi")

    for (name, step!) in _arms(mpo, D, dt, maxiter)
        psi, kry = _drive(psi0, nst, step!)
        err = maximum(abs.(magnetisation(copy(psi)) - ref))
        @printf("  %-18s %12.4e  %8d  %d\n", name, err, kry, maximum(bond_dims(psi)))
    end
end

let model = "both"
    for a in ARGS
        startswith(a, "model=") && (model = split(a, '='; limit = 2)[2])
    end
    model in ("oat", "both") && run_oat()
    model in ("xx", "both")  && run_xx()
    println()
end
