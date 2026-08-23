# IS `cbe_bug`'s TFIM DEFICIT A BUG OR A PROPERTY OF THE SCHEME?
#
# THE ANOMALY. Against `tdvp2` at the SAME `chi`, on the shipped examples:
#
#     OAT    1.406e-09  vs  1.769e-07     126x BETTER
#     XX     4.591e-05  vs  5.503e-06     8.3x  worse
#     TFIM   3.709e-06  vs  2.089e-09     1775x worse
#
# XX and TFIM are both nearest-neighbour and both free-fermion-solvable, and both runs end at
# chi = 64, so a 200x difference in where `cbe_bug` STANDS relative to `tdvp2` is not explained
# by rank, by connectivity, or by integrability. That is what this file tests.
#
# TWO CHECKS, EACH DECISIVE ON ITS OWN:
#
#   A  FULL RANK. With `maxdim` at or above the exact Schmidt rank there is no truncation left,
#      so every scheme here must agree with the exact propagator to the TIME-STEP error alone.
#      `tests/sweeps/test_sweeps.jl` already asserts this for `cbe_bug` -- on XXZ. If TFIM fails
#      it at full rank, the defect is in the sweep and NOT in selection or truncation, and the
#      model dependence localises it further.
#
#   B  dt ORDER. Both schemes are second order, so halving `dt` must QUARTER the error. A ratio
#      near 4 means TFIM is simply a harder problem with a larger constant -- a property. A
#      ratio near 2 means something in the composition is first order on this model, and a
#      PLATEAU means a floor that `dt` cannot reach: both are bugs.
#
# ⛔ THE TWO ANSWERS ARE NOT INTERCHANGEABLE. Check A can pass while B fails (a correct sweep
# composed to the wrong order) and B can pass while A fails (a consistently wrong sweep that
# still converges cleanly in dt). Both are run.
#
# Part C is the LONG-RANGE XX the ratio table above motivates: it varies interaction range while
# holding the model family fixed, which neither TFIM nor OAT does.
#
# ⛔ LONG-RANGE XX IS NOT FREE-FERMION. The JW string cancels only between ADJACENT sites; for
# |i-j| > 1, `S+_i S-_j -> c'_i (prod_k (1-2 n_k)) c_j` is not quadratic. So Part C uses exact
# diagonalisation at L=10 and NOT `free_fermion.jl`, which is nearest-neighbour only.
#
# Run:  julia --project=. benchmarks/tfim_anomaly.jl
#       julia --project=. benchmarks/tfim_anomaly.jl part=B

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h; A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J; A[2j + 1, 2j] = +2J
    end
    return A
end

function tfim_x_exact(L::Int, t::Real, dirs::Vector{Symbol}; J = 1.0, h = 1.0)
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s; G[2j, 2j - 1] = -s
    end
    R  = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    Gt = R * G * transpose(R)
    return [Gt[2j - 1, 2j] for j in 1:L]
end

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"Evolve a copy of `psi0` for `nst` steps and return `(psi, krylov)`."
function drive(psi0, nst, step!)
    psi, kry = copy(psi0), 0
    for _ in 1:nst
        kry += _kry(step!(psi))
    end
    return psi, kry
end

schemes(mpo, D, tau, maxiter) =
    ["tdvp2"      => p -> tdvp2_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                      maxiter = maxiter),
     "tdvp_cbe1s" => p -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                           maxiter = maxiter),
     "cbe_bug"    => p -> cbe_bug_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                        maxiter = maxiter, exact = true)]

# ── A. FULL RANK: no truncation left, so only the dt error may survive ────────────────────
function part_A(; L = 8, dt = 0.01, nst = 10, D = 1024, maxiter = 24)
    println("\n" * "="^78)
    @printf("A. FULL RANK  L=%d  maxdim=%d  dt=%g  %d steps (t=%g)\n", L, D, dt, nst, nst * dt)
    println("   No truncation is possible here, so anything above ~dt^2 is a SWEEP defect.")
    println("="^78)
    tau = ComplexF64(-im * dt)

    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    mtf  = tfim_mpo(L; J = 1.0, h = 1.0)
    rtf  = tfim_x_exact(L, nst * dt, dirs)
    @printf("\n  TFIM L=%d (h = J, critical)\n    %-12s %12s  %s\n", L, "scheme", "max|dX|", "chi")
    for (nm, st) in schemes(mtf, D, tau, maxiter)
        psi, _ = drive(ising_kink_state(L), nst, st)
        @printf("    %-12s %12.4e  %d\n", nm, maximum(abs.(x_profile(copy(psi)) - rtf)),
                maximum(bond_dims(psi)))
    end

    set_symmetry!(:U1)
    mxx = xxz_mpo(L; J = 1.0, delta = 0.0)
    rxx = xx_free_fermion_sz(L, nst * dt; J = 1.0)
    @printf("\n  XX L=%d (the control: same locality, same integrability)\n    %-12s %12s  %s\n",
            L, "scheme", "max|dSz|", "chi")
    for (nm, st) in schemes(mxx, D, tau, maxiter)
        psi, _ = drive(domain_wall_state(L), nst, st)
        @printf("    %-12s %12.4e  %d\n", nm, maximum(abs.(magnetisation(copy(psi)) - rxx)),
                maximum(bond_dims(psi)))
    end
end

# ── B. dt ORDER: ratio 4 is a property, ratio 2 or a plateau is a bug ─────────────────────
function part_B(; L = 16, T = 0.6, D = 64, maxiter = 16)
    println("\n" * "="^78)
    @printf("B. dt ORDER  L=%d  maxdim=%d  fixed T=%g\n", L, D, T)
    println("   Second order => each halving of dt QUARTERS the error. ratio ~2 or a")
    println("   plateau is a defect; ~4 means TFIM is merely harder.")
    println("="^78)
    dts = [0.05, 0.025, 0.0125]

    for (label, sym, mpo, psi0, ref, meas) in
        (("TFIM (h = J, critical)", :Z2,
          () -> tfim_mpo(L; J = 1.0, h = 1.0), () -> ising_kink_state(L),
          () -> tfim_x_exact(L, T, [i <= L ÷ 2 ? :plus : :minus for i in 1:L]),
          (p, r) -> maximum(abs.(x_profile(copy(p)) - r))),
         ("XX (control)", :U1,
          () -> xxz_mpo(L; J = 1.0, delta = 0.0), () -> domain_wall_state(L),
          () -> xx_free_fermion_sz(L, T; J = 1.0),
          (p, r) -> maximum(abs.(magnetisation(copy(p)) - r))))
        set_symmetry!(sym)
        m, p0, r = mpo(), psi0(), ref()
        @printf("\n  %s\n    %-12s", label, "scheme")
        for dt in dts
            @printf("  %10s", "dt=$dt")
        end
        println("      ratios")
        for nm in ("tdvp2", "tdvp_cbe1s", "cbe_bug")
            errs = Float64[]
            for dt in dts
                st = Dict(schemes(m, D, ComplexF64(-im * dt), maxiter))[nm]
                psi, _ = drive(p0, round(Int, T / dt), st)
                push!(errs, meas(psi, r))
            end
            @printf("    %-12s", nm)
            for e in errs
                @printf("  %10.3e", e)
            end
            @printf("   %5.2f %5.2f\n", errs[1] / errs[2], errs[2] / errs[3])
        end
    end
end

# ── C. LONG-RANGE XX, via pair_mpo. Reference is ED: this is NOT free-fermion. ────────────
"`Σ_{i<j} J[i,j] (Sx Sx + Sy Sy)_{ij}` densely -- the XY half only, no ZZ."
function dense_long_range_xy(L::Int, J::AbstractMatrix)
    H = zeros(ComplexF64, 2^L, 2^L)
    for i in 1:(L - 1), j in (i + 1):L
        J[i, j] == 0 && continue
        H .+= J[i, j] * 0.5 * (_embed(_SP, i, L) * _embed(_SM, j, L) +
                               _embed(_SM, i, L) * _embed(_SP, j, L))
    end
    return H
end

function part_C(; L = 10, dt = 0.05, nst = 20, D = 32, maxiter = 20,
                alphas = (99.0, 3.0, 2.0, 1.0))
    println("\n" * "="^78)
    @printf("C. LONG-RANGE XX  L=%d  maxdim=%d  dt=%g  %d steps (t=%g)\n", L, D, dt, nst, nst*dt)
    println("   J[i,j] = |i-j|^-alpha; alpha=99 IS nearest-neighbour. Range varies, model")
    println("   family does not. Reference is EXACT DIAGONALISATION -- long-range XX is NOT")
    println("   free-fermion, the JW string only cancels for adjacent sites.")
    println("="^78)
    set_symmetry!(:U1)
    psi0 = domain_wall_state(L)
    v0   = dense_state(psi0)

    for a in alphas
        Jm = [i == j ? 0.0 : 1.0 / abs(i - j)^a for i in 1:L, j in 1:L]
        mpo = pair_mpo(L, Jm, xxz_chain(L; J = 1.0, delta = 0.0))
        vex = dense_exact_propagate(dense_long_range_xy(L, Jm), v0, nst * dt)
        @printf("\n  alpha = %-5s %-12s %12s  %8s  %s\n",
                a >= 99 ? "NN" : string(a), "scheme", "infidelity", "krylov", "chi")
        for (nm, st) in schemes(mpo, D, ComplexF64(-im * dt), maxiter)
            psi, kry = drive(psi0, nst, st)
            vn = dense_state(psi)
            f  = 1 - abs2(dot(vex, vn)) / real(dot(vex, vex) * dot(vn, vn))
            @printf("  %13s %-12s %12.4e  %8d  %d\n", "", nm, f, kry, maximum(bond_dims(psi)))
        end
    end
end

let part = "all"
    for a in ARGS
        startswith(a, "part=") && (part = split(a, '='; limit = 2)[2])
    end
    part in ("A", "all") && part_A()
    part in ("B", "all") && part_B()
    part in ("C", "all") && part_C()
    println()
end
