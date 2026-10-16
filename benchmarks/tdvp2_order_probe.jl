# DOES COMPOSING 2-SITE TDVP BUY ORDER, AND WHERE? -- accuracy only, small chains.
#
# Measured first (job 17283495, test_tdvp2_order.jl): at a BINDING cap (L = 8, cap 6) plain
# tdvp2_step! converges at slope ~1.2 to its own dt -> 0 limit, and the Suzuki / Yoshida order-4
# compositions give the SAME slope -- the per-bond truncation, not the splitting, sets the error.
# This probe separates the regimes:
#
#   binding    L = 8, cap 6, trunc 0 (the cap decides the rank at every bond)
#   tolerance  L = 24, quench from a gapped (Delta = 3) ground state to Delta = 1, cap 256 (never
#              reached), trunc_thresh 1e-10: the rank is set by a tiny tolerance, far below full
#              rank (2^12 at the centre), and the start already carries a decaying spectrum, so no
#              rank has to be created from zero
#
# ⛔ Job 17283686 ran the tolerance regime at L = 12 and read errors of exactly 0. Two causes:
# (1) at L = 12 trunc 1e-10 keeps the state at effectively full rank, where the projector-splitting
# integrator is EXACT for any dt (the solution never leaves the manifold) -- there is no step error;
# (2) the overlap distance sqrt(|a|^2 + |b|^2 - 2 Re<a|b>) cancels two O(1) numbers and cannot
# resolve below ~sqrt(eps) = 1.5e-8. Hence L = 24 and an observable error.
#
# Error = max_j |<S^z_j>_dt(T) - <S^z_j>_ref(T)|, ref = the same regime at dt/8 with the Yoshida
# composition; resolves to ~1e-15. The state distance is printed beside it (floor ~1e-8). Also
# printed: the largest discarded weight of the run (the truncation each regime pays).
#
# Run:  julia --project=PKG benchmarks/tdvp2_order_probe.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

set_symmetry!(:U1)
nrm2(a) = real(overlap(a, a))
dist(a, b) = sqrt(max(nrm2(a) + nrm2(b) - 2 * real(overlap(a, b)), 0.0))
sz(a) = magnetisation(copy(a)) ./ nrm2(a)
fe(v) = join([@sprintf("%.2e", x) for x in v], " ")
fe1(v) = join([@sprintf("%.1e", x) for x in v], " ")
ff(v) = join([@sprintf("%.2f", x) for x in v], " ")

function regime(name, mpo, psi0, kw, T, dts)
    run(order, comp, dt) = (p = copy(psi0); disc = 0.0;
        for _ in 1:round(Int, T / dt)
            info = tdvp2_composed_step!(p, mpo, ComplexF64(-1im * dt); order = order,
                                        composition = comp, kw...)
            disc = max(disc, info.discarded)
        end; (p, disc))
    ref, _ = run(4, :yoshida, dts[end] / 8)
    szref = sz(ref)
    @printf("== %s  start chi %d  ref chi %d  T %.2f  kw %s\n", name, maximum(bond_dims(psi0)),
            maximum(bond_dims(ref)), T, string(kw))
    for (order, comp) in ((2, :suzuki), (4, :suzuki), (4, :yoshida))
        errs, dsts, discs, chis = Float64[], Float64[], Float64[], Int[]
        t = @elapsed for dt in dts
            p, disc = run(order, comp, dt)
            push!(errs, maximum(abs.(sz(p) .- szref))); push!(dsts, dist(p, ref))
            push!(discs, disc); push!(chis, maximum(bond_dims(p)))
        end
        sl = [log(errs[k] / errs[k + 1]) / log(dts[k] / dts[k + 1]) for k in 1:(length(dts) - 1)]
        @printf("  order %d %-8s max|dSz| %s  slopes %s  dist %s  max discarded %s  chi %s  %.0fs\n",
                order, comp, fe(errs), ff(sl), fe1(dsts), fe1(discs), string(chis), t)
        flush(stdout)
    end
end

# binding
let L = 8, cap = 6
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    kw = (maxdim = cap, trunc_thresh = 0.0, maxiter = 30)
    psi0 = neel_state(L)
    for _ in 1:20
        tdvp2_step!(psi0, mpo, ComplexF64(-0.05im); kw...)
    end
    regime("binding", mpo, psi0, kw, 0.6, (0.2, 0.1, 0.05))
end

# tolerance
let L = 24
    gs = neel_state(L)
    hg = xxz_mpo(L; J = 1.0, delta = 3.0)
    for _ in 1:40          # imaginary time to the gapped ground state, renormalised every step
        tdvp2_step!(gs, hg, ComplexF64(-0.1); maxdim = 256, trunc_thresh = 1e-12, maxiter = 30)
        canonical!(gs, 1)
        gs[1] = to_concrete((1.0 / norm(gs[1])) * gs[1])
    end
    mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
    regime("tolerance", mpo, gs, (maxdim = 256, trunc_thresh = 1e-10, maxiter = 30), 1.0,
           (0.2, 0.1, 0.05))
end
println("TDVP2-ORDER-PROBE-DONE")
