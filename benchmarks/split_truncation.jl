# TRUNCATION AT THE HALF-SWEEP SPLIT: does pruning EARLY beat pruning once at the root?
#
# `cbe_bug` has historically split with a hard `cutoff = 1e-14` and no `Nkeep`, so the K and L
# half-sweeps prune essentially nothing and EVERY rank decision is deferred to the root solve
# and `truncate_recursive!`. `tdvp_cbe1s` does the opposite: `cutoff = cut, Nkeep = maxdim` at
# every site, pruning continuously. That is the one structural difference between the two sweeps
# that is not about widening, and it has never been measured.
#
# WHY IT IS THE LEADING SUSPECT FOR `rexpand`'s REMAINING DEFICIT. `rexpand`'s `W[i]` spans `K`,
# `K1` AND the new CBE directions, so it carries strictly more rank into the root than `kaug`'s
# `orth([U_ex | svd(K1).U])` does. The root then has to discard more of it, with only the root's
# information to decide by. If that is the mechanism, pruning at the split -- where the local
# structure is still present -- should close the gap on TFIM and leave OAT and XX alone.
#
# ⛔ TFIM IS THE MODEL THAT MATTERS HERE and the other two are controls. `rexpand` already
# matches `kaug` to five figures on OAT and bit-identically on XX, so neither can show an
# improvement; they exist to catch a REGRESSION from pruning too hard.
#
# References are exact and none is a second integrator:
#   OAT  -- H diagonal in the S^z basis, one phase per basis vector.
#   XX   -- free fermions, exp(-i h t) on the single-particle Hamiltonian.
#   TFIM -- Majorana/BdG: X_l = i a_l b_l, Z_l Z_{l+1} = i b_l a_{l+1}, Gamma(t) = R Gamma R'.
#
# Run:  julia --project=. benchmarks/split_truncation.jl
#       julia --project=. benchmarks/split_truncation.jl model=tfim
#
# ⛔ NO WALL TIME. Local timing spreads 2.6-2.8x on bit-identical work; `krylov` is the cost axis.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

# ── TFIM exact, by Majorana propagation (copied from examples/tfim_z2.jl) ─────────────────
function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h
        A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J
        A[2j + 1, 2j] = +2J
    end
    return A
end

function tfim_x_profile_exact(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s
        G[2j, 2j - 1] = -s
    end
    R = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    Gt = R * G * transpose(R)
    return [Gt[2j - 1, 2j] for j in 1:L]
end

_kry(info) = hasproperty(info, :krylov_dims) ? info.krylov_dims : 0

"The split-truncation grid. `(label, split_cutoff, split_maxdim)`; `0` means no cap."
_grid(D) = [("baseline  1e-14 / none", 1e-14, 0),      # what ships
            ("cutoff    1e-12 / none", 1e-12, 0),
            ("cutoff    1e-10 / none", 1e-10, 0),
            ("cutoff     1e-8 / none", 1e-8,  0),
            ("Nkeep     1e-14 / D",    1e-14, D),
            ("Nkeep     1e-14 / 2D",   1e-14, 2D),
            ("tdvp-like 1e-10 / D",    1e-10, D)]

"""
Run every grid point plus the two reference arms, and report `(err, krylov, chi)`.

`measure(psi) -> Float64` is the error against the exact reference.
"""
function sweep(name, mpo, psi0, nst, dt, D, maxiter, measure)
    @printf("\n%s   D=%d  dt=%g  %d steps\n", name, D, dt, nst)
    @printf("  %-26s %12s  %8s  %s\n", "split cutoff / Nkeep", "err", "krylov", "chi")

    function run(step!)
        psi, kry = copy(psi0), 0
        for _ in 1:nst
            kry += _kry(step!(psi))
        end
        return measure(psi), kry, maximum(bond_dims(psi))
    end
    tau = ComplexF64(-im * dt)

    # The two references: `kaug` is what `rexpand` has to beat, and `rexpand` at the shipped
    # split truncation is the row every grid point below is trying to improve on.
    e, k, c = run(p -> cbe_bug_step!(p, mpo, tau; kaug = true, rexpand = false, maxdim = D,
                                     trunc_thresh = 1e-14, maxiter = maxiter))
    @printf("  %-26s %12.4e  %8d  %d   <- kaug, the target\n", "(kaug reference)", e, k, c)

    for (lbl, sc, sm) in _grid(D)
        e, k, c = run(p -> cbe_bug_step!(p, mpo, tau; rexpand = true, maxdim = D,
                                         trunc_thresh = 1e-14, maxiter = maxiter,
                                         split_cutoff = sc, split_maxdim = sm))
        @printf("  %-26s %12.4e  %8d  %d\n", lbl, e, k, c)
    end
end

function run_tfim(; L = 16, dt = 0.05, nst = 60, D = 32, J = 1.0, h = 1.0, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    mpo  = tfim_mpo(L; J = J, h = h)
    psi0 = ising_kink_state(L)
    ref  = tfim_x_profile_exact(L, nst * dt, dirs; J = J, h = h)
    sweep("TFIM  L=$L  :Z2  (h = J, critical)", mpo, psi0, nst, dt, D, maxiter,
          psi -> maximum(abs.(x_profile(copy(psi)) - ref)))
end

function run_xx(; L = 16, dt = 0.05, nst = 120, D = 24, J = 1.0, maxiter = 12)
    set_symmetry!(:U1)
    mpo  = xxz_mpo(L; J = J, delta = 0.0)
    psi0 = domain_wall_state(L)
    ref  = xx_free_fermion_sz(L, nst * dt; J = J)
    sweep("XX domain wall  L=$L  :U1", mpo, psi0, nst, dt, D, maxiter,
          psi -> maximum(abs.(magnetisation(copy(psi)) - ref)))
end

function run_oat(; L = 10, dt = 0.05, nst = 20, D = 6, maxiter = 30)
    set_symmetry!(:none)
    mpo   = oat_mpo(L)
    psi0  = x_polarized_state(L)
    v0    = dense_state(psi0)
    ediag = (diag(dense_total_sz(L)) .^ 2) ./ 2 .- L / 8
    vex   = exp.(-im * (nst * dt) .* ediag) .* v0
    sweep("OAT  L=$L  D=6 = the exact ceiling", mpo, psi0, nst, dt, D, maxiter,
          function (psi)
              vn = dense_state(psi)
              return 1 - abs2(dot(vex, vn)) / real(dot(vex, vex) * dot(vn, vn))
          end)
end

# ── the BUDGET sweep, after the split-truncation grid came back negative ──────────────────
#
# ⛔ THE SPLIT-TRUNCATION HYPOTHESIS IS DEAD. Measured on TFIM L=16 D=32: no grid point beat the
# `1e-14 / none` baseline and every one that bound was worse (`Nkeep = D` cost 2.6x), while
# `krylov` was 14763 for EVERY arm including `kaug`. So the deficit is not rank reaching the
# root, and there is no accuracy/cost trade hiding in the split.
#
# WHAT REPLACES IT. `budget = ceil(growth*dmax) - r` (cbe_core.jl). `rexpand`'s SECOND expansion
# runs against `U0 = orth([svd(K).U | svd(K1).U])`, which is strictly larger than the first
# expansion's `U0 = svd(K).U` that `kaug` extends. Larger `r` at fixed `growth` means a SMALLER
# budget -- possibly the floor of 1 -- and a smaller complement to search, because `K1`'s
# directions have already been absorbed into the frame. `rexpand` may be starving its own
# expansion by construction. If so, `dex` buys the gap back and `kaug` does not need it.
function budget_sweep(name, mpo, psi0, nst, dt, D, maxiter, measure)
    @printf("\n%s   D=%d  dt=%g  %d steps  -- BUDGET\n", name, D, dt, nst)
    @printf("  %-26s %12s  %8s  %s\n", "arm", "err", "krylov", "chi")
    tau = ComplexF64(-im * dt)
    function run(step!)
        psi, kry = copy(psi0), 0
        for _ in 1:nst
            kry += _kry(step!(psi))
        end
        return measure(psi), kry, maximum(bond_dims(psi))
    end
    for (lbl, kw) in [("kaug    growth=1.1",    (kaug = true, rexpand = false, growth = 1.1)),
                      ("kaug    growth=2.0",    (kaug = true, rexpand = false, growth = 2.0)),
                      ("kaug    dex=8",         (kaug = true, rexpand = false, dex = 8)),
                      ("rexpand growth=1.1",    (rexpand = true, growth = 1.1)),
                      ("rexpand dex=2",          (rexpand = true, dex = 2)),
                      ("rexpand dex=4",          (rexpand = true, dex = 4)),
                      ("rexpand dex=8",          (rexpand = true, dex = 8)),
                      ("rexpand dex=16",         (rexpand = true, dex = 16)),
                      ("rexpand growth=2.0",     (rexpand = true, growth = 2.0))]
        e, k, c = run(p -> cbe_bug_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                         maxiter = maxiter, kw...))
        @printf("  %-26s %12.4e  %8d  %d\n", lbl, e, k, c)
    end
end

# ── RSVD vs EXACT CBE, and what each one DISCARDS ────────────────────────────────────────
#
# ⛔ THE BENCHMARKS AND THE EXAMPLES HAVE NOT BEEN RUNNING THE SAME THING. `examples/common.jl`
# passes `exact = true` (the complete fused basis, no randomness anywhere); both benchmarks in
# this directory take the default `exact = false` (the sector-graded Gaussian sketch). Numbers
# from the two are therefore not directly comparable, which is what this sweep exists to fix.
#
# THE DISCARD COLUMNS ARE THE POINT, not just the error. `err_pre` is the weight the
# PRESELECTION drops, `err_fnl` the weight the FINAL selection drops for want of budget, and
# `discarded` the weight the closing `truncate_recursive!` drops. With `exact = true` the
# preselection returns the exact leading directions of `P_perp(H*Theta)`, so `err_fnl` is the
# TRUE ranked-candidate weight being thrown away -- a starvation meter with no sampling noise in
# it. Comparing it against the sketch's says whether the sketch is finding the same subspace.
#
# Phase 0 of `rsvd_param_scan.jl` measured the sketch as SATURATED (relative difference exactly
# 0.000e+00) on OAT and XX at L=16 -- but under the OLD defaults (`kaug`, `growth = 1.1`). Both
# have changed, and `rexpand` searches a smaller complement against a larger `U0`, which is
# exactly the regime where a sketch could stop being saturated. All values are maxima over the
# whole run, so one bad step cannot hide behind an average.
function exact_sweep(name, mpo, psi0, nst, dt, D, maxiter, measure)
    @printf("\n%s   D=%d  dt=%g  %d steps  -- RSVD vs EXACT\n", name, D, dt, nst)
    @printf("  %-28s %12s  %8s %4s %10s %10s %10s\n",
            "arm", "err", "krylov", "chi", "err_pre", "err_fnl", "discarded")
    tau = ComplexF64(-im * dt)
    for (lbl, kw) in [("rexpand  RSVD sketch",  (rexpand = true,)),
                      ("rexpand  EXACT cbe",    (rexpand = true, exact = true)),
                      ("kaug     RSVD sketch",  (kaug = true, rexpand = false)),
                      ("kaug     EXACT cbe",    (kaug = true, rexpand = false, exact = true))]
        psi, kry = copy(psi0), 0
        epre = efnl = disc = 0.0
        for _ in 1:nst
            info = cbe_bug_step!(psi, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                 maxiter = maxiter, kw...)
            kry += info.krylov_dims
            epre = max(epre, info.err_pre)
            efnl = max(efnl, info.err_fnl)
            disc = max(disc, info.discarded)
        end
        @printf("  %-28s %12.4e  %8d %4d %10.2e %10.2e %10.2e\n",
                lbl, measure(psi), kry, maximum(bond_dims(psi)), epre, efnl, disc)
    end
end

function exact_tfim(; L = 16, dt = 0.05, nst = 60, D = 32, J = 1.0, h = 1.0, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    mpo  = tfim_mpo(L; J = J, h = h)
    ref  = tfim_x_profile_exact(L, nst * dt, dirs; J = J, h = h)
    exact_sweep("TFIM  L=$L  :Z2", mpo, ising_kink_state(L), nst, dt, D, maxiter,
                psi -> maximum(abs.(x_profile(copy(psi)) - ref)))
end

function exact_xx(; L = 16, dt = 0.05, nst = 120, D = 24, J = 1.0, maxiter = 12)
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = J, delta = 0.0)
    ref = xx_free_fermion_sz(L, nst * dt; J = J)
    exact_sweep("XX domain wall  L=$L  :U1", mpo, domain_wall_state(L), nst, dt, D, maxiter,
                psi -> maximum(abs.(magnetisation(copy(psi)) - ref)))
end

function exact_oat(; L = 10, dt = 0.05, nst = 20, D = 6, maxiter = 30)
    set_symmetry!(:none)
    mpo   = oat_mpo(L)
    psi0  = x_polarized_state(L)
    v0    = dense_state(psi0)
    ediag = (diag(dense_total_sz(L)) .^ 2) ./ 2 .- L / 8
    vex   = exp.(-im * (nst * dt) .* ediag) .* v0
    exact_sweep("OAT  L=$L  D=6 = the exact ceiling", mpo, psi0, nst, dt, D, maxiter,
                function (psi)
                    vn = dense_state(psi)
                    return 1 - abs2(dot(vex, vn)) / real(dot(vex, vex) * dot(vn, vn))
                end)
end

function budget_tfim(; L = 16, dt = 0.05, nst = 60, D = 32, J = 1.0, h = 1.0, maxiter = 16)
    set_symmetry!(:Z2)
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    mpo  = tfim_mpo(L; J = J, h = h)
    ref  = tfim_x_profile_exact(L, nst * dt, dirs; J = J, h = h)
    budget_sweep("TFIM  L=$L  :Z2", mpo, ising_kink_state(L), nst, dt, D, maxiter,
                 psi -> maximum(abs.(x_profile(copy(psi)) - ref)))
end

let model = "both"
    for a in ARGS
        startswith(a, "model=") && (model = split(a, '='; limit = 2)[2])
    end
    model in ("tfim", "both", "all") && run_tfim()
    model in ("xx",   "both", "all") && run_xx()
    model in ("oat",  "all")         && run_oat()
    model in ("budget", "both", "all") && budget_tfim()
    if model in ("exact", "all")
        exact_tfim(); exact_xx(); exact_oat()
    end
    println()
end
