# DOES GROUND-STATE SEARCH WORK UNDER SU(2) AT ALL? -- the gate before any knob study.
#
# ⛔ WHY THIS FILE EXISTS BEFORE THE CAMPAIGN. `tests/` contains ZERO references to `GroundState`
# or `dmrg_cbe1s`, and every ground-state benchmark in this repo (`gse_rsvd_vs_exact.jl`,
# `gse_defer_truncation.jl`, `gse_comp_ratio_scan.jl`) runs `:U1` or `:Z2`. The SU(2) path has
# never been exercised for a ground state. Tuning knobs on an unvalidated solver would produce a
# table of one wrong number against another, and the wrongness would be invisible: every arm here
# reports a VARIATIONAL upper bound, so they all look plausible and all move the right way.
#
# ⛔ THE REFERENCE IS THE BETHE ANSATZ, NEVER A DIAGONALISATION. Standing rule for Heisenberg in
# this repo. `bethe_xxx_open_energy` returns `(E, lambdas, residual)` and THE RESIDUAL IS ASSERTED
# -- a Bethe solve that has not converged returns a number that looks like an energy.
#
# ⚠ chi IS IN MULTIPLETS under SU(2) and in STATES under U(1)/`:none`. The `:none` control row is
# here to prove the physics is unchanged, NOT to be compared at equal `maxdim`.
#
# ⚠ TWO ENERGIES ARE REPORTED PER ARM AND THEY ARE NOT THE SAME QUANTITY. `E_var` is the lowest
# Ritz value the sweep saw -- a bound in the space the solve actually used. `E_state` is
# `mpo_energy` of the state that came back, after truncation. They separate "the eigensolve found
# the right thing" from "the state we kept still has it", which is exactly the failure a deferred
# or aggressive truncation produces.
#
# Run:  julia -t 2 --project=. benchmarks/ground_state/su2_gate.jl [L ...]

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

const LS = isempty(ARGS) ? (8, 12) : Tuple(parse.(Int, ARGS))
const MAXDIM = parse(Int, get(ENV, "GS_MAXDIM", "64"))
const NSWEEPS = parse(Int, get(ENV, "GS_NSWEEPS", "24"))
const ETOL = parse(Float64, get(ENV, "GS_ETOL", "1e-12"))

"""
    reference(L) -> Float64

Open-chain XXX ground energy from the Bethe ansatz, with the residual asserted.
"""
function reference(L::Int)
    E, _, resid = bethe_xxx_open_energy(L)
    resid < 1e-12 || error("Bethe residual $resid at L=$L -- reference not converged")
    return E
end

"""
    two_site(psi, W; exact, grow_iters) -> (E_var, E_state, chi, matvec, sweeps)

`GroundState.solve!` -- 2-site DMRG, CBE expansion, expanding-Krylov local solve.

⛔ CALLED THROUGH `solve!` AND NOT THROUGH `cbe_dmrg!` / `rsvd_cbe_dmrg!` ON PURPOSE: those two
wrappers are typed `h::XXZChain` and a `MethodError` is all an SU(2) MPO gets from them.
"""
function two_site(psi0, W; exact::Bool, grow_iters::Int)
    psi = copy(psi0)
    opts = CBEBugOptions(; growth = 2.0, maxdim = MAXDIM, trunc_thresh = 1e-12,
                         maxiter = 30, tol = 1e-14, exact = exact)
    info = solve!(psi, W; opts = opts, n_sweeps = NSWEEPS, etol = ETOL,
                  grow_iters = grow_iters)
    return (info.energies[end], state_energy(psi, W), maximum(bond_dims(psi)),
            maximum(state_bond_dims(psi)), sum(info.krylov_dims), length(info.energies))
end

"""
    one_site(psi, W; exact, seed_expanded) -> (E_var, E_state, chi, matvec, sweeps)

`dmrg_cbe1s!` -- the line-for-line port of `DMRGSweepCBE1Si.m`. A 1-site update CANNOT change a
bond dimension, so here the CBE pass is not an accelerator: it is the ONLY source of rank.
"""
function one_site(psi0, W; exact::Bool, seed_expanded::Bool)
    psi = copy(psi0)
    run = dmrg_cbe1s!(psi, W; n_sweeps = NSWEEPS, etol = ETOL, growth = 2.0,
                      maxdim = MAXDIM, trunc_thresh = 1e-12, maxiter = 30,
                      restol = 1e-10, exact = exact, seed_expanded = seed_expanded)
    return (run.energies[end], state_energy(psi, W), maximum(bond_dims(psi)),
            maximum(state_bond_dims(psi)), sum(run.krylov_dims), length(run.energies))
end

"Energy of the state that actually came back, normalised -- see the header on E_var vs E_state."
state_energy(psi, W) = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())

function gate(L::Int)
    eref = reference(L)
    @printf("\n%s\nL = %d   Bethe E = %.12f   maxdim = %d (MULTIPLETS under SU(2))\n%s\n",
            "="^104, L, eref, MAXDIM, "="^104)
    # ⛔ TWO chi COLUMNS BECAUSE THEY ARE DIFFERENT UNITS. `bond_dims` counts MULTIPLETS,
    # `state_bond_dims` counts STATES; they coincide under `:none`/`:U1` and do NOT under SU(2)
    # (6 multiplets = 16 states here). A first version of this file read one function for the
    # 2-site arms and the other for the 1-site arms, which made a UNITS difference look like a
    # method difference -- the 2-site arms appeared to converge at chi=6 where the 1-site arms
    # needed 16, and they were the same state.
    @printf("  %-26s %-14s %14s %14s %13s %8s %6s\n",
            "arm", "symmetry", "E_var - E_ref", "E_state - E_ref",
            "chi mult/stat", "matvec", "swp")

    ok = true
    for sym in (:SU2, :none)
        set_symmetry!(sym)
        psi0 = dimer_state(L)                      # the S = 0 sector, where the ground state lives
        W = sym === :SU2 ? heisenberg_su2_mpo(L) : xxz_mpo(L; J = 1.0, delta = 1.0)

        arms = (("2-site CBE  exact", () -> two_site(psi0, W; exact = true,  grow_iters = 1)),
                ("2-site CBE  rSVD",  () -> two_site(psi0, W; exact = false, grow_iters = 1)),
                ("1-site CBE  exact", () -> one_site(psi0, W; exact = true,  seed_expanded = false)),
                ("1-site CBE  rSVD",  () -> one_site(psi0, W; exact = false, seed_expanded = false)),
                ("1-site CBE  rSVD+seed", () -> one_site(psi0, W; exact = false, seed_expanded = true)))

        for (name, f) in arms
            ev, es, chim, chis, mv, swp = try
                f()
            catch e
                @printf("  %-26s %-14s  ERROR: %s\n", name, String(sym),
                        first(split(sprint(showerror, e), '\n')))
                ok = false
                continue
            end
            # ⛔ A VARIATIONAL METHOD CANNOT GO BELOW THE TRUE GROUND ENERGY. `E_var` under the
            # reference by more than roundoff is a BUG -- an inconsistent Rayleigh-Ritz -- not a
            # lucky sweep, and it is the one failure that looks like success in every other column.
            flag = ev - eref < -1e-10 ? "  ⛔ BELOW THE BOUND" : ""
            @printf("  %-26s %-14s %14.3e %14.3e %6d /%5d %8d %6d%s\n",
                    name, String(sym), ev - eref, es - eref, chim, chis, mv, swp, flag)
            isempty(flag) || (ok = false)
            flush(stdout)
        end
    end
    return ok
end

# ⚠ A FUNCTION, NOT A TOP-LEVEL LOOP. `ok &= gate(L)` inside a top-level `for` is Julia's soft
# scope: the accumulator is treated as a NEW LOCAL, so it is undefined on the first read. It warns
# and then throws, which is the good case -- the bad one is a script that quietly reports the last
# size instead of the conjunction over all of them.
function main()
    ok = true
    for L in LS
        ok &= gate(L)
    end
    println(ok ? "\nSU(2) GROUND-STATE GATE PASSED" :
                 "\n⛔ SU(2) GROUND-STATE GATE FAILED -- do not run a knob study on this")
    flush(stdout)
    return ok
end

# ⚠ `if`, NOT `... == @__FILE__ && exit(...)`. `@__FILE__` is a MACRO and takes the rest of the
# line as its arguments, so the `&&` form is a parse error rather than a guard.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main() ? 0 : 1)
end
