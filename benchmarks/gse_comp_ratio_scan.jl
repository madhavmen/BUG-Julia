# IS THE SEEDED EIGENSOLVE INSENSITIVE TO `comp_ratio`? -- the question that actually decides
# whether it is worth having.
#
#   julia --project=. benchmarks/gse_comp_ratio_scan.jl [L] [n_sweeps]
#
# ⛔ THIS IS NOT A COST BENCHMARK, and reading it as one would miss the point. The seeded arm
# costs one EXTRA operator application per local solve, so on operations-to-accuracy it starts
# behind and may well stay behind. The claim being tested is different and is about ROBUSTNESS:
#
#   `comp_ratio` splits the RSVD probe between the complement and the isometry (`cbe_core.jl`,
#   `sector_graded_sketch`). It is a hyperparameter that has to be TUNED ONTO the relevant
#   subspace, and the right value is problem-dependent -- fine for a Heisenberg chain where 0.5
#   works, a liability for anything harder, where there is no reference to tune against.
#
#   Seeding the local eigensolve from the EXPANDED basis is supposed to make the iteration find
#   those directions itself. `cbe_core.jl:1278-1283` already argues exactly this for the
#   expanding Krylov: the basis should "converge on the relevant subspace rather than having to
#   be tuned onto it by `comp_ratio` and the oversampling."
#
# So the measurement is the SPREAD OF THE ERROR ACROSS `comp_ratio`, per arm. A flat curve for the
# seeded arm and a varying one for the default arm is the result; two flat curves would mean the
# chain is too easy to discriminate and the test must move to a harder model, NOT that the claim
# is established.
#
# `comp_ratio = nothing` is included: that is the "no projector in the probe at all" setting
# (`cbe_core.jl:234`), i.e. the extreme the tuning is supposed to become unnecessary against.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
import Random
Random.seed!(42)

const HERE = @__DIR__
include(joinpath(HERE, "..", "tests", "common", "analytic_reference.jl"))

const L        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16
const NSWEEPS  = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12
# ⛔ THE FIRST RUN OF THIS SCAN WAS A NULL RESULT, AND MAXDIM IS WHY.
# At MAXDIM = 256 the cap never binds -- cutoff 1e-8 settles at chi = 26 multiplets -- so the
# probe is never forced to CHOOSE which directions to admit, and `comp_ratio` (which only decides
# how the probe is split between complement and isometry) cannot matter. Measured: spread exactly
# 1.00x at every intermediate sweep for both arms, with sweep 1 and 2 identical to four digits
# across all six values including `nothing`. The parameter IS plumbed through -- sweep 3 shows a
# monotone trend (0.0/0.25/0.5 -> 9.306e-12, 0.75/1.0 -> 9.313e-12, none -> 9.315e-12) -- it is
# simply inert when there is enough rank for every choice to succeed.
#
# This is the same trap as [[dt-order-tests-need-rank-limited-regime]]: a test run at full rank
# measures roundoff, not the thing it was built to measure. To give the scan power, CAP the rank
# so basis quality decides the error. Overridable, and the tag goes in the filename.
const MAXDIM   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 256
const CUTOFF   = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-8
const RATIOS   = (0.0, 0.25, 0.5, 0.75, 1.0, nothing)

function main()
    set_symmetry!(:SU2)
    mpo  = heisenberg_chain_mpo(L)
    # ⛔ `bethe_xxx_open_energy` returns (energy, lambdas, residual) -- a 3-TUPLE, not a number.
    # Binding the whole tuple to `eref` type-errors on the first `abs(E - eref)`, and the residual
    # is not optional: the open solver has previously converged to residual 1e-14 while sitting
    # 0.24 BELOW ED, so a converged flag does not imply a correct value (see
    # tests/common/test_analytic_reference.jl). Same contract as `ground_reference` in
    # benchmarks/cbe_sweeps_l16.jl.
    eref, _, eres = bethe_xxx_open_energy(L)
    eres < 1e-10 || error("open Bethe solve did not converge at L=$L: residual $eres")
    @printf("L=%d  Bethe reference = %.12f (residual %.2e)  sweeps=%d  tol=%g\n",
            L, eref, eres, NSWEEPS, CUTOFF); flush(stdout)

    # MAXDIM in the filename: a capped run is a DIFFERENT experiment from the full-rank one and
    # must not overwrite it -- the full-rank null result is the evidence for why the cap is there.
    out = joinpath(HERE, "results",
                   "gse_comp_ratio_L$(L)" * (MAXDIM == 256 ? "" : "_chi$(MAXDIM)") * ".csv")
    mkpath(dirname(out))
    io = open(out, "w")
    # `maxdim`/`cutoff` ARE COLUMNS, not just filename decoration. Three caps were scanned
    # (256/12/8) and the plot globs `gse_comp_ratio_L*.csv`, so without these the runs merge into
    # one curve that silently averages three different experiments -- the reader cannot tell, and
    # neither can the script. Same failure as overlaying two symmetries in one ITE panel.
    println(io, "arm,comp_ratio,sweep,energy,err,maxbond,maxbond_states,krylov,maxdim,cutoff")

    for (arm, solver) in (("rsvd", rsvd_cbe_dmrg1s!),
                          ("rsvd_seeded", rsvd_cbe_dmrg1s_seeded!))
        for cr in RATIOS
            # A FRESH state per point: a shared one would let an earlier `comp_ratio` hand the
            # next a rank it did not have to find, which is precisely the sensitivity under test.
            psi = dimer_state(L)
            run = solver(psi, mpo; n_sweeps = NSWEEPS, maxdim = MAXDIM,
                         trunc_thresh = CUTOFF, comp_ratio = cr)
            crs = cr === nothing ? "none" : @sprintf("%.2f", cr)
            for sw in eachindex(run.energies)
                @printf(io, "%s,%s,%d,%.12f,%.6e,%d,%d,%d,%d,%g\n",
                        arm, crs, sw, run.energies[sw], abs(run.energies[sw] - eref),
                        run.max_bond_dims[sw], run.max_state_bond_dims[sw],
                        run.krylov_dims[sw], MAXDIM, CUTOFF)
            end
            flush(io)
            @printf("  %-12s comp_ratio=%-5s  final err=%.3e  chi=%d (%d states)  krylov=%d\n",
                    arm, crs, abs(run.energies[end] - eref), run.max_bond_dims[end],
                    run.max_state_bond_dims[end], run.krylov_dims[end]); flush(stdout)
        end
    end
    close(io)
    println("\nwrote ", out); flush(stdout)
end

main()
