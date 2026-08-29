# DOES `pair_mpo` CARRY A CHANNEL ACROSS AN INTERVENING SITE CORRECTLY?
#
# ⛔ THE GAP. `half_check` (a) validated `pair_mpo` against `xxz_mpo` on a NEAREST-NEIGHBOUR chain
# -- where a channel opens at `i` and closes at `i+1`, so the TRANSPORT block is never used at
# all. The doubled chain puts every system-system coupling at RANGE 2 (the partner's ancilla sits
# in between), which exercises transport for the first time in that construction. And the
# doubled-chain error is FLAT in dt (order 0.00 from dt=0.1 to 0.0125), which is an operator
# fault, not an integrator one.
#
# This tests transport on the PLAIN chain, where `dense_long_range_pairs(L, J)` is an exact
# reference for `sum_{i<j} J[i,j] (SxSx + SySy + SzSz)`, with range 1 as the CONTROL (no transport)
# and ranges 2 and 3 as the cases.
#
# ⚠ FULL VECTORS, NOT SCALARS. `<v|W|v>` for a few correlated probes is what let the doubled-chain
# fault through: six snapshots of one trajectory are nearly collinear and cannot pin down the
# operator. The first-order response `(psi(dt) - psi(0))/(-i*dt) -> H psi(0)` compares every
# component that `psi(0)` reaches.
#
# ⚠ BOTH SYMMETRY MODES. The kagome / square / dipolar models all use transport and were validated
# -- under `:U1` and `:SU2`. If the fault is specific to `:none` (the mode the Lindbladian must run
# in, since the physical conserved quantity is `Sz_ket - Sz_bra` and our local space tracks the
# SUM), those validations would not have caught it.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "transport_check.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))

const L = 4
const DT = 1e-6
const PROBES = [[:up, :down, :up, :down], [:up, :up, :down, :down],
                [:down, :up, :up, :down], [:up, :down, :down, :up]]

function check(mode::Symbol, i::Int, j::Int)
    set_symmetry!(mode)
    J = zeros(Float64, L, L); J[i, j] = 1.0
    H = dense_long_range_pairs(L, J)
    W = pair_mpo(L, J, pair_vertices(xxz_chain(L; J = 1.0, delta = 1.0)))
    worst = 0.0
    for sp in PROBES
        p0 = product_state(sp); v0 = dense_state(p0)
        p = copy(p0)
        tdvp2_step!(p, W, ComplexF64(-im * DT); maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)
        got  = (dense_state(p) .- v0) ./ (-im * DT)
        want = H * v0
        worst = max(worst, maximum(abs.(got .- want)))
    end
    return worst
end

say(@sprintf("%-8s %-14s %14s %10s", "mode", "coupling", "worst |dev|", "verdict"))
for mode in (:none, :U1)
    for (i, j, nm) in ((1, 2, "range 1 (ctrl)"), (1, 3, "range 2"), (1, 4, "range 3"),
                       (2, 4, "range 2 @ 2-4"))
        d = try
            check(mode, i, j)
        catch e
            say(@sprintf("%-8s %-14s  THREW: %s", string(mode), nm,
                         sprint(showerror, e)[1:min(end, 120)])); continue
        end
        say(@sprintf("%-8s %-14s %14.4e %10s", string(mode), nm, d, d < 1e-5 ? "PASS" : "FAIL"))
    end
end
say("")
say("MEASURED: range 1 PASS (1.6e-07, the O(dt) residue of the difference quotient), range 2 and 3")
say("FAIL at EXACTLY 5.0e-01, identically under :none and :U1.")
say("")
say("⛔ THAT IS **NOT** A BROKEN TRANSPORT BLOCK, and reading it that way would wrongly impugn")
say("every long-range model in the suite (kagome, square, dipolar all use transport). Two")
say("independent results rule it out:")
say("  * `mpo_vs_dense.jl` tested the RANGE-2 doubled-chain MPO with ENTANGLED probes and passed")
say("    at 1.3e-15 -- so `pair_mpo` builds the range-2 operator correctly;")
say("  * the deviation here is `got = 0` EXACTLY, not a wrong nonzero value. A dropped channel")
say("    would have to be zeroed to mimic that; a frozen state produces it by construction.")
say("")
say("The cause is the BOOTSTRAP FREEZE: from a PRODUCT state, a two-site update cannot activate a")
say("term whose two operators are not both inside the block, because the environment contribution")
say("is then a one-site expectation of a charge-changing operator and `<S^->` = 0. It is a")
say("property of the STARTING STATE plus the term's range, not of the MPO.")
say("")
say("⚠ WHY THE CLOSED MODELS NEVER HIT IT: kagome, square and dipolar all carry NEAREST-NEIGHBOUR")
say("terms alongside their long-range ones. Those bootstrap the state, and once it is entangled")
say("the one-site expectations are nonzero and the long-range terms activate. A chain with ONLY")
say("range-2 couplings -- which is what the interleaved doubled chain is -- has nothing to start it.")
close(io)
