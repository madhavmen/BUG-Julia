# WHERE DOES THE RANDOMISED SKETCH ACTUALLY PAY? 2D LATTICES AT LARGE CHI.
#
# Every previous measurement in this repo found rSVD NO FASTER, and one found it SLOWER
# (0.88-0.92x on U(1) Heisenberg L=12). That was not a defect in the sketch -- it is arithmetic:
#
#     budget = dex <= 0 ? ceil(growth*dmax) - r : dex        (cbe_core.jl)
#     npre   ~ 1.2 * dex
#
# At the shipped `growth = 2.0` a bond at rank `chi` gets `budget ~ chi`, so `npre ~ 1.2*chi`
# against a full local space of `d*chi = 2*chi`. THE SKETCH IS 60% OF FULL. A constant fraction
# cannot beat a constant no matter how large chi gets, and the overhead then makes it lose.
#
# ⛔ SO THE SPEEDUP NEEDS A FIXED `dex`, NOT A `growth` FRACTION. With `dex` fixed the ratio is
#
#     npre / (d*chi) ~ 1.2*dex / (2*chi)  ->  0   as chi grows
#
# which at chi = 128, dex = 8 is 3.7% -- a potential ~26x on the sketched contraction. 2D lattices
# are the first place in this study where chi is large enough for that limit to be reachable, AND
# where the MPO bond dimension is large enough for the contraction to dominate.
#
# ⛔ AND A FIXED `dex` IS NOT A HANDICAP. This repo's own budget measurement (cbe_core.jl, XX
# L=10) found `dex = 8` giving err 1.17e-06, IDENTICAL to dex = 16 and dex = 64 -- the accuracy
# saturates in the budget well below the growth schedule's width. The growth schedule is buying
# nothing at the top end and paying the sketch's whole advantage away for it.
#
# ⚠ WALL CLOCK IS THE MEASUREMENT HERE, so this must be the ONLY heavy job running. Operator
# applications cannot answer this question at all: the sketch changes the SIZE of the contraction,
# not the NUMBER of them, so `krylov` is identical between the two arms by construction. That is
# the same blindness that hid the retired `rexpand`'s cost and SU(2)'s benefit.
#
# ⚠ AND THE PREDICTION IS REPORTED NEXT TO THE MEASUREMENT. If the speedup does not track
# `(d*chi)/npre`, the mechanism above is wrong and the number is a coincidence.
#
# Run:  julia --project=. benchmarks/rsvd_speedup_2d.jl [model] [maxdim] [t_max]
#         model = square | kagome | both

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const WHICH = length(ARGS) >= 1 ? ARGS[1] : "both"
const CAP   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const TMAX  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
const DT    = 0.05
const MI    = 16

const OUT = joinpath(@__DIR__, "results", "rsvd_speedup_2d.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
const CSV = open(joinpath(@__DIR__, "results", "rsvd_speedup_2d.csv"), "w")
println(CSV, "model,dex,exact,err,states,krylov,secs,mpodim")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:U1)

function setup(nm)
    if nm == "square"
        LX, LY = 6, 4                       # 24 sites: chi can run well past the 16-site ceiling
        L = LX * LY
        Jm = square_cylinder_couplings(LX, LY)
        W  = square_cylinder_mpo(LX, LY)
        vertices = spin_vertices(L); heis = true
    else
        LX, LY = 3, 3                       # 27 sites, dipolar tail -> a WIDE MPO
        L = 3 * LX * LY
        Jm = breathing_kagome_couplings(LX, LY; h = 0.30)
        W  = breathing_kagome_mpo(LX, LY; h = 0.30)
        vertices = xxz_chain(L; J = 1.0, delta = 0.0); heis = false
    end
    bonds = [(i, j) for i in 1:L for j in (i + 1):L if Jm[i, j] != 0.0]
    probe = bonds[1:max(1, length(bonds) ÷ 16):end]
    ops = [begin
               A = zeros(Float64, L, L); A[i, j] = 1.0
               pair_mpo(L, A, vertices)
           end for (i, j) in probe]
    return (nm = nm, L = L, W = W, ops = ops, psi0 = neel_state(L), heis = heis)
end

profile(S, psi) = (n2 = max(norm(copy(psi))^2, eps());
                   [real(mpo_energy(copy(psi), op)) / n2 for op in S.ops])

function run(S; dex, exact)
    psi, kry, secs = copy(S.psi0), 0, 0.0
    for _ in 1:round(Int, TMAX / DT)
        t0 = time_ns()
        info = cbe_bug_step!(psi, S.W, ComplexF64(-im * DT);
                             exact = exact, dex = dex, krylov_basis = 3, krylov_tol = 0.0,
                             maxdim = CAP, trunc_thresh = 1e-10, maxiter = MI)
        secs += (time_ns() - t0) / 1e9
        kry += info.krylov_dims
    end
    return (prof = profile(S, psi), states = maximum(state_bond_dims(psi)),
            kry = kry, secs = secs)
end

for nm in (WHICH == "both" ? ("square", "kagome") : (WHICH,))
    S = setup(nm)
    mpod = maximum(mpo_virtual_dims(S.W))
    say("")
    say("="^104)
    say(@sprintf("%s -- %d sites, MPO virtual dim %d, maxdim %d, T %g",
                 uppercase(nm), S.L, mpod, CAP, TMAX))
    say("="^104)
    say("  A WIDE MPO IS THE POINT: the sketched contraction cost scales with it, so this is")
    say("  where shrinking the number of probe columns can actually show up in wall clock.")
    say("")
    say(@sprintf("  %-22s %10s %8s %9s %9s %11s %10s",
                 "arm", "secs", "states", "krylov", "speedup", "predicted", "agree"))
    ref = nothing
    for dex in (0, 32, 16, 8, 4)      # 0 = the shipped growth schedule
        base = nothing
        for exact in (true, false)
            r = run(S; dex = dex, exact = exact)
            ref === nothing && (ref = r.prof)
            lbl = (dex == 0 ? "growth=2.0" : "dex=$dex") * (exact ? " full SVD" : " rSVD")
            if exact
                base = r
                say(@sprintf("  %-22s %10.1f %8d %9d %9s %11s %10s",
                             lbl, r.secs, r.states, r.kry, "-", "-", "-"))
            else
                # the mechanism's own prediction: full local space over sketch width
                npre = dex == 0 ? 1.2 * r.states : 1.2 * dex
                pred = (2 * r.states) / max(npre, 1)
                say(@sprintf("  %-22s %10.1f %8d %9d %8.2fx %10.2fx %10.2e",
                             lbl, r.secs, r.states, r.kry, base.secs / r.secs, pred,
                             maximum(abs.(r.prof .- base.prof))))
            end
            @printf(CSV, "%s,%d,%s,%.8e,%d,%d,%.3f,%d\n", nm, dex, exact,
                    maximum(abs.(r.prof .- ref)), r.states, r.kry, r.secs, mpod)
            flush(CSV)
        end
    end
    say("")
    say("  READ THE `predicted` COLUMN. If the measured speedup tracks (d*chi)/npre, the")
    say("  mechanism is confirmed. If it does not, the sketch is winning or losing for some")
    say("  other reason and the explanation above is wrong.")
    say("  `agree` is |rSVD profile - full-SVD profile|: the sketch must not change the ANSWER.")
end
close(IO_); close(CSV)
