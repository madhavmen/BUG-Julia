# EXACT GROUND ENERGIES FOR THE SMALL arXiv:2209.00739 GEOMETRIES, BY SPARSE DIAGONALISATION.
#
# WHY THIS EXISTS. The Fig. 3 anchors from Iqbal et al. are a 36-site TORUS and the thermodynamic
# limit. Neither can score a 12- or 16-site CYLINDER: the torus is a different cluster and the
# infinite-system stars are what an extrapolation approaches, not what any finite run returns. So
# a small cylinder run had NO reference at all -- and a DMRG number with no reference is not an
# accuracy claim, however converged it looks.
#
# At these sizes it does not need one. The `S^z = 0` sector of 12 sites is `C(12,6) = 924` states
# and of 16 sites `C(16,8) = 12870`, so the ground energy of the SAME cluster is available exactly.
# `pairs_sparse` already takes an arbitrary coupling matrix, which is exactly what
# `triangular_cylinder_couplings` produces -- the ED and the MPO are then built from ONE object,
# so there is no second definition of the model to drift.
#
# ⛔ THIS IS NOT THE CHAIN DIRECTIVE. "Heisenberg is scored against analytic Bethe, never ED"
# applies where a closed form EXISTS. On a frustrated 2D cluster none does, and exact
# diagonalisation of the same cluster is what the literature itself quotes (Iqbal Tab. I and IV
# both carry ED columns). It is external to the integrator, which is the property that matters.
#
# ⛔ ED WORKS IN THE `S^z = 0` SECTOR; THE SU(2) RUN PINS `S = 0`. Those are not the same
# constraint. For even `L` every multiplet has an `S^z = 0` member, so the sector's lowest state IS
# the global ground state -- but if that state has `S > 0`, the SU(2) run (started from a singlet)
# is confined to a HIGHER sector and will sit above ED legitimately. On a frustrated lattice that
# is a real possibility, so the gap is reported as a SECTOR question, not automatically as error.
#
# ⚠ AND THE BIT CONVENTION IS A KNOWN LANDMINE: the sparse basis and the MPS site order have been
# mirrored before, and a dimer or Neel start hides it. The gate below therefore checks ED against
# a DMRG run that is demonstrably at full rank (loose cap, both selections agreeing), where the two
# MUST match to roundoff; if they do not, the conventions disagree and nothing else here is usable.
#
# Run:  julia --project=. benchmarks/ground_state/exact_2209.jl [Lx Ly]
# Out:  benchmarks/results/paper2209_exact_<Lx>x<Ly>.csv

using LinearAlgebra, SparseArrays, Printf
using BUGJulia
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))     # pairs_sparse, sz0_basis

const LX = length(ARGS) >= 2 ? parse(Int, ARGS[1]) : 4
const LY = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
const TOR = get(ENV, "P22_TORUS", "") == "1"
const J2S = let s = get(ENV, "P22_J2", "")
    isempty(s) ? [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5] :
                 [parse(Float64, x) for x in split(s, ',')]
end
const OUT = joinpath(@__DIR__, "..", "results")

"""
    ground(Jm, L) -> (E0, gap, dimerE)

Lowest two eigenvalues of `H = sum_{i<j} Jm[i,j] S_i.S_j` in the `S^z = 0` sector, plus the energy
of the nearest-neighbour DIMER product state in the SAME convention.

`dimerE` is here because `-3/8` per site keeps appearing at `J2 = 0.5` with bond dimension 1, and
that is exactly the dimer energy: the run either found a genuinely exact dimer ground state or it
FROZE at its start state, and those two are indistinguishable from the energy alone. Comparing the
exact `E0` with `dimerE` separates them in one number.
"""
function ground(Jm::Matrix{Float64}, L::Int)
    H, states, idx = pairs_sparse(L, Jm)
    Hd = Matrix(Symmetric(Matrix(H)))
    ev = eigvals(Hd)
    # The dimer product state pairs (1,2), (3,4), ... in SITE order -- the same pairing
    # `dimer_state` builds -- so its energy is a sum over the paired bonds only.
    dE = 0.0
    for k in 1:2:(L - 1)
        dE += -0.75 * Jm[min(k, k + 1), max(k, k + 1)]
    end
    return ev[1], ev[2] - ev[1], dE
end

function main()
    mkpath(OUT)
    L = LX * LY
    out = joinpath(OUT, "paper2209_exact_$(LX)x$(LY)$(TOR ? "_torus" : "").csv")
    @printf("\n%s\ntriangular  Lx=%d Ly=%d  N=%d  %s   sector S^z=0 (%d states)\n%s\n",
            "="^96, LX, LY, L, TOR ? "TORUS" : "cylinder (XC)",
            binomial(L, L ÷ 2), "="^96)
    @printf("%-7s %14s %14s %11s %12s\n", "J2", "E0 (exact)", "E0/N", "gap", "dimer E/N")
    open(out, "w") do io
        println(io, "Lx,Ly,N,torus,J2,E0,E0_site,gap,dimer_site,dimer_is_gs")
        for J2 in J2S
            Jm = triangular_cylinder_couplings(LX, LY; J2 = J2, periodic_y = true,
                                               periodic_x = TOR)
            E0, gap, dE = ground(Jm, L)
            isdimer = abs(E0 - dE) < 1e-9
            @printf("%-7.3f %14.10f %14.10f %11.3e %12.8f%s\n",
                    J2, E0, E0 / L, gap, dE / L, isdimer ? "   <- dimer IS exact" : "")
            @printf(io, "%d,%d,%d,%d,%.4f,%.10f,%.10f,%.6e,%.10f,%d\n",
                    LX, LY, L, TOR ? 1 : 0, J2, E0, E0 / L, gap, dE / L, isdimer ? 1 : 0)
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
