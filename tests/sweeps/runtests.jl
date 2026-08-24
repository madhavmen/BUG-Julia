# tests/sweeps/runtests.jl
#
# The structured RSVD-CBE sweeps: the models they run on, the sweeps themselves, the
# cross-symmetry requirement, and the physics against analytic references.
#
# Run with:  julia --project=. tests/sweeps/runtests.jl
#        or: sbatch scripts/run_julia.sbatch tests/sweeps/runtests.jl
#
# ORDERING IS THE DESIGN. Each file's reference is something the file before it established, so a
# red test localises instead of leaving several suspects:
#
#   1  analytic references   pinned against exact diagonalisation. FIRST, because every physics
#                            test below compares an integrator against these -- if a formula here
#                            is wrong, all of those tests are wrong in the same direction and
#                            agree with each other, which is the worst failure mode there is.
#   2  pair_mpo, tfim_mpo    the operators, pinned against `mpo_from_terms` (layout), a dense
#                            pair sum (charged transport) and a dense 2^L TFIM. No integrator
#                            involved in either -- a wrong operator would make every sweep below
#                            wrong in the same direction, so the operators are settled first.
#   3  sweeps                the three sweeps, each against something already trusted.
#   4  symmetry parity       the same physics under :none / :U1 / :SU2.
#   5  physics               Haldane-Shastry and Heisenberg, GSE / ITE / RTE, against the
#                            analytic references of step 1.
#
# Kept separate from `tests/RSVDCBEBondUpdate/runtests.jl` and from `tests/runtests.jl`, which
# cover the earlier exploratory module and the validated `BondUpdateBUG` respectively. A red test
# here must never be mistaken for a regression in either.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random
Random.seed!(42)

include(joinpath(@__DIR__, "..", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "common", "analytic_reference.jl"))
# The free-fermion closed forms (XX hopping, TFIM Majorana). Integrator tests compare against
# THESE and never against a dense propagator -- a 2^L reference caps the test at the L where it
# fits, which is routinely not the L where the effect under test is large.
include(joinpath(@__DIR__, "..", "common", "free_fermion.jl"))

@testset "sweeps (RSVD-CBE: DMRG, 1-site TDVP, fixed-rank BUG)" begin
    include(joinpath(@__DIR__, "..", "common", "test_analytic_reference.jl"))
    include("test_pair_mpo.jl")
    include("test_tfim_mpo.jl")
    include("test_sweeps.jl")
    # 3b the sweep itself, pinned against EXACT references on three
    #    models at L=12. `test_sweeps.jl` only checks it reproduces `cbe_lubich_sweep`, which is
    #    an agreement between two of our own implementations; this checks it against the answer.
    include("test_cbe_bug_exact_refs.jl")
    include("test_symmetry_parity.jl")
    include("test_physics.jl")
    # 6  one-axis twisting     the arXiv:2208.10972 Fig. 2 benchmark, at L = 10. LAST because it
    #                          is the strongest test and the most specific: OAT's `H` is diagonal
    #                          in the computational basis, so the evolved state, its Schmidt
    #                          spectrum and its entropy are ALL closed forms -- no magnon sector,
    #                          no free-fermion profile, no sparse propagation. It runs under
    #                          `:none` (the observable `S^x_tot` is charge-raising, so a U(1) run
    #                          would report a silent zero), which is also why it sits after the
    #                          symmetry-parity file rather than inside it.
    include("test_oat.jl")
end

# The symmetry mode is a GLOBAL, and several files above change it. Restore the package default
# so a later `include` of another suite in the same session is not silently run under SU(2).
set_symmetry!(:U1)
