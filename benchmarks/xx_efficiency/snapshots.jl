# benchmarks/xx_efficiency/snapshots.jl -- step 1 of xx_domain_wall_cbe_bug_efficiency_standalone_plan.md:
# Fixture M. High-accuracy domain-wall references at the predetermined snapshots J t* = 0.5 and 4,
# graded against the exact Slater state (L = 16) or the correlation matrix (L > 16), then
# deliberately compressed in U(1) sectors to modest ranks; at the predetermined bonds L/2 (front)
# and L/2 - 3 each compressed state records ||G_DD||, its DK/KD content and the achieved ranks, and
# a bond with ||G_DD|| ~ 0 is labelled TRIVIAL (plan Sec. 2). The compressed states are serialised
# for the component figures 1-3, which extract their frames from them. Not timed.
#
# Env: XXE_L (16), XXE_TIMES (0.5,4.0), XXE_RANKS (8,16,32), XXE_DT (0.025), XXE_OUT (.)

using LinearAlgebra, Printf, JSON, Serialization
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
const RC = BUGJulia.RSVDCBEBondUpdate

const COMMON = joinpath(@__DIR__, "..", "..", "tests", "common")
for f in ("free_fermion.jl", "spin_ed.jl", "ramped_tdvp.jl", "adaptive_fixtures.jl", "dd_fixtures.jl",
          "xx_efficiency_fixtures.jl")
    include(joinpath(COMMON, f))
end

const L = parse(Int, get(ENV, "XXE_L", "16"))
const TIMES = parse.(Float64, split(get(ENV, "XXE_TIMES", "0.5,4.0"), ","))
const RANKS = parse.(Int, split(get(ENV, "XXE_RANKS", "8,16,32"), ","))
const DT = parse(Float64, get(ENV, "XXE_DT", "0.025"))
const OUT = get(ENV, "XXE_OUT", ".")
const BONDS = (L ÷ 2, L ÷ 2 - 3)
# a bond whose DD action is this small next to its whole two-site action has nothing to find
const TRIVIAL = 1e-8

set_symmetry!(:U1)
mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
basis = L <= 16 ? spin_xx_sector(L, 1)[2] : nothing
mkpath(OUT)
rows = Any[]

"Observable error against `C(t)`: max |<S^z_j> - (C_jj - 1/2)|, valid at every L."
function sz_error(psi, t)
    m = spin_sz_profile(psi, 1)
    return maximum(abs.(m .- (real(diag(xx_open_correlation(L, t))) .- 0.5)))
end

for t in TIMES
    t0 = time()
    ref = xx_reference_state(L, t; dt = DT)
    tref = time() - t0
    eref = basis === nothing ? (err = NaN, err_phase = NaN, fid = NaN) : xx_state_error(ref, t, basis)
    @printf("XXE-REF L=%d t=%.2f bonds=%s err_state=%.3e err_sz=%.3e (%.0f s)\n", L, t,
            string(bond_dims(ref)), eref.err, sz_error(ref, t), tref)
    serialize(joinpath(OUT, "ref_L$(L)_t$(t).jls"), ref)
    for r in RANKS
        snap, disc = xx_snapshot(ref, r)
        name = "snap_L$(L)_t$(t)_r$(r)"
        serialize(joinpath(OUT, name * ".jls"), snap)
        for i in BONDS
            bc = xx_bond_content(copy(snap), mpo, i)
            trivial = bc.dd <= TRIVIAL * max(bc.x, eps())
            row = Dict("L" => L, "t" => t, "rank" => r, "bond" => i, "snapshot" => name,
                       "ref_err_state" => eref.err, "ref_err_sz" => sz_error(ref, t),
                       "compression_discarded" => disc, "bonds" => bond_dims(snap),
                       "old_rank" => bc.old_rank, "disc_l" => bc.disc_l, "disc_r" => bc.disc_r,
                       "norm_X" => bc.x, "G_DD" => bc.dd, "G_DK" => bc.dk, "G_KD" => bc.kd,
                       "G_KK" => bc.kk, "rank_DD" => bc.rank_dd,
                       "dk_kd_over_dd" => trivial ? NaN : (bc.dk + bc.kd) / bc.dd,
                       "trivial" => trivial)
            push!(rows, row)
            @printf("XXE-BOND t=%.2f rank=%d bond=%d old_rank=%d disc=(%d,%d) |X|=%.3e |G_DD|=%.3e |G_DK|=%.3e |G_KD|=%.3e rank_DD=%d disc_norm=%.2e%s\n", t, r, i, bc.old_rank,
                    bc.disc_l, bc.disc_r, bc.x, bc.dd, bc.dk, bc.kd, bc.rank_dd, disc,
                    trivial ? " TRIVIAL" : "")
        end
    end
    open(joinpath(OUT, "snapshots_L$(L).json"), "w") do io
        JSON.json(io, Dict("rows" => rows); allownan = true)
    end
end
println("XXE-SNAPSHOTS-DONE $(length(rows)) rows -> $OUT")
