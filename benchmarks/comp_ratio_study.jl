# Which candidate directions actually earn their place: discarded vs kept space.
#
#   sbatch --job-name=cratio scripts/run_julia.sbatch benchmarks/comp_ratio_study.jl
#
# `sector_graded_sketch` splits the sketch between the COMPLEMENT (discarded) space and the
# ISOMETRY (kept) space, in the ratio `comp_ratio` -- the reference's `CompRatio`
# (`RSVDpreBE0SiQS.m:339`: "do not project into complement space, 1-site components are also
# important for bond update"). The default 0.5 was inherited, never measured.
#
# WHY IT MATTERS HERE. The rank is throttled by a cascade: bonds 2 and 6 propose a new
# direction but KEEP only 1, because the proposal carries no weight; `room_l = chi_{i-1}*d - r`
# then caps the next bond in, and the centre saturates at 4 while 2-site TDVP needs 24 at the
# same time. If the wasted proposals are the kept-space half, comp_ratio -> 1.0 should widen
# the bonds; if they are the complement half, the opposite.
#
# comp_ratio = 1.0 is "complement only", 0.0 "isometry only". Reported against the exact
# free-fermion profile, so the accuracy number is absolute rather than relative to a
# reference simulation.

using LinearAlgebra, Printf
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include("../tests/common/free_fermion.jl")

const L    = 18
const DT   = 0.05
const TMAX = 2.0
const MAXD = 64

exact(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:(L ÷ 2))
function perr(psi, t)
    p = copy(psi); n2 = norm(p)^2
    return maximum(abs.(magnetisation(p) ./ n2 .- exact(t)))
end

function run(; comp_ratio, dex, sulz_cap)
    set_symmetry!(:U1)
    psi = domain_wall_state(L)
    h = xxz_chain(L; delta = 0.0)
    nsteps = round(Int, TMAX / DT)
    t0 = time()
    for _ in 1:nsteps
        cbe_lubich_sweep(psi, h, -im * DT; comp_ratio = comp_ratio, dex = dex,
                            sulz_cap = sulz_cap, maxdim = MAXD, trunc_thresh = 1e-12)
    end
    return perr(psi, TMAX), bond_dims(psi), time() - t0
end

println("# L=$L XX U(1) domain wall, t=$TMAX, dt=$DT, maxdim=$MAXD")
println("# 2-site TDVP reference at these settings: err 5.18e-05, maxbond 25")
println()
@printf("%-6s %-5s %-5s %10s %8s %7s  %s\n",
        "cratio", "dex", "sulz", "err", "maxbond", "wall s", "bond_dims")
println(repeat("-", 104))

# 1. the ratio, at the current budget
for cr in (1.0, 0.75, 0.5, 0.25, 0.0)
    e, bd, w = run(comp_ratio = cr, dex = 0, sulz_cap = true)
    @printf("%-6.2f %-5s %-5s %10.3e %8d %7.1f  %s\n",
            cr, "r", "on", e, maximum(bd), w, string(bd))
    flush(stdout)
end

# 2. Separate the RATIO from the BUDGET. Part 1 uses the reference's neighbourhood-coupled
#    default (`ceil(1.1*Dmax) - r`); here a fixed generous `dex` isolates whether any
#    remaining gap is candidate QUALITY (the ratio) or candidate COUNT (the budget).
println()
for (cr, dex, sc) in ((1.0, 8, false), (0.5, 8, false), (1.0, 16, false), (0.5, 16, false))
    e, bd, w = run(comp_ratio = cr, dex = dex, sulz_cap = sc)
    @printf("%-6.2f %-5d %-5s %10.3e %8d %7.1f  %s\n",
            cr, dex, "off", e, maximum(bd), w, string(bd))
    flush(stdout)
end

println("\nCRATIO_DONE")
