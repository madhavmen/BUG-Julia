# benchmarks/hq_xx_reference.jl -- the L = 16 domain-wall references of the HQ plan, Task 7.
#
# ONE SETTING PER RUN of an independent reference solver (TDVP2, `tdvp2_composed_step!`, never
# the CBE-BUG under study) on the open spin-s XX chain `spin_xx_mpo` from
# `spin_domain_wall_state`, to `J t = HQ_TF`. Records every `HQ_EVERY` (in time units):
# `<S^z_i>`, the centre-bond current, `<H>`, `Var(H)`, the norm, `S^z_tot`, the max bond and the
# summed discarded weight. `benchmarks/hq_xx_reference_band.jl` combines several settings into
# the converged reference and its uncertainty band (plan T7.2c, T7.4).
#
# s = 1/2 is ALSO graded against the closed form (tests/common/free_fermion.jl) on every record:
# that is the check that a setting which converges for s = 1/2 is a converged solver at all.
# s = 3 has no closed form -- its reference is the spread across settings, never "analytical".
#
# Frozen conventions (plan T7.4): open chain, J = 1, H = J sum (SxSx + SySy), physical index
# order 1..L left to right, sites 1..L/2 at S^z = +s, U(1) charge 2 S^z, total sector 0.
#
# Env: HQ_TWOS (6), HQ_L (16), HQ_TF (8.0), HQ_EVERY (0.1), HQ_ORDER (4), HQ_DT (0.025),
#      HQ_MAXDIM (256), HQ_TRUNC (1e-10), HQ_OUT (hq_xx_ref_<twoS>_<order>_<dt>_<maxdim>_<trunc>.json)

using LinearAlgebra, Printf, JSON
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

const TWOS = parse(Int, get(ENV, "HQ_TWOS", "6"))
const L = parse(Int, get(ENV, "HQ_L", "16"))
const TF = parse(Float64, get(ENV, "HQ_TF", "8.0"))
const EVERY = parse(Float64, get(ENV, "HQ_EVERY", "0.1"))
const ORDER = parse(Int, get(ENV, "HQ_ORDER", "4"))
const DT = parse(Float64, get(ENV, "HQ_DT", "0.025"))
const MAXDIM = parse(Int, get(ENV, "HQ_MAXDIM", "256"))
const TRUNC = parse(Float64, get(ENV, "HQ_TRUNC", "1e-10"))
const OUT = get(ENV, "HQ_OUT", "hq_xx_ref_$(TWOS)_$(ORDER)_$(DT)_$(MAXDIM)_$(TRUNC).json")

TWOS == 1 && include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

set_symmetry!(:U1)
const S = TWOS / 2
const MPO_ = spin_xx_mpo(L, TWOS)
const JMPO = spin_current_mpo(L, TWOS, L ÷ 2)
const NSTEP = round(Int, TF / DT)
const KREC = round(Int, EVERY / DT)
abs(NSTEP * DT - TF) < 1e-12 && abs(KREC * DT - EVERY) < 1e-12 ||
    error("HQ_TF and HQ_EVERY must be multiples of HQ_DT")

psi = spin_domain_wall_state(L, TWOS)
sector = psi[L].spaces[3]
recs = Any[]
record(t, disc, wall) = begin
    m = spin_sz_profile(psi, TWOS)
    r = Dict{String, Any}("t" => t, "m" => m, "j" => real(mpo_energy(psi, JMPO)),
                          "E" => real(mpo_energy(psi, MPO_)), "var" => mpo_variance(psi, MPO_),
                          "norm" => real(overlap(psi, psi)), "sz" => sum(m),
                          "sector" => psi[L].spaces[3] == sector, "D" => maximum(bond_dims(psi)),
                          "bonds" => bond_dims(psi), "discarded" => disc, "wall" => wall)
    if TWOS == 1
        C = xx_open_correlation(L, t)
        r["err_exact_m"] = maximum(abs.(m - (real(diag(C)) .- 0.5)))
        r["err_exact_j"] = abs(r["j"] - xx_open_current(C)[L ÷ 2])
    end
    push!(recs, r)
    @printf("HQREF t=%.3f D=%d m_c=%.12f j=%.12f E=%.2e dvar=%.2e dnorm=%.2e disc=%.2e%s wall=%.1f\n",
            t, r["D"], m[L ÷ 2], r["j"], r["E"], r["var"] - S^2, r["norm"] - 1, disc,
            TWOS == 1 ? @sprintf(" exact_m=%.2e exact_j=%.2e", r["err_exact_m"], r["err_exact_j"]) : "",
            wall)
    flush(stdout)
end

println("HQREF-CONFIG twoS=$TWOS L=$L tf=$TF dt=$DT order=$ORDER maxdim=$MAXDIM trunc=$TRUNC")
t0, disc = time(), 0.0
record(0.0, disc, 0.0)
for k in 1:NSTEP
    info = tdvp2_composed_step!(psi, MPO_, ComplexF64(-1im * DT); order = ORDER,
                                maxdim = MAXDIM, trunc_thresh = TRUNC)
    global disc += info.cumulative
    k % KREC == 0 && record(k * DT, disc, time() - t0)
end
open(OUT, "w") do io
    JSON.print(io, Dict("config" => Dict("twoS" => TWOS, "L" => L, "tf" => TF, "dt" => DT,
                                         "order" => ORDER, "maxdim" => MAXDIM, "trunc" => TRUNC,
                                         "solver" => "tdvp2_composed_step!", "J" => 1.0),
                        "records" => recs))
end
println("HQREF-DONE $OUT wall=$(round(time() - t0; digits = 1)) s")
