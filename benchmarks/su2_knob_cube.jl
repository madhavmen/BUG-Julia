# THE THREE-KNOB CUBE UNDER SU(2), FROM A DIMER START.
#
# Same cube as `heisenberg_tolerance.jl phase=grid` -- half-sweep `split_cutoff` x root-out
# `trunc_thresh` x Krylov depth `m` -- but under the non-abelian symmetry, which changes THREE
# things that each silently break a naive port:
#
# ⛔ 1. NEEL IS NOT SU(2)-REPRESENTABLE. A definite-`Sz` product state spans many total-spin
#       sectors, so the U(1) cube's start cannot be used. `dimer_state` is the representable
#       `S = 0` start, and the U(1) arm has to be RE-RUN on that same start or the two are not
#       the same physics.
#
# ⛔ 2. THE Sz PROFILE IS IDENTICALLY ZERO AND THEREFORE USELESS. A dimer has `<S^z_j> = 0` on
#       every site, and Heisenberg evolution is SU(2)-symmetric so it STAYS zero. The U(1) cube's
#       observable would report a flawless 0.000e+00 for every arm at every tolerance -- a
#       measurement that cannot fail and says nothing. The observable here is the BOND ENERGY
#       profile `<S_i . S_{i+1}>`, which is non-trivial on a dimer and is built through
#       `pair_mpo` so it is literally the same operator in every symmetry mode.
#
# ⛔ 3. `maxdim` COUNTS MULTIPLETS. A rank reported under SU(2) is not comparable to one under
#       U(1) unless it is converted, so `maxbond` in the CSV is `state_bond_dims` (STATES) in
#       both modes, and the multiplet count is carried in its own column. Measured at L=8 in
#       `model_ladder.jl`: 6 multiplets carrying 16 states, against U(1)'s 16 and 16.
#
# THE REFERENCE IS EXACT. `dimer_vector` is the same state in the `Sz = 0` sector, propagated by
# `expv_sparse`, and `bond_energies_dense` is the matched observable -- no second integrator, and
# nothing symmetry-specific in the reference at all.
#
# Emits the SAME CSV columns as `heisenberg_tolerance.jl`, so `plot_knob_cube.py` reads it
# unchanged.
#
# Run:  julia --project=. benchmarks/su2_knob_cube.jl [L] [sym] [t_max]

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const L     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
const SYM   = length(ARGS) >= 2 ? Symbol(ARGS[2])     : :SU2
const TMAX  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.5
const DT    = 0.05
const CAP   = 64
const MI    = 16
const TAUS  = [1e-4, 1e-6, 1e-8, 1e-10]
const SPLIT = [1e-4, 1e-6, 1e-8, 1e-10]
const DEPTHS = ("m0" => 0, "m2" => 2, "m3" => 3, "mdef" => -1)   # -1 = package defaults

const COLS = ["phase", "scheme", "L", "dt", "tau_trunc", "split_cutoff", "cbe_cutoff",
              "root_trunc", "close_trunc", "maxdim", "t", "stag", "stag_exact", "err_stag",
              "err_prof", "maxbond", "norm", "energy", "dE", "err_fnl", "discarded",
              "krylov", "seconds", "multiplets"]

const OUTDIR = joinpath(@__DIR__, "results")
mkpath(OUTDIR)

set_symmetry!(SYM)
const H_ONE = SYM === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; J = 1.0, delta = 1.0)
const W     = RSVDCBEBondUpdate.mpo_from_terms(H_ONE)
const PSI0  = dimer_state(L)

"`<S_i . S_{i+1}>` for every bond, as single-entry `pair_mpo`s -- one operator in every mode."
function bond_profile(psi)
    out = zeros(Float64, L - 1)
    for i in 1:(L - 1)
        Jm = zeros(Float64, L, L); Jm[i, i + 1] = 1.0
        out[i] = real(mpo_energy(copy(psi), pair_mpo(L, Jm, spin_vertices(L))))
    end
    return out ./ max(norm(copy(psi))^2, eps())
end

# ── the exact reference, in the Sz = 0 sector ─────────────────────────────────────────────
const HSP, STATES, IDX = heisenberg_sparse(L)
const V0 = dimer_vector(L, STATES, IDX)
const SAMPLES = collect(0.0:0.25:TMAX)
const REF = Dict(t => bond_energies_dense(
                        t == 0 ? copy(V0) : expv_sparse(HSP, ComplexF64(-im * t), V0;
                                                        m = 30, tol = 1e-13),
                        L, STATES, IDX) for t in SAMPLES)

@printf("SU(2) cube: L=%d sym=%s T=%g  dimer start, bond-energy observable\n", L, SYM, TMAX)
@printf("exact reference: %d states in the Sz=0 sector\n", length(STATES)); flush(stdout)

for (nm, m) in DEPTHS
    path = joinpath(OUTDIR, @sprintf("heis_grid_%s_L%d_dt%g_T%g.csv", nm, L, DT, TMAX))
    # A DISTINCT prefix per symmetry, or the U(1) cube's CSVs and these overwrite each other and
    # the plot silently mixes two different symmetry modes into one panel.
    path = replace(path, "heis_grid_" => "heis_grid$(SYM === :SU2 ? "su2" : "u1")_")
    io = open(path, "w"); println(io, join(COLS, ","))
    try
        for tt in TAUS, sc in SPLIT
            kw = m < 0 ? (;) : (krylov_basis = m, krylov_tol = 0.0)
            psi, kry, secs, efnl = copy(PSI0), 0, 0.0, 0.0
            e0 = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
            emit(t) = begin
                p = bond_profile(psi)
                e = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
                @printf(io, "grid,cbe_bug_%s,%d,%g,%g,%g,%g,%d,%d,%d,%g,%.10g,%.10g,%.6e,%.6e,%d,%.10g,%.10g,%.6e,%.3e,%.3e,%d,%.2f,%d\n",
                        nm, L, DT, tt, sc, 0.0, 0, 1, CAP, t, 0.0, 0.0, 0.0,
                        maximum(abs.(p .- REF[t])), maximum(state_bond_dims(psi)),
                        norm(psi), e, abs(e - e0), efnl, 0.0, kry, secs,
                        maximum(bond_dims(psi)))
                flush(io)
            end
            emit(0.0)
            for k in 1:round(Int, TMAX / DT)
                t0 = time_ns()
                info = cbe_bug_step!(psi, W, ComplexF64(-im * DT);
                                     exact = true, split_cutoff = sc, maxdim = CAP,
                                     trunc_thresh = tt, maxiter = MI, kw...)
                secs += (time_ns() - t0) / 1e9
                kry += info.krylov_dims
                efnl = max(efnl, info.err_fnl)
                t = k * DT
                any(isapprox(t, s; atol = 1e-9) for s in SAMPLES) && emit(t)
            end
            @printf("  %-5s tau=%.0e split=%.0e -> err=%.3e states=%d mult=%d kry=%d\n",
                    nm, tt, sc, maximum(abs.(bond_profile(psi) .- REF[TMAX])),
                    maximum(state_bond_dims(psi)), maximum(bond_dims(psi)), kry)
            flush(stdout)
        end
    finally
        close(io)
    end
end
println("wrote to ", OUTDIR)
