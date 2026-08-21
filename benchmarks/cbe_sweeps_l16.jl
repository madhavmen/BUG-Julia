# L = 16, T = 5: the Phase-2 smoke tests turned into a measured campaign.
#
# Six integrators, three models, three modes, and ANALYTIC references throughout:
#
#   integrators   cbe_bug (decoupled), cbe_bug (interleaved), tdvp_cbe1s, tdvp2,
#                 jan_bug, jan_bug_centre -- the first two are the new sweep's two expansion
#                 orderings, the rest are what it is being compared against.
#   modes         GSE (dmrg_cbe1s!), ITE (tau = -dt), RTE (tau = -i dt)
#
#   ⛔ EVERY REFERENCE IS A CLOSED-FORM EXPRESSION. No dense propagation, no sparse Krylov, no
#   diagonalisation -- those are other COMPUTATIONS, and agreeing with one is weaker evidence
#   than agreeing with a formula. It is also what lets this scale: the largest object any
#   reference here builds is `L x L`, never `2^L`.
#
#     GSE  open Heisenberg chain, SU(2)  -> OPEN-CHAIN BETHE ANSATZ (`bethe_xxx_open_energy`,
#                                          validated against ED to 4e-15 for L = 4..14)
#     ITE  open Heisenberg chain, Néel   -> the same Bethe ground energy
#     RTE  XX chain, domain wall         -> FREE FERMIONS. `Σ(SxSx+SySy)` is quadratic after
#                                          Jordan-Wigner, so `⟨Sz_j(t)⟩` is closed-form and the
#                                          comparison is against an expression, not a propagation.
#     RTE  Haldane-Shastry, `S^z_{j0}|GS⟩` -> the closed form `-π²(L²+5)/(24L)` for the initial
#                                          ground state, then ENERGY CONSERVATION (exact) in time.
#
# WHY THERE IS A dt-SCAN PHASE. Measured at L=8 (scratchpad/probe_phase.jl): at maxdim=8 ALL FOUR
# integrators land on err = 4.0425e-6 with a dt exponent of 0.0, and at maxdim=4 on 3.9271e-3,
# again 0.0. At fixed rank the error is set by the RANK and is dt-independent, so an "error vs dt"
# curve measured there has no slope to report -- and at FULL rank these schemes are exact for any
# dt (one step is `exp(tau H)` on the complete space), so there is no slope there either. The
# temporal order is only visible in the window where the dt error exceeds the rank floor. Phase
# `dtscan` finds that window by sweeping dt over two decades at fixed rank instead of assuming it.
#
# Output is STREAMED CSV, one row per sample, so a killed run still yields everything up to the
# kill -- these runs are long and the alternative is losing all of it.
#
# usage:  julia --project=. benchmarks/cbe_sweeps_l16.jl [phase] [L] [T] [dt]
#         phase in (all | main | dtscan | gse | ite | rte)
#         main = gse+ite+rte (the deliverable); dtscan is a separate diagnostic at L=16, T=0.5

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

const HERE = @__DIR__
# ⛔ `exact_sparse.jl` is NOT included. It supplies `expv_sparse` / `ground_energy_sparse` /
# `hs_coupling_matrix`, and nothing here consumes any of them any more: every reference is a
# closed-form expression. The include lingered because `model()` returned a coupling matrix that
# all three call sites immediately discarded.
include(joinpath(HERE, "..", "tests", "common", "analytic_reference.jl"))
# The EXACT XX solution: Jordan-Wigner to free fermions, so `⟨Sz_j(t)⟩` is a closed-form
# expression involving nothing bigger than an `L x L` matrix exponential. This is the real-time
# reference -- see `phase_evolve`.
include(joinpath(HERE, "..", "tests", "common", "free_fermion.jl"))

const OUTDIR = joinpath(HERE, "results")
const PHASE = isempty(ARGS) ? "all" : ARGS[1]
const L     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const TMAX  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5.0
const DT    = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.05

# ⛔ RANK IS SET BY THE TOLERANCE, NOT BY A CAP. The first campaign ran at maxdim = 32/64 and
# every scheme hit the cap on the first step, so all bond-growth curves were flat lines at the
# cap and every scheme did identical fixed-size work. That makes the ERROR floor identical
# (measured: all four within 0.2%) and the wall-clock comparison a measure of cost-per-step at
# pinned rank -- NOT cost to reach an accuracy, which is the only fair comparison between
# adaptive-rank integrators. `MAXDIM` is now a safety net that should never bind (full rank at
# the L=16 centre is 256), and `CUTOFFS` is what actually decides the rank each step.
const MAXDIM = 256
# ⛔ THE LADDER MUST BRACKET WHATEVER ELSE LIMITS THE ERROR, or it measures nothing.
# MEASURED on the beta=20 ITE arm: the tail decay rate is 0.4384 for every scheme at BOTH 1e-8 and
# 1e-10, constant to four digits and still not flattening. The error there is residual
# excited-state contamination from the Neel start, `0.385 exp(-0.4384 beta)` -- it is
# BETA-LIMITED, and the cutoff is not the binding constraint. At beta=20 the error is 6e-5, about
# 6000x above the 1e-8 cutoff; the fit says 1e-8 would not bind until beta ~ 40 and 1e-10 until
# beta ~ 50. So both arms land on the same curve and the only difference they show is cost:
# tol=1e-10 spends ~47% more rank (chi 94 vs 64) for a 0.4% WORSE prefactor.
#
# Two ways to make the ladder informative, and the cheap one is the second:
#   (a) push beta to ~50, which is 2.5x the steps -- a cluster job, per the local/cluster split;
#   (b) keep beta=20 and move the cutoffs to STRADDLE the 6e-5 floor, which costs LESS than the
#       current run because the ranks are smaller. 1e-3/1e-4 sit above the floor
#       (truncation-limited, arms separate); 1e-5/1e-6 sit below it (beta-limited, arms coincide).
#       That shows the crossover, which is the thing worth plotting.
# Overridable as ARGS[7] so neither variant needs an edit here.
const CUTOFFS = length(ARGS) >= 7 ?
    Tuple(parse.(Float64, split(ARGS[7], ","))) : (1e-8, 1e-10)
# Imaginary time needs beta >> 1/gap to converge. The L=16 ring gap is ~0.2, so beta = 5 left the
# energy at 6.4e-3 while DMRG reached 1.07e-4 on the same state count -- the ITE curve was
# unconverged BY CONSTRUCTION. Separate from TMAX so real time still runs to the requested T.
const BETA = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 20.0

const COLS = "suite,scheme,model,sym,maxdim,cutoff,dt,x,energy,err,fid,maxbond," *
             "maxbond_states,centrebond,err_fnl,discarded,peak,krylov,elapsed"

row(io; suite, scheme, model, sym, maxdim, cutoff, dt, x, energy, err, fid, maxbond,
    maxbond_states, centrebond, err_fnl, discarded, peak, krylov, elapsed) =
    (@printf(io, "%s,%s,%s,%s,%d,%g,%g,%g,%.10e,%.6e,%.6e,%d,%d,%d,%.3e,%.3e,%d,%d,%.2f\n",
             suite, scheme, model, sym, maxdim, cutoff, dt, x, energy, err, fid, maxbond,
             maxbond_states, centrebond, err_fnl, discarded, peak, krylov, elapsed);
     flush(io))

# ── the models, on both sides ────────────────────────────────────────────────

# ⛔ HEISENBERG IS AN OPEN CHAIN; ONLY HALDANE-SHASTRY IS A RING. Earlier runs used a PERIODIC
# Heisenberg ring, which is a different model and carries a different analytic reference.
"""
    model(name, L) -> MPO

THE MPO ONLY. It used to return `(mpo, coupling_matrix)` so a sparse many-body reference could
be built from the SAME Hamiltonian -- but every reference is analytic now (Bethe for the chain,
free fermions for XX, the Li paper for Haldane-Shastry), so nothing consumes the matrix. All
three call sites were writing `mpo, _ = model(...)`, i.e. computing `hs_coupling_matrix(L)` and
throwing it away, and that discarded value was the ONLY reason `benchmarks/exact_sparse.jl` was
still included here. Returning one thing removes the last sparse dependency from the driver.
"""
function model(name::String, L::Int)
    if name == "chain"
        return heisenberg_chain_mpo(L)
    elseif name == "hs"
        return haldane_shastry_mpo(L)
    elseif name == "xx"
        # `delta = 0` drops the `S^z S^z` term, leaving `Σ (SxSx + SySy)` -- QUADRATIC after
        # Jordan-Wigner, hence exactly solvable, and the reference is the free-fermion formula.
        return RSVDCBEBondUpdate.mpo_from_terms(xxz_chain(L; delta = 0.0))
    end
    error("unknown model $name")
end

"""
Ground energy reference -- ANALYTIC for both models, never a diagonalisation.

  - Haldane-Shastry: the closed form `-π²(L²+5)/(24L)`.
  - Open Heisenberg chain: `bethe_xxx_open_energy`, the OPEN Bethe equations (reflecting
    boundaries, so `λ_j + λ_k` mirror terms and integer quantum numbers).

⛔ THIS DELIBERATELY DOES NOT FALL BACK TO SPARSE DIAGONALISATION. It used to, because an earlier
open-chain Bethe solver was wrong (a spurious `θ₂(λ_j)` self term put it 0.24 below ED at every
`L` while reporting residual 1e-14). That solver has since been corrected and validated against ED
to 4.00e-15 at every `L` in 4…12, so the analytic reference is available and is what the campaign
compares against. ED is exact too, but it is not analytic and it does not scale.

The residual is ASSERTED, not ignored: an unconverged Newton solve must fail loudly rather than
quietly become the thing every error in the campaign is measured against.
"""
function ground_reference(name::String, L::Int)
    name == "hs" && return hs_ground_energy(L)
    if name == "chain"
        e, _, res = bethe_xxx_open_energy(L)
        res < 1e-10 || error("open Bethe solve did not converge at L=$L: residual $res")
        return e
    end
    return nothing
end

maxstate(psi) = maximum(state_bond_dims(psi); init = 0)
centre_bond(psi) = leg_dim(psi[length(psi) ÷ 2], 3)
renorm!(psi) = (psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center]); psi)

"""
    energy(psi, mpo) -> Float64

The RAYLEIGH QUOTIENT `⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩`, not the bare `mpo_energy`.

⛔ THE DIVISION IS NOT COSMETIC HERE. `mpo_energy` returns `⟨ψ|H|ψ⟩` alone, and the fixed-rank
BUG does NOT preserve the norm: the step projects `ψ` onto the frame product basis, and whatever
the basis fails to span is dropped, so `‖ψ‖` decays. MEASURED at L=8 after five real-time steps:
`‖ψ‖ = 0.9962`, i.e. `‖ψ‖² = 0.9925`. Reporting the bare value would therefore print a 0.75%
energy drift that is pure normalisation — in the panel whose whole claim is that real-time energy
is CONSERVED and any drift is integrator error. The state itself is deliberately left
un-normalised so `‖v − v_exact‖` still measures the true defect, norm loss included.
"""
energy(psi, mpo) = real(mpo_energy(psi, mpo)) / max(norm(psi)^2, eps())

# One step of each integrator, behind one signature so the loops below are shared.
function stepper(scheme::String, mpo, maxdim::Int, cutoff::Float64)
    if scheme == "bug_decoupled"
        return (p, tau) -> cbe_bug_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff)
    elseif scheme == "bug_interleaved"
        return (p, tau) -> cbe_bug_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff)
    elseif scheme == "tdvp_cbe1s"
        return (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau;
                                            maxdim = maxdim, trunc_thresh = cutoff)
    elseif scheme == "tdvp2"
        return (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff)
    elseif scheme == "jan_bug"
        # Defaults, which put the Galerkin step on the END bond.
        return (p, tau) -> jan_bug_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = cutoff)
    elseif scheme == "jan_bug_centre"
        # The SAME sweep with the root moved to the centre. The pair is the attributable test of
        # the recorded claim that root position dominates jan_bug's error -- never checked above
        # L=6, and the centre is where `cbe_bug` puts its root, so this is also the like-for-like
        # comparison between the two BUG variants.
        return (p, tau) -> jan_bug_step!(p, mpo, tau; root = length(p) ÷ 2,
                                         maxdim = maxdim, trunc_thresh = cutoff)
    end
    error("unknown scheme $scheme")
end

"`err_fnl`, `discarded`, `peak_elements`, `krylov_dims` from whichever Info type came back.

⛔ `JanBugInfo` spells it `krylov_dim` (SINGULAR -- it performs ONE 0-site exponential per step,
so the name is right for that type), while the CBE/TDVP Info types spell it `krylov_dims`. A
`hasproperty(:krylov_dims)`-only probe therefore silently returned 0 for both jan_bug variants,
which in the cost panel reads as A FREE INTEGRATOR -- the failure biases in the flattering
direction, which is the dangerous one. Every field here is probed under BOTH spellings for that
reason; a missing count must never be indistinguishable from a zero count."
info_fields(info) = (
    hasproperty(info, :err_fnl)       ? info.err_fnl       : 0.0,
    hasproperty(info, :discarded)     ? info.discarded     : 0.0,
    hasproperty(info, :peak_elements) ? info.peak_elements : 0,
    hasproperty(info, :krylov_dims)   ? info.krylov_dims   :
    hasproperty(info, :krylov_dim)    ? info.krylov_dim    : 0,
)

const ALL_SCHEMES = ("bug_decoupled", "bug_interleaved", "tdvp_cbe1s", "tdvp2",
                     "jan_bug", "jan_bug_centre")
# ARGS[5], if given, is a comma-separated subset -- so a scheme added later can be run on its own
# instead of re-running four that are already measured. The output path gets a tag in that case so
# it cannot clobber the full run's CSV.
# `String.(...)`, not bare `split`: `split` yields `SubString{String}`, which does NOT match
# `stepper(::String, ...)`. The GSE phase ignores `SCHEMES`, so this only surfaces the first time a
# scheme subset reaches an evolution phase -- i.e. exactly in the slurm array.
const SCHEMES = length(ARGS) >= 5 ? Tuple(String.(split(ARGS[5], ","))) : ALL_SCHEMES

# ── PHASE dtscan: WHERE IS THE dt ERROR ABOVE THE RANK FLOOR? ────────────────
#
# Fixed rank, fixed T, dt over two decades. Read the slope off the CSV: it should be ~2 at large
# dt and flatten to 0 at the rank floor. This is the honest form of the "second order" claim and
# it is also one of the error plots wanted, so it is measured once and used twice.
#
# ⛔ L = 8 IS THE WRONG SIZE FOR THIS AND THE FIRST RUN PROVED IT. Full bond dims at L=8 are
# `[2,4,8,16,8,4,2]`, so chi >= 16 IS FULL RANK and the scheme is EXACT there (errors ~1e-14 at
# every dt); and at chi=8 with T=1.6 the state has entangled far past the cap, so the error is
# 0.21-0.35, non-monotonic in dt, and drifts slightly UP as dt shrinks. Two regimes,
# rank-dominated or exact, with no window between them -- measured, see
# `results/cbe_sweeps_dtscan_L8_chi8.csv`.
#
# So the scan runs at L = 16, where full rank at the centre is 256 and chi = 64 is therefore a
# genuine but MILD restriction, and over a SHORT T where the exact state is nearly representable
# at that chi -- which is the only place the dt error can rise above the rank floor.
function phase_dtscan(io)
    set_symmetry!(:U1)
    Ls, T = 16, 0.5
    # XX, so the reference is the EXACT free-fermion profile rather than a sparse propagation --
    # same rule as `phase_evolve`, and it keeps the diagnostic to `L x L` work.
    mpo = model("xx", Ls)
    want = xx_free_fermion_sz(Ls, T)
    # ⛔ THE CAP IS THE POINT HERE, unlike every other phase. This diagnostic asks where the dt
    # error rises above the RANK floor, so the rank must be PINNED: `maxdim` binds and the
    # tolerance is set below it so it never does.
    for maxdim in (32, 64)
        for nst in (2, 4, 8, 16, 32)
            dt = T / nst
            for scheme in SCHEMES
                psi = domain_wall_state(Ls)
                step = stepper(scheme, mpo, maxdim, 1e-12)
                t0 = time(); efnl = 0.0; disc = 0.0; pk = 0; kd = 0
                ok = true
                try
                    for _ in 1:nst
                        info = step(psi, ComplexF64(-im * dt))
                        f = info_fields(info)
                        efnl = max(efnl, f[1]); disc = max(disc, f[2])
                        pk = max(pk, f[3]); kd += f[4]
                    end
                catch e
                    ok = false
                    @printf("  dtscan %s maxdim=%d nst=%d THREW %s\n", scheme, maxdim, nst,
                            sprint(showerror, e)[1:min(end, 90)]); flush(stdout)
                end
                ok || continue
                err = maximum(abs.([sz_expectation(psi, j) for j in 1:Ls] .- want))
                row(io; suite = "dtscan", scheme = scheme, model = "xx", sym = "U1",
                    maxdim = maxdim, cutoff = 1e-12, dt = dt, x = T,
                    energy = energy(psi, mpo),
                    err = err, fid = 0.0, maxbond = maximum(bond_dims(psi); init = 0),
                    maxbond_states = maxstate(psi), centrebond = centre_bond(psi),
                    err_fnl = efnl, discarded = disc, peak = pk, krylov = kd,
                    elapsed = time() - t0)
            end
        end
        @printf("  dtscan maxdim=%d done\n", maxdim); flush(stdout)
    end
end

# `apply_sz!` MOVED to `src/BondUpdateBUG/symmetric_mps.jl` (exported) once
# `benchmarks/hs_structure_factor.jl` needed it too. Its docstring documents a silent-failure trap
# -- contracting `Sz` onto the site leg re-sorts the charge sectors and returns a wrong state of
# the right shape -- and keeping two copies of a function like that guarantees only one gets fixed.

"""
    analytic_initial_energy(name, kind, L) -> Union{Float64, Nothing}

`⟨ψ₀|H|ψ₀⟩` IN CLOSED FORM for the product-state starts, `nothing` when there is none.

On a definite-`Sz` PRODUCT state the XY terms have zero expectation, so ONLY `S^z S^z` survives.
That makes the answer depend on the MODEL as much as on the state:

  - `"xx"` (`Δ = 0`, no `S^zS^z` at all) -- `E₀ = 0` for EVERY product state.
  - `"chain"` (Heisenberg) -- each bond contributes `±1/4`:
      * `:domain_wall` `|↑…↑↓…↓⟩`: `L-2` aligned bonds and ONE wall bond -> `(L-3)/4`
      * `:neel` `|↑↓↑↓…⟩`: all `L-1` bonds anti-aligned -> `-(L-1)/4`
    At `L = 16`: `3.25` and `-3.75`.

⛔ THE MODEL ARGUMENT IS NOT DECORATION. This function first carried only `(kind, L)` and returned
the Heisenberg value; when real time moved to XX the closed form silently became wrong by the
whole `S^zS^z` contribution. The `t = 0` assertion in `phase_evolve` caught it immediately
(measured `E(0) = 0` against a claimed `1.25` at L=8) -- which is the entire reason that check
exists rather than trusting the initial state.
"""
function analytic_initial_energy(name::String, kind::Symbol, L::Int)
    name == "xx" && return kind in (:neel, :domain_wall) ? 0.0 : nothing
    if name == "chain"
        kind === :domain_wall && return (L - 3) / 4
        kind === :neel && return -(L - 1) / 4
    end
    return nothing            # `:sz_gs` has no closed form; use the t=0 MPS value
end

"""
    initial_state(kind, L, mpo) -> psi

The initial state as an MPS.

⛔ NO DENSE VECTOR IS BUILT. The campaign compares against ANALYTIC references only; real time is
graded on CONSERVATION LAWS (see `phase_evolve`), so nothing here needs a `2^L` companion.

  - `:neel` -- `|↑↓↑↓…⟩`, the requested Heisenberg IMAGINARY-time start. `Sz = 0`, so the sector
    reference applies unchanged.
  - `:domain_wall` -- `|↑…↑↓…↓⟩` with the wall at the centre, the requested Heisenberg REAL-time
    start. A single basis state, `Sz = 0` for even `L`, and the canonical setup for a
    magnetisation light cone.
  - `:dimer` -- `⊗_k |singlet⟩`, the `S = 0` product state. Unused by the current arms but kept:
    it is the only one of the three representable under `:SU2`.

⚠ `:neel` and `:domain_wall` are ABELIAN ONLY. A definite-`Sz` product state is a superposition
of total-spin sectors and has no `:SU2` representation at all (`neel_state`/`product_state` are
guarded off in that mode), which is why both evolution phases run under U(1).
  - `:sz_gs` -- `S^z_{j0}|GS⟩`, the Haldane-Shastry protocol of arXiv:2208.10972, whose measured
    quantity is the ground-state correlator `C(x,t) = ⟨Ψ₀|**S**_x(t)·**S**_0(0)|Ψ₀⟩`. The ground
    state is SU(2) symmetric so the three components contribute equally,
    `C = 3⟨Ψ₀|S^z_x(t) S^z_0|Ψ₀⟩`, and the state to propagate is `S^z_{j0}|Ψ₀⟩`. Taking `z`
    rather than `S^-` keeps the run inside ONE symmetry sector. `|Ψ₀⟩` comes from OUR OWN
    CBE-DMRG, checked against the CLOSED FORM `-π²(L²+5)/(24L)` -- analytic, not a diagonalisation.
"""
function initial_state(kind::Symbol, L::Int, mpo)
    kind === :neel        && return neel_state(L)
    kind === :dimer       && return dimer_state(L)
    kind === :domain_wall && return domain_wall_state(L)
    kind === :sz_gs || error("unknown initial state $kind")

    # Haldane-Shastry: ground state, then a localised S^z at the centre.
    psi = dimer_state(L)
    run = rsvd_cbe_dmrg1s!(psi, mpo; n_sweeps = 40, maxdim = MAXDIM, trunc_thresh = 1e-10)
    eref = hs_ground_energy(L)
    @printf("  hs |psi0>: DMRG E=%.10f  closed form=%.10f  diff=%.2e\n",
            run.energies[end], eref, abs(run.energies[end] - eref)); flush(stdout)
    apply_sz!(psi, L ÷ 2)
    renorm!(psi)
    return psi
end

# ── PHASE gse ────────────────────────────────────────────────────────────────
#
# TWO CBE ARMS, which is the comparison that matters for the DMRG side: the RSVD selection (the
# method) against the EXACT selection (the reference arm, which forms the rank-4 `H*Theta` the
# randomised version exists to avoid). Same sweep, same tolerance, same everything else -- so a
# difference is the selection and nothing else.
function phase_gse(io)
    # Heisenberg, SU(2) only -- as requested.
    for name in ("chain",), sym in (:SU2,), cutoff in CUTOFFS
        set_symmetry!(sym)
        mpo = model(name, L)
        eref = ground_reference(name, L)
        # THREE ARMS, differing in ONE thing each, so a gap is attributable:
        #   rsvd  vs exact       -> the SELECTION (randomised sketch vs the full fused basis)
        #   rsvd  vs rsvd_reuse  -> the local eigensolve's START VECTOR, nothing else
        for (nm, solver) in (("dmrg_cbe_rsvd", rsvd_cbe_dmrg1s!),
                             ("dmrg_cbe_exact", exact_cbe_dmrg1s!),
                             ("dmrg_cbe_rsvd_reuse", rsvd_cbe_dmrg1s_seeded!))
            psi = dimer_state(L)
            t0 = time()
            local run
            try
                run = solver(psi, mpo; n_sweeps = 30, maxdim = MAXDIM, trunc_thresh = cutoff)
            catch e
                @printf("  gse %s %s %s cut=%g THREW %s
", name, sym, nm, cutoff,
                        sprint(showerror, e)[1:min(end, 90)]); flush(stdout)
                continue
            end
            # EVERY column here must be read from `run.*[sw]`, never from `psi`: `psi` is the
            # state the LAST sweep left behind, so `maxstate(psi)` stamps one number onto all
            # of them and the growth the run is supposed to demonstrate reads as flat.
            # Cumulative, so the column reads as "cost to reach THIS accuracy" rather than one
            # whole-run number repeated. Sweep 1 carries this session's JIT; the cluster job is
            # what the timing comparison is actually read off.
            cum = cumsum(run.sweep_seconds)
            for (sw, e) in enumerate(run.energies)
                row(io; suite = "gse", scheme = nm, model = name, sym = String(sym),
                    maxdim = MAXDIM, cutoff = cutoff, dt = 0.0, x = sw, energy = e,
                    err = eref === nothing ? NaN : abs(e - eref), fid = 0.0,
                    maxbond = run.max_bond_dims[sw],
                    maxbond_states = run.max_state_bond_dims[sw],
                    centrebond = run.bond_dims[sw][L ÷ 2], err_fnl = run.err_fnl[sw],
                    discarded = run.discarded[sw], peak = 0, krylov = run.krylov_dims[sw],
                    elapsed = cum[sw])
            end
            @printf("  gse %s %s %-15s cut=%-6g E=%.12f err=%.3e bd=%d %.1fs
",
                    name, sym, nm, cutoff, run.energies[end],
                    eref === nothing ? NaN : abs(run.energies[end] - eref),
                    run.max_bond_dims[end], time() - t0); flush(stdout)
        end
    end
end

# ── PHASES ite and rte ───────────────────────────────────────────────────────
#
# ITE and RTE share every line except `tau`, the initial state and what the error is measured
# against, so they share the loop: an accidental difference between the two would otherwise be
# indistinguishable from a physical one.
#
# ⛔ ANALYTIC REFERENCES ONLY -- NO DENSE, NO SPARSE PROPAGATION, ANYWHERE.
#
#   ite  `|E(β) − E₀|` against the OPEN-CHAIN BETHE ground energy.
#   rte  THE XX CHAIN'S EXACT FREE-FERMION PROFILE. `Σ (SxSx + SySy)` is QUADRATIC after
#        Jordan-Wigner, so the spin expectation has a closed form,
#
#            ⟨Sz_j(t)⟩ = Σ_{k occupied} |[exp(-i h t)]_{jk}|² − ½,     h = the L x L hopping
#
#        and the domain wall is exactly the Slater determinant with sites `1:L/2` occupied.
#        `err` is the WORST-SITE deviation of the MPS profile from that expression.
#
# WHY XX RATHER THAN HEISENBERG FOR REAL TIME. "Exactly solvable" for the XXX chain means the
# SPECTRUM yields to the Bethe ansatz -- it gives NO closed form for `exp(-iHt)|domain wall⟩`, so
# a Heisenberg state error could only ever come from dense or sparse propagation. XX is quadratic,
# so the dynamics itself is closed-form and the comparison is against an EXPRESSION rather than
# against another computation. It also costs an `L x L` matrix exponential instead of `2^L`, which
# is what removes the campaign's hard scaling wall (`dense_state` is impossible past L ≈ 24).
#
# `fid` carries `‖ψ(t)‖`, which must stay 1: the fixed-rank BUG does NOT preserve it (MEASURED
# 0.9962 at L=8), so it is a real defect measure rather than a formality.
#
# The reference is itself VALIDATED, not assumed: `tests/BondUpdateBUG/test_xx_analytic.jl` pins
# `xx_free_fermion_sz` against dense ED to 1e-11. That matters because the free-fermion formula is
# derived from the Hamiltonian on paper while the dense path is built from the integrator's own
# conventions -- agreement between two independent derivations is evidence, not a tautology.
function phase_evolve(io, mode::String)
    imag_time = mode == "ite"
    # Imaginary time runs to BETA (long enough to actually converge); real time to TMAX.
    tfinal = imag_time ? BETA : TMAX
    nst = round(Int, tfinal / DT)
    sample = max(1, round(Int, 0.25 / DT))
    #
    # THE ARM TABLE IS THE ONLY PLACE THAT DECIDES WHAT STARTS WHERE. One arm per (model,
    # symmetry, initial state), as requested:
    #
    #     ite  ("chain", :U1, :neel)         Heisenberg from the Néel state  -> Bethe
    #     rte  ("xx",    :U1, :domain_wall)  XX from the domain wall         -> free fermions
    #     rte  ("hs",    :U1, :sz_gs)        Haldane-Shastry from `S^z_{j0}|GS⟩` (2208.10972)
    #
    # ⛔ EVERYTHING RUNS UNDER U(1), AND THAT IS FORCED, NOT A PREFERENCE. Both requested
    # Heisenberg starts are definite-`Sz` PRODUCT states, and such a state is a superposition of
    # total-spin sectors -- it has no `:SU2` representation at all, which is why `neel_state` and
    # `domain_wall_state` are guarded off in that mode. An SU(2) evolution arm would therefore
    # have to start from a DIFFERENT state (the dimer) and could not be compared against these.
    # SU(2) is still exercised where it CAN be: the ground-state phase, which starts from the
    # dimer.
    arms = if imag_time
        (("chain", :U1, :neel),)
    else
        (("xx", :U1, :domain_wall), ("hs", :U1, :sz_gs))
    end
    for (name, sym, kind) in arms
        set_symmetry!(sym)
        mpo = model(name, L)
        # Built ONCE per arm and copied per run, so every scheme starts from bit-identical data
        # -- and so the Haldane-Shastry ground-state solve is not repeated once per scheme.
        psi0 = initial_state(kind, L, mpo)
        # ITE is graded against the ground energy; RTE against its OWN t=0 energy, which is what
        # conservation says must not move. The `t=0` value is exact -- no step has been taken and
        # nothing has been truncated -- and on the product starts it is CROSS-CHECKED against the
        # closed form, so a wrong initial state cannot quietly become a flat, healthy-looking
        # conservation curve.
        e0 = energy(psi0, mpo)
        eclosed = analytic_initial_energy(name, kind, L)
        if eclosed !== nothing
            @printf("  %s %s/%s/%s: E(0)=%.12f  closed form=%.12f  diff=%.2e\n",
                    mode, name, sym, kind, e0, eclosed, abs(e0 - eclosed))
            abs(e0 - eclosed) < 1e-10 ||
                error("initial energy disagrees with the closed form for $kind at L=$L")
        end
        eref = imag_time ? ground_reference(name, L) : e0
        @printf("  %s %s/%s/%s: start bd=%s  eref=%.12f\n", mode, name, sym, kind,
                string(bond_dims(psi0)), eref); flush(stdout)
        for cutoff in CUTOFFS, scheme in SCHEMES
            psi = copy(psi0)
            step = stepper(scheme, mpo, MAXDIM, cutoff)
            t0 = time()
            efnl = 0.0; disc = 0.0; pk = 0; kd = 0
            # Carried out of the loop purely so the console summary can print the SAME metric the
            # CSV recorded. The summary used to recompute `abs(E - eref)` and print `NaN` for real
            # time, since real time is not graded on energy -- so every RTE line in the log read
            # `err=NaN` and looked like a failed run while the CSV held perfectly good numbers.
            lasterr = NaN
            for n in 1:nst
                try
                    info = step(psi, imag_time ? ComplexF64(-DT) : ComplexF64(-im * DT))
                    f = info_fields(info)
                    efnl = max(efnl, f[1]); disc = max(disc, f[2])
                    pk = max(pk, f[3]); kd += f[4]
                    imag_time && renorm!(psi)
                catch e
                    @printf("  %s %s %s cut=%g step %d THREW %s
", mode, name, scheme,
                            cutoff, n, sprint(showerror, e)[1:min(end, 90)]); flush(stdout)
                    break
                end
                if n % sample == 0 || n == nst
                    e = energy(psi, mpo)
                    fid = norm(psi)          # must stay 1; the fixed-rank BUG loses it
                    err = if imag_time
                        eref === nothing ? NaN : abs(e - eref)
                    elseif name == "xx"
                        # THE EXACT FREE-FERMION PROFILE, worst site. `sz_expectation` moves the
                        # orthogonality centre, which is lossless, so measuring does not perturb
                        # the run.
                        got = [sz_expectation(psi, j) for j in 1:L]
                        maximum(abs.(got .- xx_free_fermion_sz(L, n * DT)))
                    else
                        # Haldane-Shastry has no closed-form quench profile; energy is EXACTLY
                        # conserved under unitary evolution, so drift from `E(0)` is integrator
                        # error and needs no reference state.
                        abs(e - eref)
                    end
                    lasterr = err
                    row(io; suite = mode, scheme = scheme, model = name, sym = String(sym),
                        maxdim = MAXDIM, cutoff = cutoff, dt = DT, x = n * DT, energy = e,
                        err = err, fid = fid, maxbond = maximum(bond_dims(psi); init = 0),
                        maxbond_states = maxstate(psi), centrebond = centre_bond(psi),
                        err_fnl = efnl, discarded = disc, peak = pk, krylov = kd,
                        elapsed = time() - t0)
                end
            end
            @printf("  %s %s %-16s cut=%-6g bd=%d err=%.3e %.1fs
", mode, name, scheme,
                    cutoff, maximum(bond_dims(psi); init = 0), lasterr,
                    time() - t0); flush(stdout)
        end
    end
end

# ── entry ────────────────────────────────────────────────────────────────────
function main()
    mkpath(OUTDIR)
    # The PHASE goes in the filename as well as the scheme subset. Without it an array job
    # running `ite` and `rte` for the same scheme writes both to one path and the second wins.
    # The CUTOFF ladder goes in the filename too: a run with a different ladder is a different
    # experiment, and without this it would overwrite the 1e-8/1e-10 campaign data in place.
    tag = (PHASE == "all" ? "" : "_" * PHASE) *
          (length(ARGS) >= 5 ? "_" * replace(ARGS[5], "," => "-") : "") *
          (length(ARGS) >= 7 ? "_cut" * replace(ARGS[7], "," => "-") : "")
    path = joinpath(OUTDIR, "cbe_sweeps_l$(L)_T$(TMAX)$(tag).csv")
    @printf("L=%d T=%g dt=%g phase=%s -> %s\n", L, TMAX, DT, PHASE, path); flush(stdout)
    open(path, "w") do io
        println(io, COLS); flush(io)
        # `main` = the deliverable phases without the dt scan, which is a separate question and a
        # separate system size (see `phase_dtscan`). Kept apart so a long campaign is not held up
        # behind a diagnostic.
        main = PHASE == "all" || PHASE == "main"
        (PHASE == "all" || PHASE == "dtscan") && phase_dtscan(io)
        (main || PHASE == "gse") && phase_gse(io)
        (main || PHASE == "ite") && phase_evolve(io, "ite")
        (main || PHASE == "rte") && phase_evolve(io, "rte")
    end
    println("wrote $path")
    println("MAIN_DONE")
end

main()
