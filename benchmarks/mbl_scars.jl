# CAN A RANK-ADAPTIVE BUG SWEEP REPRODUCE THE ED RESULTS THAT DEFINE THE MBL AND SCAR
# LITERATURE, IN THE REGIME WHERE A FIXED-RANK PROJECTION CANNOT?
#
# THE QUESTION, stated so it can come out either way. Two protocols whose published results
# exist only as exact diagonalisation of short chains:
#
#   mbl   the disordered XXZ chain of Rev. Mod. Phys. 91, 021001 (arXiv:1804.11065) Eq. (5)/(6)
#         -- Jordan-Wigner-identical to the interacting fermions of the Bloch-group experiment --
#         quenched from the Néel charge-density wave. Observables: the IMBALANCE `I(t)` (the
#         experiment's own), the half-chain entropy (LOGARITHMIC in the MBL phase, LINEAR in the
#         ergodic one), and the site-resolved `⟨S^z_j⟩`.
#   pxp   the Rydberg blockade chain of arXiv:2011.09486, quenched from `|Z_2⟩`. Observables: the
#         REVIVAL FIDELITY `|⟨Z_2|ψ(t)⟩|²`, the half-chain entropy (which OSCILLATES rather than
#         saturating), and `⟨S^z_j⟩`.
#
# Three integrators, one rank cap ladder, one ED reference per realisation.
#
# ⛔ WHAT WOULD FALSIFY THE HYPOTHESIS, and it must be said before the numbers arrive. The
# recorded result on the Heisenberg chain is that CBE-BUG is ~4x LESS accurate than `tdvp2` at
# IDENTICAL pinned rank, and that this is intrinsic to the local step rather than to the basis
# (measured at chi = 41/64/128 with nothing truncated). So "BUG wins" is NOT the default
# expectation here; the hypothesis is specifically that on these two protocols the ordering
# INVERTS, because both start from a chi = 1 product state and demand a rank schedule that is
# either creeping (MBL, logarithmic) or NON-MONOTONE (scars, oscillating) -- failures of a
# different kind from the local-step constant. If the cap ladder shows `tdvp2` matching or
# beating the BUG at every chi on these models too, that is the answer and it gets reported.
#
# ⛔ THE CAP IS THE INDEPENDENT VARIABLE, NOT A SAFETY NET, which is the opposite of
# `cbe_sweeps_l16.jl`'s configuration. There, rank was tolerance-driven and `maxdim` was set high
# enough never to bind, because the question was cost-to-accuracy for adaptive-rank methods. The
# question HERE is what a method does when it is not allowed enough rank -- that is the entire
# content of "robust" -- so `maxdim` binds by construction, `trunc_thresh` is set well below it,
# and every arm at a given `chi` is doing the same-sized work. All three arms take the cap through
# the SAME closing truncation, so the comparison is between what they put IN the cap.
#
# MATCHED KNOBS, because an unmatched one has already produced an inverted result once in this
# project. Identical across arms: `maxdim`, `trunc_thresh`, `maxiter = 30`, `conv_tol = 0.0` (so
# every local solve burns the same Krylov depth rather than one arm stopping adaptively), and
# `growth` for the two CBE arms. Left at the library default by OMISSION: the BUG's own
# `krylov_basis`/`krylov_tol` basis-update depth -- pinning it here would fork this campaign from
# the library the day a default changes, and depth is separately measured to be accuracy-neutral.
#
# ⛔ THE COST COLUMN IS NOT A COST AXIS. `krylov` counts operator applications in the local
# solves and CANNOT see `cbe_expand`, so it under-reports both CBE arms. It is recorded because it
# is informative about the solves; any cost claim must come from `elapsed` on a cluster node with
# one timing job running, and this driver is not that job.
#
# usage:  julia -t N --project=. benchmarks/mbl_scars.jl [phase] [L] [T] [dt] [seed] [Ws] [chis] [schemes]
#         phase   smoke | mbl | pxp | all
#         Ws      comma-separated disorder strengths (mbl only)
#         chis    comma-separated rank caps
#         schemes comma-separated subset of bug,tdvp2,cbe1s
#   env   BUG_OUTDIR   where the CSV goes. MANDATORY on the cluster -- the default is next to
#                      this file, i.e. inside $HOME, which is code-only storage.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

const HERE = @__DIR__
include(joinpath(HERE, "exact_sparse.jl"))
include(joinpath(HERE, "mbl_scars_exact.jl"))

const OUTDIR = get(ENV, "BUG_OUTDIR", joinpath(HERE, "results"))
const PHASE  = isempty(ARGS) ? "all" : ARGS[1]
const L      = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const TMAX   = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 50.0
const DT     = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 0.05
const SEED   = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 1
const WS     = length(ARGS) >= 6 ? Tuple(parse.(Float64, split(ARGS[6], ","))) : (1.0, 4.0, 8.0)
const CHIS   = length(ARGS) >= 7 ? Tuple(parse.(Int, split(ARGS[7], ","))) : (8, 16, 32, 64)
const ALL_SCHEMES = ("bug", "tdvp2", "cbe1s")
const SCHEMES = length(ARGS) >= 8 ? Tuple(String.(split(ARGS[8], ","))) : ALL_SCHEMES

# `trunc_thresh` must NOT bind -- the cap is the variable. Kept above the `split_cutoff` default
# of 1e-14 so the BUG's half-sweep split stays no looser than its closing truncation (the
# recorded hard rule: a looser split inflates rank and makes a fixed-cap comparison meaningless).
const CUTOFF = 1e-12
# The two CBE arms' expansion width, held EQUAL. 1.1 rather than the library's 2.0: growth is
# measured to be inert on the achieved chi while costing accuracy, and at these caps a wide probe
# is pure overhead.
const GROWTH = 1.1
# One Krylov budget for every local solve in every arm. `conv_tol = 0` everywhere means none of
# them stops early, so a difference cannot be a difference in how hard they tried.
const MAXITER = 30

const COLS = "suite,scheme,model,sym,L,W,seed,maxdim,dt,t,obs,obs_exact,obs_err," *
             "entropy,entropy_exact,entropy_err,sz_err,energy,energy_drift,norm," *
             "maxbond,centrebond,krylov,elapsed"

row(io; suite, scheme, model, sym, l, w, seed, maxdim, dt, t, obs, obs_exact, obs_err,
    entropy, entropy_exact, entropy_err, sz_err, energy, energy_drift, nrm,
    maxbond, centrebond, krylov, elapsed) =
    (@printf(io, "%s,%s,%s,%s,%d,%g,%d,%d,%g,%.6f,%.10e,%.10e,%.6e,%.6e,%.6e,%.6e,%.6e," *
                 "%.10e,%.6e,%.10f,%d,%d,%d,%.3f\n",
             suite, scheme, model, sym, l, w, seed, maxdim, dt, t, obs, obs_exact, obs_err,
             entropy, entropy_exact, entropy_err, sz_err, energy, energy_drift, nrm,
             maxbond, centrebond, krylov, elapsed);
     flush(io))

# ── shared plumbing ─────────────────────────────────────────────────────────────────────────

maxstate(psi) = maximum(state_bond_dims(psi); init = 0)
centre_bond(psi) = leg_dim(psi[length(psi) ÷ 2], 3)

"""
One step of each integrator behind one signature.

⛔ `tdvp2` IS THE STRONG BASELINE AND `cbe1s` IS THE STRONGER ONE AT TIGHT TOLERANCE (recorded:
1.6x better than `tdvp2` at equal chi). Both are carried; dropping either would leave the
comparison against whichever happens to lose.
"""
function stepper(scheme::String, mpo, maxdim::Int)
    if scheme == "bug"
        # `krylov_basis` / `krylov_tol` deliberately omitted -- see the header.
        return (p, tau) -> cbe_bug_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = CUTOFF,
                                         growth = GROWTH, maxiter = MAXITER, conv_tol = 0.0)
    elseif scheme == "tdvp2"
        return (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = CUTOFF,
                                       maxiter = MAXITER, conv_tol = 0.0)
    elseif scheme == "cbe1s"
        return (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim, trunc_thresh = CUTOFF,
                                            growth = GROWTH, maxiter = MAXITER, conv_tol = 0.0)
    end
    error("unknown scheme $scheme")
end

info_krylov(info) = hasproperty(info, :krylov_dims) ? info.krylov_dims :
                    hasproperty(info, :krylov_dim)  ? info.krylov_dim  : 0

"""
    mps_entropy(psi) -> Float64

Half-chain entanglement entropy in NATS, from the Schmidt spectrum at the centre bond -- the same
`entropy_from_svals` the ED side uses, so the log base cannot differ between the two.

`bond_spectrum` moves the orthogonality centre, which is lossless, so measuring does not perturb
the run.
"""
mps_entropy(psi) = entropy_from_svals(bond_spectrum(psi, length(psi) ÷ 2))

"""
    imbalance_mps(psi) -> Float64

`I(t) = (2/L) Σ_j (-1)^{j+1} ⟨S^z_j⟩`, the same expression [`imbalance_dense`](@ref) evaluates on
the ED vector, so the two are comparable term by term rather than "the same observable".
"""
function imbalance_mps(psi)
    n = length(psi)
    return (2 / n) * sum((isodd(j) ? 1 : -1) * sz_expectation(psi, j) for j in 1:n)
end

"""
    sample_steps(nst) -> Vector{Int}

Step indices to sample at: LOGARITHMIC in time, plus the endpoint.

⛔ WHY LOG AND NOT EVERY-Nth. The MBL signature IS a logarithm -- `S(t) ~ ξ log t` -- and a
linear grid puts almost every sample in the last decade, where the curve is flattest, and almost
none in the first three, where it is being established. It is also what makes the ED reference
affordable: each sample costs a dense `2^{L/2} × 2^{L/2}` SVD.

⛔ AND THE GRID IS A STEP COUNT, WHICH IS THE ONLY SAFE FORM. This project has already compared
an MPS at `t = 0.2` against an exact vector at `t = 0.25` by building the two grids
independently; here the MPS is sampled AT step `n` and the reference is advanced BY `(n - n_prev)
* dt`, so the two times are the same number by construction and not by agreement.
"""
function sample_steps(nst::Int)
    nst <= 1 && return [max(nst, 1)]
    ks = round.(Int, 10 .^ range(0.0, log10(nst); length = 14 * max(1, ceil(Int, log10(nst)))))
    return sort!(unique!(clamp!(vcat(ks, nst), 1, nst)))
end

"""
    exact_reference(H, v0, steps, dt, L, states; want_entropy) -> Vector{NamedTuple}

The ED reference at exactly the sampled times, by sparse Krylov (`expv_sparse`, which asserts its
own residual and substeps until it is met) advanced BETWEEN samples rather than restarted.

One reference per `(model, L, W, seed)`, shared by every arm and every cap -- so no arm can be
graded against a slightly different reference, and the cost is paid once.
"""
function exact_reference(H, v0::Vector{ComplexF64}, steps::Vector{Int}, dt::Float64,
                         Ls::Int, states::Vector{Int}; want_entropy::Bool,
                         kind::Symbol, z2::Union{Nothing, Vector{ComplexF64}} = nothing)
    out = NamedTuple[]
    v = copy(v0)
    prev = 0
    for n in steps
        v = expv_sparse(H, ComplexF64(-im * dt * (n - prev)), v)
        prev = n
        obs = kind === :mbl ? imbalance_dense(v, Ls, states) :
              abs2(dot(z2::Vector{ComplexF64}, v) / norm(v))
        push!(out, (; n = n,
                    obs = obs,
                    sz = sz_profile_dense(v, Ls, states),
                    ent = want_entropy ? half_chain_entropy_sector(v, Ls, states) : NaN))
    end
    return out
end

# ── PHASE smoke: the exactness gate ──────────────────────────────────────────────────────────
#
# ⛔ NOTHING FROM THE PRODUCTION PHASES IS TRUSTWORTHY UNTIL THIS PASSES, and it is cheap enough
# to run every time. At FULL RANK a one-step sweep is `exp(tau H)` on the complete space, so all
# three integrators must agree with the ED propagation to Krylov tolerance. Any disagreement is a
# defect in the MPO, the state convention, or the sweep -- never truncation -- so this is the test
# that separates "the model is wrong" from "the method is limited", which is the distinction the
# whole campaign rests on.
#
# It also pins THREE conventions that are silent when wrong:
#   * the `t = 0` energy against a CLOSED FORM that includes the disorder (a field dropped at the
#     boundary sites, or applied to mirrored sites, passes every other check here);
#   * `|Z_2⟩`'s `⟨S^z_1⟩ = +1/2`, i.e. that `|•⟩ = up` on both sides;
#   * for MBL, the `:U1` run against the `:none` run -- the symmetry must change the COST and not
#     the physics, and that is asserted rather than assumed.
function phase_smoke(io)
    Ls, T, dt = 8, 1.0, 0.05
    nst = round(Int, T / dt)
    fails = String[]

    # ---- MBL, U(1)-native, against ED in the S^z = 0 sector ----
    hz = disorder_fields(Ls, 4.0; seed = 12345)
    Hs, states, idx = disordered_xxz_sparse(Ls, hz)
    v0 = neel_vector(Ls, idx)
    steps = [nst]
    ref = exact_reference(Hs, v0, steps, dt, Ls, states; want_entropy = true, kind = :mbl)[1]

    for sym in (:U1, :none)
        set_symmetry!(sym)
        mpo = disordered_xxz_mpo(Ls; hz = hz)
        psi0 = neel_state(Ls)
        e0 = real(mpo_energy(psi0, mpo)) / max(norm(psi0)^2, eps())
        ecl = neel_energy_closed(Ls, hz)
        @printf("  smoke mbl %-5s E(0)=%.12f  closed form=%.12f  diff=%.2e\n",
                sym, e0, ecl, abs(e0 - ecl))
        abs(e0 - ecl) < 1e-10 || push!(fails, "mbl/$sym initial energy vs closed form")
        for scheme in SCHEMES
            psi = copy(psi0)
            step = stepper(scheme, mpo, 256)          # 256 >= full rank at L=8: no truncation
            t0 = time(); kd = 0
            for _ in 1:nst
                kd += info_krylov(step(psi, ComplexF64(-im * dt)))
            end
            got = [sz_expectation(psi, j) for j in 1:Ls]
            sz_err = maximum(abs.(got .- ref.sz))
            ib = imbalance_mps(psi)
            ent = mps_entropy(psi)
            @printf("  smoke mbl %-5s %-6s sz_err=%.3e  I=%.6f (exact %.6f)  S=%.6f (exact %.6f)\n",
                    sym, scheme, sz_err, ib, ref.obs, ent, ref.ent)
            sz_err < 1e-10 || push!(fails, "mbl/$sym/$scheme full-rank sz_err=$sz_err")
            abs(ent - ref.ent) < 1e-8 || push!(fails, "mbl/$sym/$scheme entropy")
            row(io; suite = "smoke", scheme = scheme, model = "mbl", sym = String(sym),
                l = Ls, w = 4.0, seed = 12345, maxdim = 256, dt = dt, t = nst * dt,
                obs = ib, obs_exact = ref.obs, obs_err = abs(ib - ref.obs),
                entropy = ent, entropy_exact = ref.ent, entropy_err = abs(ent - ref.ent),
                sz_err = sz_err, energy = real(mpo_energy(psi, mpo)) / max(norm(psi)^2, eps()),
                energy_drift = 0.0, nrm = norm(psi),
                maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                krylov = kd, elapsed = time() - t0)
        end
    end

    # ---- PXP, :none by necessity, against ED in the constrained basis ----
    set_symmetry!(:none)
    Hp, pstates, pidx = pxp_sparse(Ls)
    @printf("  smoke pxp constrained dim=%d of 2^%d=%d (%.1fx)\n",
            length(pstates), Ls, 2^Ls, 2^Ls / length(pstates))
    z2 = z2_vector(Ls, pidx)
    pref = exact_reference(Hp, z2, [nst], dt, Ls, pstates;
                           want_entropy = true, kind = :pxp, z2 = z2)[1]
    mpo = pxp_mpo(Ls)
    psi0 = z2_state(Ls)
    # `|•⟩ = up` on both sides. A mirrored convention would put the excitations on the even sites,
    # which for `|Z_2⟩` is the OTHER, equally legal, symmetry-related state -- so every scalar
    # observable would still look perfect and only the site-resolved profile would be reversed.
    abs(sz_expectation(psi0, 1) - 0.5) < 1e-12 ||
        push!(fails, "pxp |Z2> convention: <Sz_1> = $(sz_expectation(psi0, 1)), want +1/2")
    for scheme in SCHEMES
        psi = copy(psi0)
        step = stepper(scheme, mpo, 256)
        t0 = time(); kd = 0
        for _ in 1:nst
            kd += info_krylov(step(psi, ComplexF64(-im * dt)))
        end
        got = [sz_expectation(psi, j) for j in 1:Ls]
        sz_err = maximum(abs.(got .- pref.sz))
        fid = abs2(overlap(psi0, psi) / max(norm(psi), eps()))
        ent = mps_entropy(psi)
        @printf("  smoke pxp none  %-6s sz_err=%.3e  F=%.6f (exact %.6f)  S=%.6f (exact %.6f)\n",
                scheme, sz_err, fid, pref.obs, ent, pref.ent)
        sz_err < 1e-10 || push!(fails, "pxp/$scheme full-rank sz_err=$sz_err")
        abs(fid - pref.obs) < 1e-9 || push!(fails, "pxp/$scheme fidelity")
        row(io; suite = "smoke", scheme = scheme, model = "pxp", sym = "none",
            l = Ls, w = 0.0, seed = 0, maxdim = 256, dt = dt, t = nst * dt,
            obs = fid, obs_exact = pref.obs, obs_err = abs(fid - pref.obs),
            entropy = ent, entropy_exact = pref.ent, entropy_err = abs(ent - pref.ent),
            sz_err = sz_err, energy = real(mpo_energy(psi, mpo)) / max(norm(psi)^2, eps()),
            energy_drift = 0.0, nrm = norm(psi),
            maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
            krylov = kd, elapsed = time() - t0)
    end

    if isempty(fails)
        println("  SMOKE OK")
    else
        for f in fails
            println("  SMOKE FAIL: ", f)
        end
        error("smoke phase failed ($(length(fails)) checks) -- production phases are not valid")
    end
end

# ── PHASES mbl and pxp ───────────────────────────────────────────────────────────────────────
#
# One loop, because the two differ only in the model, the symmetry, the start state and which
# scalar `obs` holds -- and an accidental difference in the step loop between them would be
# indistinguishable from a physical one.
function phase_run(io, kind::Symbol)
    nst = round(Int, TMAX / DT)
    steps = sample_steps(nst)
    want_entropy = L <= 20            # the ED entropy is a dense 2^{L/2} SVD -- see its docstring
    ws = kind === :mbl ? WS : (0.0,)

    for w in ws
        # ---- the reference, once per realisation, shared by every arm and cap ----
        if kind === :mbl
            hz = disorder_fields(L, w; seed = SEED)
            Hs, states, idx = disordered_xxz_sparse(L, hz)
            v0 = neel_vector(L, idx)
            z2v = nothing
            sym = :U1
        else
            hz = Float64[]
            Hs, states, idx = pxp_sparse(L)
            v0 = z2_vector(L, idx)
            z2v = copy(v0)
            sym = :none
        end
        @printf("  %s L=%d W=%g seed=%d: ED dim=%d, %d samples to t=%g\n",
                kind, L, w, SEED, length(states), length(steps), nst * DT); flush(stdout)
        tref = time()
        ref = exact_reference(Hs, v0, steps, DT, L, states;
                              want_entropy = want_entropy, kind = kind, z2 = z2v)
        @printf("  %s reference done in %.1fs\n", kind, time() - tref); flush(stdout)

        set_symmetry!(sym)
        mpo = kind === :mbl ? disordered_xxz_mpo(L; hz = hz) : pxp_mpo(L)
        psi0 = kind === :mbl ? neel_state(L) : z2_state(L)
        e0 = real(mpo_energy(psi0, mpo)) / max(norm(psi0)^2, eps())
        if kind === :mbl
            ecl = neel_energy_closed(L, hz)
            abs(e0 - ecl) < 1e-10 || error("initial energy $e0 != closed form $ecl at L=$L W=$w")
        end

        for maxdim in CHIS, scheme in SCHEMES
            psi = copy(psi0)
            step = stepper(scheme, mpo, maxdim)
            t0 = time(); kd = 0; si = 1; n = 0; broke = false
            while n < nst
                n += 1
                try
                    kd += info_krylov(step(psi, ComplexF64(-im * DT)))
                catch e
                    @printf("  %s %s chi=%d step %d THREW %s\n", kind, scheme, maxdim, n,
                            sprint(showerror, e)[1:min(end, 120)]); flush(stdout)
                    broke = true
                    break
                end
                (si <= length(steps) && steps[si] == n) || continue
                r = ref[si]; si += 1
                got = [sz_expectation(psi, j) for j in 1:L]
                nrm = norm(psi)
                obs = kind === :mbl ? imbalance_mps(psi) :
                      abs2(overlap(psi0, psi) / max(nrm, eps()))
                ent = mps_entropy(psi)
                en = real(mpo_energy(psi, mpo)) / max(nrm^2, eps())
                row(io; suite = String(kind), scheme = scheme, model = String(kind),
                    sym = String(sym), l = L, w = w, seed = SEED, maxdim = maxdim, dt = DT,
                    t = n * DT, obs = obs, obs_exact = r.obs, obs_err = abs(obs - r.obs),
                    entropy = ent, entropy_exact = r.ent, entropy_err = abs(ent - r.ent),
                    sz_err = maximum(abs.(got .- r.sz)), energy = en,
                    energy_drift = abs(en - e0), nrm = nrm,
                    maxbond = maximum(bond_dims(psi); init = 0), centrebond = centre_bond(psi),
                    krylov = kd, elapsed = time() - t0)
            end
            @printf("  %s %-6s chi=%-4d bd=%d obs_err=%.3e S_err=%.3e %.1fs%s\n",
                    kind, scheme, maxdim, maximum(bond_dims(psi); init = 0),
                    si > 1 ? abs((kind === :mbl ? imbalance_mps(psi) :
                                  abs2(overlap(psi0, psi) / max(norm(psi), eps()))) -
                                 ref[si - 1].obs) : NaN,
                    si > 1 ? abs(mps_entropy(psi) - ref[si - 1].ent) : NaN,
                    time() - t0, broke ? "  (BROKE)" : ""); flush(stdout)
        end
    end
end

# ── entry ────────────────────────────────────────────────────────────────────────────────────
function main()
    mkpath(OUTDIR)
    tag = "$(PHASE)_L$(L)_T$(TMAX)_dt$(DT)_s$(SEED)" *
          (length(ARGS) >= 8 ? "_" * replace(ARGS[8], "," => "-") : "")
    path = joinpath(OUTDIR, "mbl_scars_$tag.csv")
    @printf("phase=%s L=%d T=%g dt=%g seed=%d threads=%d -> %s\n",
            PHASE, L, TMAX, DT, SEED, Threads.nthreads(), path); flush(stdout)
    open(path, "w") do io
        println(io, COLS); flush(io)
        # The gate runs before either production phase, always: see `phase_smoke`.
        (PHASE == "all" || PHASE == "smoke") && phase_smoke(io)
        (PHASE == "all" || PHASE == "mbl") && phase_run(io, :mbl)
        (PHASE == "all" || PHASE == "pxp") && phase_run(io, :pxp)
    end
    println("wrote $path")
    println("MAIN_DONE")
end

main()
