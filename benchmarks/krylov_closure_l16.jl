# XX chain, U(1), L = 16, domain wall -- the general-order Krylov-closure benchmark
# (krylov_closure_cbe_bug_realtime_L16_plan.md §5, IT1-IT5).
#
#     H = (J/2) sum_j (S+_j S-_{j+1} + S-_j S+_{j+1}),  J = 1, open;   |up^8 down^8>
#
# Graded against an INDEPENDENT sparse exact reference in the N_up = 8 sector (dimension
# C(16,8) = 12870): `heisenberg_sparse(L; delta = 0)` propagated by `expv_sparse`
# (benchmarks/exact_sparse.jl), with the occupations cross-checked against the free-fermion
# 16 x 16 correlation-matrix evolution (tests/common/free_fermion.jl). The MPS is read out in the
# same sector basis (`mps_sector_vector`), so the COMPLEX WAVEFUNCTION is compared -- raw (global
# phase included) and phase-aligned -- not only observables.
#
# PHASES (first argument):
#   check  conventions: the read-out reproduces the exact domain wall, the two references agree,
#          and one tight short step matches the exact propagator. Runs before every phase.
#   rho    FIRST: the unchanged order-1 step with `closure_probe = 4` -- rho_0..rho_3 on its own
#          frames every step, beside its dt convergence (T = 0.8, dt 0.2..0.025; tight and
#          default tolerances). Decides whether the closure sweeps are needed at all.
#   it1    exact real-time state, t = 0.5, 1, 2 (graded every 0.5); orders 1, 2, 4; dt = 0.05
#   it2    temporal convergence: T = 0.8, dt = 0.2 / 0.1 / 0.05 / 0.025, orders 1-4 (+ the
#          midpoint step as a known-second-order calibration arm); compression effectively off,
#          tight tol_basis / tol_expv, large maxdim
#   it3    conservation and physical profile to t = 2: norm, energy, N_up, isometry, n_j(t)
#   it4    rank adaptation: maxdim 32 / 64 / 128, orders 2 and 4, to t = 4
#   it5    work at matched accuracy, ONE process (wall clocks compare): order 1, midpoint,
#          orders 2-4 at dt = 0.2 / 0.1 / 0.05 to T = 2
#
# usage:  julia -t 2 --project=. benchmarks/krylov_closure_l16.jl PHASE OUTDIR [L]
#
# Output (streamed, so a killed job keeps its rows): OUTDIR/kl16_<phase>.csv (one row per step),
# ..._bonds.csv (one row per half-sweep bond visit, closure sweeps tagged by pass/frontier),
# ..._closure.csv (one row per frontier sweep), ..._profile.csv (n_j(t), it3). Exits non-zero if it
# graded nothing.

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))
include(joinpath(@__DIR__, "exact_sparse.jl"))

const PHASE  = isempty(ARGS) ? "check" : ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results")
const L      = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
const NUP    = L ÷ 2
PHASE in ("check", "rho", "it1", "it2", "it3", "it4", "it5") ||
    error("unknown phase $PHASE (check | rho | it1 | it2 | it3 | it4 | it5)")

const HSP, STATES, IDX = heisenberg_sparse(L; J = 1.0, delta = 0.0)   # XX: delta = 0
const V0 = domain_wall_vector(L, STATES, IDX)
const PROF = (tol_basis = 1e-8, tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = 256)

# ── exact reference ─────────────────────────────────────────────────────────────────────────

const _TRAJ = Dict{Tuple{Float64, Int}, Vector{Vector{ComplexF64}}}()
"The exact sector vector after each of `n` steps of `dt`, cached per grid."
exact_traj(dt, n) = get!(_TRAJ, (dt, n)) do
    out = Vector{Vector{ComplexF64}}(undef, n)
    v = V0
    for s in 1:n
        v = expv_sparse(HSP, ComplexF64(-im * dt), v; tol = 1e-13)
        out[s] = v
    end
    out
end

# ── reading the MPS out in the sector basis ─────────────────────────────────────────────────

const _BRA = Dict{Tuple{Int, Int, Bool}, Any}()
"Product-state site tensor at site `i` after `u` up spins, bra links retagged as in `overlap`."
_bra_site(i, u, isup) = get!(_BRA, (i, u, isup)) do
    spins = vcat([k <= u ? :up : :down for k in 1:(i - 1)], [isup ? :up : :down],
                 fill(:down, L - i))
    t = product_state(spins)[i]
    i > 1 && (t = setitag(t, 1, "BL,$i"))
    i < L && (t = setitag(t, 3, "BL,$(i + 1)"))
    to_concrete(t)
end

"""
    mps_sector_vector(psi) -> Vector{ComplexF64}

`<b|psi>` for every basis state of the N_up = L/2 sector (`sz0_basis` order: bit `j-1` set = site
`j` up), by a depth-first walk that shares the left transfer of every prefix -- `overlap`'s
contraction, one product state at a time, without rebuilding the common prefixes.
"""
function mps_sector_vector(psi::SymMPS)
    v = zeros(ComplexF64, length(STATES))
    function walk(i, u, bits, E)
        for isup in (true, false)
            u2 = u + (isup ? 1 : 0)
            (u2 <= NUP && NUP - u2 <= L - i) || continue
            b = _bra_site(i, u, isup)
            b2 = bits | (isup ? (1 << (i - 1)) : 0)
            if i == L
                tmp = contract(E, (1,), b', (1,))
                s = to_concrete(contract(tmp, (1, 2, 3), psi[L], (1, 2, 3)))
                v[IDX[b2]] = length(s) == 0 ? 0.0im : ComplexF64(s[])
            else
                E2 = i == 1 ? to_concrete(contract(b', (1, 2), psi[1], (1, 2))) :
                     to_concrete(contract(to_concrete(contract(E, (1,), b', (1,))), (1, 2),
                                          psi[i], (1, 2)))
                length(E2) == 0 && continue
                walk(i + 1, u2, b2, E2)
            end
        end
    end
    walk(1, 0, 0, nothing)
    return v
end

"`J Im<S+_j S-_{j+1}>` (J = 1), the spin current on bond `j`, from a sector vector."
function current_dense(v, j)
    bit(b, k) = (b >> (k - 1)) & 1
    acc = 0.0im
    for (k, b) in enumerate(STATES)
        if bit(b, j) == 0 && bit(b, j + 1) == 1
            acc += conj(v[IDX[b ⊻ (1 << (j - 1)) ⊻ (1 << j)]]) * v[k]
        end
    end
    return imag(acc) / max(norm(v)^2, eps())
end

"Raw and phase-aligned wavefunction errors, and the domain-wall magnetisation/current errors."
function wf_grade(psi, vex)
    v = mps_sector_vector(psi)
    raw = norm(v - vex)
    # Rotate, then subtract: `sqrt(|v|^2 + |vex|^2 - 2|<vex,v>|)` cancels to a sqrt(eps) ~ 1e-8
    # floor (IT1/IT2 first run read 0 and non-monotone values below it).
    ov = dot(vex, v)
    aligned = norm(v .* (iszero(ov) ? 1.0 : conj(ov) / abs(ov)) .- vex)
    mag(w) = sum(magnetisation_dense(w, L, STATES)[1:NUP])
    emag = abs(mag(v) - mag(vex))
    ecur = abs(current_dense(v, NUP) - current_dense(vex, NUP))
    return raw, aligned, emag, ecur
end

function iso_defect(psi)
    d = 0.0
    for i in 1:length(psi)
        i < psi.center && (d = max(d, left_isometry_defect(psi[i])))
        i > psi.center && (d = max(d, right_isometry_defect(psi[i])))
    end
    return d
end

# ── conventions ─────────────────────────────────────────────────────────────────────────────

function check_conventions()
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    e0 = norm(mps_sector_vector(domain_wall_state(L)) - V0)
    t1 = exact_traj(0.05, 20)[end]
    eff = maximum(abs.(magnetisation_dense(t1, L, STATES) .-
                       xx_free_fermion_sz(L, 1.0; J = 1.0, occupied = 1:NUP)))
    psi = domain_wall_state(L)
    cbe_bug_step!(psi, mpo, ComplexF64(-0.01im); order = 2, tol_basis = 1e-12,
                  tol_trunc = 1e-14, tol_expv = 1e-14, maxdim = 256)
    # The step is order 2, so `e1` is its local error, ~3e-8 here (UT3: 2.95e-5 at dt = 0.1,
    # times (0.1)^3). The check is about CONVENTIONS -- a wrong time direction gives
    # ~2 dt ||H psi|| ~ 1e-2 -- so it asserts the separation from the reversed propagator.
    v1 = mps_sector_vector(psi)
    e1 = norm(v1 - expv_sparse(HSP, ComplexF64(-0.01im), V0; tol = 1e-14))
    e1r = norm(v1 - expv_sparse(HSP, ComplexF64(0.01im), V0; tol = 1e-14))
    println(@sprintf("CHECK readout %.3e  sparse-vs-freefermion %.3e  one-step %.3e  reversed %.3e",
                     e0, eff, e1, e1r))
    (e0 < 1e-13 && eff < 1e-9 && e1 < 1e-6 && e1r > 1e3 * e1) ||
        (println("EXIT-CONVENTION-MISMATCH"); exit(2))
    return nothing
end

# ── one arm ─────────────────────────────────────────────────────────────────────────────────

const COLS = ["phase", "arm", "scheme", "order", "dt", "tol_basis", "tol_trunc", "tol_expv",
              "maxdim", "exact", "seed", "step", "t", "err_sz", "err_sz_ff", "wf_err",
              "wf_err_aligned", "err_mag_left", "err_current_mid", "norm_drift",
              "energy_drift", "nup_drift", "iso_defect", "chi_max", "r_old", "r_aug",
              "r_final", "krylov_dims", "n_h1", "n_h1_closure", "n_root_frontier",
              "n_h2_leak", "n_probes", "n_holdout", "n_selected", "n_krylov", "n_passes",
              "n_closure_sweeps", "closure_added", "new_sectors", "leak", "defect",
              "leak_initial", "rho_initial",
              "max_sample", "max_tail", "max_krylov_c", "max_krylov_d", "discarded_cum",
              "discarded_max", "n_compress", "proj_err", "expv_err", "expv_iters", "status",
              "unmet", "workcap", "maxdim_reached", "t_step", "t_closure", "t_cbe", "t_kry",
              "t_root", "t_trunc", "alloc_bytes"]
const BCOLS = ["phase", "arm", "order", "dt", "maxdim", "step", "pass", "frontier", "stage",
               "bond", "side", "width", "width_before", "added", "leak", "skipped", "sample",
               "tail", "krylov_c", "krylov_d", "n_probes", "n_selected", "n_krylov",
               "n_sectors", "new_sectors", "workcap", "met", "r_old", "r_final", "discarded"]
const BFIELDS = (:pass, :frontier, :stage, :bond, :side, :width, :width_before, :added, :leak,
                 :skipped, :sample, :tail, :krylov_c, :krylov_d, :n_probes, :n_selected,
                 :n_krylov, :n_sectors, :new_sectors, :workcap, :met, :r_old, :r_final,
                 :discarded)
const CCOLS = ["phase", "arm", "order", "dt", "maxdim", "step", "pass", "frontier", "enrich",
               "norm", "leak", "leak_left", "leak_right", "met", "added", "added_left",
               "added_right", "new_sectors", "skipped", "bonds", "width_before", "width_after",
               "n_h2", "n_h1", "n_probes", "n_selected", "n_krylov", "zero", "t"]
const CFIELDS = (:pass, :frontier, :enrich, :norm, :leak, :leak_left, :leak_right, :met,
                 :added, :added_left, :added_right, :new_sectors, :skipped, :bonds,
                 :width_before, :width_after, :n_h2, :n_h1, :n_probes, :n_selected,
                 :n_krylov, :zero, :t)

_f(x) = x === nothing ? "" : x isa AbstractFloat ? @sprintf("%.6e", x) : string(x)
_get(e, f) = haskey(e, f) ? getfield(e, f) : nothing

"""
    run_arm(ios, phase, arm; order, scheme, dt, T, every, seed, profile, kw...) -> Int

`T/dt` steps from the domain wall; one row per step (wavefunction graded at every `every`-th step
and the last), its bond visits and frontier sweeps. `kw` reaches the step VERBATIM.
"""
function run_arm(ios, phase, arm; order::Int = 1, scheme = "ordinary", dt::Float64,
                 T::Float64, every::Int = 0, seed::Int = 1, profile::Bool = false, kw...)
    io, bio, cio, pio = ios
    set_symmetry!(:U1)
    mpo = xxz_mpo(L; J = 1.0, delta = 0.0)
    psi = domain_wall_state(L)
    n = round(Int, T / dt)
    ex = exact_traj(dt, n)
    rng = Random.MersenneTwister(seed)
    step! = scheme == "midpoint" ? cbe_bug_midpoint_step! : cbe_bug_step!
    okw = scheme == "midpoint" ? NamedTuple() : (order = order,)
    nkw = values(kw)
    md = get(nkw, :maxdim, 128)
    E0 = real(mpo_energy(psi, mpo))
    sz(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())
    graded = 0
    for s in 1:n
        g0 = Base.gc_num()
        info = step!(psi, mpo, ComplexF64(-im * dt); parallel = true, rng = rng, okw..., kw...)
        alloc = Base.GC_Diff(Base.gc_num(), g0).allocd
        lg = info.ledger
        t = s * dt
        szm = sz(psi)
        szx = magnetisation_dense(ex[s], L, STATES)
        szf = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:NUP)
        wf = (s == n || (every > 0 && s % every == 0)) ? wf_grade(psi, ex[s]) :
             (NaN, NaN, NaN, NaN)
        nrm = norm(copy(psi))
        en = real(mpo_energy(psi, mpo)) / max(nrm^2, eps())
        cl = lg.closure_log
        row = Any[phase, arm, scheme, order, dt, get(nkw, :tol_basis, nothing),
                  get(nkw, :tol_trunc, nothing), get(nkw, :tol_expv, nothing), md,
                  get(nkw, :exact, false), seed, s, t,
                  maximum(abs.(szm .- szx)), maximum(abs.(szm .- szf)), wf...,
                  nrm - 1, en - E0, sum(szm) - sum(szx), iso_defect(psi),
                  maximum(bond_dims(psi); init = 0), lg.r_old, lg.r_aug, lg.r_final,
                  info.krylov_dims, lg.n_h1, lg.n_h1_closure, lg.n_root_frontier,
                  lg.n_h2_leak, lg.n_probes, lg.n_holdout, lg.n_selected, lg.n_krylov,
                  lg.n_passes, lg.n_closure_sweeps,
                  sum((get(c, :added, 0) for c in cl); init = 0),
                  sum((get(c, :new_sectors, 0) for c in cl); init = 0),
                  join(_f.(lg.leak), ";"), lg.defect, join(_f.(lg.leak_initial), ";"),
                  join(_f.(lg.rho_initial), ";"), lg.max_sample, lg.max_tail,
                  lg.max_krylov_c, lg.max_krylov_d, lg.discarded, info.discarded,
                  lg.n_compress, lg.proj_err, lg.expv_err, lg.expv_iters,
                  budget_status(lg), join(lg.unmet, "+"), lg.workcap_reached,
                  lg.maxdim_reached, info.t_step, lg.t_closure, info.t_cbe, info.t_kry,
                  info.t_root, info.t_trunc, alloc]
        println(io, join(_f.(row), ","))
        for e in lg.bond_log
            println(bio, join(_f.(Any[phase, arm, order, dt, md, s,
                                      (_get(e, f) for f in BFIELDS)...]), ","))
        end
        for c in cl
            println(cio, join(_f.(Any[phase, arm, order, dt, md, s,
                                      (_get(c, f) for f in CFIELDS)...]), ","))
        end
        if profile
            for j in 1:L
                println(pio, join(_f.(Any[phase, arm, order, dt, s, t, j, szm[j], szx[j]]), ","))
            end
        end
        foreach(flush, ios)
        graded += 1
    end
    fin = wf_grade(psi, ex[n])
    @info @sprintf("%s %s order=%d %s dt=%.3f: wf(T) %.3e aligned %.3e  chi %d", phase, arm,
                   order, scheme, dt, fin[1], fin[2], maximum(bond_dims(psi); init = 0))
    return graded
end

function main()
    mkpath(OUTDIR)
    check_conventions()
    PHASE == "check" && (println("GRADED-ROWS=0 (check only)"); return nothing)
    base = joinpath(OUTDIR, "kl16_$(PHASE)")
    graded = 0
    files = (base * ".csv", base * "_bonds.csv", base * "_closure.csv", base * "_profile.csv")
    ios = map(f -> open(f, "w"), files)
    try
        println(ios[1], join(COLS, ","))
        println(ios[2], join(BCOLS, ","))
        println(ios[3], join(CCOLS, ","))
        println(ios[4], "phase,arm,order,dt,step,t,site,sz_mps,sz_exact")
        go(args...; kw...) = (graded += run_arm(ios, args...; kw...))

        if PHASE == "rho"
            # The UNCHANGED order-1 step, measured: rho_j = ||(I-P) H F_j|| / ||F_j|| (j = 0..3)
            # on the frames its own CBE + local-Krylov half-sweeps built, beside its temporal
            # convergence. `closure_probe` leaves the evolved state bit-identical.
            tight = (tol_basis = 1e-12, tol_trunc = 1e-14, tol_expv = 1e-13, maxdim = 512)
            for (pname, pr) in (("tight", tight), ("prof", PROF)), dt in (0.2, 0.1, 0.05, 0.025)
                go("rho", "order1-$pname"; order = 1, dt = dt, T = 0.8, every = 1,
                   closure_probe = 4, pr...)
            end
        elseif PHASE == "it1"
            for p in (1, 2, 4)
                go("it1", "order$p"; order = p, dt = 0.05, T = 2.0, every = 10, PROF...)
            end
        elseif PHASE == "it2"
            tight = (tol_basis = 1e-12, tol_trunc = 1e-14, tol_expv = 1e-13, maxdim = 512)
            for dt in (0.2, 0.1, 0.05, 0.025)
                for p in 1:4
                    go("it2", "order$p"; order = p, dt = dt, T = 0.8, tight...)
                end
                go("it2", "midpoint"; scheme = "midpoint", dt = dt, T = 0.8, tight...)
            end
        elseif PHASE == "it3"
            for p in (2, 4)
                go("it3", "order$p"; order = p, dt = 0.05, T = 2.0, every = 5, profile = true,
                   PROF...)
            end
        elseif PHASE == "it4"
            for md in (32, 64, 128), p in (2, 4)
                go("it4", "order$p-md$md"; order = p, dt = 0.05, T = 4.0, every = 20,
                   tol_basis = 1e-8, tol_trunc = 1e-10, tol_expv = 1e-12, maxdim = md)
            end
        elseif PHASE == "it5"
            for dt in (0.2, 0.1, 0.05)
                go("it5", "order1"; order = 1, dt = dt, T = 2.0, PROF...)
                go("it5", "midpoint"; scheme = "midpoint", dt = dt, T = 2.0, PROF...)
                for p in 2:4
                    go("it5", "order$p"; order = p, dt = dt, T = 2.0, PROF...)
                end
            end
        end
    finally
        foreach(close, ios)
    end
    println("GRADED-ROWS=$graded FILE=$(files[1])")
    graded > 0 || (println("EXIT-GRADED-ZERO"); exit(1))
    return nothing
end

main()
