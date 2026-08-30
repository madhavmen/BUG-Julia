# FIGS. 4 AND 5 OF arXiv:2209.00739 -- S(q,omega) ON THE SQUARE AND TRIANGULAR CYLINDERS,
# with the three integrators measured on the SAME correlator.
#
# This supersedes `fig4_2209.jl`, which is square-only and hard-codes a square Fourier transform.
# The geometry lives in `lattice2d.jl` so Fig. 4 and Fig. 5 cannot drift apart; read that file
# first, because the triangular basis is at 120 degrees and the integer site coordinates are NOT
# Cartesian positions.
#
# The paper's protocol, quoted:
#   Eq. (5)   G(x,t) = 3 <Om| S^z_x(t) S^z_c |Om>   -- "We drop the factor of 3 in this work."
#   Eq. (6)   G(x,t) = e^{i E0 t} <Om| S^z_x e^{-iHt} S^z_c |Om>
#   Eq. (2)   S(q,w) = (1/N) sum_x int dt/2pi e^{i(wt - q.x)} G(x,t)   -- the DEFINITION
#   Eq. (14)  S(q,w) = 1/(pi sqrt(N)) int_0^inf dt sum_x cos(q.x)[cos(wt) ReG - sin(wt) ImG]
#             -- what is actually EVALUATED. Sec. IID: truncating Eq. (2) leaves an unphysical
#             imaginary part, and Eq. (14) "enforces reality" instead of discarding it afterwards.
#   Eq. (12)  Gaussian damping e^{-eta^2 t^2};  eta^2 = 0.03 square, 0.02 triangular (120-deg)
#   Sec. IID  RELIABILITY = POSITIVITY: "the spectral function must be positive for all
#             frequencies ... choose eta as small as possible so that [it] is positive",
#             with the reliable window T_max ~ 1/eta. See `F_ETA2`.
#   Eq. (19)  eps(q) := argmax_w S(q,w)
#   Eq. (20)  <w>(q) := int w S dw / int S dw
#   Sec. IIE  single-site TDVP, dt = 0.1, Tmax = 40, chi = 512, U(1) zero-magnetisation sector
#   Fig. 4    square, J2 = 0, points X = (0,pi), M = (pi,pi)
#   Fig. 5    triangular, J2 = 0, K = (4pi/3, 0)
#
# ⛔ THE GROUND STATE IS `rsvd_cbe_dmrg1s!`, NOT the 2-site solver `fig4_2209.jl` used. That is
# the campaign's choice: the same rSVD-CBE selection that the BUG time step uses, so a knob tuned
# for the expansion is tuned for both halves of the calculation rather than for the evolution
# alone. `benchmarks/ground_state/gate_l20.jl` is the gate that must pass before anything here is
# read as physics -- every G(x,t) starts from |Om>, so an unconverged ground state does not give a
# slightly-wrong spectrum, it gives the spectrum of a different state.
#
# ⛔ U(1), FORCED BY THE OBSERVABLE. `apply_sz!` selects `SECTOR_UP`/`SECTOR_DOWN` blocks of the
# physical leg -- an abelian construction. S^z is not an SU(2) scalar; it is one component of a
# rank-1 tensor operator and takes the state out of S = 0. The SU(2) route needs the vector
# operator S, whose correlator is <S.S> = 3x this one, i.e. exactly the factor the paper drops.
#
# ⚠ WHAT A SMALL CYLINDER CAN AND CANNOT SHOW. The paper is C=6, L=36 -> 216 sites. Here the
# pipeline, the sum rules, the qualitative branch and the RELATIVE COST OF THE THREE INTEGRATORS
# are meaningful; absolute dispersion values are not a reproduction of their figure. Momentum
# resolution along the open direction is 2*pi/Lx and only the circumference momenta are exact --
# `geometry_report` prints which high-symmetry points survive BEFORE any run starts.
#
# Run:  julia -t 2 --project=. benchmarks/spectral/fig45_2209.jl <square|triangle> [Lx Ly]
# Env:  F_TMAX (40) F_DT (0.1) F_D (128) F_ARMS (tdvp2,cbe1s,bug) F_J2 (0.0) F_NPTS (12)
#       F_BUG_M (3)  pinned Krylov depth m; F_BUG_M=0 uses the adaptive beta rule instead
# Out:  results/fig45_G_<lat>_<Lx>x<Ly>_<arm>.csv     G(x,t), one row per (site,time)
#       results/fig45_Sqw_<lat>_<Lx>x<Ly>_<arm>.csv   S(q,w) on the path
#       results/fig45_cost_<lat>_<Lx>x<Ly>.csv        the integrator comparison

using LinearAlgebra, Printf, Serialization
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "lattice2d.jl"))
include(joinpath(@__DIR__, "bug_knobs.jl"))

const LATNAME = length(ARGS) >= 1 ? ARGS[1] : "square"
const LX   = length(ARGS) >= 3 ? parse(Int, ARGS[2]) : 5
const LY   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
const TMAX = parse(Float64, get(ENV, "F_TMAX", "40.0"))
const DT   = parse(Float64, get(ENV, "F_DT",   "0.1"))
const DCAP = parse(Int,     get(ENV, "F_D",    "128"))
# Ground-state cap, independent of the evolution cap -- see the block at the DMRG call.
const DGS  = parse(Int,     get(ENV, "F_DGS",  string(max(DCAP, 512))))
const J2   = parse(Float64, get(ENV, "F_J2",   "0.0"))
const NPTS = parse(Int,     get(ENV, "F_NPTS", "12"))
const BUGM = parse(Int,     get(ENV, "F_BUG_M", "3"))
const ARMS = split(get(ENV, "F_ARMS", "tdvp2,cbe1s,bug"), ',')
# `F_OUT` is a subdirectory UNDER results/, so a study's CSVs and its figures stay together
# instead of accumulating in one flat directory where a `fig45_*` glob picks up every campaign
# ever run. Empty default = results/ itself, so every existing invocation is unchanged.
# ⛔ `plot_fig45_2209.py` READS THE SAME VARIABLE. Set it once in the environment for the run and
# the plotter lands beside its own data; set it for only one of the two and the plotter silently
# globs a DIFFERENT study's CSVs and captions them with this run's tag.
const OUT  = normpath(joinpath(@__DIR__, "..", "results", get(ENV, "F_OUT", "")))

const LAT = lattice(LATNAME)
const N   = LX * LY
# Eq. (12) Gaussian broadening. ⚠ THIS IS A PHYSICS KNOB, NOT COSMETIC SMOOTHING. Sec. IID makes it
# the reliability criterion: "we choose eta as small as possible so that the spectral function is
# positive", and the reliable window is then T_max ~ 1/eta. So when the positivity gate fails, this
# is the first thing to raise -- ahead of the bond dimension. Defaults to the lattice's published
# value (eta^2 = 0.03 square, 0.02 triangle).
const ETA2 = parse(Float64, get(ENV, "F_ETA2", string(LAT.eta2)))

"""
    correlator(psi0, E0, step!, nt, dt) -> (ts, G, cost)

`G[i, n] = e^{i E0 t_n} <Om| S^z_i |phi(t_n)>` with `|phi> = S^z_c |Om>`, for EVERY site `i`.

Indexed by ABSOLUTE site, not displacement: the 2D transform needs `r_i - r_c` as a vector, which
one displacement index cannot carry on a cylinder.

`cost` accumulates what the arm actually spent, so the integrator comparison is measured on the
same trajectory that produced the physics rather than in a separate microbenchmark.
"""
function correlator(psi0::SymMPS, E0::Float64, step!, nt::Int, dt::Float64)
    L = length(psi0)
    c = centre_site(LX, LY)
    phi = deepcopy(psi0)
    apply_sz!(phi, c)

    # Built ONCE: `psi0` never changes and `apply_sz!` runs two canonical sweeps, so rebuilding
    # per step would cost L*nt canonicalisations of L distinct states.
    bras = map(1:L) do i
        b = deepcopy(psi0); apply_sz!(b, i); b
    end

    ts = collect(0:nt) .* dt
    G  = zeros(ComplexF64, L, nt + 1)
    cost = (mv = Ref(0), sec = Ref(0.0), chi = Ref(0))

    row!(col) = for i in 1:L
        G[i, col] = overlap(bras[i], phi)
    end

    row!(1)
    for n in 1:nt
        t0 = time()
        info = step!(phi, ComplexF64(-im * dt))
        cost.sec[] += time() - t0
        cost.mv[]  += info === nothing ? 0 : _matvec(info)
        cost.chi[]  = max(cost.chi[], maximum(bond_dims(phi)))
        row!(n + 1)
        # `e^{i E0 t}` removes the ground state's own phase; without it every G winds at E0 and
        # the whole spectrum is rigidly shifted.
        G[:, n + 1] .*= cis(E0 * ts[n + 1])
    end
    return ts, G, cost
end

"Operator applications an integrator reports. A missing field is a HARD ERROR, never a silent
zero -- a zero would read as 'this arm is free', the exact claim under measurement."
function _matvec(info)
    hasproperty(info, :krylov_dims) || error("integrator info has no krylov_dims: $(typeof(info))")
    kd = info.krylov_dims
    return kd isa Number ? Int(kd) : sum(kd)
end

"""
    structure_factor_2d(G, ts, qs; eta2, ws) -> Matrix

`S(q,w)` by **Eq. (14)** -- the paper's own cosine transform -- with Eq. (12) Gaussian damping:

    S(q,w) = 1/(pi*sqrt(N)) int_0^inf dt sum_x cos(q.x) [ cos(w t) Re G(x,t) - sin(w t) Im G(x,t) ]

⛔ EQ. (14), NOT EQ. (2). Sec. IID: "One issue with just naively truncating Eq. (2) is that S(q,w)
generically will acquire a non-zero imaginary part that is not physical. From there, one could only
look at the real part, or the magnitude, but we propose an alternative that ENFORCES REALITY."
Eq. (14) is that alternative: the cosine transform in space and the `cos/sin` pairing in time make
`S` real BY CONSTRUCTION, instead of real by discarding an imaginary part afterwards. It rests on
the Eq. (13) properties `G(-x,t) = G(x,t)` and `G(x,-t) = conj(G(x,t))`.

⚠ THE NORMALISATION IS `1/(pi*sqrt(N))`, NOT `1/N`. The previous `1/L` left every intensity off by
a constant `2*pi/sqrt(N)`; MEASURED as a uniform `int S dw / S(q) = 0.2988 +- 0.011` across 34 q,
which is the signature of a missing constant rather than a broken transform. Dispersions and the
SIGN structure are untouched by it -- so switching to Eq. (14) does NOT fix negative weight.

⚠ WHAT DOES FIX NEGATIVE WEIGHT IS `eta`, and that is the paper's own procedure: "we use a
physically motivated criterion, namely that the spectral function must be positive for all
frequencies ... we choose eta as small as possible so that the spectral function is positive. The
maximum time T_max for which the correlation function is reliable is approximated by T_max ~
1/eta." So a positivity failure is first a statement about the RELIABLE TIME WINDOW, and only
then about the bond dimension. `F_ETA2` overrides the lattice default so eta can be scanned.

⛔ POSITIONS ARE CARTESIAN (`lattice2d.jl`). On the triangular lattice the integer `(x,y)` are
coefficients in a 120-degree basis; transforming against them computes the structure factor of a
sheared square lattice, which is wrong without being obviously wrong.

The `t = 0` sample is counted once, not twice: it is the shared endpoint of the two half-lines.
"""
function structure_factor_2d(G::Matrix{ComplexF64}, ts::Vector{Float64},
                             qs::Vector{NTuple{2, Float64}};
                             eta2::Float64, ws::Vector{Float64})
    L, nt = size(G)
    dt = length(ts) > 1 ? ts[2] - ts[1] : 1.0
    rc = site_position(LAT, centre_site(LX, LY), LY)
    dr = [(site_position(LAT, i, LY)[1] - rc[1], site_position(LAT, i, LY)[2] - rc[2]) for i in 1:L]
    Wt = [exp(-eta2 * t^2) for t in ts]                       # Eq. (12), Gaussian

    S = zeros(Float64, length(qs), length(ws))
    norm14 = 1.0 / (pi * sqrt(L))
    for (iq, (qx, qy)) in enumerate(qs)
        # sum_x cos(q.x) G(x,t). The COSINE (not `cis`) is Eq. (14): it uses G(-x,t) = G(x,t) up
        # front instead of forming F(+q) and F(-q) and hoping they agree.
        Fc = [sum(G[i, n] * cos(qx * dr[i][1] + qy * dr[i][2]) for i in 1:L) for n in 1:nt]
        for (iw, om) in enumerate(ws)
            acc = 0.0
            for n in 1:nt
                wt = (n == 1 ? 0.5 : 1.0) * Wt[n] * dt       # t=0 is the endpoint of the half-line
                acc += wt * (cos(om * ts[n]) * real(Fc[n]) - sin(om * ts[n]) * imag(Fc[n]))
            end
            S[iq, iw] = acc * norm14
        end
    end
    return S
end

"""
    static_structure_factor(G, qs) -> Vector

`S(q)` of Eq. (4), the equal-time transform -- Fig. 5 a). This is the panel that shows the
120-degree order as a peak at `K`, and it is exactly the panel a circumference not divisible by 3
cannot produce (see `lattice2d.jl`).
"""
function static_structure_factor(G::Matrix{ComplexF64}, qs::Vector{NTuple{2, Float64}})
    L = size(G, 1)
    rc = site_position(LAT, centre_site(LX, LY), LY)
    dr = [(site_position(LAT, i, LY)[1] - rc[1], site_position(LAT, i, LY)[2] - rc[2]) for i in 1:L]
    return [real(sum(G[i, 1] * cis(-(q[1] * dr[i][1] + q[2] * dr[i][2])) for i in 1:L))
            for q in qs]
end

"Open Heisenberg chain, `Jm[i,i+1] = 1`. Same matrix `benchmarks/ground_state/gate_l20.jl` builds;
written out rather than included because that file is a standalone gate with its own `ARGS`."
function chain_couplings_1d(L::Int)
    Jm = zeros(L, L)
    for i in 1:(L - 1)
        Jm[i, i + 1] = 1.0
    end
    return Jm
end

# ⛔ `LAT.wrap` AND THE COUPLING MATRIX MUST COME FROM THE SAME SYMBOL -- see `lattice2d.jl`.
#
# ⛔ THIS DISPATCHES ON `LAT.name` WITH AN ERROR FALLTHROUGH, NOT ON `!= "square"`. The earlier
# ternary sent EVERY non-square name into the triangular builder, so adding the chain to
# `lattice()` would have silently evolved a triangular cylinder under a chain's label and a
# chain's Brillouin-zone path. The energy would have looked plausible and the spectrum would have
# been of a different Hamiltonian.
function couplings()
    LAT.name == "square"   && return square_cylinder_couplings(LX, LY; periodic_y = true)
    LAT.name == "triangle" && return triangular_cylinder_couplings(LX, LY; J2 = J2,
                                                                   geometry = LAT.wrap,
                                                                   periodic_y = true)
    if LAT.name == "chain"
        LY == 1 || throw(ArgumentError("the chain is Lx=L, Ly=1; got Ly=$LY"))
        return chain_couplings_1d(LX)
    end
    error("no coupling builder for lattice '$(LAT.name)'")
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)                      # forced by S^z -- see the header
    Jm = couplings()
    W  = pair_mpo(N, Jm, spin_vertices(N))
    nt = round(Int, TMAX / DT)
    c  = centre_site(LX, LY)

    @printf("\n%s\nFig.%s  %s cylinder  Lx(length)=%d Ly(circumference)=%d  N=%d  J2=%.3f\n",
            "="^100, LATNAME == "square" ? "4" : "5", LATNAME, LX, LY, N, J2)
    @printf("U(1) S^z=0; centre c=%d at r=%s; dt=%.3f Tmax=%.1f (%d steps) D=%d eta2=%.3f\n",
            c, string(round.(site_position(LAT, c, LY); digits = 3)), DT, TMAX, nt, DCAP, ETA2)
    full = geometry_report(stdout, LAT, LX, LY)
    @printf("  MPO virtual dim max %d\n%s\n", maximum(mpo_virtual_dims(W)), "="^100)

    # ── ground state: rsvd 1-site CBE-DMRG ──────────────────────────────────────────────────
    # ⛔ THE GROUND STATE GETS ITS OWN CAP, AND IT IS NOT `DCAP`. `DCAP` is the rank the TIME
    # EVOLUTION is held to -- that is the quantity this campaign compares integrators at. Using it
    # for the ground state too would confound the two: at DCAP=64 the N=18 solve does not converge
    # (MEASURED at DCAP=256, 30 sweeps: dev=+9.9e-05 with converged=false), and every arm would
    # then be evolving a DIFFERENT STATE rather than the ground state. That corrupts the one check
    # that says whether the spectrum is right at all -- the comparison against spin-wave theory --
    # because a wrong |Om> gives a wrong dispersion with no truncation error to blame it on.
    # `F_DGS` therefore defaults to a CONVERGED cap regardless of DCAP; set it equal to DCAP only
    # to deliberately reproduce the paper's single-chi protocol.
    # ⚠ THE GROUND STATE DOES NOT DEPEND ON `DCAP`, so a campaign that sweeps the EVOLUTION cap
    # re-solves a bit-identical state for every cap. MEASURED 2026-08-30 (square 5x4, N=20,
    # DGS=512): the solve owns ~51 min of a run whose first arm then took 8313 s. `run_2209_
    # campaign.sh` loops `for D in 64 128` OUTSIDE the geometry list and repeats the same three
    # geometries again in `knobs` and `cube`, so 15 of the campaign's solves are redundant.
    #
    # ⛔ THE KEY CARRIES EVERY INPUT THAT CHANGES THE STATE and is stored INSIDE the payload, not
    # just in the filename -- a filename cannot catch a file written by an older coupling builder,
    # and `couplings()` has already been wrong once (see the block above it). On load the energy is
    # RECOMPUTED against THIS run's own `W` and must reproduce the stored one; that is what makes
    # the cache safe rather than merely fast. A cache miss is silent-wrong physics if unchecked.
    # ⛔ EVERY cache path is wrapped: a truncated, corrupt or version-skewed `.jls` must fall back
    # to solving, never take the campaign down with it. `F_GS_CACHE=0` disables the whole thing.
    gskey = (LATNAME, LX, LY, N, J2, DGS, :U1, RSVD_KNOBS.dex, RSVD_KNOBS.dover,
             RSVD_KNOBS.comp_ratio, RSVD_KNOBS.growth)
    gsfile = joinpath(OUT, "gs_$(LATNAME)_$(LX)x$(LY)_" *
                           "J2$(replace(@sprintf("%.4f", J2), "." => "p"))_Dgs$(DGS).jls")
    usecache = get(ENV, "F_GS_CACHE", "1") == "1"

    psi0   = nothing
    cached = false
    if usecache && isfile(gsfile)
        try
            blob = Serialization.deserialize(gsfile)
            if blob.key != gskey
                @printf("  GS cache %s REJECTED -- key mismatch, re-solving\n", basename(gsfile))
            else
                cand = blob.psi
                Ec   = real(mpo_energy(copy(cand), W)) / max(norm(cand)^2, eps())
                if abs(Ec - blob.E0) < 1e-8 * max(1.0, abs(blob.E0))
                    psi0, cached = cand, true
                else
                    @printf("  GS cache %s REJECTED -- E0 %.10f vs stored %.10f, re-solving\n",
                            basename(gsfile), Ec, blob.E0)
                end
            end
        catch err
            @printf("  GS cache %s UNREADABLE (%s) -- re-solving\n", basename(gsfile), string(err))
        end
    end

    tgs, nsw, conv = 0.0, 0, true
    if psi0 === nothing
        psi0 = neel_state(N)                # half filling = the S^z = 0 sector
        tgs = @elapsed gs = rsvd_cbe_dmrg1s!(psi0, W; n_sweeps = 60, etol = 1e-12,
                                             maxdim = DGS, trunc_thresh = 1e-12,
                                             maxiter = 30, restol = 1e-10,
                                             dex = RSVD_KNOBS.dex, dover = RSVD_KNOBS.dover,
                                             comp_ratio = RSVD_KNOBS.comp_ratio,
                                             growth = RSVD_KNOBS.growth)
        nsw, conv = length(gs.energies), gs.converged
    end
    E0 = real(mpo_energy(copy(psi0), W)) / max(norm(psi0)^2, eps())
    chi0 = maximum(state_bond_dims(psi0))
    if cached
        @printf("ground state: E0=%.10f  E0/N=%.10f  chi=%d  REUSED from %s (solve skipped)\n",
                E0, E0 / N, chi0, basename(gsfile))
    else
        @printf("ground state: E0=%.10f  E0/N=%.10f  chi=%d  conv=%s  sweeps=%d  %.1fs\n",
                E0, E0 / N, chi0, string(conv), nsw, tgs)
    end
    # ⛔ A GROUND STATE THAT NEVER GREW IS THE FROZEN-START FAILURE, not a rank-1 ground state.
    # This also runs on the CACHE path -- a frozen state that once got serialised must not be
    # trusted a second time just because it came off disk.
    (chi0 <= 2 && (cached || nsw <= 3)) &&
        error("ground state FROZE at the product start (chi=$chi0) -- nothing below is physics; " *
              "see gate_l20.jl")

    if usecache && !cached
        try
            Serialization.serialize(gsfile, (key = gskey, psi = psi0, E0 = E0))
            @printf("  cached ground state -> %s\n", basename(gsfile))
        catch err
            @printf("  WARNING ground state not cached (%s)\n", string(err))
        end
    end
    println()

    # ⛔ THE BUG SETTINGS COME FROM THE chi=128 SCAN WHEN ONE EXISTS, and the driver says which.
    # `F_TUNED=0` forces the campaign defaults -- use it when the point of the run IS the
    # untuned baseline, so the two are never confused in the cost table.
    bugkw, prov = get(ENV, "F_TUNED", "1") == "1" ?
        optimal_bug_kwargs(geom_key(LATNAME, LX, LY); m_default = BUGM, D = DCAP) :
        (bug_step_kwargs(BUGM), "campaign DEFAULTS (F_TUNED=0)")
    @printf("BUG settings: %s\n", prov)
    # ⛔ ONE LITERAL FORMAT STRING. `@printf` refuses a concatenation -- `"a" * "b"` throws an
    # ArgumentError at LOAD time, passes every syntax check, and kills the run minutes into JIT.
    println("  m=$(bugkw.krylov_tol == 0.0 ? string(bugkw.krylov_basis) : "adaptive") " *
            "krylov_tol=$(bugkw.krylov_tol) stol_pre=$(bugkw.stol_pre) " *
            "split=$(bugkw.split_cutoff) root=$(bugkw.root_cutoff) dex=$(bugkw.dex) " *
            "dover=$(bugkw.dover) comp=$(bugkw.comp_ratio) growth=$(bugkw.growth) " *
            "parallel=$(bugkw.parallel)\n")

    kw = (maxdim = DCAP, trunc_thresh = 1e-12, maxiter = 8)
    steps = Dict{String, Any}(
        "tdvp2" => (p, tau) -> tdvp2_step!(p, W, tau; kw...),
        "cbe1s" => (p, tau) -> tdvp_cbe1s_step!(p, W, tau; kw...),
        "bug"   => (p, tau) -> cbe_bug_step!(p, W, tau; kw..., bugkw...))

    qs, labels, dists = bz_path(LAT, NPTS)
    ws = collect(range(0.0, 4.0; length = 401))
    # ⛔ THE EVOLUTION CAP IS PART OF THE FILENAME. Without it a chi=128 run silently OVERWRITES
    # the chi=64 one -- same lattice, same Lx x Ly, same arm -- and the pair that was supposed to
    # be a convergence comparison becomes one run twice. (This is the same failure `gate_l20.jl`
    # already hit when one fixed filename let a later geometry erase an earlier one.)
    tag = "$(LATNAME)_$(LX)x$(LY)_D$(DCAP)"

    costio = open(joinpath(OUT, "fig45_cost_$tag.csv"), "w")
    println(costio, "arm,lattice,Lx,Ly,N,J2,D,dt,tmax,bug_m,E0,E0_site,chi_gs,chi,matvec," *
                    "seconds,sumrule,gc0,t_farsite,full_bz")
    for arm in ARMS
        haskey(steps, arm) || (@printf("  unknown arm %s -- skipped\n", arm); continue)
        @printf("-- %s --\n", arm)
        ts, G, cost = correlator(psi0, E0, steps[arm], nt, DT)

        # ⛔ SUM RULES BEFORE ANY SPECTRUM. `sum_i G(i,0) = <Om| S^z_tot S^z_c |Om> = 0` in the
        # S^z = 0 sector and `G(c,0) = <(S^z_c)^2> = 1/4` exactly for spin-1/2. If either fails,
        # the operator or the overlap is wrong and no S(q,w) built from it means anything.
        g0 = sum(real, G[:, 1])
        gc = real(G[c, 1])
        @printf("   sum_i G(i,0) = %+.3e (must be 0)   G(c,0) = %.6f (must be 0.25)\n", g0, gc)
        abs(gc - 0.25) < 1e-6 || error("G(c,0) = $gc is not 1/4 -- S^z normalisation is wrong")
        abs(g0) < 1e-6 || error("sum_i G(i,0) = $g0 != 0 -- not in the S^z = 0 sector")

        # ⚠ LIGHT-CONE ARRIVAL AT THE GEOMETRICALLY FARTHEST SITE, reported not hidden: past it
        # the record is contaminated by the boundary. Probing site 1 instead would report the
        # correlation arriving next door.
        rc = site_position(LAT, c, LY)
        far = argmax([hypot(site_position(LAT, i, LY)[1] - rc[1],
                            site_position(LAT, i, LY)[2] - rc[2]) for i in 1:N])
        hit = findfirst(n -> abs(G[far, n]) > 0.05 * abs(G[c, 1]), 2:(nt + 1))
        tfar = hit === nothing ? TMAX : DT * hit

        open(joinpath(OUT, "fig45_G_$(tag)_$(arm).csv"), "w") do io
            println(io, "site,x,y,rx,ry,t,ReG,ImG")
            for i in 1:N, n in 1:(nt + 1)
                ix, iy = coords(i, LY); rx, ry = site_position(LAT, i, LY)
                @printf(io, "%d,%d,%d,%.6f,%.6f,%.4f,%.10e,%.10e\n",
                        i, ix, iy, rx, ry, ts[n], real(G[i, n]), imag(G[i, n]))
            end
        end

        S  = structure_factor_2d(G, ts, qs; eta2 = ETA2, ws = ws)
        Sq = static_structure_factor(G, qs)
        open(joinpath(OUT, "fig45_Sqw_$(tag)_$(arm).csv"), "w") do io
            println(io, "iq,qx,qy,label,dist,allowed,Sq,w,S")
            for iq in eachindex(qs), iw in eachindex(ws)
                @printf(io, "%d,%.6f,%.6f,%s,%.6f,%d,%.10e,%.6f,%.10e\n",
                        iq, qs[iq][1], qs[iq][2], labels[iq], dists[iq],
                        allowed_q(LAT, qs[iq], LY) ? 1 : 0, Sq[iq], ws[iw], S[iq, iw])
            end
        end
        @printf(costio, "%s,%s,%d,%d,%d,%.4f,%d,%.3f,%.1f,%d,%.8f,%.8f,%d,%d,%d,%.2f,%.3e,%.6f,%.2f,%d\n",
                arm, LATNAME, LX, LY, N, J2, DCAP, DT, TMAX, BUGM, E0, E0 / N, chi0,
                cost.chi[], cost.mv[], cost.sec[], g0, gc, tfar, full ? 1 : 0)
        @printf("   chi=%-4d matvec=%-8d %8.1fs   far-site arrival t~%.1f\n",
                cost.chi[], cost.mv[], cost.sec[], tfar)
        flush(costio)
    end
    close(costio)
    println("\nwrote ", OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
