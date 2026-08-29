# FIG. 4 OF arXiv:2209.00739 -- S(q,omega) OF THE SQUARE-LATTICE HEISENBERG MODEL,
# and the three integrators measured on the SAME correlator.
#
# The paper's Sec. IV A: "For the square lattice, we only look at the nearest neighbor model
# (J2 = 0), which serves as a benchmark to compare this method against quantum Monte Carlo".
# Their protocol, quoted:
#
#   Eq. (3)   G(x,t) = <Om| S_x(t) . S_c(0) |Om>,  c = the CENTRE site, x measured from it
#   Eq. (5)   G(x,t) = 3 <Om| S^z_x(t) S^z_c |Om>   -- "We drop the factor of 3 in this work."
#   Eq. (6)   G(x,t) = e^{i E0 t} <Om| S^z_x e^{-iHt} S^z_c |Om>
#   Eq. (2)   S(q,w) = (1/N) sum_x int dt/2pi e^{i(wt - q.x)} G(x,t)
#   Eq. (12)  Gaussian damping e^{-eta^2 t^2};  eta^2 = 0.03 for the SQUARE lattice
#   Eq. (19)  eps(q) := argmax_w S(q,w)      <- the dispersion they focus on
#   Eq. (20)  <w>(q) := int w S dw / int S dw
#   Sec. IIE  single-site TDVP, dt = 0.1, Tmax = 40, chi = 512, U(1) zero-magnetisation sector
#
# ⛔ THIS RUNS UNDER U(1), NOT SU(2), AND THAT IS FORCED BY THE OBSERVABLE. `apply_sz!` selects the
# `SECTOR_UP`/`SECTOR_DOWN` blocks of the physical leg -- an abelian construction. Under `:SU2` the
# physical leg is ONE spin-1/2 multiplet with no such grading, because `S^z` is not an SU(2) scalar:
# it is one component of a rank-1 tensor operator and takes the state out of the S = 0 sector. The
# SU(2) extension of this figure needs the VECTOR operator `S`, whose correlator is `<S.S>` = 3x
# this one -- i.e. exactly the factor the paper drops in Eq. (5). Until that operator exists, this
# file reproduces the paper on the paper's own symmetry. It is not a limitation of the integrator.
#
# ⛔ THE 1D RING SHORTCUT IN `hs_structure_factor.jl` DOES NOT APPLY HERE, and its own docstring
# says so: that transform uses ring momenta and `C(-d,t) = C(d,t)`, which "rests on the ring's
# reflection symmetry through j0; on an OPEN chain C(-d,t) != C(d,t) and this shortcut is wrong."
# A cylinder is open along its length. So the spatial transform below runs over ABSOLUTE 2D
# positions and assumes no reflection; only the TIME conjugation is reused, and that one is exact:
# with H real symmetric in the S^z basis and |Om> real, `G(x,-t) = conj(G(x,t))` (the paper's
# Eq. (13)), because <0|A e^{iHt} B|0> = <0|B e^{iHt} A|0> by transposing real symmetric matrices.
#
# ⚠ WHAT 12 SITES CAN AND CANNOT SHOW. The paper is C=6, L=36 -> 216 sites. At 12 sites S(q,w) is a
# handful of delta functions smeared by the window: the PIPELINE, the sum rules, the qualitative
# magnon branch and the RELATIVE COST OF THE THREE INTEGRATORS are meaningful; the dispersion
# values are not a reproduction of their Fig. 4 d). Momentum resolution along the open direction is
# 2*pi/Lx, and only q_y = 2*pi*n/Ly are exact quantum numbers.
#
# Run:  julia -t 2 --project=. benchmarks/spectral/fig4_2209.jl [Lx Ly]
# Env:  F4_TMAX (40) F4_DT (0.1) F4_D (64) F4_ETA2 (0.03) F4_ARMS (tdvp2,cbe1s,bug)
# Out:  benchmarks/results/fig4_G_<Lx>x<Ly>_<arm>.csv     C(x,t), one row per (site,time)
#       benchmarks/results/fig4_cost_<Lx>x<Ly>.csv        the integrator comparison

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

const LX   = length(ARGS) >= 2 ? parse(Int, ARGS[1]) : 3
const LY   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
const TMAX = parse(Float64, get(ENV, "F4_TMAX", "40.0"))
const DT   = parse(Float64, get(ENV, "F4_DT",   "0.1"))
const DCAP = parse(Int,     get(ENV, "F4_D",    "64"))
const ARMS = split(get(ENV, "F4_ARMS", "tdvp2,cbe1s,bug"), ',')
const OUT  = joinpath(@__DIR__, "..", "results")

"`site(x,y)` -- the column-major snake the lattice models use. Keep it identical or the Fourier
transform below is taken against the wrong positions, which is silent."
site(x, y) = (x - 1) * LY + y

"Real-space position of a site index, as `(x, y)` in lattice units."
function position(i::Int)
    x = div(i - 1, LY) + 1
    y = mod(i - 1, LY) + 1
    return (Float64(x), Float64(y))
end

"""
    centre_site() -> Int

The paper takes `c` to be "the center site in the lattice". On an even-length cylinder there is no
exact centre, so this picks the lower-middle column and row -- and the choice is RECORDED in the
output rather than assumed, because every `x` in `G(x,t)` is measured from it.
"""
centre_site() = site(cld(LX, 2), cld(LY, 2))

"""
    correlator(psi0, E0, step!, nt, dt) -> (ts, G, cost)

`G[i, n] = e^{i E0 t_n} <Om| S^z_i |phi(t_n)>` with `|phi> = S^z_c |Om>`, for EVERY site `i`.

⛔ INDEXED BY ABSOLUTE SITE, NOT BY DISPLACEMENT. The 2D transform needs `r_i - r_c` as a vector,
which a single displacement index cannot carry on a cylinder; `structure_factor_2d` subtracts the
centre position itself. (`hs_structure_factor.jl` re-centres instead because its ring transform
uses the row index directly -- a different, 1D-only contract.)

`cost` accumulates what the arm actually spent, so the integrator comparison is measured on the
same trajectory that produced the physics rather than in a separate microbenchmark.
"""
function correlator(psi0::SymMPS, E0::Float64, step!, nt::Int, dt::Float64)
    L = length(psi0)
    c = centre_site()
    phi = deepcopy(psi0)
    apply_sz!(phi, c)

    # The L bras are built ONCE: `psi0` never changes and `apply_sz!` runs two full canonical
    # sweeps, so rebuilding them per step would cost L*nt canonicalisations for L distinct states.
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
        # `e^{i E0 t}` removes the ground state's own phase; without it every G(x,t) winds at E0
        # and the whole spectrum is rigidly shifted.
        G[:, n + 1] .*= cis(E0 * ts[n + 1])
    end
    return ts, G, cost
end

"Operator applications an integrator reports. The three arms name the field differently, and a
missing field must be a hard error rather than a silent zero -- a zero would read as 'this arm is
free', which is the exact claim the campaign is trying to measure."
function _matvec(info)
    hasproperty(info, :krylov_dims) || error("integrator info has no krylov_dims: $(typeof(info))")
    kd = info.krylov_dims
    return kd isa Number ? Int(kd) : sum(kd)
end

"""
    structure_factor_2d(G, ts, qs; eta2, dt) -> Matrix  (length(qs) x length(ws))

`S(q,w)` by Eq. (2), with the Gaussian damping of Eq. (12) and NO reflection assumption:

    F(q,t) = sum_i e^{-i q.(r_i - r_c)} G[i,t]
    S(q,w) = (1/N) Re int_0^inf dt [ e^{iwt} F(q,t) + e^{-iwt} conj(F(-q,t)) ] W(t)

⛔ THE `F(-q,t)` TERM IS WHY THIS IS NOT `2*Re(...)`. The usual doubling needs `F(-q,t) = F(q,t)`,
i.e. inversion symmetry of the lattice about `c`. On a finite cylinder with an even side there is
no site at the exact centre, so that symmetry is BROKEN and the doubling would mix the real and
imaginary parts of the transform -- giving an `S(q,w)` that is neither positive nor band-limited
and looks like a broken integrator. Computing `F(-q,t)` costs one more spatial sum and removes the
assumption entirely.

The `t = 0` sample is counted once, not twice: it is the shared endpoint of the two half-lines.
"""
function structure_factor_2d(G::Matrix{ComplexF64}, ts::Vector{Float64},
                             qs::Vector{Tuple{Float64, Float64}};
                             eta2::Float64, ws::Vector{Float64})
    L, nt = size(G)
    dt = length(ts) > 1 ? ts[2] - ts[1] : 1.0
    rc = position(centre_site())
    dr = [(position(i)[1] - rc[1], position(i)[2] - rc[2]) for i in 1:L]
    W  = [exp(-eta2 * t^2) for t in ts]                       # Eq. (12), Gaussian

    S = zeros(Float64, length(qs), length(ws))
    for (iq, (qx, qy)) in enumerate(qs)
        Fp = [sum(G[i, n] * cis(-(qx * dr[i][1] + qy * dr[i][2])) for i in 1:L) for n in 1:nt]
        Fm = [sum(G[i, n] * cis(+(qx * dr[i][1] + qy * dr[i][2])) for i in 1:L) for n in 1:nt]
        for (iw, om) in enumerate(ws)
            acc = 0.0 + 0.0im
            for n in 1:nt
                wt = (n == 1 ? 0.5 : 1.0) * W[n] * dt        # t=0 shared by the two half-lines
                acc += wt * (cis(om * ts[n]) * Fp[n] + cis(-om * ts[n]) * conj(Fm[n]))
            end
            S[iq, iw] = real(acc) / L
        end
    end
    return S
end

"""
The Fig. 4 a) path through the square-lattice Brillouin zone: `Gamma -> X -> M -> Gamma`.

⚠ ONLY `q_y = 2*pi*n/Ly` ARE EXACT QUANTUM NUMBERS -- the circumference is the only periodic
direction. `q_x` is sampled continuously along the open direction, where the resolution is
`2*pi/Lx`; the `exact_qy` column records which points sit on an allowed `q_y` so the figure can
mark them rather than implying every point is equally meaningful.
"""
function bz_path(npts::Int)
    corners = [(0.0, 0.0), (pi, 0.0), (pi, pi), (0.0, 0.0)]
    names   = ["G", "X", "M", "G"]
    qs = Tuple{Float64, Float64}[]; labels = String[]; dists = Float64[]
    d = 0.0
    for s in 1:(length(corners) - 1)
        a, b = corners[s], corners[s + 1]
        seg = hypot(b[1] - a[1], b[2] - a[2])
        for k in 0:(npts - 1)
            f = k / npts
            push!(qs, (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])))
            push!(labels, k == 0 ? names[s] : "")
            push!(dists, d + f * seg)
        end
        d += seg
    end
    push!(qs, corners[end]); push!(labels, names[end]); push!(dists, d)
    return qs, labels, dists
end

"`q_y` is an exact quantum number only on the circumference lattice `2*pi*n/Ly`."
exact_qy(qy) = any(abs(rem(qy - 2pi * n / LY, 2pi, RoundNearest)) < 1e-9 for n in 0:(LY - 1))

function main()
    N = LX * LY
    mkpath(OUT)
    set_symmetry!(:U1)                      # forced by S^z -- see the header
    W = square_cylinder_mpo(LX, LY)
    nt = round(Int, TMAX / DT)

    @printf("\n%s\nFig.4  square cylinder  Lx(length)=%d Ly(circumference)=%d  N=%d\n",
            "="^100, LX, LY, N)
    @printf("U(1) S^z=0 sector; centre site c=%d at r=%s; dt=%.3f Tmax=%.1f (%d steps) D=%d\n%s\n",
            centre_site(), string(position(centre_site())), DT, TMAX, nt, DCAP, "="^100)

    # ── ground state ────────────────────────────────────────────────────────────────────────
    psi0 = neel_state(N)                    # half filling = the S^z = 0 sector
    opts = CBEBugOptions(; maxdim = DCAP, trunc_thresh = 1e-12, dex = 8, dover = 4)
    tgs = @elapsed gs = GroundState.solve!(psi0, W; opts = opts, n_sweeps = 60, etol = 1e-12,
                                           grow_iters = 1)
    E0 = real(mpo_energy(copy(psi0), W)) / max(norm(psi0)^2, eps())
    @printf("ground state: E0=%.10f  E0/N=%.10f  chi=%d  conv=%s  sweeps=%d  %.1fs\n\n",
            E0, E0 / N, maximum(bond_dims(psi0)), string(gs.converged),
            length(gs.energies), tgs)

    kw = (maxdim = DCAP, trunc_thresh = 1e-12, maxiter = 8)
    steps = Dict(
        "tdvp2" => (p, tau) -> tdvp2_step!(p, W, tau; kw...),
        "cbe1s" => (p, tau) -> tdvp_cbe1s_step!(p, W, tau; kw...),
        # The campaign's BUG settings: `dex/dover` are the expansion, `grow_iters` the growth
        # passes. `krylov_basis = 3, krylov_tol = 0` is the measured-fast Krylov setting.
        "bug"   => (p, tau) -> cbe_bug_step!(p, W, tau; kw..., dex = 8, dover = 4,
                                             grow_iters = 1, krylov_tol = 0.0))

    qs, labels, dists = bz_path(12)
    ws = collect(range(0.0, 4.0; length = 401))
    eta2 = parse(Float64, get(ENV, "F4_ETA2", "0.03"))

    costio = open(joinpath(OUT, "fig4_cost_$(LX)x$(LY).csv"), "w")
    println(costio, "arm,Lx,Ly,N,D,dt,tmax,E0,E0_site,chi,matvec,seconds,sumrule,t_farsite")
    for arm in ARMS
        haskey(steps, arm) || (@printf("  unknown arm %s -- skipped\n", arm); continue)
        @printf("-- %s --\n", arm)
        ts, G, cost = correlator(psi0, E0, steps[arm], nt, DT)

        # ⛔ SUM RULE BEFORE ANY SPECTRUM. `sum_i G(i,0) = <Om| S^z_tot S^z_c |Om> = 0` in the
        # S^z = 0 sector, and `G(c,0) = <(S^z_c)^2> = 1/4` exactly for spin-1/2. If either fails,
        # the operator or the overlap is wrong and no S(q,w) computed from it means anything.
        g0 = sum(real, G[:, 1])
        gc = real(G[centre_site(), 1])
        @printf("   sum_i G(i,0) = %+.3e (must be 0)   G(c,0) = %.6f (must be 0.25)\n", g0, gc)
        abs(gc - 0.25) < 1e-6 || @printf("   ⛔ G(c,0) is NOT 1/4 -- S^z normalisation is wrong\n")
        abs(g0) < 1e-6 || @printf("   ⛔ sum_i G(i,0) != 0 -- not in the S^z = 0 sector\n")

        # ⚠ THE LIGHT-CONE ARRIVAL AT THE FARTHEST SITE, reported rather than hidden: past it the
        # record is contaminated by the boundary and is no longer the thermodynamic correlator.
        # The Gaussian eta^2=0.03 suppresses t >~ 1/eta ~ 5.8 anyway, which is why the paper's
        # Tmax=40 is harmless here -- but the number belongs in the output.
        #
        # ⛔ THE PROBE SITE MUST BE THE GEOMETRICALLY FARTHEST ONE, NOT SITE 1. On a 12-site
        # lattice site 1 sits at r=(1,1) against a centre at (2,2) -- a NEAREST neighbour -- so
        # probing it reported "recurrence at t~0.1" on the very first step, which is the
        # correlation arriving next door, not the boundary coming back.
        far = argmax([hypot(position(i)[1] - position(centre_site())[1],
                            position(i)[2] - position(centre_site())[2]) for i in 1:N])
        hit = findfirst(n -> abs(G[far, n]) > 0.05 * abs(G[centre_site(), 1]), 2:(nt + 1))
        tfar = hit === nothing ? TMAX : DT * hit

        open(joinpath(OUT, "fig4_G_$(LX)x$(LY)_$(arm).csv"), "w") do io
            println(io, "site,x,y,t,ReG,ImG")
            for i in 1:N, n in 1:(nt + 1)
                px, py = position(i)
                @printf(io, "%d,%.1f,%.1f,%.4f,%.10e,%.10e\n",
                        i, px, py, ts[n], real(G[i, n]), imag(G[i, n]))
            end
        end

        S = structure_factor_2d(G, ts, qs; eta2 = eta2, ws = ws)
        open(joinpath(OUT, "fig4_Sqw_$(LX)x$(LY)_$(arm).csv"), "w") do io
            println(io, "iq,qx,qy,label,dist,exact_qy,w,S")
            for iq in eachindex(qs), iw in eachindex(ws)
                @printf(io, "%d,%.6f,%.6f,%s,%.6f,%d,%.6f,%.10e\n",
                        iq, qs[iq][1], qs[iq][2], labels[iq], dists[iq],
                        exact_qy(qs[iq][2]) ? 1 : 0, ws[iw], S[iq, iw])
            end
        end
        @printf(costio, "%s,%d,%d,%d,%d,%.3f,%.1f,%.8f,%.8f,%d,%d,%.2f,%.3e,%.2f\n",
                arm, LX, LY, N, DCAP, DT, TMAX, E0, E0 / N, cost.chi[], cost.mv[],
                cost.sec[], g0, tfar)
        @printf("   chi=%-3d matvec=%-7d %7.1fs   far-site arrival t~%.1f\n",
                cost.chi[], cost.mv[], cost.sec[], tfar)
        flush(costio)
    end
    close(costio)
    println("\nwrote ", OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
