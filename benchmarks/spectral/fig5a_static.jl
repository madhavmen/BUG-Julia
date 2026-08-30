# FIG. 5 a) OF arXiv:2209.00739 -- THE STATIC STRUCTURE FACTOR, AS A GEOMETRY SANITY CHECK.
#
# WHY THIS FIRST, AND WHY IT IS CHEAP. `S(q)` is an EQUAL-TIME quantity: it needs the ground state
# and nothing else -- no propagation, no Krylov, no integrator. So it costs one DMRG solve and
# `N^2` overlaps, and it exercises every part of the spectral pipeline that is NOT the time
# evolution:
#
#   * the Cartesian positions -- under the paper's XC wrapping the triangle's are the ZIGZAG
#     embedding, NOT `r = x*a1 + y*a2` (`lattice2d.jl`, `triangular_site_position`)
#   * the allowed-momentum rule `q.T in 2*pi*Z` with the wrapping's own `T`
#   * the site ordering of the column-major snake
#   * the Brillouin-zone path and its high-symmetry points
#
# ⛔ AND IT HAS A SHARP PASS/FAIL, WHICH IS THE POINT. The triangular Heisenberg ground state is
# 120-degree ordered, so `S(q)` MUST peak at `K = (4*pi/3, 0)`. A peak anywhere else means the
# geometry is wrong -- sheared positions, a mis-ordered snake, the wrong wrap direction -- and
# every dynamical figure built on it would be wrong in the same way while still looking like a
# spectrum. The paper's Fig. 5 a) shows exactly this peak and the text confirms it: "In Fig. 5 a)
# we show the static structure factor obtained from DMRG, and see ordering at q = K as expected."
#
# ⛔ THE TRIANGLE IS THE PAPER'S `XC` WRAPPING (Sec. IIB), NOT `YC`. Under XC, `K` has `q_y = 0`
# and is exact at EVERY EVEN `C` -- the old "K exists only when 3 | Ly" rule was a YC artifact, not
# lattice physics, and XC is precisely what the paper chose to avoid it. **`C` must be even**: the
# zigzag ring has a two-site period, so `C = 3` is not a cylinder XC can build. `geometry_report`
# prints the wrap vector and the exact/image/interpolated breakdown before anything runs, and the
# verdict below is suppressed rather than reported as a failure when a corner is unrepresentable.
#
# ⚠ THE SQUARE LATTICE HAS THE SAME CHECK with a different answer: the Neel state orders at
# `M = (pi,pi)`, which is allowed at every even circumference. Running `square` here is the
# control -- a pipeline that puts the triangular peak at K and the square peak at M is not
# accidentally right.
#
# ⛔ THE MPS IS CHECKED AGAINST EXACT ED, NOT TRUSTED. At `N <= 20` the `S^z = 0` sector is small
# enough that `C[i,j] = <S^z_i S^z_j>` is available exactly (S^z is DIAGONAL in that basis, so it
# is a weighted sum over amplitudes, not an operator application). The two correlation matrices
# must agree before either is transformed -- otherwise a disagreement in `S(q)` cannot be
# attributed between the state and the transform.
#
# TWO DEFINITIONS ARE COMPUTED AND BOTH ARE REPORTED:
#   full     S(q) = (1/N) sum_{i,j} e^{-i q.(r_i - r_j)} C[i,j]     the textbook double sum
#   centred  S(q) = sum_i e^{-i q.(r_i - r_c)} C[i,c]               referenced to the centre site
# The centred form is what the paper's own `G(x, t=0)` gives (Eq. 6 applies `S^z_c` to `|Om>`), so
# it is the one the dynamical figure must match; the full form is translation-averaged and is the
# cleaner peak on a finite open cylinder. If the two disagree about WHERE the peak is, the finite
# cylinder's lack of translation invariance is the reason, and that belongs in the caption.
#
# Run:  julia --project=. benchmarks/spectral/fig5a_static.jl <triangle|square> [Lx Ly]
# Env:  S5_D (256) S5_SWEEPS (30) S5_J2 (0.0) S5_NQ (121)
# Out:  benchmarks/results/fig5a_<lat>_<Lx>x<Ly>.csv        S(q) on a 2D grid
#       benchmarks/results/fig5a_path_<lat>_<Lx>x<Ly>.csv   S(q) along the BZ path

using LinearAlgebra, SparseArrays, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))
include(joinpath(@__DIR__, "lattice2d.jl"))

const LATNAME = length(ARGS) >= 1 ? ARGS[1] : "triangle"
const LX   = length(ARGS) >= 3 ? parse(Int, ARGS[2]) : 6
const LY   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const DCAP = parse(Int,     get(ENV, "S5_D",      "256"))
const NSW  = parse(Int,     get(ENV, "S5_SWEEPS", "30"))
const J2   = parse(Float64, get(ENV, "S5_J2",     "0.0"))
const NQ   = parse(Int,     get(ENV, "S5_NQ",     "121"))
const LAT  = lattice(LATNAME)
const N    = LX * LY
const OUT  = joinpath(@__DIR__, "..", "results")

# ⛔ `LAT.wrap` AND THE COUPLING MATRIX MUST COME FROM THE SAME SYMBOL. The positions used by the
# Fourier transform are `triangular_site_position(...; geometry = LAT.wrap)`; if the Hamiltonian
# were built with a different wrapping, S(q) would transform a lattice the state never lived on.
couplings() = LATNAME == "square" ?
    square_cylinder_couplings(LX, LY; periodic_y = true) :
    triangular_cylinder_couplings(LX, LY; J2 = J2, geometry = LAT.wrap, periodic_y = true)

"""
    sq_full(C, dr) -> function

`S(q) = (1/N) sum_{i,j} e^{-i q.(r_i - r_j)} C[i,j]`, real by construction (`C` is symmetric and
the phases conjugate in pairs), so the imaginary part is a roundoff check rather than a result.
"""
function sq_full(C::Matrix{Float64}, r::Vector{NTuple{2, Float64}})
    L = size(C, 1)
    return function (q)
        acc = 0.0 + 0.0im
        for i in 1:L, j in 1:L
            acc += C[i, j] * cis(-(q[1] * (r[i][1] - r[j][1]) + q[2] * (r[i][2] - r[j][2])))
        end
        return real(acc) / L
    end
end

"`S(q)` referenced to the centre site -- the paper's `G(x, t=0)` transformed."
function sq_centred(C::Matrix{Float64}, r::Vector{NTuple{2, Float64}}, c::Int)
    L = size(C, 1)
    return function (q)
        acc = 0.0 + 0.0im
        for i in 1:L
            acc += C[i, c] * cis(-(q[1] * (r[i][1] - r[c][1]) + q[2] * (r[i][2] - r[c][2])))
        end
        return real(acc)
    end
end

"Exact `C[i,j] = <S^z_i S^z_j>` from the sparse ground vector. `S^z` is DIAGONAL here, so this is
a weighted sum over amplitudes and costs `N^2 * dim` flops with no operator ever built."
function exact_corr(states::Vector{Int}, Om::Vector{Float64}, L::Int)
    w = Om .^ 2
    sz = [sz_diag(states, i) for i in 1:L]
    C = zeros(Float64, L, L)
    for i in 1:L, j in i:L
        v = sum(w[k] * sz[i][k] * sz[j][k] for k in eachindex(w))
        C[i, j] = v; C[j, i] = v
    end
    return C
end

"`C[i,j]` from an MPS, by building `S^z_i|psi>` once per site and overlapping the pairs."
function mps_corr(psi, L::Int)
    bras = map(1:L) do i
        b = deepcopy(psi); apply_sz!(b, i); b
    end
    C = zeros(Float64, L, L)
    for i in 1:L, j in i:L
        v = real(overlap(bras[i], bras[j]))
        C[i, j] = v; C[j, i] = v
    end
    return C
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)
    Jm = couplings()
    W  = pair_mpo(N, Jm, spin_vertices(N))
    c  = centre_site(LX, LY)
    r  = [site_position(LAT, i, LY) for i in 1:N]

    @printf("\n%s\nFig.5a  STATIC STRUCTURE FACTOR  --  %s  Lx=%d Ly=%d  N=%d  J2=%.3f  D=%d\n",
            "="^104, LATNAME, LX, LY, N, J2, DCAP)
    full_bz = geometry_report(stdout, LAT, LX, LY)
    @printf("  centre site c=%d at r=%s;  MPO virtual dim max %d\n%s\n",
            c, string(round.(r[c]; digits = 3)), maximum(mpo_virtual_dims(W)), "="^104)
    flush(stdout)

    # ── exact reference ─────────────────────────────────────────────────────────────────────
    t0 = time()
    H, states, = pairs_sparse(N, Jm)
    E0x, Om, resid, = sparse_ground(H, length(states))
    resid < 1e-9 || error("Lanczos residual $resid -- reference not converged")
    Cx = exact_corr(states, Om, N)
    @printf("exact  E0/N=%.12f  resid=%.2e   C built  %.1fs\n", E0x / N, resid, time() - t0)
    flush(stdout)

    # ── the MPS ground state ────────────────────────────────────────────────────────────────
    psi = neel_state(N)
    # ⛔ `restol = 1e-6`, NOT THE 1e-10 DEFAULT. MEASURED on the 5x4 square cylinder at D=128
    # (`dmrg_knobs.jl`), against exact ED, with matvec deterministic so this is not clock noise:
    #
    #   restol 1e-10 (default)   dev/site 5.037e-07   matvec 5467   121.6 s
    #   restol 1e-6              dev/site 5.023e-07   matvec 2704    54.6 s
    #   maxiter 10 (vs 30)       dev/site 5.029e-07   matvec 2703    60.0 s
    #
    # HALF THE OPERATOR APPLICATIONS FOR THE SAME ENERGY -- and marginally BETTER, so it is not a
    # trade-off. A DMRG sweep does not need its local eigensolve converged to 1e-10: the sweep
    # re-solves the same site on the next pass, so an over-tight local tolerance buys iterations
    # that the following sweep discards. (This is the DMRG analogue of the Krylov-depth result
    # for the time integrator, and the opposite conclusion from the one that holds there, where
    # a direction not admitted is gone for good.)
    tgs = @elapsed run = rsvd_cbe_dmrg1s!(psi, W; n_sweeps = NSW, etol = 1e-12, maxdim = DCAP,
                                          trunc_thresh = 1e-12, dex = 8, dover = 4,
                                          comp_ratio = 1.0, maxiter = 30, restol = 1e-6)
    E = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    @printf("MPS    E0/N=%.12f  dev=%+.3e  chi=%d  sweeps=%d  conv=%s  %.1fs\n",
            E / N, E - E0x, maximum(state_bond_dims(psi)), length(run.energies),
            string(run.converged), tgs)
    flush(stdout)

    Cm = mps_corr(psi, N)
    dC = maximum(abs.(Cm .- Cx))
    # ⛔ TWO FAILURES LOOK IDENTICAL HERE AND HAVE OPPOSITE FIXES, so the energy decides which.
    # A CONVENTION mismatch (mirrored bit order, wrong snake, wrong wrap) misplaces sites and is
    # visible in `C` while the ENERGY still matches -- the Hamiltonian is a scalar and does not
    # care how the basis is labelled. An UNCONVERGED ground state instead shows `dev > 0` (the
    # variational bound) with `converged = false`, and `C` is simply the correlation matrix of a
    # different, higher state. Chasing a convention bug when the solve merely needs more sweeps
    # or a larger cap is a long detour; the message below refuses to guess.
    dev = E - E0x
    unconverged = dev > 1e-7 || !run.converged
    verdict = dC < 1e-7 ? "PASS" :
              unconverged ?
                  @sprintf("⛔ FAIL -- but dev=%+.3e (>0) and converged=%s, so this is an " *
                           "UNCONVERGED ground state, NOT a convention mismatch: raise " *
                           "S5_D / S5_SWEEPS", dev, string(run.converged)) :
                  "⛔ FAIL -- energy is converged, so this IS a state or site convention mismatch"
    @printf("correlation gate: max|C_mps - C_exact| = %.3e   %s\n", dC, verdict)
    flush(stdout)
    # ⛔ TRANSFORM ONLY WHAT AGREES. If the two correlation matrices differ, an S(q) built from
    # either cannot be attributed between the state and the geometry.
    dC < 1e-7 || error("correlation matrices disagree by $dC")

    # ⛔ SUM RULE ON C ITSELF, before any transform: sum_ij C[i,j] = <(S^z_tot)^2> = 0 in this
    # sector, and C[c,c] = 1/4 for spin-1/2. Both are properties of the OPERATOR, so a failure
    # here is `apply_sz!`, not the physics.
    @printf("sum rules: sum_ij C = %+.3e (must be 0)   C[c,c] = %.6f (must be 0.25)\n\n",
            sum(Cx), Cx[c, c])
    flush(stdout)

    Sfull, Scen = sq_full(Cx, r), sq_centred(Cx, r, c)
    SfullM = sq_full(Cm, r)

    # ── the BZ path ─────────────────────────────────────────────────────────────────────────
    qs, labels, dists = bz_path(LAT, 24)
    open(joinpath(OUT, "fig5a_path_$(LATNAME)_$(LX)x$(LY).csv"), "w") do io
        # `status` distinguishes a MEASUREMENT from the paper's Eq. (10) symmetry fill-in from a
        # plain interpolation -- see `q_status` in lattice2d.jl. A figure that lumps the three
        # together presents continuation as data.
        println(io, "iq,qx,qy,label,dist,allowed,status,S_full,S_centred,S_full_mps")
        for (iq, q) in enumerate(qs)
            @printf(io, "%d,%.6f,%.6f,%s,%.6f,%d,%s,%.10e,%.10e,%.10e\n",
                    iq, q[1], q[2], labels[iq], dists[iq],
                    allowed_q(LAT, q, LY) ? 1 : 0, q_status(LAT, q, LY),
                    Sfull(q), Scen(q), SfullM(q))
        end
    end

    # ── a 2D map over the zone ──────────────────────────────────────────────────────────────
    qmax = LATNAME == "triangle" ? 5.0 : 2pi
    grid = range(-qmax, qmax; length = NQ)
    best = (-Inf, (0.0, 0.0)); best_allowed = (-Inf, (0.0, 0.0))
    open(joinpath(OUT, "fig5a_$(LATNAME)_$(LX)x$(LY).csv"), "w") do io
        println(io, "qx,qy,allowed,status,S_full,S_centred")
        for qx in grid, qy in grid
            q = (qx, qy); s = Sfull(q); al = allowed_q(LAT, q, LY)
            @printf(io, "%.6f,%.6f,%d,%s,%.10e,%.10e\n",
                    qx, qy, al ? 1 : 0, q_status(LAT, q, LY), s, Scen(q))
            s > best[1] && (best = (s, q))
            al && s > best_allowed[1] && (best_allowed = (s, q))
        end
    end

    # ── the verdict ─────────────────────────────────────────────────────────────────────────
    target = LATNAME == "triangle" ? (4pi / 3, 0.0) : (pi, pi)
    tname  = LATNAME == "triangle" ? "K=(4π/3,0)" : "M=(π,π)"
    # ⚠ THE PEAK IS COMPARED UP TO THE POINT GROUP. `K` has six images and `M` two, so a peak at
    # a symmetry partner is the SAME answer; comparing against one representative alone would
    # report a correct run as a failure.
    imgs = LATNAME == "triangle" ?
        [(4pi / 3 * cos(t), 4pi / 3 * sin(t)) for t in (0:5) .* (pi / 3)] :
        [(pi, pi), (pi, -pi), (-pi, pi), (-pi, -pi)]
    d = minimum(hypot(best[2][1] - m[1], best[2][2] - m[2]) for m in imgs)
    tol = 2 * (2qmax / (NQ - 1))                    # within a couple of grid steps
    @printf("peak of S(q):        q = (%+.4f, %+.4f)  S = %.6f\n", best[2]..., best[1])
    @printf("nearest %s image:  distance %.4f  (grid step %.4f)\n",
            tname, d, 2qmax / (NQ - 1))
    @printf("peak over ALLOWED q: q = (%+.4f, %+.4f)  S = %.6f\n",
            best_allowed[2]..., best_allowed[1])
    if !full_bz
        println("\n⚠ VERDICT SUPPRESSED: this circumference cannot represent $tname, so the " *
                "peak is not expected there and its absence is geometry, not a defect.")
    elseif d <= tol
        println("\n✅ PASS -- S(q) peaks at $tname, the expected ordering wavevector.")
    else
        println("\n⛔ FAIL -- S(q) does NOT peak at $tname. The geometry is wrong (sheared " *
                "positions, mis-ordered snake, or wrong wrap); no dynamical figure built on " *
                "it can be trusted.")
    end
    println("\nwrote ", OUT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
