# THE L = 20 GROUND-STATE GATE FOR THE arXiv:2209.00739 CAMPAIGN (Fig. 3 at J2 = 0).
#
# NOTHING IN THE SPECTRAL CAMPAIGN IS MEANINGFUL UNTIL THIS PASSES. Figs. 4 and 5 are
# `G(x,t) = <Om| S^z_x e^{-iHt} S^z_c |Om>` -- every one of them starts from `|Om>`, so a
# ground state that has not converged does not produce a slightly-wrong spectrum, it produces
# a spectrum of the wrong state. This driver answers one question per geometry: DOES
# `rsvd_cbe_dmrg1s!` REACH THE EXACT GROUND ENERGY AT N = 20, AND AT WHICH `chi`.
#
# ⛔ WHY N = 20 IS THE RIGHT SIZE, AND IT IS NOT A COMPROMISE. The `S^z = 0` sector of 20 sites
# is `C(20,10) = 184756` states with ~30 off-diagonal entries per row. A dense `eigen` is out of
# reach (184756^2 Float64 is 273 TB) but a SPARSE LANCZOS is seconds, so the exact ground energy
# of the SAME CLUSTER exists for ALL THREE geometries -- chain, square cylinder, triangular
# cylinder. That is strictly better than the 12-site figures already shipped, where the square
# and triangular arms had ED and the published anchors were a different cluster entirely. At
# N = 20 there is no "we try to compare": every row here is scored against an exact number.
#
# ⚠ AND THE CHAIN CERTIFIES THE CERTIFIER. The open Heisenberg chain has the Bethe ansatz, so
# the chain row computes BOTH `bethe_xxx_open_energy(20)` and the sparse Lanczos and asserts they
# agree. That is what licenses the Lanczos to be the reference for the two 2D geometries, where
# no closed form exists. A reference nobody checked is not a reference.
#
# ⛔ FULL RANK AT N = 20 IS chi = 2^10 = 1024. The centre bond of a 20-site spin-1/2 MPS cannot
# exceed 1024, so `maxdim = 1024` IS AN EXACT SIMULATION and cannot truncate -- the `bound`
# column below reports this per row rather than leaving a reader to infer that a "chi = 1024 vs
# chi = 512" comparison had one arm doing no truncation at all. `chi = 512` is half full rank.
# This is not an argument against the ladder the campaign asked for; it is the reason the ladder
# has to be READ with the `bound` column, because a row where the cap never bound measures the
# solver's floor and not the cap.
#
# ⛔ U(1), NOT SU(2), AND THAT IS THE PAPER'S OWN CHOICE. Sec. IIE runs DMRG "in the zero
# magnetisation sector"; more decisively, Figs. 4/5 apply `S^z`, which is NOT an SU(2) scalar --
# `apply_sz!` selects `SECTOR_UP`/`SECTOR_DOWN` physical-leg blocks that do not exist under
# `:SU2`. Running the gate under SU(2) and the spectrum under U(1) would certify a state the
# spectral driver never uses. Under U(1) `maxdim` is in STATES, directly comparable to the
# paper's `chi = 512`, with none of the multiplet-counting ambiguity.
#
# ⛔ ED AND THE MPO ARE BUILT FROM ONE COUPLING MATRIX. `pairs_sparse` and `pair_mpo` both read
# `Jm`, so there is no second definition of the model to drift apart. The bit-convention landmine
# (`exact_sparse.jl` header) is why this matters: two independent model definitions that disagree
# produce a deviation column that looks like solver error.
#
# Run:  julia --project=. benchmarks/ground_state/gate_l20.jl [chain|square|triangle ...]
# Env:  G20_CHI      comma list, overrides the per-geometry ladder
#       G20_SWEEPS   max DMRG sweeps (default 40)
#       G20_ETOL     sweep-to-sweep energy tolerance (default 1e-12)
#       G20_EXACTCBE "1" adds the exact-complement A/B arm next to the rSVD arm
#       G20_J2       J2/J1 for the triangular geometry (default 0.0 -- the Fig. 3 gate point)
# Out:  benchmarks/results/gate_l20.csv

using LinearAlgebra, SparseArrays, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))                      # pairs_sparse, sz0_basis
include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

# ⛔ THE CIRCUMFERENCE IS NOT FREE ON THE TRIANGULAR LATTICE. The 120-degree ordering wavevector
# ⛔ THE TRIANGLE IS THE PAPER'S `XC` WRAPPING (Sec. IIB), AND `C` MUST BE EVEN. Under XC,
# `K = (4*pi/3, 0)` has `q_y = 0` and is an exact momentum at EVERY even circumference -- the old
# "only when 3 | Ly" rule was an artifact of the YC wrapping this file used to build. See
# `benchmarks/spectral/lattice2d.jl` and the 30-check `benchmarks/spectral/xc_gate.jl`.
#
# ⚠ AND THE PAPER TAKES `L >> C`, SPECIFICALLY `L = C^2` (Fig. 3 caption; Sec. IIB "we take
# L >> C"). That is `N = 64` at `C = 4` and `N = 216` at `C = 6` -- both far past any exact
# diagonalisation, which is why `ED_MAX_N` exists below. A short cylinder like `L = 3, C = 6`
# (`L/C = 0.5`) is a perfectly good COST measurement and an invalid Fig. 5 geometry; both are
# useful and they must not be conflated.
const LX      = parse(Int, get(ENV, "G20_LX", "5"))
const LY      = parse(Int, get(ENV, "G20_LY", "4"))
const N       = LX * LY
const WANT    = let a = filter(x -> x in ("chain", "square", "triangle"), ARGS)
    isempty(a) ? ["chain", "square", "triangle"] : a
end
const NSWEEPS = parse(Int, get(ENV, "G20_SWEEPS", "40"))
const ETOL    = parse(Float64, get(ENV, "G20_ETOL", "1e-12"))
const EXCBE   = get(ENV, "G20_EXACTCBE", "") == "1"
const J2      = parse(Float64, get(ENV, "G20_J2", "0.0"))
const CHIENV  = let s = get(ENV, "G20_CHI", "")
    isempty(s) ? nothing : [parse(Int, x) for x in split(s, ',')]
end
const OUT     = joinpath(@__DIR__, "..", "results")

# ⛔ THE LARGEST `N` THAT STILL HAS AN EXACT REFERENCE. The `S^z = 0` sector is `binomial(N, N/2)`
# and one complex vector is 16 bytes per state: N=20 -> 2.8 MB, N=24 -> 41 MB, N=26 -> 159 MB,
# N=28 -> 612 MB. Past this the gate reports chi-convergence only and SAYS SO -- it does not fall
# back on its own finest grid and call it truth.
# ⚠ It is also a HARD stop, not a slow path: `sz0_basis` enumerates `0:2^N-1`, so N=64 never
# returns rather than merely taking a long time.
const ED_MAX_N = parse(Int, get(ENV, "G20_ED_MAX_N", "26"))

# The campaign's ladders. The chain is cheap and its interesting range is low; the 2D cylinders
# carry the circumference-4 entanglement and are where the paper's chi = 512 lives.
# ⛔ THE TOP RUNG IS HALF OF FULL RANK ON EVERY GEOMETRY, AND THAT IS THE POINT OF THE LADDER.
# Full rank at the centre bond is `2^(N/2)`: 1024 at N=20, 512 at N=18. A cap at or above it
# cannot truncate, so such a row measures the solver's floor rather than the cap and cannot carry
# a truncation claim. The ladders below are chosen so the top rung BINDS everywhere:
#
#     chain    N=20   64, 128, 512      full rank 1024
#     square   N=20   128, 512          full rank 1024
#     triangle N=18   128, 256          full rank  512   <- 512 would be exact, hence 256
const CHI_CHAIN  = [64, 128, 512]
const CHI_SQUARE = [128, 512]
const CHI_TRI    = [128, 256]

# ⛔ FULL RANK AT THE CENTRE BOND, so a cap at or above this cannot truncate and the row measures
# the solver floor rather than the cap. It is `2^(N/2)`, and at these sizes that is NOT a
# formality:
#
#     N = 20  ->  1024      chi = 512 binds (half of full rank)
#     N = 18  ->   512      chi = 512 IS FULL RANK -- the triangular runs at C=3 and C=6 are
#                           EXACT there, which makes them a free reference and a null measurement
#                           of truncation at the same time. Both facts belong in the write-up.
# ⚠ `2^(N/2)` OVERFLOWS Int64 once N > 124, and is already astronomically past any reachable cap
# well before that, so it saturates rather than wrapping NEGATIVE -- a negative FULL_RANK would
# make `binding()` drop every cap as "at or above full rank".
const FULL_RANK = (N ÷ 2) >= 62 ? typemax(Int) : 2^(N ÷ 2)

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE EXACT REFERENCE -- SPARSE LANCZOS, RESTARTED, WITH THE RESIDUAL ASSERTED
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    sparse_ground(H, n; m, restarts, tol) -> (E0, resid, E1_est, iters)

Lowest eigenvalue of a real symmetric sparse `H` by restarted Lanczos with FULL
reorthogonalisation, plus the true residual `||H x - E0 x||` and an estimate of the second
Ritz value.

⛔ THE RESIDUAL IS RETURNED SO THE CALLER CAN ASSERT IT, and every caller here does. An
eigensolver that quietly stops short returns a number that is above the ground energy by an
unknown amount, and since the DMRG arm is VARIATIONAL (also above), the deviation column would
then be a difference of two upper bounds -- which can be either sign and means nothing. A
DMRG energy BELOW this reference is not a good run; it is proof the reference is unconverged.

⛔ FULL REORTHOGONALISATION IS NOT OPTIONAL AT THIS SIZE. Plain three-term Lanczos loses
orthogonality after ~30 steps on a 184756-dimensional problem and starts returning GHOST copies
of the lowest eigenvalue, which look exactly like a converged degenerate ground state. The cost
is `m` dot products per step against a basis that fits in memory (`m = 150` here is 150 x 184756
Float64 = 222 MB), which is cheap next to being wrong.

⚠ `E1_est` IS AN ESTIMATE AND IS LABELLED AS ONE. The second Ritz value of a Lanczos run
targeting the first converges much later; it is reported because the J2 = 0.5 freeze on the
12-site cylinder tracked DEGENERACY (gap 1.5e-14), so a near-zero gap is a warning worth having.
It is not asserted and must not be quoted as a spectral gap.
"""
function sparse_ground(H::SparseMatrixCSC{Float64, Int}, n::Int;
                       m::Int = 150, restarts::Int = 30, tol::Float64 = 1e-11)
    rng = MersenneTwister(0xC0FFEE)
    x = randn(rng, n); x ./= norm(x)
    theta = Inf; E1 = Inf; iters = 0
    V = Matrix{Float64}(undef, n, m)

    for _ in 1:restarts
        V[:, 1] = x
        alpha = Float64[]; beta = Float64[]
        k = m
        for j in 1:m
            w = H * view(V, :, j)
            iters += 1
            a = dot(view(V, :, j), w)
            push!(alpha, a)
            # Two passes of Gram-Schmidt against the WHOLE basis: one pass loses orthogonality
            # once the vectors start to align on the dominant direction.
            for _pass in 1:2, i in 1:j
                w .-= dot(view(V, :, i), w) .* view(V, :, i)
            end
            b = norm(w)
            if j < m
                if b < 1e-13                      # genuine invariant subspace, not convergence
                    k = j
                    break
                end
                push!(beta, b)
                V[:, j + 1] = w ./ b
            end
        end
        T = SymTridiagonal(alpha[1:k], beta[1:(k - 1)])
        F = eigen(T)
        theta = F.values[1]
        E1 = length(F.values) >= 2 ? F.values[2] : Inf
        x = view(V, :, 1:k) * F.vectors[:, 1]
        x ./= norm(x)
        r = norm(H * x - theta * x)
        r < tol && return (theta, r, E1, iters)
    end
    return (theta, norm(H * x - theta * x), E1, iters)
end

"Energy of the nearest-neighbour dimer product state under `Jm`, in the site pairing `dimer_state` uses."
dimer_energy(Jm::AbstractMatrix, L::Int) =
    sum(-0.75 * Jm[k, k + 1] for k in 1:2:(L - 1); init = 0.0)

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE THREE GEOMETRIES, ALL AT N = 20, ALL FROM ONE COUPLING MATRIX
# ══════════════════════════════════════════════════════════════════════════════════════════════

struct Geom
    name::String
    Jm::Matrix{Float64}
    chis::Vector{Int}
    note::String
end

"Open Heisenberg chain: `Jm[i,i+1] = J`. The only geometry here with a closed form."
function chain_couplings(L::Int; J::Float64 = 1.0)
    Jm = zeros(L, L)
    for i in 1:(L - 1)
        Jm[i, i + 1] = J
    end
    return Jm
end

function build(name::String)
    if name == "chain"
        return Geom(name, chain_couplings(N), something(CHIENV, CHI_CHAIN),
                    "open Heisenberg chain, L=20")
    elseif name == "square"
        return Geom(name, square_cylinder_couplings(LX, LY; periodic_y = true),
                    something(CHIENV, CHI_SQUARE),
                    "square cylinder XC, Lx=$LX Ly=$LY (C=$LY)")
    elseif name == "triangle"
        return Geom(name, triangular_cylinder_couplings(LX, LY; J2 = J2, geometry = :xc,
                                                        periodic_y = true),
                    something(CHIENV, CHI_TRI),
                    "triangular cylinder XC (zigzag ring, T=(0,√3C/2)), Lx=$LX Ly=$LY (C=$LY), " *
                    "J2=$J2")
    end
    error("unknown geometry $name")
end

"""
    binding(chis) -> Vector{Int}

Drop caps at or above full rank, loudly.

⛔ A CAP THAT CANNOT TRUNCATE IS NOT A CHEAP DATA POINT, IT IS A DIFFERENT EXPERIMENT. It runs
the state at full rank and reports the solver's floor; put it beside a binding cap in the same
table and the pair reads as a convergence study when one of the two never converged anything.
Dropping it here rather than flagging it downstream means no such row can reach a figure by
accident. `G20_CHI` still overrides -- an explicit request is honoured, and still flagged by the
`bound` column.
"""
function binding(chis::Vector{Int})
    keep = filter(D -> D < FULL_RANK, chis)
    for D in chis
        # ⛔ `println`, NOT `@printf` WITH A CONCATENATED FORMAT. `@printf` requires a LITERAL
        # format string: `"a" * "b"` throws an ArgumentError at LOAD time, which every syntax
        # check passes and which kills the run several minutes into JIT.
        D < FULL_RANK || println("  ⚠ dropping D=$D: full rank at N=$N is $FULL_RANK, so this " *
                                 "cap cannot truncate (set G20_CHI to force it)")
    end
    isempty(keep) && error("every requested cap is at or above full rank ($FULL_RANK) at N=$N")
    return keep
end

# ══════════════════════════════════════════════════════════════════════════════════════════════
# THE DMRG ARM
# ══════════════════════════════════════════════════════════════════════════════════════════════

"""
    GS_KNOBS

Expansion settings for the ground-state solve, stated EXPLICITLY rather than inherited.

⛔ THE LIBRARY DEFAULT IS A TIME-INTEGRATOR CONSTANT AND MUST NOT BE INHERITED HERE. Leaving
`dex` unset gives `dex = 0`, which hands the budget to the growth schedule
`budget = ceil(growth*dmax) - r` at `growth = 2.0` -- and `cbe_core.jl:609` says in as many
words that 2.0 is OURS, chosen for the time integrator, against the DMRG reference's 1.1: "The
reference is DMRG, where each sweep re-converges the same state, so a 10% ratchet compounds over
sweeps and costs only iterations. A TIME INTEGRATOR gets one shot per step." At `chi = 128` the
schedule asks for a 128-direction budget, a 154-column sketch and a bond expanded to 256 before
truncation; `dex = 8, dover = 4` asks for 12 columns and a bond of 136.

The values here are the ones `benchmarks/ground_state/dmrg_knobs.jl` measures. Change them there,
against exact ED, not by taste.

⛔ `restol = 1e-6`, NOT THE 1e-10 DEFAULT -- A FREE FACTOR OF TWO. MEASURED on the 5x4 square
cylinder at D=128 against exact ED (`dmrg_knobs.jl`), with `matvec` DETERMINISTIC so this is not
the box's ~1.8x wall-clock noise:

    restol 1e-10   dev/site 5.037e-07   matvec 5467   153.0 s
    restol 1e-8    dev/site 5.039e-07   matvec 4243    97.3 s
    restol 1e-6    dev/site 5.023e-07   matvec 2704    54.6 s

Accuracy is FLAT across the axis, so tightening the local eigensolve buys nothing at all. A DMRG
sweep re-solves the same site on the next pass, so an over-tight local tolerance buys iterations
the following sweep discards. ⚠ THE OPPOSITE HOLDS FOR THE TIME INTEGRATOR, where a direction not
admitted in a step is gone for good -- same knob, opposite conclusion, because one algorithm gets
a second chance and the other does not.
"""
const GS_KNOBS = (dex = 8, dover = 4, comp_ratio = 1.0, maxiter = 30, restol = 1e-6)

"""
    run_arm(kind, psi0, W, D) -> NamedTuple

One `rsvd_cbe_dmrg1s!` (or `exact_cbe_dmrg1s!`) solve at cap `D`.

⛔ THE ENERGY IS RECOMPUTED FROM THE STATE, not taken from `run.energies[end]`. The sweep's
number is the lowest Ritz value seen at any bond, which is a variational bound on the state at
that moment inside the sweep; the state that leaves the function has been truncated since.
`mpo_energy` on the returned state is what the next stage of the campaign will actually evolve.
"""
function run_arm(kind::Symbol, psi0, W, D::Int)
    psi = copy(psi0)
    solver = kind === :rsvd ? rsvd_cbe_dmrg1s! : exact_cbe_dmrg1s!
    t0 = time()
    run = solver(psi, W; n_sweeps = NSWEEPS, etol = ETOL, maxdim = D,
                 trunc_thresh = 1e-12, GS_KNOBS...)
    sec = time() - t0
    E = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    chi = maximum(state_bond_dims(psi); init = 0)
    return (E = E, chi = chi, sweeps = length(run.energies),
            matvec = sum(run.krylov_dims), sec = sec,
            conv = run.converged, psi = psi)
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)
    # ⛔ NAMED BY WHAT IT CONTAINS. One fixed filename meant a later geometry silently OVERWROTE
    # an earlier one -- the chain rows vanished the moment the square ran.
    out = joinpath(OUT, "gate_l$(N)_$(join(WANT, '-'))_$(LX)x$(LY).csv")

    @printf("\n%s\n", "="^108)
    @printf("N=%d GROUND-STATE GATE  --  arXiv:2209.00739 Fig.3 at J2=%.3f   (L=%d, C=%d, L/C=%.2f)\n",
            N, J2, LX, LY, LX / LY)
    # `binomial` OVERFLOWS Int64 past N ~ 66, so the sector size is printed via BigInt.
    println("U(1) S^z=0 sector ($(binomial(big(N), N ÷ 2)) states);  full rank at the centre bond " *
            "is chi=$FULL_RANK, a cap >= that CANNOT truncate")
    N <= ED_MAX_N || println("⚠ N=$N > ED_MAX_N=$ED_MAX_N: NO exact reference. Rows are scored " *
                             "by chi-convergence and the variational bound only.")
    LX >= 2 * LY || println("⚠ L/C = $(round(LX/LY; digits=2)): the paper takes L >> C " *
                            "(specifically L = C^2). This aspect ratio is a COST geometry, " *
                            "not a Fig. 4/5 physics geometry.")
    @printf("%s\n", "="^108)

    open(out, "w") do io
        println(io, "geom,J2,N,arm,D,bound,E,E_site,E_exact,E_exact_site,dev,dev_site," *
                    "chi,sweeps,matvec,seconds,converged,ref,resid")

        for name in WANT
            g = build(name)
            @printf("\n%s\n%s   [%s]\n%s\n", "-"^108, uppercase(name), g.note, "-"^108)

            # ── the reference ──────────────────────────────────────────────────────────────
            # ⛔ ABOVE `ED_MAX` THERE IS NO EXACT NUMBER, AND THE OUTPUT MUST SAY SO RATHER THAN
            # QUIETLY SCORE AGAINST SOMETHING WEAKER. The paper's own geometries (L = C^2, so
            # N = 64 at C=4 and N = 216 at C=6) are far past any diagonalisation: the S^z=0
            # sector at N=64 is 1.8e18 states. `sz0_basis` would also LOOP over `0:2^N-1`, so
            # this is not merely slow, it never returns.
            #
            # ⚠ WHAT REPLACES IT IS STRICTLY WEAKER, AND IT IS NOT "our finest grid is truth".
            # Two things still bite: E(chi) is VARIATIONAL, so it decreases monotonically in chi
            # and a row BELOW a converged larger-chi row is a bug; and the chi-to-chi decrement
            # bounds how far from converged the ladder is. Neither is an accuracy certificate.
            # An external anchor for these cylinders would have to come from the paper's Fig. 3
            # curve itself -- see the `external-reference-anchors` rule.
            E0, resid, E1, its, ted = NaN, NaN, NaN, 0, 0.0
            refname = "chi-convergence"
            have_ed = N <= ED_MAX_N
            if have_ed
                t0 = time()
                H, = pairs_sparse(N, g.Jm)
                nst = size(H, 1)
                E0, resid, E1, its = sparse_ground(H, nst)
                ted = time() - t0
                resid < 1e-9 || error("Lanczos residual $resid at $name -- reference NOT " *
                                      "converged, no row scored against it is quotable")
                refname = "ed"
            else
                println("  ⚠ NO EXACT REFERENCE: N=$N is past ED_MAX_N=$ED_MAX_N. Rows are " *
                        "scored by chi-CONVERGENCE and the variational bound only -- `dev` " *
                        "columns are blank and no row here is an accuracy claim.")
            end

            if have_ed && name == "chain"
                # ⛔ THE GATE ON THE GATE. The only geometry with a closed form checks the sparse
                # solver at exactly the size the other two need it. If these disagree, the ED is
                # not a reference for the cylinders either.
                Eb, _, rb = bethe_xxx_open_energy(N)
                rb < 1e-12 || error("Bethe residual $rb at L=$N")
                dbe = abs(Eb - E0)
                @printf("  reference: Bethe %.12f   sparse-ED %.12f   |diff| = %.3e  %s\n",
                        Eb, E0, dbe, dbe < 1e-9 ? "AGREE -- ED certified at N=20" : "⛔ DISAGREE")
                dbe < 1e-9 || error("Bethe and sparse ED disagree by $dbe at L=$N; " *
                                    "the ED path is not usable as the cylinder reference")
                refname = "bethe+ed"
            end

            dE = dimer_energy(g.Jm, N)
            if have_ed
                @printf("  exact E0 = %.12f  (E0/N = %.12f)   resid %.2e   %d matvec   %.1fs\n",
                        E0, E0 / N, resid, its, ted)
                @printf("  gap estimate %.3e (NOT asserted)   dimer E/N = %.8f%s\n",
                        E1 - E0, dE / N,
                        abs(E0 - dE) < 1e-9 ? "   <- dimer IS the exact ground state here" : "")
            else
                @printf("  dimer E/N = %.8f  (a variational START, not a reference)\n", dE / N)
            end

            # ── the DMRG arms ──────────────────────────────────────────────────────────────
            psi0 = neel_state(N)
            W    = pair_mpo(N, g.Jm, spin_vertices(N))
            @printf("  MPO virtual dims: max %d\n", maximum(mpo_virtual_dims(W)))
            flush(stdout)
            caps = CHIENV === nothing ? binding(g.chis) : g.chis
            @printf("\n  %-6s %5s %6s %16s %12s %6s %6s %9s %9s %s\n",
                    "arm", "D", "bound", "E/N", have_ed ? "dev/site" : "ΔE/N vs D-",
                    "chi", "swp", "matvec", "secs", "")

            arms = EXCBE ? [:rsvd, :exact] : [:rsvd]
            prevE = Dict{Symbol, Float64}()          # last E at a smaller cap, per arm
            for D in caps, arm in arms
                r = run_arm(arm, psi0, W, D)
                bound = D < FULL_RANK
                dev = r.E - E0
                # ⛔ A VARIATIONAL ARM BELOW AN EXACT REFERENCE IS A BUG, NOT A GOOD RUN.
                # Without ED the same logic still applies ACROSS CAPS: raising `chi` enlarges the
                # variational manifold, so E must not INCREASE. A rise means the larger-cap solve
                # is the worse one -- a stuck sweep, not a tighter bound.
                drop = get(prevE, arm, NaN) - r.E
                flag = (have_ed && dev < -1e-9) ?
                           "  ⛔ BELOW EXACT -- operator or convention mismatch" :
                       (!have_ed && !isnan(drop) && drop < -1e-9) ?
                           @sprintf("  ⛔ E ROSE by %.2e as chi grew -- larger cap solved WORSE",
                                    -drop) :
                       (r.chi <= 2 && r.sweeps <= 3) ? "  ⛔ FROZEN at the start state" :
                       (!bound ? "  (cap cannot bind: full rank)" : "")
                prevE[arm] = r.E
                if have_ed
                    @printf("  %-6s %5d %6s %16.10f %12.3e %6d %6d %9d %9.1f%s\n",
                            arm, D, bound ? "yes" : "NO", r.E / N, dev / N, r.chi, r.sweeps,
                            r.matvec, r.sec, flag)
                else
                    @printf("  %-6s %5d %6s %16.10f %12s %6d %6d %9d %9.1f%s\n",
                            arm, D, bound ? "yes" : "NO", r.E / N,
                            isnan(drop) ? "--" : @sprintf("%.3e", drop / N),
                            r.chi, r.sweeps, r.matvec, r.sec, flag)
                end
                # ⛔ REDIRECTED STDOUT IS BLOCK-BUFFERED. Without this a long run shows NOTHING
                # until it exits, so "is it working or wedged?" is unanswerable and the only
                # recourse is watching CPU time. Costs nothing; buys observability.
                flush(stdout)
                @printf(io, "%s,%.4f,%d,%s,%d,%d,%.12f,%.12f,%.12f,%.12f,%.6e,%.6e,%d,%d,%d,%.3f,%d,%s,%.3e\n",
                        name, J2, N, arm, D, bound ? 1 : 0, r.E, r.E / N, E0, E0 / N,
                        dev, dev / N, r.chi, r.sweeps, r.matvec, r.sec,
                        r.conv ? 1 : 0, refname, resid)
                flush(io)
            end
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
