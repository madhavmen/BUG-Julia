# DOES FOLDING `Omega` INTO THE CONTRACTION GIVE THE SAME ANSWER AS PROJECTING FIRST -- and does
# it cost less? Correctness first; the seconds are worthless until it passes.
#
# ⛔ THE TWO ORDERINGS ARE AN IDENTITY, NOT AN APPROXIMATION, so the bar here is ROUNDOFF and not
# "close enough". `P_perp` acts on `(link_l, site_l)`, `Om` on `(site_r, link_r)` -- disjoint
# legs, so `P_perp((H Theta) Om') == (P_perp (H Theta)) Om'` exactly. The two paths contract the
# same tensors in a different ORDER, so they differ only by floating-point association and must
# agree to ~1e-12 relative. Anything larger is a leg-order, prime-level or boundary-case bug in
# `_fold_sketch_closures`, not a tolerance to be widened.
#
# ⛔ THE SKETCH CLOSURES ARE COMPARED DIRECTLY, NOT THE EXPANSION THEY FEED. `cbe_expand` runs an
# SVD and a selection on top, and a selection can be IDENTICAL while the object under it is
# wrong: near-degenerate singular values get reordered, small directions get truncated away, and
# a discrepancy below `stol_fnl` vanishes from the output entirely. Comparing `skl(Om)` against
# `skl(Om)` puts the assertion on the quantity this change actually alters.
#
# ⚠ BOUNDARY BONDS ARE SEPARATE CASES AND ARE TESTED SEPARATELY. `apply_h_two_site` has three
# `nothing`-environment branches (`lenv.E`, `renv.E`, and the singleton MPO legs they leave), and
# the folded chain has to reproduce each one in a different contraction order. Interior bonds
# alone would exercise none of them, and bond 1 / bond L-1 are exactly where an MPS sweep spends
# its boundary steps.
#
# ⚠ THE STATE MUST NOT BE A PRODUCT STATE. At chi = 1 the projector `P_perp` is rank-deficient in
# a way that makes almost any leg order look right, and `d*chi_r` vs `Dpre` is 2 vs 2. The state
# is evolved first so the ranks are non-trivial and asymmetric.
#
#   julia --project=. benchmarks/fold_omega_check.jl [L] [nsteps]

using Printf, LinearAlgebra, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
# ⚠ `R` REACHES THE UNEXPORTED INTERNALS, and the two sketch closure builders are exactly that:
# `_sketch_closures` and `_fold_sketch_closures` are implementation detail, which is why this
# check lives beside them rather than in the public test suite.
const R = BUGJulia.RSVDCBEBondUpdate

const L      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const NSTEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4
const DT     = 0.05
const DEX    = 8
const DOVER  = 4

say(l) = (println(l); flush(stdout))

"""
Both sketch closures at one bond, from an IDENTICAL frame and an IDENTICAL probe.

⛔ THE PROBE IS DRAWN ONCE AND HANDED TO BOTH PATHS. Re-seeding two `MersenneTwister`s and
trusting them to produce the same `Om` would silently compare two different probes the moment
either path draws a different NUMBER of randoms -- and `sector_graded_sketch` allocates its
columns per SECTOR, so that count depends on the frame. One draw, two consumers.
"""
function compare_bond(psi, mpo, i; symmetry)
    canonical!(psi, i)
    f = R.bond_frame(psi, i)
    lst = R.left_env_stack(psi, mpo; upto = i)
    rst = R.right_env_stack(psi, mpo; downto = i + 2)
    lenv = R.left_channels(lst, i)
    renv = R.right_channels(rst, i + 2)

    sk_p = R._sketch_closures(f, () -> R.apply_h_two_site(R.frame_theta(f), mpo, i, lenv, renv))
    sk_f = R._fold_sketch_closures(f, mpo, i, lenv, renv)

    # ⚠ THE PROBE TAKES `V0` / `U0`, NOT THE FRAME -- `cbe_expand` calls `probe(V0, :right)` and
    # `probe(U0, :left)` (cbe_core.jl:698,705). `Om` is shaped like the frame factor it will
    # STAND IN FOR, with `Dpre` columns where that factor has `r`, which is why the left sketch
    # is probed with a right-frame-shaped object and vice versa.
    rng = MersenneTwister(0xC0FFEE)
    npre = DEX + DOVER
    OmR = R.sector_graded_sketch(f.V0, :right, npre; comp_ratio = 1.0, rng = rng)
    OmL = R.sector_graded_sketch(f.U0, :left,  npre; comp_ratio = 1.0, rng = rng)

    # ⛔ THE SCALE TO DIVIDE BY IS THE UNPROJECTED SKETCH, NOT `norm(Y)`.
    # `Y` is the DISCARDED component `P_perp (H Theta) Om'`. Where the frame already spans the
    # local space -- which is the ORDINARY case at an interior bond once the rank has grown --
    # `P_perp` annihilates almost everything and `norm(Y)` collapses to roundoff. Dividing the
    # difference by that is dividing roundoff by roundoff: MEASURED, it reported `rel 8.7e-06` on
    # a bond whose `|Y|` was 2.2e-16 and whose ABSOLUTE difference was ~1e-21, i.e. it flagged
    # perfect agreement as a failure. Every absolute difference in that run was ~1e-16 whether the
    # row passed or failed, which is the tell.
    # ⇒ `(H Theta) Om'` BEFORE the projector is the physically meaningful magnitude, and building
    # it here is free: this is the object the fold path exists to avoid, and a test is exactly
    # where it is legitimate to form it.
    HT = R.apply_h_two_site(R.frame_theta(f), mpo, i, lenv, renv)

    out = Tuple{String,Float64,Float64,Float64}[]
    if OmR !== nothing
        a, b = sk_p[1](OmR), sk_f[1](OmR)
        ref = to_concrete(contract(HT, (3, 4), OmR', (2, 3)))     # (link_l, site_l, g)
        d = norm(to_concrete(a - b))
        push!(out, ("skl", d, max(norm(ref), eps()), norm(a)))
    end
    if OmL !== nothing
        a, b = sk_p[2](OmL), sk_f[2](OmL)
        ref = to_concrete(contract(HT, (1, 2), OmL', (1, 2)))      # (site_r, link_r, g)
        d = norm(to_concrete(a - b))
        push!(out, ("skr", d, max(norm(ref), eps()), norm(a)))
    end
    return out
end

function main()
    ok = true
    for symmetry in (:none, :U1)
        set_symmetry!(symmetry)
        say(@sprintf("\n== symmetry = %s, L = %d ==", symmetry, L))
        mpo = xxz_mpo(L; J = 1.0, delta = 1.0)
        # ⚠ DOMAIN WALL, NOT NEEL: a Neel state is mirror-symmetric about the centre bond, so a
        # left/right leg-order swap in the folded chain can cancel against itself and pass. The
        # wall is asymmetric, which is the standing rule for this repo's dense/sparse checks and
        # applies just as much here.
        psi = domain_wall_state(L)
        # ⛔ DO NOT PIN `tol = 0` HERE. Without a convergence exit the Krylov recursion runs the
        # full `maxiter` even when the block is SMALLER than that, and the degenerate basis dies
        # constructing a rank-0 `TLArray` (`MethodError: no method matching TLArray(::Tuple{}, ...)`
        # inside `_view_scale` <- `lanczos_expv`). MEASURED HERE at L=10, `:none`, where the
        # zero-site root block is tiny -- and already recorded in `rsvd_parallel.jl`, which pins
        # `tol = 0` deliberately and documents that its TDVP2 arm cannot.
        # ⚠ Pinning exists to make operator COUNTS comparable between timed arms. This script
        # times nothing: it only needs a state with non-trivial, asymmetric ranks.
        for _ in 1:NSTEPS
            cbe_bug_step!(psi, mpo, ComplexF64(-im * DT);
                          dex = DEX, dover = DOVER, comp_ratio = 1.0,
                          krylov_basis = 3,
                          maxdim = 64, trunc_thresh = 1e-10, maxiter = 8)
        end
        say(@sprintf("   chi after %d steps: %s", NSTEPS, string(bond_dims(psi))))

        for i in unique([1, 2, L ÷ 2, L - 2, L - 1])
            (1 <= i <= L - 1) || continue
            edge = (i == 1 || i == L - 1) ? "  <- BOUNDARY" : ""
            for (which, d, scale, ynrm) in compare_bond(psi, mpo, i; symmetry = symmetry)
                rel = d / scale
                pass = rel < 1e-11
                ok &= pass
                # `|Y|/|HTh Om|` is printed because it EXPLAINS the rows rather than decorating
                # them: where it is ~1e-13 the projector has removed everything and the frame is
                # already complete at that bond, which is why `norm(Y)` cannot be the yardstick.
                # ⛔ ONE LITERAL. `@sprintf` REJECTS A CONCATENATED FORMAT -- "First argument to
                # @sprintf must be a format string" -- because it parses the format at MACRO
                # expansion, before `*` has run. It is a compile-time error, so it takes out the
                # whole script rather than the one line, after a full JIT warmup.
                say(@sprintf("   bond %-3d %-4s |diff| %.3e / |HTh Om| %.3e = %.3e  [|Y| frac %.1e]  %s%s",
                             i, which, d, scale, rel, ynrm / scale,
                             pass ? "PASS" : "*** FAIL ***", edge))
            end
        end
    end

    say("")
    say(ok ? "ALL BONDS AGREE -- fold_omega is an exact reordering, timings are meaningful." :
             "*** MISMATCH -- do NOT quote any fold_omega timing until this passes. ***")
    return ok
end

main() || exit(1)
