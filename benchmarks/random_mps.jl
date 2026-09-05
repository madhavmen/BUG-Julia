# RANDOM SYMMETRIC MPS AT A TARGET BOND DIMENSION — for cost benchmarking only.
#
# WHY THIS EXISTS. Measuring the rSVD, the CBE expansion and BLAS threading at chi = 1024 / 5000
# needs a state at that rank, and EVOLVING one there is self-defeating: the ramp costs about as
# much as a step at the target, because the cost is dominated by the largest chi touched. Worse,
# which chi you can reach depends on the physics — the XX domain wall entangles as S ~ (1/6)ln t
# and STALLS at chi = 35 after 40 untruncated steps (job 16326142), so its chi = 512 cap is
# unreachable at any sane T. A random state decouples the cost measurement from the physics.
#
# ⛔ THE SECTOR STRUCTURE IS INHERITED, NEVER INVENTED. A hand-rolled U(1) MPS has to choose how
# many states sit in each charge sector on each bond, and a wrong choice is invisible: the code
# runs, the bond dimension is right, and every block is the wrong SHAPE — so the cost profile,
# which is a sum over per-sector matrix products, is wrong in exactly the quantity being measured.
# So we never construct sectors. We take a state the code itself produced and grow it by DIRECT
# SUM with a randomised copy of itself, which doubles every bond while reproducing the sparsity
# pattern exactly (`_randomize_like` copies qlabels/wmatdata/wmatinfo/inds/spaces and refills only
# the block payloads).
#
# ⚠ THIS IS NOT A PHYSICAL STATE and must never be scored for accuracy. Gaussian block payloads
# have no relation to any Hamiltonian's ground state or any time-evolved state; the Schmidt
# spectrum is flat-ish rather than decaying. That is FINE for cost — the kernels do the same work
# regardless of the numbers in the blocks — and WRONG for anything else. In particular a
# truncation threshold behaves completely differently on a flat spectrum, so benchmark with
# `trunc_thresh = 0.0` and let `maxdim` set the rank.

using Random
using Printf
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using Telum: TLArray, oplus, symm, to_concrete, contract, svd, setitag

"Gaussian payloads on `A`'s exact sparsity pattern. Mirrors `cbe_core.jl:_randomize_like`."
function randomize_like(A, rng::AbstractRNG)
    T = eltype(A) <: Complex ? eltype(A) : ComplexF64
    return TLArray(symm(A), copy(A.qlabels), copy(A.wmatdata), copy(A.wmatinfo),
                   [randn(rng, T, size(r)) for r in A.RMTs], A.inds, A.spaces)
end

"Rescale to unit norm. ⛔ REQUIRED BETWEEN DOUBLINGS: each `oplus` adds a Gaussian block of
the same size as the state, so the norm grows geometrically — measured 1.5e4 after two
doublings from chi=15. Eight doublings (what chi=10000 needs) overflows to Inf and every
subsequent timing measures NaN propagation instead of arithmetic."
function normalize_mps!(psi::SymMPS)
    canonical!(psi, 1)
    n = sqrt(real(BondUpdateBUG.overlap(psi, psi)))
    isfinite(n) && n > 0 || error("normalize_mps!: norm is $n — the state already overflowed")
    psi[1] = to_concrete((1 / n) * psi[1])
    return psi
end

"""
    double_chi!(psi, rng) -> psi

`psi <- psi (+) psi_random`, doubling every internal bond. Site tensors carry legs
`(link_l, site, link_r)`; the direct sum concatenates the LINK legs only — the physical leg must
stay `d`, so leg 2 is never in the `oplus` tuple. Boundary tensors have only one internal link.
"""
function double_chi!(psi::SymMPS, rng::AbstractRNG)
    L = length(psi)
    # ⛔ CANONICALISE FIRST. The bond arrow direction depends on which sweep last touched the
    # tensor, and `oplus` on two tensors whose links point opposite ways fails with "No leg in
    # TLArray matches leg ..." — or, worse, concatenates a leg that was not meant to grow.
    canonical!(psi, 1)
    for i in 1:L
        A = psi[i]
        B = randomize_like(A, rng)
        legs = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        psi[i] = to_concrete(oplus([A, B], legs))
    end
    canonical!(psi, 1)
    return psi
end

"""
    random_mps(L, target_chi; seed = 1, delta = 1.0, seed_dt = 0.5) -> psi

A random symmetric MPS of length `L` whose maximum bond dimension is at least `target_chi`.

Seeded from a Néel product state and given a few UNTRUNCATED evolution steps first — that is what
opens up the charge sectors. A product state has one state per bond and doubling it only ever
reproduces that single sector, so the result would be a rank-`chi` state living in one charge
sector: the right bond dimension and completely the wrong block structure.
"""
function random_mps(L::Int, target_chi::Int; seed::Int = 1, delta::Float64 = 1.0,
                    seed_steps::Int = 3, max_seed_steps::Int = 60,
                    seed_dt::Float64 = 0.5, seed_maxdim::Int = 256,
                    verbose::Bool = true)
    rng = MersenneTwister(seed)
    t0 = time()
    log(msg) = verbose && (@printf("[random_mps %7.1fs] %s\n", time() - t0, msg); flush(stdout))

    psi = BondUpdateBUG.neel_state(L)
    mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = delta)

    # ⛔ SEED UNTIL THE SECTOR COUNT SATURATES, NOT FOR A FIXED NUMBER OF STEPS.
    # Doubling reproduces whatever sector structure it is handed, so the seed decides it for
    # good. Measured at L=30: three steps opened only 7 sectors on the middle bond and the
    # count then stayed at 7 for every chi, while a converged state there carries ~16 (the
    # left half of 30 sites spans Sz = -7.5 ... 7.5). A 7-sector state at chi=1024 has blocks
    # about twice too large and half too few -- which is a direct bias on the ONE measurement
    # this file exists to support, since the block-size histogram is what decides whether
    # parallelism belongs inside each gemm or across them.
    # The target is COUNTABLE, so count it rather than watching for a plateau. On the left
    # link of site `mid` the left block holds `b = mid - 1` spins, so its charge runs over
    # Sz = -b/2 ... +b/2 in integer steps -- but the right block must cancel it, capping
    # |Sz| at (L-b)/2. Hence `min(b, L-b) + 1` reachable sectors: 15 for L=30.
    # ⚠ A stall detector was tried first and stopped at 9 of 15: sector opening is not
    # monotone step to step, so "two quiet steps" fires long before the light cone has
    # crossed the bond.
    mid = L ÷ 2
    b = mid - 1
    target_sectors = min(b, L - b) + 1
    nsec_of(p) = length(p[mid].RMTs)

    # ⛔ SEED AT LARGE `dt` AND SMALL `maxdim`. What opens a charge sector is TIME, not rank:
    # the weight in a high-|Sz| sector is suppressed like t^k, so at t < 1 it sits under the
    # SVD's own 1e-14 floor and the sector is dropped no matter how much bond dimension is on
    # offer. Measured: dt=0.05 with maxdim=4096 was still at 9 of 15 sectors after 10 steps,
    # with chi already at 283 and climbing — the loop was buying rank, which is free later via
    # doubling, at the price of the one thing doubling cannot manufacture.
    # So: `seed_dt = 0.5` reaches t = 10 in twenty steps, and `seed_maxdim = 64` keeps each of
    # those steps in the milliseconds. Seeding accuracy is irrelevant — the payloads are
    # replaced by Gaussians immediately afterwards.
    # ⛔ KEEP THE BEST STATE SEEN, DO NOT ASSUME THE COUNT ONLY RISES. Measured at
    # seed_maxdim=64: sectors reached 11 at t=2.5 and then FELL to 9 and stayed there out to
    # t=30. Truncation removes low-weight sectors faster than time opens new ones once the
    # state thermalises, so running longer actively makes the structure worse — and the loop
    # then burned all 60 steps waiting for a count it had already passed.
    best_n, best = -1, psi
    stall = 0
    for k in 1:max_seed_steps
        RSVDCBEBondUpdate.tdvp2_step!(psi, mpo, ComplexF64(-im * seed_dt);
                                      maxdim = seed_maxdim, trunc_thresh = 0.0, maxiter = 8)
        n = nsec_of(psi)
        if n > best_n
            best_n, best, stall = n, deepcopy(psi), 0
        else
            stall += 1
        end
        k % 5 == 0 && log(@sprintf("seed step %d  t=%.1f  chi=%d  sectors@mid=%d/%d (best %d)",
                                   k, k * seed_dt, maximum(BondUpdateBUG.bond_dims(psi)),
                                   n, target_sectors, best_n))
        if best_n >= target_sectors && k >= seed_steps
            log(@sprintf("all %d sectors open after %d seed steps (t=%.1f)", best_n, k,
                         k * seed_dt))
            break
        end
        # No improvement for a long stretch: the count has peaked and more time only costs.
        if stall >= 15 && k >= seed_steps
            log(@sprintf("sector count peaked at %d/%d after %d steps; keeping the best state",
                         best_n, target_sectors, k))
            break
        end
    end
    psi = best
    best_n < target_sectors && @warn "random_mps: only $best_n of $target_sectors sectors " *
        "opened — block shapes are larger and fewer than a physical state's, which biases " *
        "any per-sector cost measurement. Raise seed_maxdim (truncation is what closes them)."

    while maximum(BondUpdateBUG.bond_dims(psi)) < target_chi
        double_chi!(psi, rng)
        normalize_mps!(psi)
        log(@sprintf("doubled -> chi=%d (target %d)",
                     maximum(BondUpdateBUG.bond_dims(psi)), target_chi))
    end

    # Doubling overshoots — chi=32 was requested and chi=60 came back. Trim to EXACTLY the
    # target so a cost curve has a controlled x-axis. Truncating a flat Schmidt spectrum is
    # meaningless as physics and irrelevant here: the state was never physical.
    trim_to!(psi, target_chi, log)
    normalize_mps!(psi)
    log(@sprintf("done  chi=%d  elements=%d",
                 maximum(BondUpdateBUG.bond_dims(psi)), mps_elements(psi)))
    return psi
end

"Truncating canonical sweep to bring every bond to at most `nkeep`."
function trim_to!(psi::SymMPS, nkeep::Int, log = _ -> nothing)
    L = length(psi)
    canonical!(psi, 1)
    for i in 1:(L - 1)
        res = svd(psi[i], (1, 2); cutoff = 0.0, Nkeep = nkeep)
        M = res.S * res.Vd
        B = contract(M, (2,), psi[i + 1], (1,))
        psi[i]     = to_concrete(setitag(res.U, 3, "L,$(i + 1)"))
        psi[i + 1] = to_concrete(setitag(B, 1, "L,$(i + 1)"))
    end
    psi.center = L
    log(@sprintf("trimmed to Nkeep=%d  chi=%d", nkeep, maximum(BondUpdateBUG.bond_dims(psi))))
    return psi
end

"Total stored complex entries — the honest memory figure for a block-sparse MPS."
mps_elements(psi::SymMPS) = sum(sum(length(r) for r in psi[i].RMTs; init = 0)
                                for i in 1:length(psi))
