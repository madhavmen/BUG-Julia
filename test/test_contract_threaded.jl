# THREADED `contract` MUST BE BIT-IDENTICAL TO THE SERIAL PATH.
#
# The threading in `Telum.contract` (step 5a) parallelises over OUTPUT SECTORS. Nothing about
# the arithmetic changes: accumulation order within a sector is untouched and sectors write to
# disjoint slots, so equality here is exact, not approximate. Testing with `isapprox` would
# hide precisely the bug this guards against — a race in the shared scratch buffer or in the
# permuted-RMT caches, which corrupts a few entries of one sector and leaves a result that is
# still "close".
#
# ⛔ THE RACE THIS PROTECTS AGAINST HAS NO OTHER SYMPTOM. `_cached_prepared_sector_rmt!`
# mutates on a miss, and the threaded loop relies on step 4 having warmed every entry it will
# read. If that stops being true the numbers go wrong silently — no throw, no warning.

using Test
using LinearAlgebra
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using Telum: TELUM_CONTRACT_THREADS

"Every stored block of an MPS, in sector order, as one flat vector — for exact comparison."
function _fingerprint(psi)
    out = ComplexF64[]
    for i in 1:length(psi)
        for r in psi[i].RMTs
            append!(out, vec(ComplexF64.(r)))
        end
    end
    return out
end

@testset "threaded contract == serial contract" begin
    if Threads.nthreads() < 2
        @warn "julia is running with $(Threads.nthreads()) thread(s); the threaded path " *
              "cannot be exercised and this testset only checks that the knob is honoured"
    end

    for sym in (:U1, :none)
        BondUpdateBUG.set_symmetry!(sym)
        L = 8
        mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)

        # Enough rank for the middle bonds to carry several charge sectors — a product state
        # has one sector per bond and would exercise none of the threading.
        base = BondUpdateBUG.neel_state(L)
        for _ in 1:6
            RSVDCBEBondUpdate.tdvp2_step!(base, mpo, ComplexF64(-im * 0.3);
                                          maxdim = 64, trunc_thresh = 0.0, maxiter = 6)
        end
        # ⚠ THE VACUITY GUARD ONLY APPLIES UNDER A SYMMETRY. `:none` has no charge sectors at
        # all — every tensor is one dense block — so a contraction there has ONE output sector
        # and `ntasks = min(1, ...) = 1`: the threaded path cannot be exercised, by
        # construction. Asserting `nsec >= 3` there fails on a correct implementation, which
        # is what it did. Under `:none` this testset checks only that the code still runs and
        # still agrees; the real comparison is the `:U1` half.
        nsec = length(base[L ÷ 2].RMTs)
        if sym === :U1
            @test nsec >= 3      # otherwise the comparison below is vacuous
        else
            @test nsec >= 1
        end

        for arm in (:tdvp2, :cbe1s, :bug)
            step! = (p, ) -> if arm === :tdvp2
                RSVDCBEBondUpdate.tdvp2_step!(p, mpo, ComplexF64(-im * 0.05);
                    maxdim = 64, trunc_thresh = 0.0, maxiter = 6)
            elseif arm === :cbe1s
                RSVDCBEBondUpdate.tdvp_cbe1s_step!(p, mpo, ComplexF64(-im * 0.05);
                    maxdim = 64, trunc_thresh = 0.0, maxiter = 6, exact = false)
            else
                RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-im * 0.05);
                    maxdim = 64, trunc_thresh = 0.0, maxiter = 6, exact = false)
            end

            TELUM_CONTRACT_THREADS[] = 1
            pser = deepcopy(base); step!(pser)
            fser = _fingerprint(pser)

            TELUM_CONTRACT_THREADS[] = typemax(Int)
            pthr = deepcopy(base); step!(pthr)
            fthr = _fingerprint(pthr)

            @test length(fser) == length(fthr)
            @test BondUpdateBUG.bond_dims(pser) == BondUpdateBUG.bond_dims(pthr)
            @test fser == fthr                     # EXACT, deliberately not isapprox
        end
    end
    TELUM_CONTRACT_THREADS[] = typemax(Int)
    BondUpdateBUG.set_symmetry!(:U1)
end
