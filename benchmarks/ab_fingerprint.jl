# A FINGERPRINT OF A TRAJECTORY, FOR COMPARING TWO REVISIONS OF THE SOURCE.
#
# Two of the changes made on 2026-08-27 have OPPOSITE expected effects on the numbers, and the only
# way to tell a correct one from a broken one is to fix what the output should be BEFORE running it:
#
#   * SHARING `H*Theta` across `cbe_expand`'s three sketch calls must be **BIT-IDENTICAL**. The same
#     tensors are contracted in the same order; they are simply not rebuilt. ANY difference here is
#     a bug in the caching.
#   * RUNNING THE HALF-SWEEPS CONCURRENTLY must agree to ROUNDOFF but NOT bit-for-bit, because each
#     sweep now re-gauges its own copy of the state and `canonical!` therefore starts from a
#     different centre. Demanding bit-identity there would be demanding the wrong thing.
#
# So this prints full-precision observables and the caller diffs two revisions' output. `git stash`
# the source, run, restore, run, `diff`.
#
#   julia --project=. benchmarks/ab_fingerprint.jl [model] [sites] [chi] [nsteps] [parallel]

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "study_models.jl"))

const MODEL = length(ARGS) >= 1 ? ARGS[1] : "xx"
const SITES = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const CHI   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 32
const NSTEP = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 6
const PAR   = length(ARGS) >= 5 ? parse(Bool, ARGS[5]) : false
const DT    = 0.05

const S = setup(MODEL; sites = SITES, dt = DT, cap = CHI)

step!(p, exact) = cbe_bug_step!(p, S.W, S.tau; exact = exact, parallel = PAR,
                                dex = 8, dover = 0, comp_ratio = 1.0,
                                krylov_basis = 3, krylov_tol = 0.0,
                                hermitian = (S.d == 2), maxdim = CHI,
                                trunc_thresh = 1e-10, maxiter = 20, tol = 0.0)

@printf("model %s  L=%d  d=%d  chi=%d  steps=%d  parallel=%s\n",
        MODEL, S.L, S.d, CHI, NSTEP, PAR)
for exact in (true, false)
    p = copy(S.psi0)
    nmv = 0
    for _ in 1:NSTEP
        info = step!(p, exact)
        nmv += info.krylov_dims
        S.post!(p)
    end
    # ⚠ FULL PRECISION (`%.17e`), because the claim under test is BIT-IDENTITY. Printing six
    # digits would let a change of 1e-10 pass as "the same", which is exactly the size of error a
    # broken cache would introduce.
    pr = S.profile(p)
    @printf("  arm exact=%-5s krylov=%d bonds=%s\n", exact, nmv,
            string(state_bond_dims(p)))
    for (k, v) in enumerate(pr)
        @printf("    obs[%2d] = %+.17e\n", k, v)
    end
end
