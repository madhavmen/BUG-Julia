# WHERE DO THE 142 GB PER STEP ACTUALLY COME FROM?
#
# The kernel breakdown says GC/allocation is ~67% of ACTIVE samples at chi=1024 while BLAS
# gemm is ~12%, and the step allocates ~142 GB against a 0.08 GB state — roughly 1800x churn.
# That says "reduce allocation", but not WHICH allocation, and the obvious suspect (fresh
# intermediates inside `apply_one_site`) is a hypothesis, not a measurement.
#
# ⛔ THIS FILE EXISTS BECAUSE THE OBVIOUS ATTRIBUTION HAS BEEN WRONG ALL NIGHT. `other` looked
# like unclassified work and was idle threads. The sector loop looked like the bottleneck and
# threading it made everything slower. So: count bytes per primitive at the real shapes,
# multiply by the real call counts, and check the total against the measured step. If the
# reconstruction does not add up to ~142 GB, the hypothesis is incomplete and the missing
# allocation is somewhere I have not looked.
#
# ⚠ `@allocated` counts the CALLING task only. Everything here runs on one task, which is
# what we want — this measures the allocation of a primitive, not of a parallel sweep.

using LinearAlgebra
using Printf
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: one_site_h, apply_one_site, boundary_channels,
                                  right_env_stack, right_channels, left_env_stack,
                                  left_channels, zero_site_h, apply_zero_site
using Telum: contract, to_concrete, svd

include(joinpath(@__DIR__, "random_mps.jl"))

const L    = parse(Int, get(ENV, "AA_L", "30"))
const CHI  = parse(Int, get(ENV, "AA_CHI", "1024"))
const MI   = parse(Int, get(ENV, "AA_MAXITER", "8"))
gb(b) = b / 2^30
mb(b) = b / 2^20

BondUpdateBUG.set_symmetry!(:U1)
BLAS.set_num_threads(parse(Int, get(ENV, "AA_BLAS", "8")))

@printf("L=%d chi=%d  building state...\n", L, CHI); flush(stdout)
psi = random_mps(L, CHI; seed = 1, delta = 1.0)
mpo = RSVDCBEBondUpdate.xxz_mpo(L; J = 1.0, delta = 1.0)
canonical!(psi, 1)
@printf("state: %d elements, %.3f GB\n\n", mps_elements(psi), gb(16 * mps_elements(psi)))

# ── one `apply_one_site`, the Krylov matvec ──────────────────────────────────
# Build the environments at a mid-chain bond so the shapes are the widest the sweep sees.
i = L ÷ 2
canonical!(psi, i)
rstack = right_env_stack(psi, mpo; downto = i + 1)
lstack = left_env_stack(psi, mpo; upto = i - 1)
H1 = one_site_h(mpo, i, left_channels(lstack, i), right_channels(rstack, i + 1))
A = psi[i]

apply_one_site(H1, A)                                    # warm up / compile
a1 = @allocated apply_one_site(H1, A)
t1 = @elapsed for _ in 1:5; apply_one_site(H1, A); end
@printf("apply_one_site at the mid bond : %8.1f MB   %6.1f ms\n", mb(a1), 1000 * t1 / 5)

# ── the pieces inside it, to see which contraction dominates ─────────────────
X1 = to_concrete(contract(H1.l, (3,), A, (1,)))
aX1 = @allocated to_concrete(contract(H1.l, (3,), A, (1,)))
@printf("  step 1  env_l * A            : %8.1f MB\n", mb(aX1))
aX2 = @allocated to_concrete(contract(X1, (2, 3), H1.w, (1, 2)))
X2 = to_concrete(contract(X1, (2, 3), H1.w, (1, 2)))
@printf("  step 2  (.) * W              : %8.1f MB\n", mb(aX2))
aX3 = @allocated to_concrete(contract(X2, (4, 2), H1.r, (2, 3)))
@printf("  step 3  (.) * env_r          : %8.1f MB\n", mb(aX3))

# ── reconstruct the step ─────────────────────────────────────────────────────
# tdvp2 does maxiter matvecs per solve; the sweep visits ~2(L-1) bonds. This is the
# order-of-magnitude check, not an exact call count -- if it lands near the measured 142 GB
# the matvec really is the whole story, and if it lands far short something else allocates.
nmv_est = 2 * (L - 1) * MI
@printf("\nreconstruction: %d matvecs x %.1f MB = %.1f GB\n", nmv_est, mb(a1),
        gb(nmv_est * a1))
@printf("measured tdvp2 step at chi=1024 was ~142 GB\n")
@printf("=> matvec accounts for %.0f%% of it\n", 100 * nmv_est * a1 / (142 * 2^30))

# ── what a Krylov solve costs, and how much of it is reusable ────────────────
# Every iteration allocates the SAME shapes. If the intermediates were caller-owned buffers,
# all but the first iteration's allocation would disappear -- that is the size of the prize
# for an in-place `contract!` in the Telum fork.
@printf("\nper Krylov solve (maxiter=%d): %.1f MB allocated, of which %.1f MB is re-allocating\n",
        MI, mb(MI * a1), mb((MI - 1) * a1))
@printf("=> an in-place matvec would remove up to %.0f%% of the step's allocation\n",
        100 * (MI - 1) / MI * nmv_est * a1 / (142 * 2^30))
