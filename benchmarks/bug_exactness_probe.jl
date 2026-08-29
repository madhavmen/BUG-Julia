# IS BUG'S TRACE LOSS STRUCTURAL, OR AN IMPLEMENTATION BUG?
#
# ⛔ THE DECISIVE CASE IS L = 2 FUSED. There is ONE bond. The left frame maps `(1 x d) -> b_L` so
# `b_L <= 4`, the right frame likewise `b_R <= 4`, and the Galerkin space `b_L x b_R` is therefore
# at most 16 -- which IS the whole Hilbert space, `4^2 = 16`. So once both frames reach rank 4:
#
#     * the Galerkin basis is COMPLETE, `P = I`;
#     * `H0 = (W (x) Z)^dag L (W (x) Z)` is the FULL generator, not a projection of it;
#     * `exp(tau*H0) S` is the EXACT propagator;
#     * `|I>>` is trivially IN the basis, so `Tr` must be conserved with NO correction.
#
# If BUG is exact here, the loss at larger L is the manifold projection -- structural, and the fix
# is `|I>>` in the basis. If BUG is NOT exact here, something in the implementation is wrong and
# no amount of rescaling is the answer.
#
# ⚠ `growth` IS THE CONFOUND AND IS SWEPT. `budget = ceil(growth*dmax) - r` starts from `chi = 1`,
# so a small `growth` reaches rank 4 only after several steps and the early, genuinely rank-starved
# steps leave an error that has nothing to do with the question being asked. `growth = 16` puts the
# frame at full rank on step ONE.
#
# ⚠ AND THE ACHIEVED BASIS WIDTH IS REPORTED, not assumed: `info.expanded` is the frame width the
# sweep actually reached. A test that assumes completeness and does not check it would read a
# rank-starved run as an implementation bug.
#
#   julia --project=. benchmarks/bug_exactness_probe.jl

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "bug_exactness_probe.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const L = 2; const T = 0.3; const GAMMA = 0.5
const SPINS = [:up, :down]
const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]

function fused_dense_state(psi)
    n = length(psi); P = _full_tensor(psi)
    v = zeros(ComplexF64, 4^n)
    for idx in 0:(4^n - 1)
        cfg = fused_product_state([(idx ÷ 4^(n - j)) % 4 + 1 for j in 1:n])
        v[idx + 1] = tensor_inner(_full_tensor(cfg), P)
    end
    return v
end
function idx_to_ab(idx::Int, n::Int)
    a = 0; b = 0
    for j in 1:n
        d = (idx ÷ 4^(n - j)) % 4
        a |= (d >> 1) << (n - j); b |= (d & 1) << (n - j)
    end
    return a, b
end
function dense_super(gamma)
    D = 2^L
    sop(op, j) = (m = ComplexF64[1;;]; for k in 1:L; m = kron(m, k == j ? op : _I2); end; m)
    SXm = ComplexF64[0 .5; .5 0]; SYm = ComplexF64[0 -.5im; .5im 0]
    H = sum(JM[i,j] * (sop(SXm,i)*sop(SXm,j) + sop(SYm,i)*sop(SYm,j))
            for i in 1:L for j in (i+1):L if JM[i,j] != 0.0)
    S = zeros(ComplexF64, D*D, D*D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a,b] = 1.0
        R = -im * (H*E - E*H)
        for j in 1:L
            Z = sop(ComplexF64.(_SZ), j)
            R += gamma * (Z*E*Z - 0.5*(Z*Z*E + E*Z*Z))
        end
        S[:, (b-1)*D + a] = vec(R)
    end
    perm = [(ab = idx_to_ab(idx, L); ab[2]*D + ab[1] + 1) for idx in 0:(4^L - 1)]
    return S[perm, perm]
end

const W, SHIFT = fused_lindblad(L, JM, fill(GAMMA, L))
const IMPS = identity_superket(L)
const SREF = dense_super(GAMMA)
rho0() = rho0_product(SPINS)
const WANT = exp(SREF * T) * fused_dense_state(rho0())

say(@sprintf("L = %d FUSED sites. Galerkin space at the single bond is b_L x b_R <= 4 x 4 = 16,", L))
say(@sprintf("and the FULL Hilbert space is 4^%d = %d. So a full-rank frame => COMPLETE basis.", L, 4^L))
say("")
say(@sprintf("  %-30s %6s %13s %16s %6s %6s %6s",
             "arm", "steps", "state dev", "Tr rho", "chi", "minbL", "minbR"))
for g in (2.0, 4.0, 8.0, 16.0), m in (0, 3, 8)
    for n in (3, 48)
        p = copy(rho0()); dt = T / n
        # ⛔ MINIMUM OVER STEPS, NOT MAXIMUM. "The basis was complete" is a claim about EVERY
        # step; a max reports the best step and hides a starved first one, which is exactly how a
        # rank-limited run can be mistaken for a defect. `typemax` seeds the min.
        frame = typemax(Int); fright = typemax(Int)
        for _ in 1:n
            info = cbe_bug_step!(p, W, ComplexF64(dt); exact = true, growth = g,
                                 krylov_basis = m, krylov_tol = 0.0, hermitian = false,
                                 maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)
            # the frame width the sweep ACTUALLY reached -- checked, never assumed
            # ⛔ BOTH FRAME WIDTHS. The Galerkin space is `b_L x b_R`, and `expanded[c]` records
            # the LEFT one only -- judging completeness from it alone can call a basis complete
            # when it is not. `root_right` was added for exactly this.
            # ⛔ `init` FOR A MINIMUM IS `typemax`, NOT 0. `minimum(x; init = 0)` seeds the fold at
            # zero and therefore ALWAYS returns 0 -- it silently reported "b_L = 0" for every run,
            # which reads as a catastrophically starved basis and is pure artefact. `init = 0` is
            # right for `maximum` and wrong here; the two are not symmetric.
            frame  = min(frame,  hasproperty(info, :expanded) ?
                                 minimum(info.expanded; init = typemax(Int)) : typemax(Int))
            fright = min(fright, hasproperty(info, :root_right) ? info.root_right : 0)
            SHIFT != 0.0 && apply_shift!(p, exp(dt * SHIFT))
        end
        d  = maximum(abs.(fused_dense_state(p) .- WANT))
        tr = real(trace_rho(IMPS, p, L))
        say(@sprintf("  growth %-5.1f m=%-2d %-12s %6d %13.4e %16.12f %6d %6d %6d",
                     g, m, "", n, d, tr, maximum(state_bond_dims(p)), frame, fright))
    end
end
say("")
say("VERDICT:")
say("  b_L = b_R = 4 (COMPLETE, 4x4 = 16 = the whole space) AND dev ~ 1e-14 AND Tr = 1 ->")
say("     complete. The loss at larger L is then the MANIFOLD PROJECTION, and the fix is |I>> in")
say("     the basis -- not rescaling, which is neither principled nor scalable.")
say("  b_L = b_R = 4 but state dev >> 1e-14 -> AN IMPLEMENTATION BUG. A complete")
say("     Galerkin basis makes exp(tau*H0)S the exact propagator; anything else is a defect.")
close(io)
