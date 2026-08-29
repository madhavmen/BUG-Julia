# IS `cbe_bug` RANK-STARVED ON A d = 4 CHAIN, OR GENUINELY FIRST ORDER THERE?
#
# MEASURED in `fused_lindblad.jl`: on the fused chain `tdvp2` reaches 1.9e-14 while `cbe_bug m=0`
# sits at 2.95e-03 with convergence order 1.00, at BOTH L = 2 and L = 3 -- so it is systematic,
# not an L = 2 degeneracy. In the BLOCKED spin-1/2 test the same `cbe_bug m=0` reached 2.6e-15,
# so it is also not intrinsic to the scheme.
#
# ⛔ THE SUSPECT IS THE EXPANSION BUDGET, WHICH IS CALIBRATED FOR d = 2.
#     budget = ceil(growth * dmax) - r,  growth = 2.0
# "a bond at the neighbourhood max may DOUBLE" -- which is the right unit when one site can at
# most double a bond. A d = 4 site can QUADRUPLE it, so the schedule that tracks the reachable
# space on a spin-1/2 chain throttles it on a fused one. This repo already records that
# `growth = 2.0` is a PER-PROBLEM starting point to be re-derived, not a constant.
#
# ⚠ DISTINGUISHING THE TWO IS THE WHOLE POINT, and the achieved chi is what does it:
#     chi BELOW full rank  -> starvation; more budget must fix it
#     chi AT full rank and the error still O(dt) -> a real first-order step on this generator,
#                                                  and no amount of budget will help
# Full rank for `rho = |psi><psi|` on fused sites is `chi_ket^2`, i.e. 4 at the centre for L = 3 --
# printed alongside so the reader is not asked to remember it.
#
#   julia --project=. benchmarks/fused_growth_scan.jl

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "fused_growth_scan.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const L = 3; const T = 0.3
const SPINS = [:up, :down, :up]
const JM = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]

"`4^L` amplitudes, site 1 most significant, by one overlap per basis configuration."
function fused_dense_state(psi)
    n = length(psi)
    P = _full_tensor(psi)
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
        if gamma != 0.0
            for j in 1:L
                Z = sop(ComplexF64.(_SZ), j)
                R += gamma * (Z*E*Z - 0.5*(Z*Z*E + E*Z*Z))
            end
        end
        S[:, (b-1)*D + a] = vec(R)
    end
    perm = [(ab = idx_to_ab(idx, L); ab[2]*D + ab[1] + 1) for idx in 0:(4^L - 1)]
    return S[perm, perm]
end

rho0() = rho0_product(SPINS)
const IMPS = identity_superket(L)

for gamma in (0.0, 0.5)
    W, shift = fused_lindblad(L, JM, fill(gamma, L))
    S = dense_super(gamma)
    v0 = fused_dense_state(rho0())
    want = exp(S * T) * v0
    say(@sprintf("── gamma = %.2f, L = %d fused sites; full rank for |psi><psi| is chi = %d ──",
                 gamma, L, 4))
    say(@sprintf("  %-22s %6s %12s %8s %14s %8s", "config", "steps", "dev", "chi", "Tr rho", "order"))
    for (nm, mk) in (
        ("tdvp2 (reference)", (dt) -> (p, W) -> tdvp2_step!(p, W, ComplexF64(dt);
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30, hermitian = false)),
        ("cbe_bug m=0 g=2",   (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        krylov_basis = 0, krylov_tol = 0.0, growth = 2.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)),
        ("cbe_bug m=0 g=4",   (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        krylov_basis = 0, krylov_tol = 0.0, growth = 4.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)),
        ("cbe_bug m=0 g=8",   (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        krylov_basis = 0, krylov_tol = 0.0, growth = 8.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)),
        ("cbe_bug m=3 g=4",   (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        krylov_basis = 3, krylov_tol = 0.0, growth = 4.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)),
        # ⛔ THE CANONICAL ARM, AND THE ONE THE STUDY ACTUALLY QUOTES. `bug_interleaved` is
        # `cbe_bug_step!` at LIBRARY DEFAULTS -- `krylov_basis = 30` (a CEILING) with
        # `krylov_tol = 1e-6` (Saad's stopping rule), i.e. ADAPTIVE depth, not `m = 0`. The
        # first-order result above was measured on the m = 0 arm and says nothing about this one;
        # including it here is what stops that mistake from being repeated.
        ("bug_interleaved g=2", (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        growth = 2.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)),
        ("bug_interleaved g=4", (dt) -> (p, W) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                        growth = 4.0, hermitian = false,
                        maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)))
        prev = NaN
        for n in (3, 12, 48)
            dt = T / n
            p = copy(rho0())
            d, chi, tr = try
                f = mk(dt)
                for _ in 1:n
                    f(p, W)
                    shift != 0.0 && apply_shift!(p, exp(dt * shift))
                end
                (maximum(abs.(fused_dense_state(p) .- want)),
                 maximum(state_bond_dims(p)), real(trace_rho(IMPS, p, L)))
            catch e
                say("    $nm THREW: " * sprint(showerror, e)[1:min(end, 180)]); break
            end
            ord = isnan(prev) ? "" : @sprintf("%.2f", log2(prev / d) / 2)
            say(@sprintf("  %-22s %6d %12.4e %8d %14.10f %8s", nm, n, d, chi, tr, ord))
            prev = d
        end
    end
    say("")
end
say("chi < 4 with an O(dt) error = STARVATION (more budget fixes it).")
say("chi = 4 with an O(dt) error  = a genuine first-order step on this generator.")
close(io)
