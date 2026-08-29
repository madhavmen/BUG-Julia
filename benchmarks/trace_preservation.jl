# WHY DOES `bug_interleaved` LOSE `Tr rho`, AND CAN IT BE MADE TO PRESERVE IT?
#
# `Tr rho` is the conservation law of a Lindbladian: `<<I|L = 0` exactly, because every term is a
# commutator or a jump-plus-anticommutator and `<<I|` annihilates each. `tdvp2` inherits that
# EXACTLY -- its sweep is a product of local exponentials `exp(tau*h_i)` and each `h_i` separately
# annihilates `<<I|`, so the product does too. A GALERKIN scheme does not: it evolves
# `exp(tau * P L P)` in a truncated basis, and
#
#     Tr(P rho) = <<I|P|rho>> = <<P I|rho>>
#
# so the trace survives the projection IF AND ONLY IF `P|I>> = |I>>`, i.e. the variational basis
# CONTAINS the identity superket. Nothing in CBE/BUG puts it there.
#
# ⛔ TWO CANDIDATE CAUSES, AND THEY CALL FOR DIFFERENT FIXES:
#     (a) TRUNCATION discarding weight that carries trace  -> more rank / a trace-aware truncation
#     (b) the GALERKIN projection itself, even at full rank -> the basis must contain |I>>
# Distinguishing them is what this file does, by running the SAME arm at a cap that cannot bind
# (full rank) and at a cap that certainly does.
#
# ⚠ THE dt SCALING IS THE OTHER HALF OF THE ANSWER. A truncation-driven loss is set by the
# DISCARDED WEIGHT and barely moves with dt at fixed rank; an integrator-driven one falls like a
# power of dt. Both are printed.
#
# ⚠ AND `|I>>` IS A PRODUCT STATE IN THE FUSED LAYOUT (chi = 1, MEASURED: bond dims [1]) -- so if
# (b) is the cause, the structural fix costs ONE extra direction per bond, not a rank blow-up.
#
#   julia --project=. benchmarks/trace_preservation.jl

using Printf, LinearAlgebra
include(joinpath(@__DIR__, "fused_common.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "trace_preservation.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const L = 3; const T = 0.3; const GAMMA = 0.5
const SPINS = [:up, :down, :up]
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
const V0 = fused_dense_state(rho0())
const WANT = exp(SREF * T) * V0

"""
Run one arm. `fix` selects the per-step trace correction: `:none`, `:rescale`, or `:along_I`.

⚠ "FIXES `Tr`" AND "IMPROVES THE STATE" ARE DIFFERENT CLAIMS, which is why both columns are
printed. `rho/Tr(rho)` is the physical density matrix and every observable is `Tr(O rho)/Tr(rho)`,
so rescaling is legitimate -- but it multiplies the TRACELESS part too, and if the error is not a
pure trace deficit it overcorrects. `:along_I` moves only along `|I>>`, leaving the traceless part
untouched. Quoting `Tr = 1` after either as evidence of accuracy would be circular.
"""
function arm(name, mk, cap, cut, nsteps; fix::Symbol = :none)
    p = copy(rho0()); dt = T / nsteps
    f = mk(cap, cut)
    for _ in 1:nsteps
        f(p, dt)
        SHIFT != 0.0 && apply_shift!(p, exp(dt * SHIFT))
        # :rescale  divides the WHOLE state by Tr -- overcorrects, the traceless part included.
        # :along_I  adds the deficit back ONLY along |I>>, leaving the traceless part untouched.
        #           That is the minimal correction that fixes the conservation law, and the whole
        #           point of testing it separately is that "fixes Tr" and "improves the state" are
        #           different claims.
        fix === :rescale && normalise_trace!(IMPS, p, L)
        fix === :along_I && restore_trace!(IMPS, p, L; maxdim = cap, cutoff = cut)
    end
    d = maximum(abs.(fused_dense_state(p) .- WANT))
    return d, real(trace_rho(IMPS, p, L)), maximum(state_bond_dims(p))
end

mk_tdvp2(cap, cut) = (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = cap,
                          trunc_thresh = cut, maxiter = 30, hermitian = false)
mk_bug(cap, cut)   = (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true, growth = 4.0,
                          hermitian = false, maxdim = cap, trunc_thresh = cut, maxiter = 30)

say(@sprintf("L = %d fused sites, gamma = %.2f, T = %.2f. FULL RANK for |psi><psi| is chi = 4.",
             L, GAMMA, T))
say("")
for (label, cap, cut) in (("cap 256, cut 1e-14  (cannot bind: FULL RANK)", 256, 1e-14),
                          ("cap 2               (BINDS: forced truncation)", 2, 1e-14))
    say("── $label ──")
    say(@sprintf("  %-26s %6s %12s %16s %7s %10s",
                 "arm", "steps", "state dev", "Tr rho", "chi", "1-Tr"))
    for (nm, mk, rn) in (("tdvp2", mk_tdvp2, :none),
                         ("bug_interleaved", mk_bug, :none),
                         ("bug +rescale Tr", mk_bug, :rescale),
                         ("bug +restore along |I>", mk_bug, :along_I))
        for n in (3, 12, 48)
            d, tr, chi = try
                arm(nm, mk, cap, cut, n; fix = rn)
            catch e
                say("    $nm THREW: " * sprint(showerror, e)[1:min(end, 160)]); break
            end
            say(@sprintf("  %-26s %6d %12.4e %16.12f %7d %10.2e", nm, n, d, tr, chi, abs(1 - tr)))
        end
    end
    say("")
end
say("READING IT:")
say("  `1-Tr` falling like dt^2 at FULL RANK  -> the GALERKIN PROJECTION, not truncation.")
say("  `1-Tr` roughly FLAT in dt at cap 2     -> TRUNCATION, set by discarded weight.")
say("  If both appear, both fixes are needed: |I>> in the basis, and a trace-aware truncation.")
say("  BOTH fixes pin Tr to 1. The question is the STATE dev:")
say("    `+rescale`  divides the whole state -> MEASURED 1.91x WORSE, it overcorrects.")
say("    `+restore along |I>` moves only along the identity direction, leaving the traceless")
say("    part exactly alone. If it fixes Tr WITHOUT costing state accuracy, it is the fix.")
close(io)
