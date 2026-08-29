# THE VECTORISED LINDBLADIAN ON FUSED d = 4 SITES -- the layout that works for BOTH dynamics
# and observables.
#
# Each system site becomes ONE chain site carrying the pair `(ket, bra)`, basis index
# `2*k + b + 1` with `0 = up`, `1 = down`:  1 = uu, 2 = ud, 3 = du, 4 = dd.
#
# ⛔ WHY NOT THE TWO SPIN-1/2 LAYOUTS, BOTH MEASURED AND BOTH REJECTED.
#
#   INTERLEAVED (ket 2j-1, ancilla 2j) -- the layout `PLAN` §3 originally chose, to keep the MPO
#   narrow. It puts every system-system term at RANGE 2, so NO two-site block contains both of a
#   term's operators; the block sees the term only through an environment channel, which for a
#   half-open term is a ONE-SITE EXPECTATION of a charge-changing operator, and `<S^-> = 0` on a
#   product state. MEASURED: the superket NEVER MOVES -- `max|rho_mps(T) - rho(0)| = 0.0000e+00`
#   exactly, error flat in dt (order 0.00 across a 16x range), identical for `tdvp2` and
#   `cbe_bug`, and equal to `|rho(T)-rho(0)|` to three digits. The MPO is NOT at fault: with
#   ENTANGLED probes it reproduces the dense superoperator to 1.3e-15.
#
#   BLOCKED (ket 1..L, ancilla L+1..2L) -- fixes the dynamics completely. MEASURED: 7.5e-14
#   (tdvp2) / 2.6e-15 (cbe_bug) at gamma = 0, and clean order-3 convergence at gamma = 0.5. But
#   `|I>>` is MAXIMALLY ENTANGLED across the ket/ancilla cut there (chi = 2^L), and a SPATIAL cut
#   needs TWO MPS cuts -- so operator entanglement, the observable this study is being asked for,
#   is not readable from a single bond.
#
# FUSED gets all three: `H` is RANGE 1 (wholly inside a two-site update, so the dynamics starts),
# the jump term is ON-SITE, `|I>>` is a PRODUCT state, and a spatial cut IS one MPS bond.
#
# ⚠ THE ON-SITE JUMP TERM STILL GOES THROUGH `pair_mpo`, which is two-body only: it is written as
# `O_j (x) I_{j+1}` on bonds 1..L-1 plus `I_{L-1} (x) O_L` on the last bond -- the same
# distribute-over-bonds trick `ising_bond_gates` uses for its field, done with vertices. Per-vertex
# coupling matrices (already in `pair_mpo`) are what make two different supports expressible at once.
#
# ⚠ OPERATOR NORMALISATION IS WRITTEN OUT HERE, NOT INHERITED. Telum's `q.Sp` carries a 1/sqrt(2)
# (see `gates.jl`), which silently rescaled a first attempt at `|I>>` -- `Tr rho0` came back -0.5
# instead of +1. These matrices are given explicitly, so there is nothing to inherit.
using LinearAlgebra, SparseArrays, Printf
using LurCGT, Telum
using Telum: _getLocalSpace_no_symmetry
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "fused_lindblad.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

const I2 = Matrix{Float64}(LinearAlgebra.I, 2, 2)
const SP = Float64[0 1; 0 0]
const SM = Float64[0 0; 1 0]
const SZD = Float64[0.5 0; 0 -0.5]

"""
The d = 4 local space for one fused (ket, bra) pair, under `:none`.

⚠ `kron(A, B)` puts A on the KET (the more significant index), matching `2*k + b + 1`.
`CUP` is the operator that turns `|uu>` into `(|uu> + |dd>)/sqrt2`; it is written SYMMETRIC on
purpose, so it acts identically whether the contraction convention applies it or its transpose.
"""
function fused_space()
    c = 1 / sqrt(2)
    CUP = zeros(Float64, 4, 4)
    CUP[1, 1] = c; CUP[4, 1] = c; CUP[1, 4] = c; CUP[4, 4] = c
    ops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}(
        :I   => sparse(kron(I2, I2)),
        :Kp  => sparse(kron(SP, I2)), :Km => sparse(kron(SM, I2)), :Kz => sparse(kron(SZD, I2)),
        :Bp  => sparse(kron(I2, SP)), :Bm => sparse(kron(I2, SM)), :Bz => sparse(kron(I2, SZD)),
        :JMP => sparse(kron(SZD, SZD)),
        :CUP => sparse(CUP))
    return _getLocalSpace_no_symmetry(ops, ("s", "s", "op"))
end
const Q4 = fused_space()

"A product state over fused sites; `idxs[j]` in 1..4. Mirrors `_product_state_none`."
function fused_product_state(idxs::Vector{Int})
    L = length(idxs)
    tensors = Any[]
    prev = (getvac(Q4.I, ("L,0", "L,1")), 2)
    for i in 1:L
        (1 <= idxs[i] <= 4) || throw(ArgumentError("fused index must be 1..4, got $(idxs[i])"))
        F = to_concrete(getIdentity(prev, (Q4.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = to_concrete(getsub(F, 3, s -> idxs[i]))
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

"Apply a one-site operator; contract the operator LEG 1, as `apply_gate` does."
function site_apply!(p, k::Int, M)
    t = p[k].inds[2].itags
    op = to_concrete(setitag(setitag(M, 1, t), 2, t))
    p[k] = to_concrete(permutedims(contract(op, (1,), p[k], (2,)), (2, 1, 3)))
    return p
end

"`4^L` amplitudes, site 1 most significant, by one overlap per basis configuration."
function fused_dense_state(psi)
    L = length(psi)
    P = _full_tensor(psi)
    v = zeros(ComplexF64, 4^L)
    for idx in 0:(4^L - 1)
        cfg = fused_product_state([(idx ÷ 4^(L - j)) % 4 + 1 for j in 1:L])
        v[idx + 1] = tensor_inner(_full_tensor(cfg), P)
    end
    return v
end

"Fused chain index -> (ket config, bra config), both `L`-bit, site 1 most significant."
function idx_to_ab(idx::Int, L::Int)
    a = 0; b = 0
    for j in 1:L
        d = (idx ÷ 4^(L - j)) % 4
        a |= (d >> 1) << (L - j)
        b |= (d & 1)  << (L - j)
    end
    return a, b
end

"""
The Lindbladian MPO on fused sites, and the zero-body constant it cannot hold.

`Jm` is the SYSTEM coupling (real; the ancilla half is `+i*H^T` and is written as `+i*Jm`).
"""
function fused_lindblad(L::Int, Jm::AbstractMatrix, gammas::Vector{Float64}; delta::Float64 = 0.0)
    PV = RSVDCBEBondUpdate.PairVertex
    Jk = zeros(ComplexF64, L, L); Jb = zeros(ComplexF64, L, L)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        Jk[i, j] = -im * Jm[i, j]
        Jb[i, j] = +im * Jm[i, j]
    end
    # the on-site jump, distributed over bonds: O_j (x) I_{j+1} for j < L, then I (x) O_L
    Jj1 = zeros(ComplexF64, L, L); Jj2 = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        Jj1[j, j + 1] = gammas[j]
    end
    Jj2[L - 1, L] = gammas[L]
    verts = PV[PV(Q4.Kp, Q4.Km, 0.5), PV(Q4.Km, Q4.Kp, 0.5), PV(Q4.Kz, Q4.Kz, delta),
               PV(Q4.Bp, Q4.Bm, 0.5), PV(Q4.Bm, Q4.Bp, 0.5), PV(Q4.Bz, Q4.Bz, delta),
               PV(Q4.JMP, Q4.I, 1.0), PV(Q4.I, Q4.JMP, 1.0)]
    Js = Any[Jk, Jk, Jk, Jb, Jb, Jb, Jj1, Jj2]
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I), -sum(gammas) / 4
end

# ══ VALIDATION ═══════════════════════════════════════════════════════════════════════════
const SPINS = Dict(2 => [:up, :down], 3 => [:up, :down, :up], 4 => [:up, :down, :up, :up])
"`rho0 = |psi0><psi0|`: site j in |s,s> -- index 1 for up, 4 for down."
rho0_of(sp) = fused_product_state([s === :up ? 1 : 4 for s in sp])

function dense_super(L, Jm, gammas)
    D = 2^L
    sop(op, j) = (m = ComplexF64[1;;]; for k in 1:L; m = kron(m, k == j ? op : I2); end; m)
    SXm = ComplexF64[0 .5; .5 0]; SYm = ComplexF64[0 -.5im; .5im 0]
    H = sum(Jm[i,j] * (sop(SXm,i)*sop(SXm,j) + sop(SYm,i)*sop(SYm,j))
            for i in 1:L for j in (i+1):L if Jm[i,j] != 0.0)
    S = zeros(ComplexF64, D*D, D*D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a,b] = 1.0
        R = -im * (H*E - E*H)
        for j in 1:L
            gammas[j] == 0.0 && continue
            Z = sop(ComplexF64.(SZD), j)
            R += gammas[j] * (Z*E*Z - 0.5*(Z*Z*E + E*Z*Z))
        end
        S[:, (b-1)*D + a] = vec(R)
    end
    perm = [(ab = idx_to_ab(idx, L); ab[2]*D + ab[1] + 1) for idx in 0:(4^L - 1)]
    return S[perm, perm]
end

for L in (2, 3)
    say(@sprintf("== L = %d fused sites (dim 4^%d = %d) ==", L, L, 4^L))
    # ── GATE 0: |I>> is a PRODUCT state and IS vec(identity) ──────────────────────────────
    Imps = rho0_of([:up for _ in 1:L])
    for j in 1:L
        site_apply!(Imps, j, Q4.CUP)
    end
    dI = fused_dense_state(Imps) .* 2.0^(L / 2)
    want = [idx_to_ab(idx, L)[1] == idx_to_ab(idx, L)[2] ? 1.0 + 0im : 0.0 + 0im
            for idx in 0:(4^L - 1)]
    d0 = maximum(abs.(dI .- want))
    say(@sprintf("  GATE 0  |I>> = vec(identity), bond dims %s : %10.3e  %s",
                 string(state_bond_dims(Imps)), d0, d0 < 1e-12 ? "PASS" : "FAIL"))
    # ── GATE 1/2: traces of a product rho0 ────────────────────────────────────────────────
    sp = SPINS[L]; r0 = rho0_of(sp)
    trf(rho; at = 0) = begin
        o = copy(Imps); at > 0 && site_apply!(o, at, Q4.Kz)
        2.0^(L / 2) * overlap(o, rho)
    end
    t = real(trf(r0))
    say(@sprintf("  GATE 1  Tr rho0 = %+.12f (expect +1)  %s", t,
                 isapprox(t, 1.0; atol = 1e-10) ? "PASS" : "FAIL"))
    for j in 1:L
        z = real(trf(r0; at = j)); w = sp[j] === :up ? 0.5 : -0.5
        say(@sprintf("  GATE 2  Tr(S^z_%d rho0) = %+.12f (expect %+.1f)  %s", j, z, w,
                     isapprox(z, w; atol = 1e-10) ? "PASS" : "FAIL"))
    end
    # ── GATE 3: the dynamics, dt-refined against the exact dense propagator ───────────────
    Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    for gamma in (0.0, 0.5)
        gammas = fill(gamma, L)
        W, shift = fused_lindblad(L, Jm, gammas)
        S = dense_super(L, Jm, gammas)
        T = 0.3
        v0 = fused_dense_state(r0)
        wantv = exp(S * T) * v0
        say(@sprintf("  GATE 3  gamma = %.2f  MPO width %d  |rho(T)-rho(0)| = %.4e",
                     gamma, maximum(mpo_virtual_dims(W)), maximum(abs.(wantv .- v0))))
        for (nm, f) in (
            ("tdvp2",       (p, dt) -> tdvp2_step!(p, W, ComplexF64(dt); maxdim = 256,
                                trunc_thresh = 1e-14, maxiter = 30, hermitian = false)),
            ("cbe_bug m=0", (p, dt) -> cbe_bug_step!(p, W, ComplexF64(dt); exact = true,
                                krylov_basis = 0, krylov_tol = 0.0, hermitian = false,
                                maxdim = 256, trunc_thresh = 1e-14, maxiter = 30)))
            prev = NaN
            for n in (3, 12, 48)
                p = copy(r0); dt = T / n
                d, trv = try
                    for _ in 1:n
                        f(p, dt)
                        shift != 0.0 && (p[1] = to_concrete(ComplexF64(exp(dt*shift)) * p[1]))
                    end
                    (maximum(abs.(fused_dense_state(p) .- wantv)), real(trf(p)))
                catch e
                    say("      $nm THREW: " * sprint(showerror, e)[1:min(end, 200)]); break
                end
                ord = isnan(prev) ? "" : @sprintf("%.2f", log2(prev / d) / 2)
                say(@sprintf("      %-12s %3d steps  dev %10.3e  Tr rho %.10f  order %s",
                             nm, n, d, trv, ord))
                prev = d
            end
        end
    end
    say("")
end
say("GATE 0 bond dims all 1 = |I>> really is a PRODUCT state, which is the whole point of fusing.")
say("GATE 3 order ~ 2+ with Tr rho pinned at 1 = a working, trace-preserving open-system integrator.")
close(io)
