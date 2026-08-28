# THE DISSIPATIVE ANISOTROPIC HEISENBERG CHAIN OF arXiv Comms Phys (2026) 9:45 -- Hryniuk &
# Szymanska, "Variational approach to open quantum systems with long-range competing interactions".
#
#   julia -t 2 --project=. benchmarks/dissipative_heisenberg.jl [certify|spectrum|paper]
#
# ⛔ WHY THIS MODEL. It is the THIRD model of the campaign and the one that CANNOT use a
# logarithmic grid. The other two (Heisenberg chain, Heisenberg cylinder) are IMAGINARY-time ground
# state searches: `H` is Hermitian, its spectrum is REAL, `exp(-H tau)` is a monotone decay, and a
# log grid is safe because there is nothing to alias. A Lindbladian is NOT Hermitian, its spectrum
# is COMPLEX, and `exp(L t)` is a decay TIMES an oscillation -- so the log grid's enormous late
# steps alias it. `spectrum` below draws exactly that, which is the figure the claim needs.
#
# ⚠ THE STEADY STATE IS REACHED IN REAL TIME, NOT IMAGINARY TIME. The NESS is the eigenvector of
# `L` with eigenvalue 0, and plain `exp(L t)` converges to it because every other eigenvalue has
# `Re lambda < 0`. There is no imaginary-time analogue here, and the rate is set by the Liouvillian
# GAP -- which is why "run to the steady state" is a statement about `t`, not about the integrator.
#
# ── THE PAPER'S PROTOCOL (Fig. 3a-c), AND WHERE OURS DIFFERS ────────────────────────────────────
#
#   H     = -sum_i (Jx sx_i sx_{i+1} + Jy sy_i sy_{i+1} + Jz sz_i sz_{i+1}) - h sum_i sz_i   [Eq 5]
#   Gamma_k = sqrt(gamma)/2 (sx_k - i sy_k) = sqrt(gamma) * sigma^-_k         (spin DECAY)
#   (Jx, Jy, Jz) = (-1.0, -0.9, -1.2),  gamma = -h = 1,  start = product state <sx> = +1
#   N = 200, chi = 20, t in [0, 4];  their exact check is N = 10 via QuantumOptics.jl
#   observables: bulk <s^a(t)> and C^aa_d(t) = <s^a_i s^a_{i+d}> - <s^a_i><s^a_{i+d}>, d = 1, 2
#
# ⛔ THEY USE PAULI MATRICES AND WE USE SPIN OPERATORS. `s^a = 2 S^a`, so `s^a s^a = 4 S^a S^a` and
# `h s^z = 2h S^z`. Getting this wrong does NOT throw -- it produces a smooth, plausible curve at
# the WRONG FREQUENCY, which is indistinguishable from a converged result unless you already know
# the answer. In `S` operators, using `S+S- + S-S+ = 2(SxSx + SySy)` and
# `S+S+ + S-S- = 2(SxSx - SySy)`:
#
#   H = -sum_i [ (Jx+Jy)(S+_i S-_{i+1} + S-_i S+_{i+1})
#              + (Jx-Jy)(S+_i S+_{i+1} + S-_i S-_{i+1})
#              + 4 Jz S^z_i S^z_{i+1} ]  -  2h sum_i S^z_i
#
# ⛔ `Jx != Jy` IS WHAT ADDS THE `S+S+`/`S-S-` CHANNEL, and no existing builder here has it:
# `fused_lindblad` is XXZ (`Kp/Km` pairs plus `Kz Kz`). It is also what DESTROYS the symmetry --
# `S+S+` changes `S^z` by 2. ⚠ NOTE WHICH TERM IS GUILTY: the spin-decay DISSIPATOR is innocent.
# `sigma^- rho sigma^+` is `kron(SM, SM)` in the fused layout, lowering ket and bra TOGETHER, so it
# PRESERVES the difference charge `S^z_ket - S^z_bra` that a fused superket would exploit. Only the
# anisotropic Hamiltonian breaks it. ⇒ a U(1) arm needs `Jx = Jy` (XXZ) and is a deliberate
# ONE-PARAMETER departure from the paper, never a claim about their model.
#
# ⛔ AND `pair_mpo` APPLIES THE TRANSPOSE (the defect that froze the boundary-driven model for a
# whole diagnostic ladder). Three consequences, all load-bearing:
#   * `PV(Kp,Km)` + `PV(Km,Kp)` is invariant as a SUM under the swap, which is why XX never noticed.
#   * `PV(Kp,Kp)` + `PV(Km,Km)` is invariant for the SAME reason -- so the new anisotropic channel
#     is safe, but only because BOTH are present with the SAME coefficient.
#   * `kron(SM,SM)^T = kron(SP,SP)`, so the DECAY jump must be written `Q4.JP`. Writing the
#     "obvious" `Q4.JM` gives GAIN, and the anticommutator then cancels it exactly.
#
# ⚠ EVERY ONE OF THOSE IS CERTIFIED AGAINST A DENSE `4^L` LIOUVILLIAN BELOW RATHER THAN ARGUED.
# Four conventions stack here (Pauli-vs-spin, the -i/+i on ket/bra, the transpose, the zero-body
# shift) and each is individually plausible when wrong. `certify` builds the superoperator from
# first principles at L = 2..4 and compares the MPO's action to it.

using LinearAlgebra, Printf
include(joinpath(@__DIR__, "fused_common.jl"))

const PAPER = (Jx = -1.0, Jy = -0.9, Jz = -1.2, h = -1.0, gamma = 1.0)

# ── the fused MPO ───────────────────────────────────────────────────────────────────────────────

"""
    xyz_decay_lindblad(L; Jx, Jy, Jz, h, gamma) -> (mpo, shift)

Fused d=4 Liouvillian of the paper's chain. `shift` is the zero-body piece `pair_mpo` cannot hold.

⚠ `gamma` IS UNIFORM OVER ALL SITES, unlike `fused_boundary_lindblad` where only the two ends are
driven. The one-body terms therefore ride EVERY bond (`Jone[j, j+1]` for `j = 1..L-1`) plus one
extra vertex for the last site, since bond `j` only ever reaches site `j+1`.
"""
function xyz_decay_lindblad(L::Int; Jx::Float64 = PAPER.Jx, Jy::Float64 = PAPER.Jy,
                            Jz::Float64 = PAPER.Jz, h::Float64 = PAPER.h,
                            gamma::Float64 = PAPER.gamma)
    L >= 2 || throw(ArgumentError("need at least two sites"))
    PV = RSVDCBEBondUpdate.PairVertex
    verts = PV[]; Js = Any[]
    nn(c) = begin                       # nearest-neighbour coupling matrix with entry `c`
        A = zeros(ComplexF64, L, L)
        for j in 1:(L - 1); A[j, j + 1] = c; end
        A
    end
    push2(opk, opb, coeff, c) = begin   # one Hamiltonian term, ket half (-i) and bra half (+i)
        push!(verts, PV(opk[1], opk[2], coeff)); push!(Js, nn(-im * c))
        push!(verts, PV(opb[1], opb[2], coeff)); push!(Js, nn(+im * c))
    end
    # ⚠ THE HAMILTONIAN CARRIES AN OVERALL MINUS (Eq. 5), so each coefficient below is `-(...)`.
    xy  = -(Jx + Jy)              # (S+S- + S-S+)
    aniso = -(Jx - Jy)            # (S+S+ + S-S-)  -- zero iff Jx == Jy
    zz  = -4 * Jz                 # S^z S^z
    push2((Q4.Kp, Q4.Km), (Q4.Bp, Q4.Bm), 1.0, xy)
    push2((Q4.Km, Q4.Kp), (Q4.Bm, Q4.Bp), 1.0, xy)
    if aniso != 0
        # ⛔ BOTH OR NEITHER. `pair_mpo` transposes, mapping `(Kp,Kp) <-> (Km,Km)`; the SUM is
        # invariant only because both are pushed with the SAME coefficient. Dropping one would
        # silently evolve `S+S+` as `S-S-`.
        push2((Q4.Kp, Q4.Kp), (Q4.Bp, Q4.Bp), 1.0, aniso)
        push2((Q4.Km, Q4.Km), (Q4.Bm, Q4.Bm), 1.0, aniso)
    end
    push2((Q4.Kz, Q4.Kz), (Q4.Bz, Q4.Bz), 1.0, zz)
    # ── one-body terms, on EVERY site ──────────────────────────────────────────────────────────
    one_body!(op, c) = begin
        c == 0 && return
        A = zeros(ComplexF64, L, L); B = zeros(ComplexF64, L, L)
        for j in 1:(L - 1); A[j, j + 1] = c; end     # sites 1..L-1
        B[L - 1, L] = c                              # site L needs its own vertex
        push!(verts, PV(op, Q4.I, 1.0)); push!(Js, A)
        push!(verts, PV(Q4.I, op, 1.0)); push!(Js, B)
    end
    # ⛔ THE DECAY JUMP IS `Q4.JP`, NOT `Q4.JM` -- see the header. `kron(SM,SM)^T = kron(SP,SP)`.
    one_body!(Q4.JP, gamma)
    # ⛔ ANTICOMMUTATOR + FIELD, COMBINED ON THE SAME OPERATOR.
    #   -gamma/2 {S+S-, .} with `S+S- = n_up = I/2 + S^z` gives `-gamma/2` on `Kz` AND on `Bz`,
    #   plus an identity part `-gamma/4` PER LEG = `-gamma/2` per site -> the zero-body `shift`.
    #   the field `-2h S^z` enters the ket half as `-i*(-2h) = +2ih` and the bra half as `-2ih`.
    one_body!(Q4.Kz, -gamma / 2 + 2im * h)
    one_body!(Q4.Bz, -gamma / 2 - 2im * h)
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I), -gamma * L / 2
end

# ── the dense reference, for certification ONLY ─────────────────────────────────────────────────

const _SP2 = ComplexF64[0 1; 0 0]
const _SM2 = ComplexF64[0 0; 1 0]
const _SZ2 = ComplexF64[0.5 0; 0 -0.5]
_at(op, k, L) = kron(Matrix{ComplexF64}(I, 2^(k - 1), 2^(k - 1)), ComplexF64.(op),
                     Matrix{ComplexF64}(I, 2^(L - k), 2^(L - k)))

"""
Dense `4^L x 4^L` Liouvillian built from first principles, in SPIN operators.

⚠ `vec` IS COLUMN-MAJOR: `vec(A rho B) = kron(transpose(B), A) vec(rho)`, so the transpose sits on
the BRA leg. A dephasing model cannot detect a missing transpose (`S^z` is real symmetric);
`sigma^- rho sigma^+` can, which is what makes this model worth certifying.

⛔ ONLY FOR `L <= 5`. `4^6 = 4096` is already a 4096x4096 superoperator; this exists to pin
conventions at tiny `L` and is never used at benchmark sizes.
"""
function dense_xyz_decay(L::Int; Jx = PAPER.Jx, Jy = PAPER.Jy, Jz = PAPER.Jz,
                         h = PAPER.h, gamma = PAPER.gamma)
    L <= 5 || throw(ArgumentError("dense reference is for L <= 5 only, got $L"))
    d = 2^L
    Sx(k) = _at((_SP2 + _SM2) / 2, k, L)
    Sy(k) = _at((_SP2 - _SM2) / (2im), k, L)
    Sz(k) = _at(_SZ2, k, L)
    H = zeros(ComplexF64, d, d)
    for k in 1:(L - 1)
        H .-= 4 .* (Jx .* Sx(k) * Sx(k + 1) .+ Jy .* Sy(k) * Sy(k + 1) .+ Jz .* Sz(k) * Sz(k + 1))
    end
    for k in 1:L
        H .-= 2 * h .* Sz(k)
    end
    Id = Matrix{ComplexF64}(I, d, d)
    Ls = -im .* (kron(Id, H) .- kron(transpose(H), Id))
    for k in 1:L
        jump = sqrt(gamma) .* _at(_SM2, k, L)          # sigma^- == S^-
        LdL = jump' * jump
        Ls .+= kron(transpose(jump'), jump) .- 0.5 .* kron(Id, LdL) .- 0.5 .* kron(transpose(LdL), Id)
    end
    return Ls
end

# ── entry points ────────────────────────────────────────────────────────────────────────────────

say(s) = (println(s); flush(stdout))

"""
Exact `<sigma^z_j(t)>` of the dense model, from a Neel product start. PAULI normalisation.

⚠ `2 *` BECAUSE THE PAPER PLOTS PAULI EXPECTATIONS AND `trace_rho` RETURNS `<S^z>`. A factor of 2
between the two sides is the single easiest way to produce two smooth curves that disagree by
exactly a factor nobody notices on a log axis.
"""
function dense_sz_traj(L::Int, ts; kwargs...)
    Ld = dense_xyz_decay(L; kwargs...)
    d = 2^L
    rho0 = ComplexF64[1;;]
    for j in 1:L
        p = isodd(j) ? ComplexF64[1 0; 0 0] : ComplexF64[0 0; 0 1]    # Neel |up down ...>
        rho0 = kron(rho0, p)
    end
    v0 = vec(rho0)
    out = Vector{Vector{Float64}}()
    for t in ts
        v = exp(Matrix(Ld) .* float(t)) * v0
        rho = reshape(v, d, d)
        tr = real(LinearAlgebra.tr(rho))
        push!(out, [2 * real(LinearAlgebra.tr(_at(_SZ2, j, L) * rho)) / tr for j in 1:L])
    end
    return out
end

"""
Certify the fused MPO by EVOLVING both sides from the same product state and comparing observables.

⛔ WHY A DYNAMICAL COMPARISON AND NOT AN ENTRYWISE ONE. `dense_state` is hardcoded for `d = 2` and
cannot represent a fused superket at all, and `vec` need not agree with the fused `(ket, bra)`
index order anyway -- the two dense conventions in this repo are already known to be MIRRORED.
Comparing OBSERVABLES sidesteps the question entirely: `<sigma^z_j>` is a physical number, computed
on the MPS side through `trace_rho` (the same path the benchmarks use) and on the dense side through
`tr(sigma^z_j rho)`. If the two agree, the MPO is the Lindbladian AND the observable path is right.

⛔ A NEEL START, NOT `|I>>`. `|I>>` is precisely the state that hid the boundary-driven freeze for a
whole diagnostic ladder: the generator was wrong and `1-|<I|q>|^2` still read `4.4e-16`, because a
wrong operator happened to annihilate it. A Neel product state has weight on the channels that
matter and is buildable IDENTICALLY on both sides (`rho0_product` here, a Kronecker product there),
so nothing is assumed about the mapping between them.

⚠ THIS DOES FOLD IN THE INTEGRATOR'S `dt` ERROR, which is why it is run as a dt-REFINEMENT rather
than a single comparison: a correct MPO shows the deviation FALLING with `dt`, a wrong one shows a
floor that does not move -- the same signature that caught both defects in the XX model.
"""
function certify(; L::Int = 3, t::Float64 = 0.25, dts = (0.05, 0.025, 0.0125, 0.00625))
    set_symmetry!(:none)
    say("\n=== CERTIFY: fused MPO vs DENSE Liouvillian, by evolving both ===")
    say("  H = -sum (Jx sx sx + Jy sy sy + Jz sz sz) - h sum sz ;  Gamma = sqrt(gamma) sigma^-")
    say(@sprintf("  (Jx,Jy,Jz) = (%.1f,%.1f,%.1f)  h = %.1f  gamma = %.1f   [the paper's values]",
                 PAPER.Jx, PAPER.Jy, PAPER.Jz, PAPER.h, PAPER.gamma))
    say(@sprintf("  L = %d, Neel start, compare <sigma^z_j> at t = %g\n", L, t))
    exact = dense_sz_traj(L, (t,))[1]
    say("  exact <sigma^z_j> = " * join([@sprintf("%+.8f", x) for x in exact], " "))
    W, shift = xyz_decay_lindblad(L)
    Imps = identity_superket(L)
    say(@sprintf("\n  %-8s %-8s %-13s %-8s", "dt", "t_act", "max|d sigma^z|", "order"))
    prev = NaN; ok = false
    for dt in dts
        rho = rho0_product([isodd(j) ? :up : :down for j in 1:L])
        tnow = 0.0
        for _ in 1:round(Int, t / dt)
            tdvp2_step!(rho, W, ComplexF64(dt); maxdim = 256, trunc_thresh = 1e-14,
                        maxiter = 30, hermitian = false)
            apply_shift!(rho, exp(dt * shift))
            tnow += dt
        end
        trr = real(trace_rho(Imps, rho, L))
        prof = [2 * real(trace_rho(Imps, rho, L; at = j)) / trr for j in 1:L]
        dev = maximum(abs.(prof .- exact))
        p = isnan(prev) ? NaN : log2(prev / dev)
        say(@sprintf("  %-8.5f %-8.4f %-13.3e %-8s", dt, tnow, dev,
                     isnan(p) ? "--" : @sprintf("%.2f", p)))
        # ⛔ AT FULL RANK THE RIGHT ANSWER IS ROUNDOFF, NOT A CONVERGENCE RATE, AND GATING ON THE
        # RATE WOULD FAIL A PERFECT RESULT. With `maxdim = 256` at `L = 3` the MPS manifold IS the
        # whole `4^3 = 64`-dimensional space, so TDVP's tangent projector is the IDENTITY and the
        # integrator is exact up to the Krylov exponential -- which is also exact there. MEASURED:
        # `2.054e-15` at the very first `dt`. A `dt`-order test in that regime measures the
        # double-precision floor, which is the standing rule about needing a RANK-LIMITED regime.
        #
        # ⚠ SO THERE ARE TWO WAYS TO PASS, AND ONLY ONE WAY TO FAIL. Either the deviation is AT the
        # floor (the MPO is exactly right), or it FALLS with `dt` (right, with a resolvable
        # discretisation error). What fails is a deviation that is neither -- a floor well above
        # roundoff that does not move, which is exactly the signature both XX defects showed.
        ok = dev < 1e-12 || (!isnan(p) && p > 1.5)
        prev = dev
    end
    say(ok ? "\nCERTIFY PASSED -- the fused MPO reproduces the dense Liouvillian" :
             "\nCERTIFY FAILED -- deviation neither at roundoff nor falling with dt")
    return ok
end

"""
    spectrum(; Ls) -> writes results/dissip_heis_spectrum.csv

THE FIGURE THAT JUSTIFIES THE GRID CHOICE FOR ALL THREE MODELS, in one comparison.

⛔ THE CLAIM IS NOT "LOG GRIDS ARE BAD". It is that a log grid is safe for a generator whose
spectrum is REAL and unsafe for one whose spectrum is COMPLEX, and the two live in the same
campaign. So both generators are built HERE, for the SAME chain, and their eigenvalues are written
side by side:

  * `-H`  (imaginary time, models 1 and 2). `H` is Hermitian ⇒ every eigenvalue is REAL ⇒
    `exp(-H tau)` is a sum of MONOTONE decays. There is no oscillation, so there is nothing a large
    late step can alias, and the log grid's exponentially-growing steps cost nothing.
  * `L`   (the Lindbladian, model 3). Non-Hermitian ⇒ eigenvalues are COMPLEX ⇒ `exp(L t)` is a sum
    of decaying OSCILLATIONS at angular frequencies `|Im lambda|`. A step larger than
    `pi / max|Im lambda|` cannot resolve the fastest of them, and the log grid's last step is
    orders of magnitude larger than that.

⚠ THE BOUND IS `pi / max|Im lambda|` FOR A LINEAR OBSERVABLE. A QUADRATIC observable (a correlation
function, a current) beats pairs of eigenvalues together and its fastest frequency is
`2 max|Im lambda|`, halving the allowed step. The XX model's bound is `pi/4` and not `pi/2` for
exactly this reason, and using the linear bound there would have drawn half the aliasing arms as
resolving. Both are reported so the figure can state which observable it applies to.

⚠ SMALL `L` ON PURPOSE. This is a statement about the STRUCTURE of the spectrum -- real versus
complex -- and that structure is set by the generator, not by the system size. `L = 3, 4` makes the
point at `4^L = 64, 256`, and going larger would only add eigenvalues to the same picture at
exponential cost.
"""
function spectrum(; Ls = (3, 4))
    say("\n=== SPECTRUM: why the OPEN model cannot use a logarithmic grid ===")
    out = joinpath(@__DIR__, "results", "dissip_heis_spectrum.csv")
    mkpath(dirname(out))
    open(out, "w") do io
        println(io, "generator,L,idx,re,im")
        for L in Ls
            # the Lindbladian
            Ld = dense_xyz_decay(L)
            el = eigvals(Matrix(Ld))
            for (k, z) in enumerate(el)
                @printf(io, "lindblad,%d,%d,%.10g,%.10g\n", L, k, real(z), imag(z))
            end
            # the SAME chain's Hamiltonian, the imaginary-time generator
            d = 2^L
            Sx(k) = _at((_SP2 + _SM2) / 2, k, L)
            Sy(k) = _at((_SP2 - _SM2) / (2im), k, L)
            Sz(k) = _at(_SZ2, k, L)
            H = zeros(ComplexF64, d, d)
            for k in 1:(L - 1)
                H .-= 4 .* (PAPER.Jx .* Sx(k) * Sx(k + 1) .+ PAPER.Jy .* Sy(k) * Sy(k + 1) .+
                            PAPER.Jz .* Sz(k) * Sz(k + 1))
            end
            for k in 1:L; H .-= 2 * PAPER.h .* Sz(k); end
            eh = eigvals(Hermitian(Matrix(H)))
            for (k, z) in enumerate(eh)
                @printf(io, "hamiltonian,%d,%d,%.10g,%.10g\n", L, k, -real(z), 0.0)
            end
            mim = maximum(abs.(imag.(el)))
            gap = -maximum(real(z) for z in el if real(z) < -1e-9)
            say(@sprintf("\n  L = %d   (4^L = %d)", L, 4^L))
            say(@sprintf("    HAMILTONIAN  max|Im lambda| = %.3e   -> monotone, NO Nyquist limit",
                         maximum(abs.(imag.(eh)))))
            say(@sprintf("    LINDBLADIAN  max|Im lambda| = %.6f", mim))
            say(@sprintf("      Nyquist step, LINEAR observable    pi/max|Im|  = %.6f", pi / mim))
            say(@sprintf("      Nyquist step, QUADRATIC observable pi/(2max|Im|) = %.6f",
                         pi / (2 * mim)))
            say(@sprintf("    Liouvillian gap = %.6f  -> steady state needs t ~ %.1f (3 e-foldings)",
                         gap, 3 / gap))
            # what a log grid would actually do over that horizon
            tmax = 3 / gap
            for n in (8, 12, 20, 50)
                r = (tmax / 0.02)^(1 / (n - 1))
                ts = [0.02 * r^k for k in 0:(n - 1)]
                big = maximum(vcat(ts[1], diff(ts)))
                say(@sprintf("      log n=%-4d largest step %10.3f   %s", n, big,
                             big <= pi / mim ? "resolves" :
                             @sprintf("ALIASES (%.0fx over)", big * mim / pi)))
            end
        end
    end
    say("\nwrote " * out)
    return true
end

# ── the paper's observables, and the certification of the path that computes them ───────────────

"""
Dense `rho(t)` from a product start: `:neel` (`rho0_product`) or `:plusx` (`rho0_plusx`).

⚠ `:plusx` EXISTS FOR THE `sigma^y` SIGN AND NOTHING ELSE. Under a Neel start this model keeps
`<sigma^x> = <sigma^y> = 0` identically, so a `sigma^y` comparison there is `0` against `0`.
"""
function dense_rho(L::Int, t::Real; start::Symbol = :neel, kwargs...)
    d = 2^L
    rho0 = ComplexF64[1;;]
    px = ComplexF64[0.5 0.5; 0.5 0.5]
    for j in 1:L
        rho0 = kron(rho0, start === :plusx ? px :
                          isodd(j) ? ComplexF64[1 0; 0 0] : ComplexF64[0 0; 0 1])
    end
    return reshape(exp(Matrix(dense_xyz_decay(L; kwargs...)) .* float(t)) * vec(rho0), d, d)
end

"Dense `(<sx_j>, <sy_j>, <sz_j>)`, PAULI normalisation."
function dense_pauli1(rho, L::Int, j::Int)
    tr = real(LinearAlgebra.tr(rho))
    px = _at(_SP2 + _SM2, j, L)
    py = _at(-im .* (_SP2 - _SM2), j, L)
    pz = _at(2 .* _SZ2, j, L)
    return (real(LinearAlgebra.tr(px * rho)) / tr, real(LinearAlgebra.tr(py * rho)) / tr,
            real(LinearAlgebra.tr(pz * rho)) / tr)
end

"Dense `(<sx_j sx_k>, <sy_j sy_k>, <sz_j sz_k>)`, PAULI normalisation."
function dense_pauli2(rho, L::Int, j::Int, k::Int)
    tr = real(LinearAlgebra.tr(rho))
    p(a, s) = _at(a, s, L)
    X = _SP2 + _SM2; Y = -im .* (_SP2 - _SM2); Z = 2 .* _SZ2
    f(a) = real(LinearAlgebra.tr(p(a, j) * p(a, k) * rho)) / tr
    return (f(X), f(Y), f(Z))
end

"""
Certify the OBSERVABLE path -- `trace_op`, `pauli_1site`, `pauli_2site` -- against dense.

⛔ WHY THIS IS A SEPARATE GATE FROM `certify`. That one pins the GENERATOR using `<sigma^z>` alone,
and `sigma^z` is DIAGONAL: it reads back the same number whether `site_apply!` applies `M` or `M'`.
So does `sigma^x`, which is real symmetric. `sigma^y` is the ONLY one of the three that can tell
them apart (`sigma^y' = -sigma^y`), and the correlators are the only place a two-site product can
be mis-ordered. A generator certified to 2e-15 says nothing about either.

⚠ TWO INDEPENDENT CHECKS, because each catches what the other cannot:
  (a) `t = 0` on `rho0_plusx`, where `<sx> = 1`, `<sy> = <sz> = 0` and every CONNECTED correlator
      vanishes exactly (a product state). Catches normalisation and a sign on `sigma^y`, with no
      evolution involved at all -- so a failure here cannot be blamed on the integrator.
  (b) `t = 0.25` from a Neel start against `exp(L t)`. Catches operator ORDER and site placement,
      which a symmetric product state at `t = 0` cannot see.
"""
function certify_obs(; L::Int = 4, t::Float64 = 0.25, dt::Float64 = 0.0125)
    set_symmetry!(:none)
    say("\n=== CERTIFY OBSERVABLES: trace_op / pauli_1site / pauli_2site vs DENSE ===")
    say("  `certify` pinned the GENERATOR with <sigma^z> only, and sigma^z is diagonal --")
    say("  it cannot see the leg-1 transpose. sigma^y can. That is what this gate is for.\n")
    Imps = identity_superket(L)
    ok = true

    say("  (a) t = 0 on rho0_plusx: <sx>=1, <sy>=<sz>=0, connected correlators = 0 EXACTLY")
    r0 = rho0_plusx(L)
    tr0 = real(trace_rho(Imps, r0, L))
    m0 = [pauli_1site(Imps, r0, L, j; tr = tr0) for j in 1:L]
    da = maximum(max(abs(m[1] - 1), abs(m[2]), abs(m[3])) for m in m0)
    c0 = pauli_2site(Imps, r0, L, 1, 2; tr = tr0)
    conn = (c0[1] - m0[1][1] * m0[2][1], c0[2] - m0[1][2] * m0[2][2], c0[3] - m0[1][3] * m0[2][3])
    dc = maximum(abs.(conn))
    say(@sprintf("      Tr rho = %.12f    max|<sa> - target| = %.3e    max|C_1 connected| = %.3e",
                 tr0, da, dc))
    say(@sprintf("      site 1: <sx> = %+.10f  <sy> = %+.10f  <sz> = %+.10f",
                 m0[1][1], m0[1][2], m0[1][3]))
    (da < 1e-10 && dc < 1e-10) || (ok = false; say("      *** (a) FAILED ***"))

    say(@sprintf("\n  (b) t = %.3f from a Neel start, vs exp(L t)   [dt = %g, full rank]", t, dt))
    rhod = dense_rho(L, t)
    W, shift = xyz_decay_lindblad(L)
    rho = rho0_product([isodd(j) ? :up : :down for j in 1:L])
    for _ in 1:round(Int, t / dt)
        tdvp2_step!(rho, W, ComplexF64(dt); maxdim = 4^L, trunc_thresh = 1e-14,
                    maxiter = 30, hermitian = false)
        apply_shift!(rho, exp(dt * shift))
    end
    trm = real(trace_rho(Imps, rho, L))
    say(@sprintf("      %-6s %-14s %-14s %-14s %-14s %-14s %-14s", "site",
                 "<sx> mps", "<sx> dense", "<sy> mps", "<sy> dense", "<sz> mps", "<sz> dense"))
    d1 = 0.0
    for j in 1:L
        a = pauli_1site(Imps, rho, L, j; tr = trm)
        b = dense_pauli1(rhod, L, j)
        d1 = max(d1, maximum(abs.(a .- b)))
        say(@sprintf("      %-6d %+-14.9f %+-14.9f %+-14.9f %+-14.9f %+-14.9f %+-14.9f",
                     j, a[1], b[1], a[2], b[2], a[3], b[3]))
    end
    say(@sprintf("\n      %-8s %-14s %-14s %-14s %-14s %-14s %-14s", "pair",
                 "<xx> mps", "<xx> dense", "<yy> mps", "<yy> dense", "<zz> mps", "<zz> dense"))
    d2 = 0.0
    for (j, k) in ((1, 2), (1, 3), (2, 4))
        k > L && continue
        a = pauli_2site(Imps, rho, L, j, k; tr = trm)
        b = dense_pauli2(rhod, L, j, k)
        d2 = max(d2, maximum(abs.(a .- b)))
        say(@sprintf("      (%d,%d)    %+-14.9f %+-14.9f %+-14.9f %+-14.9f %+-14.9f %+-14.9f",
                     j, k, a[1], b[1], a[2], b[2], a[3], b[3]))
    end
    say(@sprintf("\n      max |1-site dev| = %.3e     max |2-site dev| = %.3e", d1, d2))
    # ⚠ NOT roundoff-tight: this is a dt = 0.0125 Trotter/Krylov comparison against an EXACT
    # matrix exponential, so O(dt^2) ~ 1e-4 is the honest bar. A transpose error would show up
    # as O(1) on sigma^y, not as a small excess -- the two failure sizes are not close.
    (d1 < 1e-4 && d2 < 1e-4) || (ok = false; say("      *** (b) FAILED ***"))

    # ⛔ (c) EXISTS BECAUSE (a) AND (b) BOTH LEAVE THE ONE-SITE `sigma^y` SIGN UNTESTED, and that
    # is the single most likely place for this stack to be wrong. In (a) the state is `|+x>` and
    # in (b) it is Neel; BOTH have `<sigma^y> = 0` identically, so each compares 0 against 0. The
    # `<yy>` correlator in (b) does exercise `sigma^y`, but a sign enters it SQUARED
    # (`sigma^y sigma^y = -(s^+ - s^-)(s^+ - s^-)`), so it is blind to a global flip.
    # ⇒ evolve `|+x>`, which the field rotates INTO y, and compare a NONZERO `<sigma^y>`.
    say(@sprintf("\n  (c) t = %.3f from |+x>^L -- the ONLY gate where <sigma^y> is nonzero", t))
    rhod = dense_rho(L, t; start = :plusx)
    rho = rho0_plusx(L)
    for _ in 1:round(Int, t / dt)
        tdvp2_step!(rho, W, ComplexF64(dt); maxdim = 4^L, trunc_thresh = 1e-14,
                    maxiter = 30, hermitian = false)
        apply_shift!(rho, exp(dt * shift))
    end
    trm = real(trace_rho(Imps, rho, L))
    say(@sprintf("      %-6s %-14s %-14s %-14s %-14s", "site",
                 "<sy> mps", "<sy> dense", "<sx> mps", "<sx> dense"))
    d3 = 0.0; ymax = 0.0
    for j in 1:L
        a = pauli_1site(Imps, rho, L, j; tr = trm)
        b = dense_pauli1(rhod, L, j)
        d3 = max(d3, maximum(abs.(a .- b))); ymax = max(ymax, abs(b[2]))
        say(@sprintf("      %-6d %+-14.9f %+-14.9f %+-14.9f %+-14.9f", j, a[2], b[2], a[1], b[1]))
    end
    say(@sprintf("\n      max |dev| = %.3e   with max|<sigma^y>| = %.4f", d3, ymax))
    # ⛔ THE GATE IS TWO-PART. A small deviation proves nothing if `<sigma^y>` never left zero --
    # that is precisely the vacuous pass (a) and (b) give. Require the signal to be PRESENT first.
    if ymax < 1e-3
        ok = false
        say("      *** (c) VACUOUS -- <sigma^y> stayed ~0, so the sign is STILL untested ***")
    elseif d3 >= 1e-4
        ok = false
        say("      *** (c) FAILED ***")
    end

    say(ok ? "\n  OBSERVABLES CERTIFIED -- sigma^y (sign included) and the two-site order." :
             "\n  *** OBSERVABLE PATH FAILED -- do NOT report correlators until this passes. ***")
    return ok
end

# ── the protocol run: BUG vs TDVP on a UNIFORM grid ─────────────────────────────────────────────

"""
⛔ THERE IS NO SYMMETRY ARM FOR THIS MODEL, AND IT IS NOT A CHOICE ABOUT THE PHYSICS.

The obvious objection to the paper's chain is that `Jx != Jy` breaks U(1) -- true, and an XXZ
variant (`Jx == Jy`) would restore it, since the DISSIPATOR preserves the difference charge
`n_ket - n_bra`. But that is not the binding constraint. `fused_space()` builds the d = 4 site
through `_getLocalSpace_no_symmetry`, the same Telum route `z2_ising.jl` had to take because
`getLocalSpace` derives its space from a KNOWN symmetry type and cannot be handed an arbitrary
one. So EVERY fused superket in this repo is `:none`, XXZ or not: the symmetric d = 4 space
(sectors `q = n_ket - n_bra` in `{-1, 0, 0, +1}`) does not exist yet.

⇒ the symmetry axis of this campaign is carried by the CHAIN and the CYLINDER (models 1 and 2),
where `dimer_state`/`cbe_bug` now run under `:U1`. Model 3 compares BUG against TDVP at `:none`
on a uniform grid, which is what it was chosen to show in the first place.

⚠ AND THERE IS NO EXACT REFERENCE AT L = 16 -- `4^16` is out of reach, so the "err" column that
models 1 and 2 carry has no counterpart here. The accuracy claim is a CONVERGENCE claim: run the
same arm at `dt` and `dt/2` and compare. Do not silently substitute our own finest grid for an
anchor; the PUBLISHED anchor is the paper's own figure, and agreement with it is qualitative.
"""
function protocol(; L::Int = 16, tmax::Float64 = 4.0, dt::Float64 = 0.01, cap::Int = 128,
                  nsave::Int = 41,
                  # ⛔ `tdvp_cbe1s` ADDED 2026-08-28 AND IT IS NOT OPTIONAL FOR THE CLAIM. It is the
                  # PUBLISHED baseline (CBE-TDVP, arXiv:2208.10972 -- the same paper this repo's CBE
                  # machinery comes from), so leaving it out compares CBE-BUG only against the older
                  # 2-site sweep, which is the easier target. The three-way comparison is the point.
                  arms = ("tdvp2", "tdvp_cbe1s", "bug_rsvd", "bug_par"),
                  # ⛔ `maxiter` IS THE DOMINANT COST, NOT A SAFETY MARGIN. `arnoldi_expv` exits
                  # on BREAKDOWN only, so EVERY solve burns the full budget -- there is no early
                  # convergence exit. MEASURED on the L=8 smoke of THIS model at `maxiter = 30`:
                  # 70 s PER STEP and `krylov = 755` per step. 16 is a straight ~2x on the run.
                  # ⚠ `certify` keeps 30: there the question is exactness at L=3, where cost is free.
                  maxiter::Int = 16,
                  growth::Float64 = 4.0, out::String = "")
    set_symmetry!(:none)
    out = isempty(out) ? joinpath(@__DIR__, "results",
                                  @sprintf("dissip_heis_protocol_L%d_dt%g.csv", L, dt)) : out
    mkpath(dirname(out))
    W, shift = xyz_decay_lindblad(L)
    Imps = identity_superket(L)
    kw = (maxdim = cap, trunc_thresh = 1e-12, maxiter = maxiter, hermitian = false)
    rs = (exact = false, dover = 4, growth = growth)
    steppers = Dict{String, Function}(
        "tdvp2"      => (p, h) -> tdvp2_step!(p, W, ComplexF64(h); kw...),
        "tdvp_cbe1s" => (p, h) -> tdvp_cbe1s_step!(p, W, ComplexF64(h); kw...),
        "bug_rsvd"   => (p, h) -> cbe_bug_step!(p, W, ComplexF64(h); kw..., rs...),
        "bug_par"    => (p, h) -> cbe_bug_step!(p, W, ComplexF64(h); kw..., rs..., parallel = true))
    # bulk window: drop a quarter from each end so the reported averages are not edge-dominated
    blk = (div(L, 4) + 1):(L - div(L, 4))
    pairs(d) = [(j, j + d) for j in blk if (j + d) in blk]

    say("\n=== PROTOCOL: dissipative Heisenberg, UNIFORM grid, BUG vs TDVP ===")
    say(@sprintf("  L = %d  t in [0, %g]  dt = %g (%d steps)  cap = %d  growth = %g",
                 L, tmax, dt, round(Int, tmax / dt), cap, growth))
    say(@sprintf("  (Jx,Jy,Jz) = (%.1f,%.1f,%.1f)  h = %.1f  gamma = %.1f   start = |+x>^L",
                 PAPER.Jx, PAPER.Jy, PAPER.Jz, PAPER.h, PAPER.gamma))
    say(@sprintf("  bulk sites %d..%d;  correlator pairs d=1: %d, d=2: %d",
                 first(blk), last(blk), length(pairs(1)), length(pairs(2))))
    say("  ⛔ symmetry = :none for EVERY arm -- the fused d=4 space has no symmetric variant.\n")

    open(out, "w") do io
        println(io, "arm,L,dt,t,chi,krylov,seconds,trace_pre,sx,sy,sz," *
                    "cxx1,cyy1,czz1,cxx2,cyy2,czz2")
        for arm in arms
            haskey(steppers, arm) || throw(ArgumentError("unknown arm $arm"))
            step! = steppers[arm]
            rho = rho0_plusx(L)
            enable_krylov_log()
            tnow = 0.0; sec = 0.0; trb = 1.0
            saveat = [tmax * k / (nsave - 1) for k in 0:(nsave - 1)]
            si = 1
            say(@sprintf("  -- %s --", arm))
            while si <= nsave
                if tnow >= saveat[si] - 1e-9
                    trr = real(trace_rho(Imps, rho, L))
                    m = [pauli_1site(Imps, rho, L, j; tr = trr) for j in blk]
                    sx = sum(x[1] for x in m) / length(m)
                    sy = sum(x[2] for x in m) / length(m)
                    sz = sum(x[3] for x in m) / length(m)
                    mi = Dict(j => m[k] for (k, j) in enumerate(blk))
                    cc = Float64[]
                    for d in (1, 2)
                        ps = pairs(d)
                        acc = zeros(Float64, 3)
                        for (j, k) in ps
                            c = pauli_2site(Imps, rho, L, j, k; tr = trr)
                            for a in 1:3
                                acc[a] += c[a] - mi[j][a] * mi[k][a]
                            end
                        end
                        append!(cc, acc ./ max(1, length(ps)))
                    end
                    chi = maximum(bond_dims(rho))
                    kry = sum(get_krylov_log())
                    @printf(io, "%s,%d,%g,%.6f,%d,%d,%.3f,%.10g,%.10g,%.10g,%.10g,%s\n",
                            arm, L, dt, tnow, chi, kry, sec, trb, sx, sy, sz,
                            join([@sprintf("%.10g", x) for x in cc], ","))
                    flush(io)
                    si <= 3 || si % 10 == 0 ?
                        say(@sprintf("     t=%5.2f  chi=%-4d kry=%-8d %6.1fs  <sx>=%+.6f <sz>=%+.6f",
                                     tnow, chi, kry, sec, sx, sz)) : nothing
                    si += 1
                    continue
                end
                sec += @elapsed begin
                    step!(rho, dt)
                    apply_shift!(rho, exp(dt * shift))
                    trb = restore_trace!(Imps, rho, L; maxdim = cap)
                end
                tnow += dt
            end
            disable_krylov_log()
        end
    end
    say("\nwrote " * out)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "certify" : ARGS[1]
    mode in ("all", "certify")  && certify()
    mode in ("all", "obs")      && certify_obs()
    mode in ("all", "spectrum") && spectrum()
    mode == "protocol" && protocol(;
        L     = parse(Int,     get(ENV, "DH_L",  "16")),
        dt    = parse(Float64, get(ENV, "DH_DT", "0.01")),
        cap   = parse(Int,     get(ENV, "DH_CAP", "128")),
        tmax  = parse(Float64, get(ENV, "DH_TMAX", "4.0")),
        # ⚠ `nsave` MUST NOT EXCEED `tmax/dt + 1`, or two checkpoints fall inside one step and the
        # save loop emits duplicate rows at the same `t` -- harmless arithmetically, but it makes a
        # smoke run look like it produced more data than it did.
        nsave = parse(Int,     get(ENV, "DH_NSAVE", "41")),
        maxiter = parse(Int,   get(ENV, "DH_MAXITER", "16")))
end
