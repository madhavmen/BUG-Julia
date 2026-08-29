# THE DISSIPATIVE ISING CHAIN OF CUI, CIRAC & BANULS -- and the argument their conclusion sets up.
#
# Paper: Jian Cui, J. Ignacio Cirac, Mari Carmen Banuls, "Variational Matrix Product Operators for
# the Steady State of Dissipative Quantum Systems", PRL 114, 220601 (2015), arXiv:1501.06786.
#
#     H = V/4 sum_i sz_i sz_{i+1} + sum_i ( Omega/2 sx_i - (V-Delta)/2 sz_i ) + V/4 (sz_1 + sz_N)
#     L_i = sqrt(gamma) sigma^+_i                       gamma = 1, V = 5, Omega = 1.5, Delta varies
#
# Their Fig. 2: sqrt(<M_z^2>) against Delta (left) and the steady state's PURITY against Delta
# (right), for N = 10..50, converged by D <= 20, with M_z = sum_i (-1)^i sz_i / N.
#
# ── WHY THIS MODEL IS IN THE CAMPAIGN ──────────────────────────────────────────────────────────
#
# ⛔ THEIR METHOD AND OURS FAIL DIFFERENTLY, AND THAT IS THE POINT. They target the steady state
# DIRECTLY, minimising `|| L |Phi(rho)> ||^2` subject to `|| Phi || = 1` -- i.e. the smallest
# eigenvalue of `L^dag L`. That constraint is EUCLIDEAN (Hilbert-Schmidt): it enforces neither
# positivity nor unit trace. With a ONE-dimensional null space that costs nothing, because the
# unique null vector is proportional to the physical steady state and normalising it by the trace
# recovers a state. Their own conclusion says what happens otherwise, verbatim:
#
#     "When the steady state is degenerate, the simplest method described in this paper might have
#      problems to find a valid guess for the steady state. Specially in the situation of several
#      dark states, the null subspace of the Lindbladian contains infinitely many vectors which do
#      not correspond to positive operators and hence do not constitute valid physical states."
#
# ✅ AN EVOLUTION-BASED METHOD CANNOT FAIL THAT WAY, AND NOT BY LUCK. `exp(L t)` is CPTP for every
# `t`, so `rho(t)` is positive semidefinite and trace-one at every step, and any `t -> inf` limit of
# physical states is physical. We do not ENFORCE positivity; we inherit it from the generator.
# Degeneracy stops being a pathology and becomes a SELECTION RULE -- the initial state picks which
# steady state is reached:
#
#     rho(inf) = sum_m Tr(P_m rho_0) * I_m / d_m        (P_m the projector on dark sector m)
#
# ⚠ REAL TIME, NOT IMAGINARY. The NESS is the `lambda = 0` eigenvector of `L` and is reached by
# `exp(L t)`. Imaginary time is what models 1 and 2 (closed, Hermitian `H`) use. The argument above
# is unaffected, but the two must not be labelled interchangeably in a write-up.
#
# ⚠ AND THE COMPARISON IS NOT "OUR INTEGRATOR BEATS THEIR ALGORITHM". Theirs is faster than
# evolution WHEN the steady state is unique and low-rank -- that is the result of their paper and it
# stands. The claim here is narrower and about ROBUSTNESS: where their functional has no way to
# return a physical answer, ours has no way to return an unphysical one.
#
# Run:  julia --project=. benchmarks/imaginary_time/dissipative_ising.jl [certify|fig2]

using LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "fused_common.jl"))

"The paper's parameters (their Fig. 2). `Delta` is the axis, so it is not fixed here."
const CCB = (V = 5.0, Omega = 1.5, gamma = 1.0)

# ── the fused MPO ───────────────────────────────────────────────────────────────────────────────

"""
    ising_pump_lindblad(N; V, Omega, Delta, gamma) -> (mpo, shift)

Fused d=4 Liouvillian of the Cui-Cirac-Banuls chain. `shift` is the zero-body piece `pair_mpo`
cannot hold.

⛔ PAULI -> SPIN, ONCE, HERE. `sz = 2 S^z` and `sx = S^+ + S^-`, so

    V/4 sz sz      -> V S^z S^z                  (the 4 cancels the 1/4)
    Omega/2 sx     -> Omega/2 (S^+ + S^-)
    -(V-Delta)/2 sz -> -(V-Delta) S^z
    V/4 sz         -> V/2 S^z                    (boundary, sites 1 and N only)

⛔ THE PUMPING JUMP IS WRITTEN `Q4.JM`, WHICH IS THE OPPOSITE OF WHAT IT LOOKS LIKE. `pair_mpo`
contracts operator leg 1 and therefore applies the TRANSPOSE, and `kron(SP,SP)' = kron(SM,SM)`
exactly. `sigma^+ rho sigma^-` IS `kron(SP,SP)` = `Q4.JP` (see `fused_space`), so to APPLY it the
operator PASSED must be its transpose, `Q4.JM`. Getting this backwards does not throw -- it
silently evolves DECAY instead of PUMPING, and against a `sigma^+` drive the anticommutator then
carries the wrong sign too, which is the exact mechanism that froze the boundary-driven model
([[fused-jump-transpose-freeze]]). `certify_ising` is what pins it.

⚠ THE ANTICOMMUTATOR SIGN FLIPS RELATIVE TO A DECAY MODEL. `L^dag L = sigma^- sigma^+ = I/2 - S^z`
for a raising jump, against `I/2 + S^z` for a lowering one, so `S^z` enters `-gamma/2 * (I/2 - S^z)`
with a PLUS. `dissipative_heisenberg.jl` has the minus; they are different models, not a typo.
"""
function ising_pump_lindblad(N::Int; V::Float64 = CCB.V, Omega::Float64 = CCB.Omega,
                             Delta::Float64 = 0.0, gamma::Float64 = CCB.gamma)
    N >= 2 || throw(ArgumentError("need at least two sites"))
    PV = RSVDCBEBondUpdate.PairVertex
    verts = PV[]; Js = Any[]

    # two-body, uniform over bonds
    nn(c) = begin
        A = zeros(ComplexF64, N, N)
        for j in 1:(N - 1); A[j, j + 1] = c; end
        A
    end
    push2(opk, opb, c) = begin
        push!(verts, PV(opk[1], opk[2], 1.0)); push!(Js, nn(-im * c))
        push!(verts, PV(opb[1], opb[2], 1.0)); push!(Js, nn(+im * c))
    end

    # ⚠ ONE-BODY TERMS RIDE THE BONDS, with a PER-SITE coefficient. Site `j < N` sits on bond
    # (j, j+1) as `O (x) I`; site `N` has no bond to its right, so it rides (N-1, N) as `I (x) O`.
    # A per-site vector (rather than one scalar) is what lets the uniform field and the paper's
    # BOUNDARY term `V/4 (sz_1 + sz_N)` use the same mechanism instead of two.
    push1(op, coeffs::Vector{ComplexF64}) = begin
        length(coeffs) == N || throw(ArgumentError("need $N per-site coefficients"))
        A = zeros(ComplexF64, N, N)
        for j in 1:(N - 1); A[j, j + 1] = coeffs[j]; end
        # ⛔ NEVER PUSH AN IDENTICALLY-ZERO COUPLING MATRIX -- `pair_mpo` opens and carries a
        # channel that contributes nothing, and its bookkeeping is indexed by vertex.
        any(!iszero, A) && (push!(verts, PV(op, Q4.I, 1.0)); push!(Js, A))
        if !iszero(coeffs[N])
            B = zeros(ComplexF64, N, N); B[N - 1, N] = coeffs[N]
            push!(verts, PV(Q4.I, op, 1.0)); push!(Js, B)
        end
    end

    # ── Hamiltonian ─────────────────────────────────────────────────────────────────────────────
    push2((Q4.Kz, Q4.Kz), (Q4.Bz, Q4.Bz), V)                 # V S^z S^z

    unif(c) = ComplexF64[c for _ in 1:N]
    bnd(c)  = ComplexF64[(j == 1 || j == N) ? c : 0.0 for j in 1:N]

    # transverse field: BOTH S^+ and S^- with the SAME coefficient, which is what makes it safe
    # under the transpose convention (S^+ and S^- exchange; an equal-weight sum is invariant).
    push1(Q4.Kp, unif(-im * Omega / 2)); push1(Q4.Km, unif(-im * Omega / 2))
    push1(Q4.Bp, unif(+im * Omega / 2)); push1(Q4.Bm, unif(+im * Omega / 2))

    # longitudinal field -(V-Delta) S^z, PLUS the boundary V/2 S^z, PLUS the anticommutator gamma/2
    push1(Q4.Kz, unif(-im * -(V - Delta) + gamma / 2) .+ bnd(-im * V / 2))
    push1(Q4.Bz, unif(+im * -(V - Delta) + gamma / 2) .+ bnd(+im * V / 2))

    # ── dissipator: L = sqrt(gamma) sigma^+ ─────────────────────────────────────────────────────
    push1(Q4.JM, unif(ComplexF64(gamma)))     # applies kron(SP,SP) = sigma^+ rho sigma^-

    return pair_mpo(N, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I), -gamma * N / 2
end

# ── the dense reference (N <= 5 only) ───────────────────────────────────────────────────────────

const _SP2 = ComplexF64[0 1; 0 0]
const _SM2 = ComplexF64[0 0; 1 0]
const _SZ2 = ComplexF64[0.5 0; 0 -0.5]
_at(op, k, N) = kron(Matrix{ComplexF64}(I, 2^(k - 1), 2^(k - 1)), ComplexF64.(op),
                     Matrix{ComplexF64}(I, 2^(N - k), 2^(N - k)))

"Dense Liouvillian, built straight from the PAULI form so it shares no algebra with the MPO."
function dense_ising_pump(N::Int; V = CCB.V, Omega = CCB.Omega, Delta = 0.0, gamma = CCB.gamma)
    N <= 5 || throw(ArgumentError("dense reference is for N <= 5 only, got $N"))
    d = 2^N
    sx(k) = _at(_SP2 + _SM2, k, N)
    sz(k) = _at(2 .* _SZ2, k, N)
    H = zeros(ComplexF64, d, d)
    for k in 1:(N - 1); H .+= (V / 4) .* sz(k) * sz(k + 1); end
    for k in 1:N;       H .+= (Omega / 2) .* sx(k) .- ((V - Delta) / 2) .* sz(k); end
    H .+= (V / 4) .* (sz(1) .+ sz(N))
    Id = Matrix{ComplexF64}(I, d, d)
    Ls = -im .* (kron(Id, H) .- kron(transpose(H), Id))
    for k in 1:N
        jump = sqrt(gamma) .* _at(_SP2, k, N)          # sigma^+ : PUMPING
        LdL = jump' * jump
        Ls .+= kron(transpose(jump'), jump) .- 0.5 .* kron(Id, LdL) .- 0.5 .* kron(transpose(LdL), Id)
    end
    return Ls
end

"Dense `rho(t)` from an all-DOWN product start (the natural start against a raising jump)."
function dense_ising_rho(N::Int, t::Real; kwargs...)
    d = 2^N
    rho0 = ComplexF64[1;;]
    for _ in 1:N; rho0 = kron(rho0, ComplexF64[0 0; 0 1]); end     # |down><down|
    return reshape(exp(Matrix(dense_ising_pump(N; kwargs...)) .* float(t)) * vec(rho0), d, d)
end

# ── observables ─────────────────────────────────────────────────────────────────────────────────

say(s) = (println(s); flush(stdout))

"""
    staggered_mz2(Imps, rho, N; tr) -> Float64

`<M_z^2>` with `M_z = sum_i (-1)^i sigma^z_i / N` -- the paper's Fig. 2 left panel (they plot its
square root).

⚠ THE DIAGONAL `i == j` TERMS ARE NOT OPTIONAL AND ARE NOT CORRELATORS. `(sigma^z_i)^2 = I`, so
each contributes exactly `1`, giving a floor of `1/N` that survives to `Delta -> +-inf`. Dropping
them (by summing only `i < j`, the natural thing to write) removes a term that DOMINATES at large
`|Delta|`, where the off-diagonal correlations are what vanish.
"""
function staggered_mz2(Imps, rho, N::Int; tr::Float64)
    acc = 0.0
    for i in 1:N
        acc += 1.0                                     # i == j : (sigma^z)^2 = I
        for j in (i + 1):N
            zz = pauli_2site(Imps, rho, N, i, j; tr = tr)[3]
            acc += 2 * (-1)^(i + j) * zz               # 2x for the (j,i) term
        end
    end
    return acc / N^2
end

"""
    purity(rho, N) -> Float64

`Tr(rho^2)`, which in the fused superket language is simply `<<rho|rho>>`.

⚠ THE NORMALISATION IS THE WHOLE SUBTLETY. `identity_superket` represents `vec(I)/2^(N/2)`, and
every other superket here is scaled to match, so a raw `overlap(rho, rho)` is `Tr(rho^2)` in that
same convention -- no `2^N` factor. The check that settles it: the maximally mixed state has
`Tr(rho^2) = 2^-N`, and a pure product state has `1`. `certify_ising` asserts both.
"""
purity(rho, N::Int) = real(overlap(rho, rho))

# ── gate: the MPO against the dense Liouvillian ─────────────────────────────────────────────────

"""
Certify `ising_pump_lindblad` by evolving it and the dense Liouvillian from the same product state.

⛔ THE START IS ALL-DOWN, NOT `|I>>`. `|I>>` has hidden a wrong generator before in this stack: a
wrong operator happened to annihilate it and `1 - |<I|q>|^2` still read `4.4e-16`. Against a
`sigma^+` (raising) jump, all-down is the state the dissipator acts on hardest, so a sign error in
either the jump or the anticommutator moves the answer immediately.
"""
function certify_ising(; N::Int = 4, t::Float64 = 0.4, Delta::Float64 = 0.0,
                       dts = (0.05, 0.025, 0.0125))
    set_symmetry!(:none)
    say("\n=== CERTIFY: Cui-Cirac-Banuls dissipative Ising, fused MPO vs DENSE ===")
    say("  H = V/4 sum sz sz + sum (Omega/2 sx - (V-Delta)/2 sz) + V/4 (sz_1 + sz_N)")
    say(@sprintf("  V = %.1f  Omega = %.1f  gamma = %.1f  Delta = %.1f   L_i = sqrt(gamma) sigma^+",
                 CCB.V, CCB.Omega, CCB.gamma, Delta))
    say(@sprintf("  N = %d, all-down start, compare <sigma^z_j> at t = %g\n", N, t))

    rhod = dense_ising_rho(N, t; Delta = Delta)
    exact = [real(LinearAlgebra.tr(_at(2 .* _SZ2, j, N) * rhod)) /
             real(LinearAlgebra.tr(rhod)) for j in 1:N]
    say("  exact <sigma^z_j> = " * join([@sprintf("%+.8f", x) for x in exact], " "))

    W, shift = ising_pump_lindblad(N; Delta = Delta)
    Imps = identity_superket(N)
    ok = true

    # ⛔ THE PURITY NORMALISATION IS PINNED FIRST, and separately, because it is a CONVENTION and
    # not a dynamical result -- if it is wrong, every number in Fig. 2's right panel is wrong by a
    # constant and the curve still looks entirely reasonable.
    mixed = copy(Imps); apply_shift!(mixed, 1.0 / 2.0^(N / 2))
    pure  = rho0_product([:down for _ in 1:N])
    say(@sprintf("\n  purity gate: maximally mixed = %.10g (want %.10g);  pure product = %.10g (want 1)",
                 purity(mixed, N), 2.0^(-N), purity(pure, N)))
    (abs(purity(mixed, N) - 2.0^(-N)) < 1e-10 && abs(purity(pure, N) - 1) < 1e-10) ||
        (ok = false; say("      *** PURITY NORMALISATION FAILED ***"))

    say(@sprintf("\n  %-9s %-9s %-13s %-8s", "dt", "t_act", "max|d sigma^z|", "order"))
    prev = NaN
    for dt in dts
        rho = rho0_product([:down for _ in 1:N])
        tnow = 0.0
        for _ in 1:round(Int, t / dt)
            tdvp2_step!(rho, W, ComplexF64(dt); maxdim = 4^N, trunc_thresh = 1e-14,
                        maxiter = 30, hermitian = false)
            apply_shift!(rho, exp(dt * shift))
            tnow += dt
        end
        trr = real(trace_rho(Imps, rho, N))
        prof = [2 * real(trace_rho(Imps, rho, N; at = j)) / trr for j in 1:N]
        dev = maximum(abs.(prof .- exact))
        p = isnan(prev) ? NaN : log2(prev / dev)
        say(@sprintf("  %-9.5f %-9.4f %-13.3e %-8s", dt, tnow, dev,
                     isnan(p) ? "--" : @sprintf("%.2f", p)))
        prev = dev
    end
    # ⛔ AT FULL RANK THE RIGHT ANSWER IS ROUNDOFF, NOT A CONVERGENCE RATE. With `maxdim = 4^N` the
    # MPS manifold IS the whole space, so TDVP's projector is the identity; gating on `order > 1.5`
    # would FAIL a perfect result. Two ways to pass, one way to fail -- a floor ABOVE roundoff that
    # does not move with `dt`.
    ok &= prev < 1e-10 || (!isnan(prev) && prev < 1e-4)
    say(ok ? "\n  CERTIFIED." : "\n  *** FAILED -- do not run fig2 until this passes. ***")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "certify" : ARGS[1]
    mode in ("all", "certify") && certify_ising()
end
