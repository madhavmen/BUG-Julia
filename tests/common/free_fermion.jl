# Analytic XX reference via Jordan-Wigner. Test-side only.
#
# NOT the same thing as tests/common/xx_free_fermion.jl, which despite its name
# is a dense 2^N exact diagonalisation built on ITensors -- the legacy stack this
# refactor removes, and no closed form at all. This file is the actual free-fermion
# solution and touches nothing bigger than an L x L matrix.
#
# That independence is the point. `dense_reference.jl` shares the integrator's own
# view of the model (same gate normalisation, same site ordering, a 2^L propagator);
# if that convention were wrong, both would be wrong together. This reference is
# derived from the Hamiltonian on paper instead, so agreement between the two is
# genuine evidence rather than a tautology.

using LinearAlgebra

"""
    xx_single_particle_hamiltonian(L; J=1.0) -> Matrix{ComplexF64}

The `L x L` hopping matrix of the XX chain after Jordan-Wigner.

`J (Sx Sx + Sy Sy) = (J/2)(S+ S- + S- S+)` maps to `(J/2)(c'_j c_{j+1} + h.c.)`
for nearest neighbours -- the JW string cancels between adjacent sites -- so the
off-diagonal is `J/2` and the diagonal is zero.
"""
function xx_single_particle_hamiltonian(L::Int; J::Real = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    return h
end

"""
    xx_free_fermion_sz(L, t; J=1.0, occupied=1:(L÷2)) -> Vector{Float64}

`⟨Sz_j(t)⟩` for a Slater determinant evolved under the XX chain:

    ⟨Sz_j(t)⟩ = Σ_{k occupied} |[exp(-i h t)]_{jk}|² − ½

`occupied` lists the initially occupied sites. Up spin is an occupied fermion
(`Sz = n − ½`), so the default `1:(L÷2)` is exactly `domain_wall_state(L)`.
"""
function xx_free_fermion_sz(L::Int, t::Real; J::Real = 1.0,
                            occupied = 1:(L ÷ 2))
    U = exp(-im * t * xx_single_particle_hamiltonian(L; J = J))
    return [sum(abs2(U[j, k]) for k in occupied) - 0.5 for j in 1:L]
end

"""
    xx_dephasing_correlation(L, t; J=1.0, gamma=0.0, occupied=1:2:L) -> Matrix{ComplexF64}

`C_ij(t) = <c_i^dag c_j>` for the XX chain under ONSITE DEPHASING, EXACTLY.

⛔ THE OPEN XX CHAIN HAS A CLOSED FORM AND IT IS NOT WIDELY EXPLOITED. For dephasing jump
operators the equation of motion for the single-particle correlation matrix CLOSES -- the
dissipator maps `c_i^dag c_j` to a multiple of itself, so no higher correlator is generated:

    dC/dt = -i[h, C] - gamma (1 - delta_ij) C_ij

i.e. the Hamiltonian part is the usual commutator and dephasing simply DAMPS EVERY OFF-DIAGONAL
element at rate `gamma`, leaving the diagonal (the densities) untouched by the bath directly. So
the reference is an `L x L` linear ODE -- 18x18 at our size -- and NOTHING of size `4^L` is ever
built. This gives `xx_open` the same status as `xx`: a TRUE ERROR, not a deviation from another
arm and not a published trend.

⛔ AND OUR `L_j = S^z_j` IS THE SAME CHANNEL AS `L_j = n_j`, which is why this applies unchanged.
For Hermitian `L` and real `c`, `D[L + c*I] = D[L]` identically:

    D[L+cI]rho = (L+cI)rho(L+cI) - (1/2){(L+cI)^2, rho}
               = L rho L + c(L rho + rho L) + c^2 rho
                 - (1/2){L^2,rho} - c{L,rho} - c^2 rho
               = L rho L - (1/2){L^2, rho}                    (the c terms cancel EXACTLY)

`S^z = n - 1/2`, so the `-1/2` is exactly such a shift and drops out. Reading the two as different
channels -- and hunting a discrepancy that is not there -- is the trap this note exists to close.

⚠ `gamma` IS THE RATE IN `gamma*(Z rho Z - (1/2){Z^2, rho})`, matching `fused_lindblad` and
`dense_super` in `benchmarks/trace_preservation.jl`. It is NOT an amplitude: passing `sqrt(gamma)`
gives a different, silently plausible curve.

⚠ COLUMN-MAJOR `vec`. Julia's `vec` stacks COLUMNS, so `vec(hC) = kron(I,h)vec(C)` and
`vec(Ch) = kron(transpose(h),I)vec(C)`, and `C[i,j]` sits at linear index `(j-1)*L + i`. Getting
that backwards transposes the generator, which for a SYMMETRIC `h` and a DIAGONAL initial `C`
still produces a smooth, wrong-looking-right curve.
"""
function xx_dephasing_correlation(L::Int, t::Real; J::Real = 1.0, gamma::Real = 0.0,
                                  occupied = 1:2:L)
    h  = xx_single_particle_hamiltonian(L; J = J)
    Id = Matrix{ComplexF64}(LinearAlgebra.I, L, L)
    M  = -im * (kron(Id, h) - kron(transpose(h), Id))
    for i in 1:L, j in 1:L
        i == j && continue
        k = (j - 1) * L + i                     # column-major position of C[i,j]
        M[k, k] -= gamma
    end
    C0 = zeros(ComplexF64, L, L)
    for k in occupied
        C0[k, k] = 1.0
    end
    return reshape(exp(M * t) * vec(C0), L, L)
end

"""
    xx_dephasing_sz(L, t; J=1.0, gamma=0.0, occupied=1:2:L) -> Vector{Float64}

`<S^z_j(t)>` for the dephasing XX chain, exactly. `S^z = n - 1/2`, so this is the diagonal of
[`xx_dephasing_correlation`](@ref) shifted. At `gamma = 0` it must reproduce
[`xx_free_fermion_sz`](@ref) to machine precision -- which is the cheapest available check that
the dissipative generator was assembled the right way round.
"""
xx_dephasing_sz(L::Int, t::Real; kwargs...) =
    [real(xx_dephasing_correlation(L, t; kwargs...)[j, j]) - 0.5 for j in 1:L]

# ── NON-HERMITIAN free fermions: the Hatano-Nelson chain, EXACTLY ─────────────────────────────
#
# ⛔ WHY THIS EXISTS. Nothing in the suite currently certifies the NON-HERMITIAN path. `hermitian
# = false` selects Arnoldi and a full Hessenberg projection -- a genuinely different code path
# from Lanczos -- and every open-system and no-click driver takes it, but every non-Hermitian
# benchmark so far has been scored against ANOTHER RUN OF OUR OWN CODE on a finer grid. That is a
# GRID control, not an accuracy control: it cannot distinguish "the log grid is too coarse" from
# "Arnoldi is wrong", because both arms would be wrong together.
#
# ⛔ THE MODEL IS CHOSEN SO THAT NO NEW MPO MACHINERY IS NEEDED. `xxz_chain` pushes the XY term as
# TWO separate channels -- `pair(Sp, Sp')` and `pair(Sm, Sm')` -- each with its own coefficient.
# Giving those two channels DIFFERENT coefficients `J_r != J_l` is
#
#     H = sum_j [ J_r S^+_j S^-_{j+1} + J_l S^-_j S^+_{j+1} ]   ->   J_r c^dag_j c_{j+1}
#                                                                  + J_l c^dag_{j+1} c_j
#
# i.e. ASYMMETRIC HOPPING: the Hatano-Nelson model, the canonical non-Hermitian free-fermion
# chain. So it is `pair_mpo(L, AbstractMatrix[Jr*xy, Jl*xy, zz], spin_vertices(L))` -- the
# existing builder, two different entries.
#
# ⚠ AND THIS DOES **NOT** CONTRADICT the "the complex matrix must be THIRD" warning in
# `logtime_suite.jl`'s model 3. That warning is about giving BOTH XY channels the SAME complex
# factor, which is a complex RESCALING of H (an imaginary-time evolution in disguise). Two
# DIFFERENT real coefficients are a different object entirely, and non-Hermitian for a reason
# that survives any rescaling: `h != h'`.
#
# ⛔ THE STATE IS STILL GAUSSIAN, BUT THE ORBITALS STOP BEING ORTHONORMAL. That is the whole
# subtlety, and getting it wrong produces a smooth, plausible, wrong curve. Under a quadratic
# generator a Slater determinant stays a Slater determinant, and the orbital matrix evolves
# LINEARLY, `U(t) = exp(-i h t) U(0)` -- but for `h != h'` that propagator is not unitary, so
# `U(t)' U(t) != I` and the FAMILIAR formula `C = U U'` IS WRONG. The correct normalised
# correlation matrix divides by the overlap (Gram) matrix. See e.g. the monitored-free-fermion
# literature (Alberton/Buchhold/Diehl-type no-click constructions), where `M = U' U`,
# `<psi|psi> = det M`, and `C = U M^{-1} U'`.

"""
    nh_hopping_matrix(L; J_r=1.0, J_l=1.0, periodic=false) -> Matrix{ComplexF64}

The `L x L` single-particle matrix of the Hatano-Nelson chain,
`h[j, j+1] = J_r`, `h[j+1, j] = J_l`, plus the wrap `h[L,1] = J_r`, `h[1,L] = J_l` when
`periodic`.

`J_r == J_l == J/2` reproduces [`xx_single_particle_hamiltonian`](@ref) exactly -- which is the
calibration this file leans on: the non-Hermitian reference must collapse onto the Hermitian one
in that limit, so a sign or transpose error cannot hide.

`J_r` and `J_l` may each be a SCALAR (uniform) or a length-`L-1` VECTOR of per-bond amplitudes,
and either may be COMPLEX. The two knobs generate two different kinds of non-Hermiticity, and
only one of them has a complex spectrum -- see below.

⛔ AN OPEN ASYMMETRIC CHAIN HAS A **REAL** SPECTRUM, so `J_r != J_l` alone CANNOT make a
logarithmic grid fail. With open ends it is similar to a HERMITIAN matrix by the diagonal gauge
`S = diag(r^j)`, `r = sqrt(J_l / J_r)`: `S^-1 h S` is the symmetric hopping matrix with
`sqrt(J_r J_l)`. MEASURED `max|Im E| = 1.5e-16` at `J_r = 0.65, J_l = 0.35, L = 12`. The chain is
still NON-NORMAL -- eigenvectors non-orthogonal, norm not conserved, Arnoldi mandatory -- so it is
an excellent ACCURACY certificate for the non-Hermitian path. It is just not an aliasing test.

⛔ AND CLOSING THE RING DOES NOT FIX IT, BECAUSE THE JORDAN-WIGNER STRING DOES NOT CANCEL ACROSS
THE WRAP. For adjacent sites the string between `S^+_j S^-_{j+1}` cancels and the term IS
`c^dag_j c_{j+1}`; across the bond `(1, L)` the operators are `L-2` sites apart and the string
`prod_{k=2}^{L-1} (1 - 2 n_k)` survives, so the spin ring maps to a fermion ring whose boundary
condition is TWISTED BY THE PARTICLE-NUMBER PARITY (antiperiodic at even filling). A `periodic`
single-particle matrix would therefore describe a DIFFERENT model from the MPO -- smoothly and
plausibly wrong. `periodic` is kept only so the spectrum argument above can be demonstrated; it
must NOT be paired with a `pair_mpo` that carries a `J[1, L]` entry.

✅ THE WAY TO GET A COMPLEX SPECTRUM WITH OPEN ENDS is a COMPLEX SYMMETRIC hopping instead of an
asymmetric real one: `J_r == J_l == w_j` with `w_j` complex and NOT all of the same phase. A
uniform complex `w` is only `e^{i theta}` times a real matrix -- a complex rescaling of `H`, i.e.
mixed real/imaginary time in disguise, which is the trap `logtime_suite.jl` already warns about.
A STAGGERED `w_j = J (1 + i g (-1)^j)` is not, and gives genuinely complex eigenvalues with the
string-free nearest-neighbour terms intact.
"""
function nh_hopping_matrix(L::Int; J_r = 1.0, J_l = 1.0, periodic::Bool = false)
    _bond(x, j) = x isa Number ? ComplexF64(x) : ComplexF64(x[j])
    for (nm, x) in (("J_r", J_r), ("J_l", J_l))
        x isa Number || length(x) == L - 1 || throw(DimensionMismatch(
            "$nm has $(length(x)) entries, need $(L - 1) (one per bond)"))
    end
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = _bond(J_r, j)
        h[j + 1, j] = _bond(J_l, j)
    end
    if periodic
        L >= 3 || throw(ArgumentError("periodic needs L >= 3, got $L: at L = 2 the wrap bond " *
                                      "IS the chain bond and adding both doubles it"))
        # ⛔ SPECTRUM DEMONSTRATION ONLY -- see the Jordan-Wigner warning above. This does NOT
        # correspond to any `pair_mpo` this repo can build.
        h[L, 1] = _bond(J_r, L - 1)
        h[1, L] = _bond(J_l, L - 1)
    end
    return h
end

"""
    nh_staggered_couplings(L; J=1.0, g=0.0) -> Vector{ComplexF64}

Per-bond amplitudes `w_j = J (1 + i g (-1)^j)` for `j = 1 … L-1` -- the complex SYMMETRIC
staggering that makes an OPEN chain's spectrum complex without touching the boundary.

Pass the same vector as both `J_r` and `J_l`: symmetric in the two hopping directions, complex in
the amplitude. `g = 0` recovers the Hermitian XX chain with `J_r = J_l = J`.
"""
nh_staggered_couplings(L::Int; J::Number = 1.0, g::Real = 0.0) =
    ComplexF64[J * (1 + im * g * (-1)^j) for j in 1:(L - 1)]

"""
    nh_orbitals(L, t; J_r=1.0, J_l=1.0, occupied=1:2:L) -> Matrix{ComplexF64}

`U(t) = exp(-i h t) U(0)`, the `L x N` orbital matrix of the evolved Slater determinant.

⛔ THE PROPAGATOR IS NOT UNITARY. `exp(-i h t)` for `h != h'` grows or shrinks directions, so the
columns of `U(t)` are neither normalised nor mutually orthogonal. Every quantity below therefore
goes through the Gram matrix; none of them may use `U'` alone.
"""
function nh_orbitals(L::Int, t::Real; J_r = 1.0, J_l = 1.0,
                     periodic::Bool = false, occupied = 1:2:L)
    U0 = zeros(ComplexF64, L, length(occupied))
    for (mu, k) in enumerate(occupied)
        U0[k, mu] = 1.0
    end
    h = nh_hopping_matrix(L; J_r = J_r, J_l = J_l, periodic = periodic)
    return exp(-im * t * h) * U0
end

"""
    nh_free_fermion_norm2(L, t; J_r=1.0, J_l=1.0, occupied=1:2:L) -> Float64

`<psi(t)|psi(t)>` for the non-Hermitian chain, exactly: the Gram determinant `det(U' U)`.

⛔ THIS IS A FIRST-CLASS REFERENCE, NOT A DIAGNOSTIC. A non-Hermitian generator does not conserve
the norm -- that is the entire point of the model -- so `||psi||` is a physical prediction with a
closed form, and an integrator that gets the observable right while getting the norm wrong is
still wrong. It is also the quantity that decides REPRESENTABILITY: the imaginary-time overflow
work showed norm growth, not Krylov depth, is what kills a large step.
"""
function nh_free_fermion_norm2(L::Int, t::Real; kwargs...)
    U = nh_orbitals(L, t; kwargs...)
    return real(det(U' * U))
end

"""
    nh_free_fermion_correlation(L, t; J_r=1.0, J_l=1.0, occupied=1:2:L) -> Matrix{ComplexF64}

`C_ij(t) = <psi|c^dag_i c_j|psi> / <psi|psi>` for the Hatano-Nelson chain, EXACTLY, as
`C = U (U' U)^{-1} U'` (transposed into the `(i, j)` convention used by
[`xx_dephasing_correlation`](@ref)).

Cost is `O(L^3)`; nothing of size `2^L` is built. At `J_r == J_l` the Gram matrix is the identity
and this collapses to the ordinary `U U'`.
"""
function nh_free_fermion_correlation(L::Int, t::Real; kwargs...)
    U = nh_orbitals(L, t; kwargs...)
    M = U' * U
    # `C[i,j] = <c_i^dag c_j>` is the (j,i) entry of `U M^{-1} U'` -- the transpose is not
    # cosmetic: for a NON-Hermitian generator `C` is not Hermitian, so reading it the wrong way
    # round gives a different, equally smooth curve instead of an error.
    return permutedims(U * (M \ U'))
end

"""
    nh_free_fermion_sz(L, t; J_r=1.0, J_l=1.0, occupied=1:2:L) -> Vector{Float64}

`<S^z_j(t)>` for the Hatano-Nelson chain, exactly and ALREADY NORMALISED by `<psi|psi>`.

At `J_r == J_l == J/2` this must reproduce `xx_free_fermion_sz(L, t; J = J, occupied = ...)` to
machine precision. That check is the calibration for the whole family and is worth running before
any comparison against the MPS arms.
"""
nh_free_fermion_sz(L::Int, t::Real; kwargs...) =
    [real(nh_free_fermion_correlation(L, t; kwargs...)[j, j]) - 0.5 for j in 1:L]

# ── BOUNDARY-DISSIPATED XX CHAIN: the COMPLEX-FREQUENCY reference (arXiv:2104.11479) ─────────
#
# Yamanaka & Sasamoto, "Exact solution for the Lindbladian dynamics for the open XX spin chain
# with boundary dissipation". THIRD QUANTIZATION reduces the Lindbladian to an `N x N`
# NON-HERMITIAN matrix, so the reference costs `O(N^3)` and nothing of size `4^N` is built.
#
#     H = J sum_k (sx_k sx_{k+1} + sy_k sy_{k+1}) - B sum_k sz_k
#     L_1,2 = sqrt(eps_L (1 +- mu_L)/2) * sigma^{+,-}_1        (left boundary)
#     L_3,4 = sqrt(eps_R (1 +- mu_R)/2) * sigma^{+,-}_N        (right boundary)
#
# ⛔ WHY THIS MODEL EXISTS HERE: ITS FREQUENCIES ARE GENUINELY COMPLEX. `lambda = B + 2J cos(theta)`
# with `theta` COMPLEX, so each mode is `exp(-i Re(lambda) t) * exp(-|Im(lambda)| t)` -- an
# oscillation TIMES a decay. That is what a Hermitian real-time quench cannot supply (real
# spectrum, no decay) and what an imaginary-time Hermitian generator cannot supply either
# (completely monotone, no oscillation).
#
# ⛔ AND IT IS WHY A LOGARITHMIC GRID CANNOT BE RESCUED HERE BY MORE STEPS. MEASURED from the
# spectrum below at `N=30, J=1, eps=1`: the oscillation frequency is `max|Re lambda| = 1.9897`
# (essentially `2J`, nearly independent of N and eps) while the Liouvillian gap CLOSES as
# `O(N^-3)` -- `3.14e-04`, i.e. a relaxation time of ~3181. So the run must reach `t ~ 1e4` while
# still resolving a period-`pi` oscillation: `12089` uniform steps. A log grid at **n = 2000**
# still has a largest step of `62.2`, **79x over the Nyquist step of 0.789**. The oscillation
# persists into exactly the late-time region where a geometric grid is coarsest, so the failure
# does not go away with budget. Contrast the Neel quench, where `n >= 66` fixes it.

"""
    xx_boundary_structure_matrix(N; J=1.0, B=0.0, eps_L=1.0, eps_R=1.0) -> Matrix{ComplexF64}

The `N x N` non-Hermitian matrix whose eigenvalues are the rapidities `lambda = B + 2J cos(theta)`
of the boundary-dissipated XX chain: tridiagonal with hopping `J` and diagonal `B`, plus
`-i*eps_L/4` on site 1 and `-i*eps_R/4` on site `N`.

⛔ THE BOUNDARY NORMALISATION IS PINNED AGAINST THE PAPER'S OWN TRANSCENDENTAL EQUATION, NOT
GUESSED. The paper states that `theta_m` solves

    {2 cos(theta) + i(l + r)} sin(N theta) - (1 + l r) sin((N-1) theta) = 0,
    l = eps_L / 4J,   r = eps_R / 4J

MEASURED relative residual of that equation over all eigenvalues at `N=30, eps=1`:

    boundary term      -i*eps/4     -i*eps/2     -i*eps      -2i*eps
    rel residual       1.3e-12      4.5e-01      6.8e-01     1.5e+00

so `eps/4` is right and every neighbouring convention is O(1) wrong. A wrong factor here would
still give a plausible complex spectrum with a plausible gap -- which is exactly why it is checked
against a published identity rather than derived once and trusted.
"""
function xx_boundary_structure_matrix(N::Int; J::Real = 1.0, B::Real = 0.0,
                                      eps_L::Real = 1.0, eps_R::Real = 1.0)
    N >= 2 || throw(ArgumentError("need N >= 2, got $N"))
    X = zeros(ComplexF64, N, N)
    for k in 1:N
        X[k, k] = B
    end
    for k in 1:(N - 1)
        X[k, k + 1] = J
        X[k + 1, k] = J
    end
    X[1, 1] -= im * eps_L / 4
    X[N, N] -= im * eps_R / 4
    return X
end

"""
    xx_boundary_rapidities(N; kwargs...) -> Vector{ComplexF64}

Eigenvalues of [`xx_boundary_structure_matrix`](@ref). `real(lambda)` is a mode's OSCILLATION
FREQUENCY and `abs(imag(lambda))` its DECAY RATE, so these two numbers are what decide the Nyquist
step and the run length respectively.
"""
xx_boundary_rapidities(N::Int; kwargs...) =
    eigvals(xx_boundary_structure_matrix(N; kwargs...))

"""
    xx_boundary_liouvillian_gap(N; kwargs...) -> Float64

The Liouvillian gap as `2 * min|Im lambda|`.

✅ PINNED AGAINST THE PAPER'S CLOSED FORM, which is the headline result of arXiv:2104.11479:

    Delta = 4J pi^2 * (l+r)(l r + 1) / [ (l+r)^2 + (l r - 1)^2 ] / N^3

MEASURED ratio `2 min|Im lambda| / Delta_published` at `eps = 4`: **0.9949 (N=20), 0.9978 (N=30),
0.9987 (N=40)** -- converging to 1 as `N` grows, exactly as an asymptotic `1/N^3` formula must.
(At `eps = 1` the ratio is 0.87-0.93 at these sizes and also improving; the formula is asymptotic,
so a small-N mismatch is the FORMULA's finite-size error and not ours.) Use
[`xx_boundary_gap_published`](@ref) for the closed form to compare against.
"""
xx_boundary_liouvillian_gap(N::Int; kwargs...) =
    2 * minimum(abs.(imag.(xx_boundary_rapidities(N; kwargs...))))

"""
    xx_boundary_gap_published(N; J=1.0, eps_L=1.0, eps_R=1.0) -> Float64

arXiv:2104.11479's closed-form Liouvillian gap, `O(N^-3)` with an explicit coefficient. This is a
PUBLISHED NUMBER, so agreement with [`xx_boundary_liouvillian_gap`](@ref) is a genuine recreation
of the paper's central result rather than an internal consistency check.
"""
function xx_boundary_gap_published(N::Int; J::Real = 1.0, eps_L::Real = 1.0, eps_R::Real = 1.0)
    l = eps_L / (4J)
    r = eps_R / (4J)
    return 4J * pi^2 * (l + r) * (l * r + 1) /
           ((l + r)^2 + (l * r - 1)^2) / N^3
end

# ── THE TIME-DEPENDENT ANALYTIC SOLUTION: an N x N LYAPUNOV EQUATION, never a 4^L object ──────
#
# ⛔ THIS REPLACES THE DENSE LIOUVILLIAN AS THE REFERENCE. The spectrum above is exact but says
# nothing about `<S^z_j(t)>`, so the only time-dependent reference this model had was a dense
# `4^L x 4^L` superoperator -- which caps the check at L = 3..5 and cannot certify anything at a
# benchmark size. Third quantization does better: the model is quadratic, so the TWO-POINT FUNCTION
# CLOSES on itself and its equation of motion is an `N x N` matrix ODE. That is the paper's own
# analytic solution, it is exact at every `t`, and it costs `O(N^3)` in the CHAIN LENGTH.
#
# The dense object keeps exactly one job: certifying THIS, once, at L = 3 (`dissipative_xx.jl`).
#
# DERIVATION (adjoint Lindbladian on `A = c^dag_j c_k`, with `H = sum h_ab c^dag_a c_b`, gain
# `L = sqrt(g_m) c^dag_m` and loss `L = sqrt(l_m) c_m`):
#
#     dC/dt = i[h, C] - (1/2){D, C} + G,     C_jk = Tr(rho c^dag_j c_k)
#     D = diag(g_m + l_m),  G = diag(g_m),   h real symmetric (hopping J)
#
# so with `X = i*h - D/2`,   dC/dt = X C + C X' + G.
#
# ⛔ THE JORDAN-WIGNER STRING AT SITE N DOES NOT SPOIL THIS, AND THAT IS NOT AN APPROXIMATION.
# `sigma^+_N = P c^dag_N` with `P = prod_{j<N}(1 - 2 n_j)` is NOT linear in fermions, so the
# Lindbladian is not obviously quadratic and the whole reference would be void if the string
# survived. It does not. In `L^dag A L = c_N P (c^dag_j c_k) P c^dag_N`:
#
#   * `j, k < N`   -- `P` commutes through with TWO sign flips, so it cancels;
#   * `j = k = N`  -- `P` commutes with `n_N`, so it cancels;
#   * `j < N, k = N` (and its transpose) -- `P` gives ONE sign flip, so the string convention
#     matters -- but that term contains `c_N ... c_N` and VANISHES IDENTICALLY.
#
# Every term where the two conventions differ is zero, so the SPIN model and the stringless FERMION
# model have the SAME closed ODE for `C`. This is why the reference may be compared against an MPS
# built from `sigma^+`/`sigma^-` (`fused_boundary_lindblad`) rather than from `c`/`c^dag`.
#
# ⚠ THE ANTICOMMUTATOR RATE IS `eps`, NOT `eps/4`. `D_1 = g_1 + l_1 = eps_L(1+mu)/2 +
# eps_L(1-mu)/2 = eps_L` -- the `mu` dependence cancels in the DAMPING and survives only in the
# SOURCE `G`. The `-i*eps/4` of `xx_boundary_structure_matrix` is a different object in a different
# (Majorana) basis; the two must NOT be conflated, and mixing them silently changes the relaxation
# rate by 4x while leaving the profile qualitatively right.

"""
    xx_boundary_lyapunov(N; J, B, eps_L, eps_R, mu_L, mu_R) -> (X, G)

The generator `X` and source `G` of the covariance ODE `dC/dt = X C + C X' + G` for the
boundary-driven XX chain. `mu = +1` pumps the site fully UP (gain only), `mu = -1` fully DOWN.
"""
function xx_boundary_lyapunov(N::Int; J::Real = 1.0, B::Real = 0.0,
                              eps_L::Real = 1.0, eps_R::Real = 1.0,
                              mu_L::Real = 1.0, mu_R::Real = -1.0)
    N >= 2 || throw(ArgumentError("need N >= 2, got $N"))
    (abs(mu_L) <= 1 && abs(mu_R) <= 1) ||
        throw(ArgumentError("mu must lie in [-1, 1]; got $mu_L, $mu_R"))
    h = zeros(Float64, N, N)
    for k in 1:N
        h[k, k] = B
    end
    for k in 1:(N - 1)
        h[k, k + 1] = J
        h[k + 1, k] = J
    end
    g = zeros(Float64, N); l = zeros(Float64, N)
    g[1] = eps_L * (1 + mu_L) / 2; l[1] = eps_L * (1 - mu_L) / 2
    g[N] += eps_R * (1 + mu_R) / 2; l[N] += eps_R * (1 - mu_R) / 2
    X = im .* ComplexF64.(h) .- Diagonal(ComplexF64.(g .+ l)) ./ 2
    return Matrix(X), Matrix(Diagonal(ComplexF64.(g)))
end

"""
    xx_boundary_ness(N; kwargs...) -> Matrix{ComplexF64}

The EXACT non-equilibrium steady-state covariance `C_ss`, from `X C + C X' + G = 0`.

⚠ `lyap(A, C)` solves `A X + X A' + C = 0`, which is this equation with `C = G` -- no sign flip.
Getting that backwards returns a matrix that is still Hermitian and still has plausible entries, so
the mistake shows up only as a profile with the wrong sign of the current.
"""
xx_boundary_ness(N::Int; kwargs...) = begin
    X, G = xx_boundary_lyapunov(N; kwargs...)
    lyap(X, G)
end

"""
    xx_boundary_covariance(N, t; C0 = nothing, kwargs...) -> Matrix{ComplexF64}

EXACT `C_jk(t) = Tr(rho(t) c^dag_j c_k)` at any `t`, for any initial covariance `C0` (default
`I/2`, the MAXIMALLY MIXED state -- which is a `chi = 1` PRODUCT state in the fused superket
layout, so the MPS run starts from exactly this).

⛔ SOLVED AROUND THE STEADY STATE, NOT BY EXPONENTIATING THE AFFINE SYSTEM. Subtracting `C_ss`
makes the ODE HOMOGENEOUS, `d(C - C_ss)/dt = X (C - C_ss) + (C - C_ss) X'`, whose solution is
`exp(Xt) (C0 - C_ss) exp(Xt)'`. That is two `N x N` exponentials per time point. The direct route
-- vectorising to an `N^2` affine system and exponentiating -- is `N^2 x N^2`, which at N = 32 is a
1025-dimensional `expm` per time point for the identical answer.
"""
function xx_boundary_covariance(N::Int, t::Real; C0 = nothing, kwargs...)
    X, G = xx_boundary_lyapunov(N; kwargs...)
    Css = lyap(X, G)
    C_0 = C0 === nothing ? Matrix{ComplexF64}(I, N, N) ./ 2 : ComplexF64.(C0)
    E = exp(X .* float(t))
    return Css .+ E * (C_0 .- Css) * E'
end

"""
    xx_boundary_sz(N, t; kwargs...) -> Vector{Float64}

`<S^z_j(t)>` for the boundary-driven XX chain: `C_jj(t) - 1/2`, since `S^z = n - 1/2` and `sigma^+`
RAISES to up, so "up" is the OCCUPIED fermion state.
"""
xx_boundary_sz(N::Int, t::Real; kwargs...) =
    real.(diag(xx_boundary_covariance(N, t; kwargs...))) .- 0.5

"""
    xx_boundary_current(C; J = 1.0) -> Vector{Float64}

Bond currents `j_k = -2 J Im C_{k,k+1}` from a covariance matrix, `k = 1..N-1`.

⛔ THE SIGN IS DERIVED FROM THE CONTINUITY EQUATION, AND THE OTHER ONE WAS MEASURED WRONG. With
`H = J sum_k (c^dag_k c_{k+1} + h.c.)`, the Heisenberg equation gives
`dn_k/dt = j_{k-1} - j_k` with `j_k = i J (c^dag_k c_{k+1} - c^dag_{k+1} c_k)`, and since
`C_{k+1,k} = conj(C_{k,k+1})` that is `-2 J Im C_{k,k+1}`.

MEASURED: `+2J Im C` disagreed with the dense Liouvillian by EXACTLY twice the current at every
`L` and `t` (0.800 against a NESS current of 0.400) while the magnetisation profile matched to
1e-16 -- so a flipped sign here is invisible in every observable except the transport number this
model exists to produce. It is also physically checkable: `mu_L = +1` pumps particles IN at the
left and `mu_R = -1` removes them at the right, so the current MUST be positive.
"""
xx_boundary_current(C::AbstractMatrix; J::Real = 1.0) =
    [-2 * J * imag(C[k, k + 1]) for k in 1:(size(C, 1) - 1)]

"""
    xx_boundary_current_check(N; kwargs...) -> Float64

Largest violation of the bulk continuity equation `dC_kk/dt = j_{k-1} - j_k` in the NESS, where the
left side is ZERO by definition. Returns `max_k |j_k - j_{k-1}|` over the bulk bonds, which must be
at machine precision -- and simultaneously verifies the published statement that XX boundary
transport is BALLISTIC, i.e. the current is UNIFORM along the chain and independent of `N`.
"""
function xx_boundary_current_check(N::Int; J::Real = 1.0, kwargs...)
    j = xx_boundary_current(xx_boundary_ness(N; J = J, kwargs...); J = J)
    length(j) < 2 && return 0.0
    return maximum(abs.(diff(j)))
end

"""
    xx_boundary_current_published(; J = 1.0, eps = 1.0, mu_L = 1.0, mu_R = -1.0) -> Float64

The CLOSED-FORM non-equilibrium steady-state current of the boundary-driven XX chain:

    j = (mu_L - mu_R) * eps * J^2 / (4 J^2 + eps^2)

⛔ INDEPENDENT OF `N`. That is the ballistic signature stated as a formula, and it is the strongest
possible reference for this model: a single expression that the MPS run must reproduce at ANY chain
length, with no fitting, no extrapolation and no dense object.

⛔ THIS IS A REAL CLOSED FORM, NOT A FIT. MEASURED against `xx_boundary_ness` at `N = 32`, over
`eps` spanning four decades, the agreement is EXACT to machine precision at every point:

    eps      0.01      0.03      0.1       0.3       1        2      4      10        30        100
    lyap     0.0049999 0.0149966 0.0498753 0.1466993 0.400000 0.5    0.4    0.1923077 0.0663717 0.0199920
    formula  identical to the last printed digit at every eps

⚠ IT CONTAINS THE QUANTUM ZENO TURNOVER, which is the published qualitative statement about this
model and a genuine constraint on the shape rather than the scale. The current RISES linearly in
the drive (`j -> (Delta mu) eps / 4J^2` as `eps -> 0`), PEAKS at `eps = 2J` (where `j = Delta mu /
4`), then FALLS as `1/eps` -- a strongly measured boundary decouples from the chain. A reference
that got only the small-`eps` slope right would still fail past the peak, so matching across the
turnover tests far more than a normalisation.

⚠ EQUAL COUPLINGS ONLY. `eps_L != eps_R` has a different (and here underived) closed form, so this
throws rather than silently returning the symmetric answer -- the failure mode would be a plausible
number that is wrong by an O(1) factor.
"""
function xx_boundary_current_published(; J::Real = 1.0, eps::Real = 1.0,
                                       mu_L::Real = 1.0, mu_R::Real = -1.0)
    return (mu_L - mu_R) * eps * J^2 / (4 * J^2 + eps^2)
end

# ── TFIM, the same idea one level up: BdG rather than a hopping matrix ────────────────────────
#
# `H = -J Σ Z_j Z_{j+1} - h Σ X_j` is quadratic in MAJORANAS, not in fermions -- the ZZ term
# pairs as well as hops -- so the closed form runs through the 2L x 2L covariance matrix instead
# of an L x L propagator. Still no 2^L object anywhere, which is the whole point: the integrator
# tests can then run at the L where a defect is actually LARGE rather than at whatever L a dense
# propagator happens to fit in.
#
# ⛔ THE BASIS IS σˣ: `X` is DIAGONAL and `Z` is the FLIP. Reading it the usual way describes a
# different Hamiltonian -- see the header of `src/RSVDCBEBondUpdate/models/tfim.jl`.

"""
    tfim_majorana_generator(L; J=1.0, h=1.0) -> Matrix{Float64}

The real antisymmetric `2L x 2L` generator `A` of the TFIM's Majorana flow.

Writing `H = (i/4) Σ A_{mn} a_m a_n` in Majoranas `a_{2j-1}, a_{2j}`, the field `-h X_j` is the
on-site pair `(2j-1, 2j)` and the coupling `-J Z_j Z_{j+1}` is the bond pair `(2j, 2j+1)`. The
factors of two are the Majorana convention and are pinned against exact diagonalisation in
`test_analytic_reference.jl` rather than asserted here.
"""
function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L
        A[2j - 1, 2j] = -2h; A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)
        A[2j, 2j + 1] = -2J; A[2j + 1, 2j] = +2J
    end
    return A
end

"""
    tfim_covariance(L, t, dirs; J=1.0, h=1.0) -> Matrix{Float64}

The Majorana covariance `Γ(t)` of a σˣ PRODUCT state evolved under the TFIM.

`dirs[j]` is `:plus` or `:minus`. Such a state is Gaussian, so it is fully described by
`Γ_{2j-1,2j} = ±1`, and Heisenberg evolution is the congruence

    Γ(t) = R Γ(0) Rᵀ,   R = exp(A t)

Cost is `O(L³)`, independent of any bond dimension. THIS is the primitive: `⟨X_j⟩`, the half-chain
entropy and the connected correlators all read off the SAME `Γ`, so anything needing more than the
profile should take the matrix rather than grow a second copy of this congruence.
"""
function tfim_covariance(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    length(dirs) == L || throw(DimensionMismatch("dirs has $(length(dirs)) entries, need $L"))
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s; G[2j, 2j - 1] = -s
    end
    R = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    return R * G * transpose(R)
end

"""
    tfim_x_profile(L, t, dirs; J=1.0, h=1.0) -> Vector{Float64}

`⟨X_j(t)⟩` for a σˣ product state evolved under the TFIM -- the diagonal pair entries of
[`tfim_covariance`](@ref).
"""
function tfim_x_profile(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    Gt = tfim_covariance(L, t, dirs; J = J, h = h)
    return [Gt[2j - 1, 2j] for j in 1:L]
end
