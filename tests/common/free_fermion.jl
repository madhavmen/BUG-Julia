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
    xx_open_propagator(L, t; J=1.0) -> Matrix{ComplexF64}

The open-chain one-body propagator in closed form (HQ plan T7.1),

    u_ij(t) = 2/(L+1) sum_m sin(pi m i/(L+1)) sin(pi m j/(L+1)) exp(-i J t cos(pi m/(L+1))),

the eigen-decomposition of `xx_single_particle_hamiltonian` written out, NOT computed from it.
"""
function xx_open_propagator(L::Int, t::Real; J::Real = 1.0)
    k = [pi * m / (L + 1) for m in 1:L]
    return [2 / (L + 1) * sum(sin(k[m] * i) * sin(k[m] * j) * cis(-J * t * cos(k[m]))
                              for m in 1:L) for i in 1:L, j in 1:L]
end

"""
    xx_open_correlation(L, t; J=1.0, occupied=1:(L÷2)) -> Matrix{ComplexF64}

`C(t) = u(t) C(0) u(t)'` with `C_ij = <c_j^dag c_i>` (plan T7.1's convention), from the Slater
determinant occupying `occupied`: `<S^z_i> = C_ii - 1/2`.
"""
function xx_open_correlation(L::Int, t::Real; J::Real = 1.0, occupied = 1:(L ÷ 2))
    u = xx_open_propagator(L, t; J = J)
    C0 = Diagonal([i in occupied ? 1.0 : 0.0 for i in 1:L])
    return u * C0 * u'
end

"""
    xx_open_current(C; J=1.0) -> Vector{Float64}

The bond currents `j_b = (iJ/2) <S^+_b S^-_{b+1} - h.c.> = -J Im C_{b+1,b}` of the spin-1/2 XX chain
(`S^+_b S^-_{b+1} = c_b^dag c_{b+1}`: the string cancels on adjacent sites), `C` in the convention of
`xx_open_correlation`. Signed so that `dC_bb/dt = j_{b-1} - j_b`.
"""
xx_open_current(C::AbstractMatrix; J::Real = 1.0) = [-J * imag(C[b + 1, b]) for b in 1:(size(C, 1) - 1)]

"""
    xx_slater_amplitudes(L, t, basis; J=1.0, occupied=1:(L÷2)) -> Vector{ComplexF64}

The exact many-body state of the evolved Slater determinant, amplitude by amplitude, on `basis`
(`spin_xx_sector`'s: `basis[k][i] = 2 m_i`, up = occupied): `det V[I, :]` with
`V = u(t)[:, occupied]` and `I` the up sites in ascending order. With the string on the sites left
of `j`, `c'_{I_1} ... c'_{I_N} |0>` IS `S^+_{I_1} ... S^+_{I_N} |down...>`, so no sign enters
(checked against sector ED, xx-efficiency plan S1). A state-fidelity oracle for an arbitrary MPS,
which the correlation matrix alone is not.
"""
function xx_slater_amplitudes(L::Int, t::Real, basis; J::Real = 1.0, occupied = 1:(L ÷ 2))
    V = xx_open_propagator(L, t; J = J)[:, collect(occupied)]
    return ComplexF64[det(V[findall(>(0), b), :]) for b in basis]
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

include(joinpath(@__DIR__, "free_fermion_open.jl"))
