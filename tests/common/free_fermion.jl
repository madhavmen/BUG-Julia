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
