# free_fermion_open.jl: the time-dependent Lyapunov solution and the TFIM BdG reference (from `free_fermion.jl`, included right after it).

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
    xx_open_generator(N; gamma = 0.0, kwargs...) -> (M, g)

The boundary-driven XX chain WITH BULK DEPHASING `L_j = sqrt(gamma) n_j` on every site (plan OS-1),
as the vectorised affine ODE `d vec(C)/dt = M vec(C) + g`. The two-point function still closes:
dephasing damps every OFF-diagonal `C_jk` at rate `gamma` and leaves the diagonal alone
(`xx_dephasing_correlation`), so `dC/dt = X C + C X' + G - gamma (C - Diagonal(C))` with `X, G` of
[`xx_boundary_lyapunov`](@ref) (`kwargs`: `J, B, eps_L, eps_R, mu_L, mu_R`). Column-major `vec`:
`vec(X C) = kron(I, X) vec(C)`, `vec(C X') = kron(conj(X), I) vec(C)`. `gamma = 0` is
[`xx_boundary_covariance`](@ref) exactly. `N^2 x N^2` dense: N = 64 is 4096^2, fine for a reference.
"""
function xx_open_generator(N::Int; gamma::Real = 0.0, kwargs...)
    X, G = xx_boundary_lyapunov(N; kwargs...)
    Id = Matrix{ComplexF64}(I, N, N)
    M = kron(Id, X) + kron(conj(X), Id)
    for i in 1:N, j in 1:N
        i == j || (M[(j - 1) * N + i, (j - 1) * N + i] -= gamma)
    end
    return M, vec(G)
end

"The exact NESS covariance of [`xx_open_generator`](@ref): `M vec(C) + g = 0`."
xx_open_ness(N::Int; kwargs...) = (Mg = xx_open_generator(N; kwargs...);
                                   reshape(-(Mg[1] \ Mg[2]), N, N))

"""
    xx_open_covariance(N, ts; C0 = nothing, kwargs...) -> Vector{Matrix{ComplexF64}}

EXACT `C(t)` of [`xx_open_generator`](@ref) at every time in `ts` (ascending, from `t = 0`), for the
initial covariance `C0` (default `I/2`, maximally mixed). Solved around the NESS, so the ODE is
homogeneous; one `exp(M dt)` per distinct spacing, cached, so a uniform output grid costs one.
"""
function xx_open_covariance(N::Int, ts::AbstractVector{<:Real}; C0 = nothing, kwargs...)
    M, g = xx_open_generator(N; kwargs...)
    css = -(M \ g)
    c = vec(C0 === nothing ? Matrix{ComplexF64}(I, N, N) ./ 2 : ComplexF64.(C0)) .- css
    props = Dict{Float64, Matrix{ComplexF64}}()
    out = Matrix{ComplexF64}[]
    tprev = 0.0
    for t in ts
        dt = round(float(t) - tprev; sigdigits = 12)
        dt < 0 && throw(ArgumentError("ts must be ascending from 0"))
        dt > 0 && (c = get!(() -> exp(M .* dt), props, dt) * c)
        push!(out, reshape(c .+ css, N, N))
        tprev = float(t)
    end
    return out
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
