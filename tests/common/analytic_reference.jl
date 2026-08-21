# ANALYTIC REFERENCES for the Haldane-Shastry and Heisenberg benchmarks.
#
# TEST-SIDE ONLY, and sharing no code whatsoever with `src/` -- that is the only reason any of
# it counts as a reference. Nothing here builds an MPS, an MPO or an environment.
#
# WHAT IS ACTUALLY ANALYTIC, AND WHAT IS NOT. This distinction is load-bearing and is stated
# here rather than discovered later:
#
#   GROUND STATE        Haldane-Shastry has a CLOSED FORM. The Heisenberg ring has the Bethe
#                       ansatz -- exact, but the solution of a transcendental system rather than
#                       a formula, so `bethe_xxx_ring_energy` solves it by Newton iteration.
#   IMAGINARY TIME      converges to the ground state, so the same references apply.
#   REAL TIME           SU(2) XXX from a GENERIC state has NO analytic solution, and neither
#                       does Haldane-Shastry. What IS analytic is the MAGNON SECTOR: for any
#                       translation-invariant two-body `Σ_{i<j} J(i-j) S_i·S_j` on a ring, the
#                       one-magnon states are EXACT eigenstates with a closed-form dispersion,
#                       so any superposition of them evolves by a plain Fourier sum. That is
#                       the analytic real-time reference used, plus the exact conservation laws.
#                       Anything outside the magnon sectors has only a numerically exact
#                       reference, and where that is what a test uses, the test says so.
#
# EVERY FORMULA HERE IS CHECKED AGAINST EXACT DIAGONALISATION BEFORE IT IS USED. A constant
# quoted from memory is not a reference; `test_analytic_reference.jl` pins each one at small `L`
# against a dense diagonalisation, and the constants below are the ones that survived that.

using LinearAlgebra

# ── Haldane-Shastry ──────────────────────────────────────────────────────────

"""
    hs_couplings_analytic(L; J=1.0) -> Vector{Float64}

`J(r) = J * pi^2 / (L^2 sin^2(pi r / L))` for `r = 1 … L-1`, written from the formula in
arXiv:2208.10972 and NOT read from `haldane_shastry_couplings` -- the point of a reference is
that it is a second implementation.

Note `J(r) = J(L-r)`: the coupling is periodic, which is the ring geometry. The pair `(1, L)`
is therefore as strong as a nearest-neighbour pair, not the weakest pair in the chain.
"""
hs_couplings_analytic(L::Int; J::Float64 = 1.0) =
    [J * pi^2 / (L^2 * sin(pi * r / L)^2) for r in 1:(L - 1)]

"""
    hs_ground_energy(L; J=1.0) -> Float64

The exact Haldane-Shastry ground energy for the normalisation
`H = J Σ_{l<l'} pi^2 / (L^2 sin^2(pi(l-l')/L)) S_l·S_l'`:

    E_0 / J  =  -pi^2 (L^2 + 5) / (24 L)

**Checked against exact diagonalisation at `L = 4, 6, 8` before use** -- see
`test_analytic_reference.jl`. The reason that check is not optional is that the Haldane-Shastry
constant depends on the convention for the coupling prefactor and on whether the Hamiltonian
carries a `-1/4` shift per pair; several conventions circulate and they differ by exactly the
sort of additive constant a plausibility argument will not catch.
"""
hs_ground_energy(L::Int; J::Float64 = 1.0) = -J * pi^2 * (L^2 + 5) / (24 * L)

# ── the one-magnon sector: exact, for ANY translation-invariant two-body H ────
#
# For `H = Σ_{i<j} J(i-j) S_i·S_j` on a ring of `L` sites, the fully polarised state
# `|F> = |↑↑…↑>` is an eigenstate, and the states `|k> = Σ_j e^{i k j} S_j^- |F>` with
# `k = 2πn/L` diagonalise `H` inside the one-down-spin sector EXACTLY -- translation invariance
# is the whole argument, and it needs no integrability. So a wavepacket built from them evolves
# as `Σ_n c_n e^{-i E(k_n) t} |k_n>`, which is a closed-form real-time solution.
#
# THIS IS THE ANALYTIC REAL-TIME REFERENCE, and it works identically for Haldane-Shastry and for
# the Heisenberg ring, because it only ever uses `J(r)`. What it does NOT do is exercise rank
# growth hard -- a one-magnon state has bond dimension 2. It is therefore paired in the tests
# with the conservation laws and with a richer-state comparison, and the tests label which is
# which rather than presenting a one-magnon agreement as a general accuracy claim.

"""
    magnon_reference_energy(Jr; L) -> Float64

Energy of the fully polarised state `|↑↑…↑>` for `H = Σ_{i<j} J(i-j) S_i·S_j` on a ring:
`E_F = (L/4) Σ_{r=1}^{L-1} J(r) / 2`, i.e. `(1/4)` per pair times the number of pairs, each pair
counted once.

Written as a sum over `(r, i)` divided by two rather than as a closed form, so it cannot
disagree with the pair enumeration `magnon_energy` uses.
"""
magnon_reference_energy(Jr::AbstractVector{<:Real}; L::Int) =
    0.25 * L * sum(Jr) / 2

"""
    magnon_energy(Jr, n; L) -> Float64

Exact energy of the one-magnon state with momentum `k = 2πn/L`:

    E(k) = E_F - (1/2) Σ_{r=1}^{L-1} J(r) (1 - cos(k r))

The `1/2` is the spin-flip matrix element of `S^+ S^-`, and the `(1 - cos kr)` is the hopping
minus the diagonal cost of having moved the down spin. `n = 0` gives `E_F` exactly, which is the
cheapest available check that the normalisation is right: `k = 0` is the fully polarised state
rotated, so it must be degenerate with it.
"""
function magnon_energy(Jr::AbstractVector{<:Real}, n::Int; L::Int)
    k = 2 * pi * n / L
    return magnon_reference_energy(Jr; L = L) -
           0.5 * sum(Jr[r] * (1 - cos(k * r)) for r in 1:length(Jr))
end

"""
    magnon_amplitudes(c, t, Jr; L) -> Vector{ComplexF64}

The one-magnon wavepacket `Σ_n c[n] e^{-i E(k_n) t} |k_n>` re-expressed in the SITE basis:
entry `j` is the amplitude of `S_j^- |F>`.

`c` is indexed by `n = 0 … L-1`. This is the exact real-time solution, so an integrator's error
on a one-magnon initial state is the difference from this vector and nothing else.
"""
function magnon_amplitudes(c::AbstractVector{<:Number}, t::Real,
                           Jr::AbstractVector{<:Real}; L::Int)
    length(c) == L || throw(DimensionMismatch("c must have L = $L entries, got $(length(c))"))
    a = zeros(ComplexF64, L)
    for n in 0:(L - 1)
        c[n + 1] == 0 && continue
        e = magnon_energy(Jr, n; L = L)
        ph = c[n + 1] * cis(-e * t)
        for j in 1:L
            a[j] += ph * cis(2 * pi * n * j / L)
        end
    end
    return a ./ sqrt(L)
end

# ── the Heisenberg ring: Bethe ansatz ────────────────────────────────────────
#
# The XXX ring has no closed-form finite-`L` ground energy, but it is INTEGRABLE, so the exact
# answer is the solution of the Bethe equations rather than a diagonalisation. For the ground
# state of `H = J Σ_i S_i·S_{i+1}` on `L` sites (L even) there are `M = L/2` real rapidities
# `λ_j` satisfying
#
#     L * atan(2 λ_j)  =  π I_j  +  Σ_{k≠j} atan(λ_j - λ_k)
#
# with the ground-state quantum numbers
#
#     I_j = j - (M+1)/2,     j = 1 … M
#
# and
#
#     E = J ( L/4 - Σ_j 2 / (4 λ_j^2 + 1) ).
#
# ⛔ THE `I_j` ARE HALF-ODD-INTEGERS WHEN `M` IS EVEN, AND THAT IS NOT A CONVENTION -- IT IS THE
# GROUND STATE. This was wrong here as `I_j = 2j - M - 1`, exactly a factor of two too large,
# which for `L = 4` asks for `I = (-1, +1)` where the ground state has `I = (-1/2, +1/2)`. The
# solver then chases a configuration that is not a solution at all: measured `E = -1.3e-14`
# against the exact `-2`, with the iteration residual stuck at `1.0`. The residual is what caught
# it, which is the reason this function returns one.
#
# Newton, not fixed-point iteration. Writing the equations as
#
#     F_j(λ) = 2 L atan(2 λ_j) - 2π I_j - Σ_{k≠j} 2 atan(λ_j - λ_k) = 0
#
# the Jacobian is available in closed form,
#
#     ∂F_j/∂λ_j = 4L/(1 + 4λ_j^2) - Σ_{k≠j} 2/(1 + (λ_j-λ_k)^2),
#     ∂F_j/∂λ_k = +2/(1 + (λ_j-λ_k)^2),
#
# so the solve is quadratically convergent and reaches `1e-15` in a handful of iterations at any
# `L`. The damped fixed-point map it replaces needed thousands of sweeps and still left `res` at
# `172` for `L = 64`, which made the Hulthen check untestable.
#
# `test_analytic_reference.jl` pins it against exact diagonalisation at `L = 4, 6, 8, 10`. If it
# ever disagrees, the diagonalisation is right and this is wrong.

"""
    bethe_xxx_ring_energy(L; J=1.0, iters=200, tol=1e-14) -> (energy, lambdas, residual)

Ground energy of the periodic XXX chain `H = J Σ_i S_i·S_{i+1}` (with `S_{L+1} = S_1`) from the
Bethe equations, for even `L`.

Returns the residual alongside the energy on purpose: an iterative solve that has not converged
must not be able to masquerade as an analytic reference, so the caller can assert on it. The
residual returned is `‖F(λ)‖_∞` of the equations themselves, NOT the step size -- a step can be
small because the solver has stalled as easily as because it has converged.
"""
function bethe_xxx_ring_energy(L::Int; J::Float64 = 1.0, iters::Int = 200,
                               tol::Float64 = 1e-14)
    iseven(L) || throw(ArgumentError("bethe_xxx_ring_energy needs even L, got $L"))
    M = L ÷ 2
    Is = Float64[j - (M + 1) / 2 for j in 1:M]
    # Free-magnon start: drop the scattering term, i.e. solve `2L atan(2λ) = 2π I`.
    lam = Float64[0.5 * tan(pi * Is[j] / L) for j in 1:M]

    F = zeros(Float64, M)
    Jac = zeros(Float64, M, M)
    res = Inf
    for _ in 1:iters
        fill!(Jac, 0.0)
        for j in 1:M
            F[j] = 2 * L * atan(2 * lam[j]) - 2 * pi * Is[j]
            Jac[j, j] = 4 * L / (1 + 4 * lam[j]^2)
            for k in 1:M
                k == j && continue
                d = lam[j] - lam[k]
                F[j] -= 2 * atan(d)
                w = 2 / (1 + d^2)
                Jac[j, j] -= w
                Jac[j, k] = w
            end
        end
        res = maximum(abs, F)
        res < tol && break
        lam .-= Jac \ F
    end
    e = J * (L / 4 - sum(2 / (4 * l^2 + 1) for l in lam))
    return e, lam, res
end

"""
    bethe_xxx_open_energy(L; J=1.0, iters=300, tol=1e-14) -> (energy, lambdas, residual)

Ground energy of the OPEN XXX chain `H = J Σ_{i=1}^{L-1} S_i·S_{i+1}` (free ends, `L-1` bonds)
from the Bethe equations, for even `L`. Same contract as the ring solver: the residual comes back
so an unconverged solve cannot pose as an analytic reference.

WHY THE OPEN CASE IS NOT THE RING CASE. Open boundaries REFLECT, so each rapidity scatters off the
MIRROR IMAGE of every other one as well as off it directly — hence the `λ_j + λ_k` term the ring
lacks — and the quantum numbers become positive integers `I_j = j` rather than the ring's
symmetric half-integers, because the rapidities come in `±` pairs and only `λ > 0` is independent.

    2L·θ₁(λ_j) = 2π I_j + Σ_{k≠j} [ θ₂(λ_j − λ_k) + θ₂(λ_j + λ_k) ],   θ₁(x) = 2atan(2x),
                                                                        θ₂(x) = 2atan(x)
    E = J [ (L−1)/4 − Σ_j 2/(4λ_j² + 1) ]

⛔ THERE IS NO SELF-REFLECTION TERM, and getting that wrong is why the first attempt at this
failed. It carried an extra `−2·atan(λ_j)` on the right, i.e. `θ₂(λ_j)` — which is neither the
`k = j` member of the `+` product (`θ₂(2λ_j)`) nor nothing at all. It converged to residual 1e-14
and sat ~0.24 BELOW exact diagonalisation at every `L` from 4 to 12: a CLEAN CONVERGENCE TO THE
WRONG NUMBER, which is exactly what a wrong equation form looks like and exactly why the residual
alone must never be treated as evidence of correctness. The form above was then found by sweeping
the candidate family against ED rather than recalled, and matches at **4.00e-15 for every `L` in
4…12** (`scratchpad/bethe_open_search.jl`). Note `θ₂(2λ) ≡ θ₁(λ)`, so a self term would be
indistinguishable from bumping the length factor to `2L+1` — the sweep found that pair degenerate,
as it must be.
"""
function bethe_xxx_open_energy(L::Int; J::Float64 = 1.0, iters::Int = 300,
                               tol::Float64 = 1e-14)
    iseven(L) || throw(ArgumentError("bethe_xxx_open_energy needs even L, got $L"))
    M = L ÷ 2
    Is = Float64[j for j in 1:M]
    # Free-magnon start: drop the scattering terms, i.e. solve `2L atan(2λ) = π I` on (0, π/2).
    lam = Float64[0.5 * tan(pi * Is[j] / (2 * (L + 1))) for j in 1:M]

    F = zeros(Float64, M)
    Jac = zeros(Float64, M, M)
    res = Inf
    for _ in 1:iters
        fill!(Jac, 0.0)
        for j in 1:M
            lj = lam[j]
            # `2L·θ₁(λ) = 2L · 2atan(2λ) = 4L·atan(2λ)`. Writing `2L*atan(2λ)` here -- dropping
            # the 2 inside θ₁ -- leaves a solver that does not converge at all (residuals of
            # order 10, not 1e-14), so it fails loudly rather than quietly, unlike the earlier
            # self-term bug. Kept explicit because the ring solver's line looks deceptively similar.
            F[j] = 4 * L * atan(2 * lj) - 2 * pi * Is[j]
            Jac[j, j] = 8 * L / (1 + 4 * lj^2)
            for k in 1:M
                k == j && continue
                d, s = lj - lam[k], lj + lam[k]
                F[j] -= 2 * atan(d) + 2 * atan(s)
                wd, ws = 2 / (1 + d^2), 2 / (1 + s^2)
                Jac[j, j] -= wd + ws
                # `+λ_k` enters with the OPPOSITE sign in `∂/∂λ_k` to `−λ_k`: the mirror moves
                # the other way. Dropping that sign is a silent way to lose quadratic convergence.
                Jac[j, k] = wd - ws
            end
        end
        res = maximum(abs, F)
        res < tol && break
        # DAMPED, AND THE DAMPING IS NOT OPTIONAL. Undamped Newton on these equations walks the
        # rapidities towards 0; there `λ_j − λ_k` and `λ_j + λ_k` coincide, so `Jac[j,k] = wd − ws`
        # vanishes, the Jacobian goes exactly diagonal and the solve throws `SingularException`.
        # MEASURED at L = 4. Capping the step keeps the iterate where the rapidities stay
        # separated, which is where the ground-state solution lives.
        #
        # ⚠ DO NOT "HELP" THIS BY REFLECTING `λ` BACK POSITIVE. The rapidities are positive at the
        # solution, but folding negatives with `abs` mid-iteration makes the sequence oscillate
        # instead of converge -- MEASURED: residuals of 6-146 at every L, versus 1e-14 without it.
        step = try
            Jac \ F
        catch
            # Levenberg nudge rather than a hard failure: a singular Jacobian here means the
            # iterate wandered into the degenerate region, not that the problem is unsolvable.
            (Jac + 1e-8 * I) \ F
        end
        nrm = norm(step)
        nrm > 1.0 && (step .*= 1.0 / nrm)
        lam .-= step
    end
    e = J * ((L - 1) / 4 - sum(2 / (4 * l^2 + 1) for l in lam))
    return e, lam, res
end

"""
    hulthen_energy_density(; J=1.0) -> Float64

`E/L -> J (1/4 - ln 2) = -0.4431471805599453` -- Hulthen's thermodynamic-limit energy density
for `H = J Σ_i S_i·S_{i+1}`.

A LIMIT, not a finite-`L` value: finite-size corrections fall off as `1/L^2`, so this is the
right check on a large-`L` run and the wrong check on a small one. It is here so a `L = 6`
number is never compared against it by accident.
"""
hulthen_energy_density(; J::Float64 = 1.0) = J * (0.25 - log(2))

# ── one-axis twisting: the CLOSED-FORM state (arXiv:2208.10972 Fig. 2) ───────
#
# OAT is the one benchmark in the paper that is analytic ALL THE WAY, not just in its ground
# state or in a magnon sector -- and that is worth being explicit about, because it makes this
# the strongest reference in the suite. `H_OAT = (Σ_ℓ S^z_ℓ)^2 / 2` is DIAGONAL in the
# computational basis, so `exp(-i t H)` is a phase per configuration and the evolved state is
# known exactly for every `L` and every `t`:
#
#     |Ψ(t)⟩ = 2^(-L/2) Σ_s exp(-i t m_s^2 / 2) |s⟩,        m_s = Σ_ℓ s_ℓ ∈ {-L/2 … L/2}
#
# There is no diagonalisation and no propagation anywhere below. Every function here is that
# formula, so an integrator compared against them is compared against the exact answer rather
# than against a second numerical scheme with its own error.
#
# WHY THE SCHMIDT SPECTRUM IS ALSO CLOSED-FORM, which is what makes panels (b) and (c) checkable
# and not just panel (a). Cut the chain after site `n`. With `m = a + b` (`a` in the left block,
# `b` in the right),
#
#     exp(-i t (a+b)^2 / 2) = exp(-i t a^2/2) · exp(-i t b^2/2) · exp(-i t a b)
#
# The first two factors are LOCAL to one block and only rephase its basis vectors. All the
# entanglement is in the cross term, and `exp(-i t a b)` depends on the blocks only through the
# two BLOCK MAGNETISATIONS -- so the state's Schmidt rank is at most the number of distinct `a`,
# i.e. `min(n, L-n) + 1`, for ALL time. OAT never entangles past that ceiling, which is a strong
# and easily-checked statement about panel (c): a run reporting a larger rank is reporting noise.

"Binomial coefficient as a `Float64`, via `BigInt` so `L` in the hundreds cannot overflow."
_binom(n::Int, k::Int) = Float64(binomial(big(n), big(k)))

"""
    oat_total_sx(L, t) -> Float64

`S^x_tot(t) = (L/2) cos^(L-1)(t/2)`, the exact one-axis-twisting signal quoted in
arXiv:2208.10972 -- the periodic collapses and revivals of Fig. 2(a).

The revival period is `4π` (`cos^(L-1)` at `t = 2π` returns `-S^x_tot(0)` for even `L`, and the
full positive revival is at `4π`), so the paper's `t = 12π` is three cycles.
"""
oat_total_sx(L::Int, t::Real) = (L / 2) * cos(t / 2)^(L - 1)

"""
    oat_schmidt_matrix(L, n, t) -> Matrix{ComplexF64}

The exact `(n+1) x (L-n+1)` coefficient matrix of `|Ψ(t)⟩` in the normalised block-magnetisation
bases either side of the cut after site `n`. Its singular values ARE the Schmidt values.

Rows are indexed by the number of up spins `k = 0 … n` in the left block, columns by `m = 0 …
L-n` in the right; the weights `sqrt(C(n,k))` count the configurations each block basis vector
sums over, and `2^(-L/2)` is the amplitude every configuration of `⊗|→⟩` carries.
"""
function oat_schmidt_matrix(L::Int, n::Int, t::Real)
    1 <= n <= L - 1 || throw(ArgumentError("cut $n is outside 1:$(L - 1)"))
    nB = L - n
    M = zeros(ComplexF64, n + 1, nB + 1)
    for k in 0:n, m in 0:nB
        a, b = k - n / 2, m - nB / 2
        M[k + 1, m + 1] = 2.0^(-L / 2) * sqrt(_binom(n, k) * _binom(nB, m)) *
                          cis(-t * (a + b)^2 / 2)
    end
    return M
end

"""
    oat_schmidt_values(L, n, t) -> Vector{Float64}

Exact Schmidt spectrum of `|Ψ(t)⟩` across the cut after site `n`, largest first. Sums to 1 in
squares. Its length is capped at `min(n, L-n) + 1` -- see the header for why that ceiling holds
for all time.
"""
oat_schmidt_values(L::Int, n::Int, t::Real) = svdvals(oat_schmidt_matrix(L, n, t))

"""
    oat_entropy(L, n, t; base=ℯ) -> Float64

Exact von Neumann entanglement entropy `-Σ σ² log σ²` across the cut after site `n`.

`base = ℯ` (the default) is what arXiv:2208.10972 Fig. 2(b) plots; pass `base = 2` for bits.
Zero at `t = 0` -- `⊗|→⟩` is a product state -- and bounded by `log(min(n, L-n) + 1)` forever.
"""
function oat_entropy(L::Int, n::Int, t::Real; base::Real = ℯ)
    p = abs2.(oat_schmidt_values(L, n, t))
    return -sum(q * log(base, q) for q in p if q > 1e-300; init = 0.0)
end

"""
    oat_rank(L, n, t; tol=1e-12) -> Int

Number of Schmidt values above `tol` across the cut after site `n`: the bond dimension an
EXACT MPS of `|Ψ(t)⟩` needs there. The reference for Fig. 2(c).
"""
oat_rank(L::Int, n::Int, t::Real; tol::Float64 = 1e-12) =
    count(>(tol), oat_schmidt_values(L, n, t))
