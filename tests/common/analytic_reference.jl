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
# with the ground-state quantum numbers `I_j = 2j - M - 1` (`j = 1 … M`), and
#
#     E = J ( L/4 - Σ_j 2 / (4 λ_j^2 + 1) ).
#
# Solved below by damped fixed-point iteration on `λ`, which for the ground-state configuration
# converges monotonically from the free-magnon guess. This is EXACT up to the iteration
# tolerance -- it is not a fit and not an extrapolation.
#
# `test_analytic_reference.jl` pins it against exact diagonalisation at `L = 4, 6, 8, 10`. If it
# ever disagrees, the diagonalisation is right and this is wrong: the risk here is a convention
# (the `I_j` parity, the sign of the scattering term), not the physics.

"""
    bethe_xxx_ring_energy(L; J=1.0, iters=20000, tol=1e-14) -> (energy, lambdas, residual)

Ground energy of the periodic XXX chain `H = J Σ_i S_i·S_{i+1}` (with `S_{L+1} = S_1`) from the
Bethe equations, for even `L`.

Returns the residual alongside the energy on purpose: an iterative solve that has not converged
must not be able to masquerade as an analytic reference, so the caller can assert on it.
"""
function bethe_xxx_ring_energy(L::Int; J::Float64 = 1.0, iters::Int = 20_000,
                               tol::Float64 = 1e-14)
    iseven(L) || throw(ArgumentError("bethe_xxx_ring_energy needs even L, got $L"))
    M = L ÷ 2
    Is = Float64[2j - M - 1 for j in 1:M]
    # Free-magnon start: ignore the scattering term entirely.
    lam = Float64[0.5 * tan(pi * Is[j] / (2 * L)) for j in 1:M]
    # A rapidity at exactly 0.5*tan(0) = 0 is fine; a pair colliding is not, and the
    # ground-state `I_j` are distinct so they do not.
    res = Inf
    for _ in 1:iters
        new = similar(lam)
        for j in 1:M
            s = 0.0
            for k in 1:M
                k == j && continue
                s += atan(lam[j] - lam[k])
            end
            new[j] = 0.5 * tan((pi * Is[j] + s) / L)
        end
        res = maximum(abs.(new .- lam))
        # Damping: the map is contractive near the solution but overshoots from the free guess.
        lam = 0.5 .* lam .+ 0.5 .* new
        res < tol && break
    end
    e = J * (L / 4 - sum(2 / (4 * l^2 + 1) for l in lam))
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
