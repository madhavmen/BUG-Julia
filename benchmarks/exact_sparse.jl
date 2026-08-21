# EXACT REFERENCE FOR THE HEISENBERG CHAIN, BY SPARSE KRYLOV -- NOT BY DIAGONALISATION.
#
# The distinction matters and it is easy to get wrong. Real-time evolution needs
# `exp(-iHt)|psi0>`, NOT the spectrum. Dense diagonalisation of the Sz = 0 sector at L = 18 is
# C(18,9) = 48620 states, so a dense `eigen` is a 48620^2 Float64 matrix (~19 GB) and is out of
# reach -- but the vector is only 48620 complex numbers (~780 kB) and `H` is sparse, ~L/2
# off-diagonal entries per row. Applying `H` costs a sparse matvec. So the exact state is
# perfectly affordable at L = 18 and the reference here is EXACT to Krylov tolerance, not a
# better-converged MPS.
#
# This is what lets the benchmark measure ACCURACY rather than mutual consistency. It also
# serves BOTH symmetry runs from one computation: `:none` and `:SU2` evolve the same state under
# the same Hamiltonian, so they share a reference, and that is exactly the claim under test --
# symmetry changes the cost, not the physics.
#
# The dimer start is a total-spin singlet, hence in Sz = 0, so the sector is closed under H.

using LinearAlgebra, SparseArrays

"""
    sparse_to_dense_index(b, L) -> Int

Translate a bit pattern of THIS file into a `dense_state` (tests/common/dense_reference.jl) index.

⛔ THE TWO FILES USE MIRRORED BIT CONVENTIONS. Both are documented, both internally consistent,
and comparing them without this conversion is silently wrong:

  - HERE (`sz0_basis`): site `j` sits at bit `j-1` — site 1 is the LEAST significant bit — and a
    SET bit means UP.
  - `dense_state`: *"Site 1 is the MOST significant index and local index 1 is spin up"*, so site
    `j` sits at bit `L-j` and a SET bit means DOWN (up is local index 0, contributing nothing).

    L=6, `↑↓↑↑↓↓`:  here  = up{1,3,4}   = 0b001101 = 13
                    dense = down{2,5,6} = 0b010011 = 19      ← MEASURED: `dense_state` gives 19

⛔ WHY THIS HID FOR SO LONG, AND WHY IT MATTERS. The conventions differ by REVERSING the sites and
FLIPPING up↔down. That composition is a symmetry of the open Heisenberg chain, and the Néel state,
the domain wall and the dimer are each INVARIANT under it — so every state used until now was a
fixed point of the exact transformation that distinguishes the two, and the comparison came out
right by accident (MEASURED: 0.0 error for all three either way). On an ASYMMETRIC product state
the unconverted comparison gives 1.414, i.e. the two vectors are orthogonal. The first real
casualty was `S^z_{j0}|GS⟩`, which read in the wrong convention becomes `−S^z` at the MIRROR site:
overlap 0.9146 instead of 1. It would also silently MIRROR every site-resolved observable.
"""
sparse_to_dense_index(b::Int, L::Int) =
    sum(((b >> (j - 1)) & 1) == 0 ? (1 << (L - j)) : 0 for j in 1:L; init = 0)

"""
Basis of the `Sz = 0` sector as bit patterns, plus the index lookup.
Bit `j-1` set means site `j` is up.
"""
function sz0_basis(L::Int)
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    return states, Dict(s => k for (k, s) in enumerate(states))
end

"Sparse Heisenberg `H = sum_j S_j . S_{j+1}` (open chain) in the `Sz = 0` sector."
function heisenberg_sparse(L::Int; J::Float64 = 1.0, delta::Float64 = 1.0)
    states, idx = sz0_basis(L)
    n = length(states)
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states)
        diag = 0.0
        for j in 1:(L - 1)
            sj, sj1 = bit(b, j), bit(b, j + 1)
            diag += J * delta * (sj == sj1 ? 0.25 : -0.25)
            if sj != sj1
                # S+S-/2 + h.c. : flip the antiparallel pair
                bb = b ⊻ (1 << (j - 1)) ⊻ (1 << j)
                push!(I, idx[bb]); push!(Jd, k); push!(V, J / 2)
            end
        end
        push!(I, k); push!(Jd, k); push!(V, diag)
    end
    return sparse(I, Jd, V, n, n), states, idx
end

"""
    pairs_sparse(L, Jm) -> (H, states, idx)

`H = Σ_{i<j} Jm[i,j] S_i·S_j` in the `Sz = 0` sector, for an ARBITRARY coupling matrix -- so
Haldane-Shastry and the periodic ring get an EXACT reference, not just the open chain that
`heisenberg_sparse` covers. `Jm` need only be filled on the strict upper triangle.

At `L = 16` the sector is `C(16,8) = 12870` states, so this is cheap and `expv_sparse` on it is
exact to Krylov tolerance -- which is what makes it a legitimate reference for a real-time run
rather than a second approximation.
"""
function pairs_sparse(L::Int, Jm::AbstractMatrix{Float64})
    states, idx = sz0_basis(L)
    n = length(states)
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states)
        diag = 0.0
        for i in 1:(L - 1), j in (i + 1):L
            Jij = Jm[i, j]
            Jij == 0.0 && continue
            si, sj = bit(b, i), bit(b, j)
            diag += Jij * (si == sj ? 0.25 : -0.25)
            if si != sj
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                push!(I, idx[bb]); push!(Jd, k); push!(V, Jij / 2)
            end
        end
        push!(I, k); push!(Jd, k); push!(V, diag)
    end
    return sparse(I, Jd, V, n, n), states, idx
end

"Coupling matrix of the PERIODIC nearest-neighbour ring, matching `heisenberg_ring_mpo`."
function ring_coupling_matrix(L::Int; J::Float64 = 1.0)
    Jm = zeros(Float64, L, L)
    for i in 1:(L - 1)
        Jm[i, i + 1] = J
    end
    Jm[1, L] += J
    return Jm
end

"Coupling matrix of Haldane-Shastry on a ring, matching `haldane_shastry_mpo`."
function hs_coupling_matrix(L::Int; J::Float64 = 1.0)
    Jm = zeros(Float64, L, L)
    for i in 1:(L - 1), j in (i + 1):L
        r = j - i
        Jm[i, j] = J * pi^2 / (L^2 * sin(pi * r / L)^2)
    end
    return Jm
end

"""
The dimer covering `⊗_k |singlet>_{2k-1,2k}` as a dense vector in the `Sz = 0` sector.

Built combinatorially rather than by projection: each pair contributes `(|up,down> -
|down,up>)/sqrt(2)`, so the state is a sum of `2^(L/2)` bit patterns with signs `(-1)^{#flipped}`.
This is the SAME state `dimer_state(L)` produces on the MPS side, and `check_dimer.jl` pins that
agreement through the bond-energy profile.
"""
function dimer_vector(L::Int, states::Vector{Int}, idx::Dict{Int, Int})
    v = zeros(ComplexF64, length(states))
    npair = L ÷ 2
    amp = (1 / sqrt(2))^npair
    for m in 0:(2^npair - 1)
        b = 0
        sign = 1
        for k in 1:npair
            lo, hi = 2k - 1, 2k          # sites of pair k
            if (m >> (k - 1)) & 1 == 0
                b |= 1 << (lo - 1)       # |up, down>
            else
                b |= 1 << (hi - 1)       # |down, up>, the term carrying the minus
                sign = -sign
            end
        end
        v[idx[b]] += sign * amp
    end
    return v
end

"""
The Néel state `|↑↓↑↓…⟩` as a vector in the `Sz = 0` sector — a single basis state, so it is a
unit column and needs only a lookup. Matches `neel_state(L)` on the MPS side.

`Sz = 0` for even `L`, so it lives in the same sector as the dimer and shares this file's
machinery. ⚠ It is NOT `:SU2`-representable (a definite-`Sz` product state is a superposition of
total-spin sectors), so a Néel start is an abelian-only experiment.
"""
function neel_vector(L::Int, idx::Dict{Int, Int})
    v = zeros(ComplexF64, length(idx))
    b = sum(1 << (j - 1) for j in 1:2:L)     # odd sites up, matching `neel_state`
    v[idx[b]] = 1.0
    return v
end

"""
`S^z_{j0} |v⟩`, NOT normalised. `S^z` is DIAGONAL in the bit basis, so this is a rescaling of the
amplitudes and — crucially — is CHARGE-NEUTRAL, keeping the state inside `Sz = 0`.

This is the Haldane-Shastry initial state of arXiv:2208.10972. The paper evolves the full
`C(x,t) = ⟨Ψ₀| **S**_x(t)·**S**_0(0) |Ψ₀⟩`; SU(2) symmetry of the ground state makes the three
components equal, `C = 3⟨Ψ₀|S^z_x(t) S^z_0|Ψ₀⟩`, so the vector that must be propagated is
`S^z_{j0}|Ψ₀⟩`. Taking the `z` component rather than `S^-` is what keeps the run in ONE symmetry
sector and lets the same `pairs_sparse` reference serve it.
"""
function sz_on(v::Vector{ComplexF64}, states::Vector{Int}, j0::Int)
    return ComplexF64[v[k] * ((((states[k] >> (j0 - 1)) & 1) - 0.5)) for k in eachindex(states)]
end

"""
The domain wall `|↑…↑↓…↓⟩` as a vector in the `Sz = 0` sector: a SINGLE basis state, so the
vector is a unit column and needs no construction beyond a lookup.

Still `Sz = 0` for even `L` (`L/2` up, `L/2` down), so it lives in the same sector as the dimer
and shares all of this file's machinery. Its natural observable is `⟨Sz_j⟩`, which — unlike the
dimer's bond energy — is NOT available under `:SU2`, so a domain-wall run is an abelian-only
experiment (`:none`, `:U1`). That is a restriction on which comparison it can serve, not a
defect: it buys the canonical magnetisation light cone in exchange for the symmetry sweep.
"""
function domain_wall_vector(L::Int, states::Vector{Int}, idx::Dict{Int, Int})
    v = zeros(ComplexF64, length(states))
    b = sum(1 << (j - 1) for j in 1:(L ÷ 2))     # sites 1..L/2 up
    v[idx[b]] = 1.0
    return v
end

"`⟨Sz_j⟩` for every site, matching `magnetisation` on the MPS side."
function magnetisation_dense(v::Vector{ComplexF64}, L::Int, states::Vector{Int})
    p = abs2.(v)
    nrm = sum(p)
    return [sum(p[k] * (((states[k] >> (j - 1)) & 1) - 0.5) for k in eachindex(states)) / nrm
            for j in 1:L]
end

"""
ONE Lanczos projection of `exp(t*A)*v`, plus an a-posteriori error estimate.

Returns `(out, err)`. `err` is the standard Saad residual estimate
`beta0 * beta_next * |coef[end]|`: the weight the projected exponential puts on the LAST Krylov
vector, times the norm of the residual direction the space had to drop. It is what tells the
caller whether `|t|` was small enough for this `m` -- without it the routine cannot know it is
wrong, which is exactly the failure this file used to ship (see `expv_sparse`).

`err == 0` means the Krylov space closed on an invariant subspace, i.e. the answer is exact.
"""
function _lanczos_expv(A, t::ComplexF64, v::Vector{ComplexF64}, m::Int, tol::Float64)
    beta0 = norm(v)
    V = Vector{Vector{ComplexF64}}(undef, m + 1)
    alpha, beta = Float64[], Float64[]
    V[1] = v / beta0
    closed = false
    for j in 1:m
        w = A * V[j]
        a = real(dot(V[j], w)); push!(alpha, a)
        w -= a * V[j]
        j > 1 && (w -= beta[j - 1] * V[j - 1])
        # full reorthogonalisation: cheap here and the sector is ill-conditioned enough to need it
        for i in 1:j
            w -= dot(V[i], w) * V[i]
        end
        b = norm(w)
        # Invariant subspace reached: the Krylov space has closed, and continuing would divide
        # by a numerically zero `b`. Relative to `beta0` so `tol` means the same thing whatever
        # the vector norm is.
        if b < max(tol * beta0, 1e-14)
            V = V[1:j]; closed = true; break
        end
        push!(beta, b)
        V[j + 1] = w / b
        j == m && (V = V[1:(m + 1)])
    end
    k = length(alpha)
    T = SymTridiagonal(alpha, beta[1:(k - 1)])
    F = eigen(T)
    coef = F.vectors * (exp.(t * F.values) .* (F.vectors' * [i == 1 ? 1.0 + 0im : 0.0 + 0im
                                                            for i in 1:k]))
    out = zeros(ComplexF64, length(v))
    for i in 1:k
        out .+= (beta0 * coef[i]) .* V[i]
    end
    # beta[k] is the leftover coupling to V[k+1], the direction the projection discarded. It
    # exists only when the space did NOT close, in which case there is nothing discarded.
    err = (closed || length(beta) < k) ? 0.0 : beta0 * beta[k] * abs(coef[k])
    return out, err
end

"""
`exp(t*A)*v` by Lanczos, `A` symmetric sparse and `t` either imaginary (`-im*dt`, real time) or
real negative (`-dt`, imaginary time). Restarted in substeps so the Krylov space stays small and
the result stays accurate for large `|t|`.

`m` is the Krylov dimension per substep; the residual estimate `beta*|coef[end]|` decides
convergence, and a substep that does not converge is halved.

WHY THE SUBSTEPPING IS NOT OPTIONAL. A degree-`m` Krylov approximation to `exp(t*A)` is only
accurate while `|t| * ||A|| <~ m`; past that it does not degrade gracefully, it returns a
confidently wrong vector. It also stays in the symmetry sector and keeps whatever spatial
symmetry the initial condition had, so the wrong answer looks entirely plausible -- an L=18
Heisenberg domain wall at `t=20` (`||A||*|t| ~ 160`) came back with a smooth, correctly
antisymmetric `<Sz_j>` profile that was off by 7e-02, which reads as a truncation error in the
METHOD being benchmarked rather than a defect in the reference. So the substep count is chosen
up front from `||A||`, and the residual estimate is then checked and the step halved if the
estimate says the choice was optimistic. A reference that cannot detect its own failure is worse
than no reference.
"""
function expv_sparse(A, t::ComplexF64, v::Vector{ComplexF64};
                     m::Int = 30, tol::Float64 = 1e-12)
    beta0 = norm(v)
    (beta0 == 0 || t == 0) && return copy(v)

    total = abs(t)
    dir   = t / total          # unit direction: -im for real time, -1 for imaginary time

    # Cheap DETERMINISTIC upper bound on the spectral radius: for symmetric `A`,
    # rho(A) <= ||A||_inf = max row sum of |A_ij|. Over-estimating only costs substeps;
    # under-estimating would silently reintroduce the bug, so a bound beats an estimate.
    nrmA = opnorm(A, Inf)
    # Keep |t|*||A|| per substep at ~m/4: comfortably inside the radius where a degree-m
    # Krylov space converges, so the halving branch below is a safety net and not the norm.
    step = nrmA > 0 ? min(total, 0.25 * m / nrmA) : total

    w    = copy(v)
    done = 0.0
    while done < total * (1 - 1e-12)
        h = min(step, total - done)
        out, err = _lanczos_expv(A, dir * h, w, m, tol)
        # A substep that does not converge is halved -- and retried, not accepted.
        if err > tol * beta0 && h > 1e-10 * total
            step = h / 2
            continue
        end
        w = out
        done += h
    end
    return w
end

"Repeatedly apply `expv_sparse` in steps of `dt`, calling `obs(step, vector)` after each."
function propagate!(A, v0::Vector{ComplexF64}, tau_step::ComplexF64, nsteps::Int, obs;
                    normalise::Bool = false, m::Int = 30)
    v = copy(v0)
    for s in 1:nsteps
        v = expv_sparse(A, tau_step, v; m = m)
        normalise && (v ./= norm(v))
        obs(s, v)
    end
    return v
end

"`<v| S_j.S_{j+1} |v>` for every bond, matching `bond_energies` on the MPS side."
function bond_energies_dense(v::Vector{ComplexF64}, L::Int,
                             states::Vector{Int}, idx::Dict{Int, Int};
                             J::Float64 = 1.0, delta::Float64 = 1.0)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v))
    out = zeros(Float64, L - 1)
    for j in 1:(L - 1)
        acc = 0.0 + 0im
        for (k, b) in enumerate(states)
            sj, sj1 = bit(b, j), bit(b, j + 1)
            acc += conj(v[k]) * v[k] * J * delta * (sj == sj1 ? 0.25 : -0.25)
            if sj != sj1
                bb = b ⊻ (1 << (j - 1)) ⊻ (1 << j)
                acc += conj(v[idx[bb]]) * v[k] * (J / 2)
            end
        end
        out[j] = real(acc) / nrm
    end
    return out
end

"Lowest eigenvalue by Lanczos -- the ground-state reference. No dense matrix is formed."
function ground_energy_sparse(A; m::Int = 200, tol::Float64 = 1e-12, maxrestart::Int = 60)
    n = size(A, 1)
    v = randn(Float64, n); v ./= norm(v)
    prev = Inf
    for _ in 1:maxrestart
        V = Vector{Vector{Float64}}(undef, m + 1)
        alpha, beta = Float64[], Float64[]
        V[1] = v
        kmax = m
        for j in 1:m
            w = A * V[j]
            a = dot(V[j], w); push!(alpha, a)
            w -= a * V[j]
            j > 1 && (w -= beta[j - 1] * V[j - 1])
            for i in 1:j
                w -= dot(V[i], w) * V[i]
            end
            b = norm(w)
            if b < 1e-12
                kmax = j; break
            end
            push!(beta, b); V[j + 1] = w / b
        end
        k = length(alpha)
        F = eigen(SymTridiagonal(alpha, beta[1:(k - 1)]))
        e0 = F.values[1]
        # restart from the Ritz vector
        v = zeros(Float64, n)
        for i in 1:k
            v .+= F.vectors[i, 1] .* V[i]
        end
        v ./= norm(v)
        abs(prev - e0) < tol && return e0, v
        prev = e0
    end
    return prev, v
end
