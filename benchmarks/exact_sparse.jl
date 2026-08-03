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
