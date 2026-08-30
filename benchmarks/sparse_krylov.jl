# EXACT GROUND STATE AND EXACT REAL-TIME PROPAGATION IN THE S^z = 0 SECTOR, BY SPARSE KRYLOV.
#
# ⛔ THIS IS WHAT MAKES N = 20 A CERTIFIED SIZE FOR THE WHOLE arXiv:2209.00739 CAMPAIGN, ground
# state AND spectrum. The sector is `C(20,10) = 184756` states -- a dense `eigen` is 273 TB and
# out of reach, but a VECTOR is 1.5 MB (real) / 3 MB (complex) and `H` has ~30 nonzeros per row.
# So both `|Om>` and `e^{-iHt} S^z_c |Om>` are affordable EXACTLY, and every MPS row in the
# campaign -- every integrator, every knob setting -- can be scored against truth rather than
# against another MPS run.
#
# That distinction is the whole point. A knob study scored against "our own best run" answers
# "which setting agrees with our other settings", which is not the question. At 12 sites the
# ground-state figure had ED; the SPECTRUM did not, and its integrator comparison could only
# report that the three arms agreed with each other to 1e-11. Here it can report the error.
#
# ⛔ S^z IS DIAGONAL IN THIS BASIS, which is why the correlator costs nothing beyond the
# propagation: `S^z_i |b> = (+-1/2) |b>` reading bit `i-1` of the pattern. No operator is ever
# built.
#
# ⚠ THE BIT CONVENTION IS THE ONE IN `exact_sparse.jl`: site `j` at bit `j-1`, SET means UP. It is
# MIRRORED relative to `dense_state`, and the composition (reverse sites + flip up/down) is a
# symmetry of the open Heisenberg chain -- so Neel, dimer and domain-wall states are all fixed
# points of the difference and hide it. `S^z_{j0}|GS>` is the first object that does NOT hide it.

using LinearAlgebra, SparseArrays, Random

"""
    sparse_ground(H, n; m, restarts, tol) -> (E0, vec, resid, E1_est, iters)

Lowest eigenpair of a real symmetric sparse `H` by restarted Lanczos with FULL
reorthogonalisation, returning the true residual `||H x - E0 x||`.

⛔ THE RESIDUAL IS RETURNED SO THE CALLER CAN ASSERT IT, and every caller must. A DMRG arm is
variational (above the truth); if the reference is also above the truth by an unknown amount, the
deviation column is a difference of two upper bounds and can be either sign. A variational energy
BELOW this reference is never a good run -- it is proof the reference is unconverged, or that the
two model definitions disagree.

⛔ FULL REORTHOGONALISATION IS NOT OPTIONAL AT THIS SIZE. Plain three-term Lanczos loses
orthogonality after ~30 steps on a 184756-dimensional problem and returns GHOST copies of the
lowest eigenvalue, which look exactly like a converged degenerate ground state.

⚠ `E1_est` IS AN ESTIMATE AND IS LABELLED ONE. The second Ritz value of a run targeting the first
converges much later. It is reported because the J2 = 0.5 freeze on the 12-site cylinder tracked
DEGENERACY (gap 1.5e-14), so a near-zero gap is a warning worth having. It is not asserted.
"""
function sparse_ground(H::SparseMatrixCSC{Float64, Int}, n::Int;
                       m::Int = 150, restarts::Int = 30, tol::Float64 = 1e-11)
    rng = MersenneTwister(0xC0FFEE)
    x = randn(rng, n); x ./= norm(x)
    theta = Inf; E1 = Inf; iters = 0
    V = Matrix{Float64}(undef, n, m)

    for _ in 1:restarts
        V[:, 1] = x
        alpha = Float64[]; beta = Float64[]
        k = m
        for j in 1:m
            w = H * view(V, :, j)
            iters += 1
            push!(alpha, dot(view(V, :, j), w))
            for _pass in 1:2, i in 1:j
                w .-= dot(view(V, :, i), w) .* view(V, :, i)
            end
            b = norm(w)
            if j < m
                if b < 1e-13                  # genuine invariant subspace, not convergence
                    k = j
                    break
                end
                push!(beta, b)
                V[:, j + 1] = w ./ b
            end
        end
        T = SymTridiagonal(alpha[1:k], beta[1:(k - 1)])
        F = eigen(T)
        theta = F.values[1]
        E1 = length(F.values) >= 2 ? F.values[2] : Inf
        x = view(V, :, 1:k) * F.vectors[:, 1]
        x ./= norm(x)
        r = norm(H * x - theta * x)
        r < tol && return (theta, x, r, E1, iters)
    end
    return (theta, x, norm(H * x - theta * x), E1, iters)
end

"""
    sparse_expv(H, v, tau; m, tol) -> (w, applications)

`exp(tau*H) * v` for a real symmetric sparse `H` and a complex `v`, by Lanczos with full
reorthogonalisation and Saad's stopping estimate.

⛔ THE STOPPING TEST IS THE CONTRIBUTION `beta_m * |[exp(tau*T_m)]_{m,1}|`, NOT `beta` alone --
the same rule `cbe_bug_step!` uses and for the same reason: on a generic `H` the recursion never
breaks down, so a `beta` threshold never fires and the basis always runs to the cap. What decides
whether a vector matters is how much weight the propagator puts on it.

⚠ `H` REAL SYMMETRIC BUT `v` COMPLEX. The Lanczos coefficients stay real (they are quadratic
forms of a real operator) while the basis is complex; splitting into real and imaginary parts and
running twice would be equivalent but doubles the matvecs, which are the cost.
"""
function sparse_expv(H::SparseMatrixCSC{Float64, Int}, v::Vector{ComplexF64},
                     tau::ComplexF64; m::Int = 30, tol::Float64 = 1e-12)
    b0 = norm(v)
    b0 == 0 && return (copy(v), 0)
    V = Vector{Vector{ComplexF64}}()
    push!(V, v ./ b0)
    alpha = Float64[]; beta = Float64[]
    apps = 0
    k = m
    for j in 1:m
        w = H * V[j]
        apps += 1
        a = real(dot(V[j], w))
        push!(alpha, a)
        for _pass in 1:2, i in 1:j
            w .-= dot(V[i], w) .* V[i]
        end
        bj = norm(w)
        if j >= 2
            T = SymTridiagonal(alpha[1:j], beta[1:(j - 1)])
            E = exp(tau * Matrix(T))
            if bj * abs(E[j, 1]) < tol
                k = j
                break
            end
        end
        if j == m || bj < 1e-14
            k = j
            break
        end
        push!(beta, bj)
        push!(V, w ./ bj)
    end
    T = SymTridiagonal(alpha[1:k], beta[1:(k - 1)])
    E = exp(tau * Matrix(T))
    out = zeros(ComplexF64, length(v))
    for j in 1:k
        out .+= (b0 * E[j, 1]) .* V[j]
    end
    return (out, apps)
end

"""
    sz_diag(states, i) -> Vector{Float64}

The diagonal of `S^z_i` in the `S^z = 0` basis: `+1/2` where site `i` is up, `-1/2` where down.
Site `i` sits at bit `i-1` and a SET bit means UP -- the convention of `exact_sparse.jl`.
"""
sz_diag(states::Vector{Int}, i::Int) =
    [((b >> (i - 1)) & 1) == 1 ? 0.5 : -0.5 for b in states]

"""
    exact_correlator(H, states, Om, E0, c, L, ts) -> Matrix{ComplexF64}

`G[i, n] = e^{i E0 t_n} <Om| S^z_i e^{-i H t_n} S^z_c |Om>` -- the paper's Eq. (6), exactly.

⛔ THE SAME PHASE REMOVAL AS THE MPS PATH. Without `e^{i E0 t}` every `G` winds at `E0` and the
whole spectrum is rigidly shifted; the two paths must apply it identically or the comparison
measures the phase convention.

⛔ THE TIME GRID IS STEPPED, NOT RE-EXPONENTIATED FROM `t = 0`. `exp(-iH t_n) v` computed afresh
at each `t_n` needs a Krylov space that grows with `t_n` and silently loses accuracy at large `t`;
stepping by `dt` keeps every exponential small. This mirrors what the integrators do, so the
comparison is like for like.
"""
function exact_correlator(H::SparseMatrixCSC{Float64, Int}, states::Vector{Int},
                          Om::Vector{Float64}, E0::Float64, c::Int, L::Int,
                          ts::Vector{Float64}; m::Int = 30, tol::Float64 = 1e-12)
    szc = sz_diag(states, c)
    phi = ComplexF64.(szc .* Om)
    bras = [sz_diag(states, i) .* Om for i in 1:L]
    nt = length(ts)
    G = zeros(ComplexF64, L, nt)
    apps = 0
    for i in 1:L
        G[i, 1] = dot(bras[i], phi)
    end
    for n in 2:nt
        dt = ts[n] - ts[n - 1]
        phi, a = sparse_expv(H, phi, ComplexF64(-im * dt); m = m, tol = tol)
        apps += a
        ph = cis(E0 * ts[n])
        for i in 1:L
            G[i, n] = ph * dot(bras[i], phi)
        end
    end
    return G, apps
end
