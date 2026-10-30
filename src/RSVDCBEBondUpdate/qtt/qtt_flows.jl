# ── full nonlinear stages: BUG roots and the TDVP2 baselines (QTT plan, Tasks 2.3A and 2.4) ──
# A FLOW is the vector field d_t psi = F(t, psi) = pref * (H[psi, t] psi - mu[psi] psi):
#   GPFlow, real time:      pref = -i - gamma, mu = 0 (the model's own mu sits in H0)
#   GPFlow, imaginary time: pref = -1, mu[psi] = <psi, H psi> / <psi, psi> (normalised gradient flow)
#   LinearFlow:             a state-independent H (linear GP, Heston), mu = 0
# H[psi, t] is rebuilt from the state it acts on at EVERY evaluation -- every Radau stage iterate,
# every TDVP substep in the consistent mode. The real-linear Jacobian is
#   DF[d] = pref * ((J1 - mu) d + J2 conj(d)),  J1 = H0 + V_pad + 2 g diag(rho),  J2 = g diag(psi^2)
# (dmu is left out of the imaginary-time Jacobian: a Newton-like solve whose RESIDUAL is exact).
# Everything below runs on dense cores under `:none` (`qtt_cores`), never on a 2^L object.

abstract type QTTFlow end

"""
    GPFlow(m; imaginary = false)

The damped GP flow of `m` in real time, or the normalised imaginary-time gradient flow
`d_tau psi = -(H[psi] - mu[psi]) psi`.
"""
struct GPFlow <: QTTFlow
    m::GPModel
    pref::ComplexF64
    normalized::Bool
end
GPFlow(m::GPModel; imaginary::Bool = false) =
    imaginary ? GPFlow(m, ComplexF64(-1), true) : GPFlow(m, ComplexF64(-1im - m.gamma), false)

"`F = pref H psi` for a fixed operator (cores)."
struct LinearFlow <: QTTFlow
    H::Vector{Array{ComplexF64, 4}}
    pref::ComplexF64
end
LinearFlow(H::AbstractVector, pref::Number = 1.0) =
    LinearFlow([ComplexF64.(w) for w in H], ComplexF64(pref))

"The operator `H[psi, t]` (cores) and its info."
flow_op(fl::GPFlow, A, t) = gp_hamiltonian_op(fl.m, A, t)
flow_op(fl::LinearFlow, A, t) = (fl.H, (chi_rho = 0, rho_err = 0.0, t_rho = 0.0, retried = false,
                                        w = maximum(tt_ranks(fl.H); init = 1)))

"`(J1, J2)` of the real-linear Jacobian (file header); `J2 = nothing` when it vanishes."
function flow_jac_ops(fl::GPFlow, A, t)
    m = fl.m
    Hl = gp_linear_op(m, t)
    m.g == 0 && return Hl, nothing
    rho, _ = gp_density(m, A)
    psi2 = first(tt_compress(tt_hadamard(A, A); tol = m.rho_tol))
    return op_lincomb([(1.0, Hl), (2 * m.g, op_diag(rho))]), op_scale(op_diag(psi2), m.g)
end
flow_jac_ops(fl::LinearFlow, A, t) = (fl.H, nothing)

"A frozen flow: the linear flow of `H[psi_n, t_n]` (the `:frozen_density` / lagged ablation)."
function frozen_flow(fl::QTTFlow, A, t)
    H, _ = flow_op(fl, A, t)
    return LinearFlow(H, fl.pref)
end

# ── dense contractions ──

"`A` contracted with `B` over `ca` / `cb`; free legs of `A` first, then those of `B`."
function _tc(A::AbstractArray, ca::Tuple, B::AbstractArray, cb::Tuple)
    fa = Tuple(d for d in 1:ndims(A) if !(d in ca))
    fb = Tuple(d for d in 1:ndims(B) if !(d in cb))
    na = prod((size(A, d) for d in fa); init = 1)
    nk = prod((size(A, d) for d in ca); init = 1)
    M = reshape(permutedims(A, (fa..., ca...)), na, nk) *
        reshape(permutedims(B, (cb..., fb...)), nk, :)
    return reshape(M, (map(d -> size(A, d), fa)..., map(d -> size(B, d), fb)...))
end

"Left environment step `(bra, w, ket)` through site cores `Bk` (bra), `Hk`, `Ck` (ket)."
function _env_left(E, Bk, Hk, Ck)
    T = _tc(E, (3,), Ck, (1,))                     # (a', b', s', c)
    T = _tc(T, (2, 3), Hk, (1, 3))                 # (a', c, s, b)
    return permutedims(_tc(conj(Bk), (1, 2), T, (1, 3)), (1, 3, 2))    # (a, b, c)
end

"Right environment step, the mirror of `_env_left`."
function _env_right(F, Bk, Hk, Ck)
    T = _tc(Ck, (3,), F, (3,))                     # (c, s', a', b')
    T = _tc(T, (2, 4), Hk, (3, 4))                 # (c, a', b, s)
    return permutedims(_tc(conj(Bk), (2, 3), T, (4, 2)), (1, 3, 2))    # (a, b, c)
end

_unit_env(T = ComplexF64) = ones(T, 1, 1, 1)

"Left environments `E[k]` = sites `< k` (k = 1..L+1) of `<B| H |C>`."
function _env_stack_left(B, H, C)
    E = Vector{Array{ComplexF64, 3}}(undef, length(C) + 1)
    E[1] = _unit_env()
    for k in eachindex(C)
        E[k + 1] = _env_left(E[k], B[k], H[k], C[k])
    end
    return E
end

"Right environments `F[k]` = sites `>= k` (k = 1..L+1)."
function _env_stack_right(B, H, C)
    L = length(C)
    F = Vector{Array{ComplexF64, 3}}(undef, L + 1)
    F[L + 1] = _unit_env()
    for k in L:-1:1
        F[k] = _env_right(F[k + 1], B[k], H[k], C[k])
    end
    return F
end

"`<C| W |C>` by one environment sweep, `O(L chi^3 w)` -- never `W C` compressed."
function _op_expect(W, C)
    E = _unit_env()
    for k in eachindex(C)
        E = _env_left(E, C[k], W[k], C[k])
    end
    return E[1, 1, 1]
end

"Two-site effective action at `(i, i+1)`."
function _eff2(E, Hi, Hj, F, Th)
    X = _tc(E, (3,), Th, (1,))                     # (a, b, t1, t2, e)
    X = _tc(X, (2, 3), Hi, (1, 3))                 # (a, t2, e, s1, b2)
    X = _tc(X, (2, 5), Hj, (3, 1))                 # (a, e, s1, s2, b3)
    return _tc(X, (2, 5), F, (3, 2))               # (a, s1, s2, d)
end

"One-site effective action."
function _eff1(E, Hk, F, M)
    X = _tc(E, (3,), M, (1,))                      # (a, b, t, e)
    X = _tc(X, (2, 3), Hk, (1, 3))                 # (a, e, s, b')
    return _tc(X, (2, 4), F, (3, 2))               # (a, s, d)
end

# ── frames: the root space of a BUG step ──

"""
    QTTFrames(W, Z)

The Galerkin frames of a BUG step as dense cores: left isometries `W[1..c]` `(link, site, bond)` and
right isometries `Z[1..L-c]` for sites `c+1..L` `(bond, site, link)`. A root matrix `Y` (`r_L x
r_R`) is the state `W Y Z`.
"""
struct QTTFrames
    W::Vector{Array{ComplexF64, 3}}
    Z::Vector{Array{ComplexF64, 3}}
end

"Dense frames of a step's `KLFrames` (`cbe_bug_step!(...; frames_report)`)."
function QTTFrames(fr::KLFrames)
    # `W` holds sites 1..c; `Z` has length L - 1 and `Z[j]` (j >= c) absorbs site j + 1
    W = [ComplexF64.(_none_block(fr.W[i])) for i in 1:fr.c]
    Z = [ComplexF64.(_none_block(fr.Z[j])) for j in fr.c:length(fr.Z)]
    return QTTFrames(W, Z)
end

"The TT cores of `W Y Z`."
function frame_state(fr::QTTFrames, Y::AbstractMatrix)
    c = length(fr.W)
    Wc = fr.W[c]
    root = reshape(reshape(Wc, :, size(Wc, 3)) * Y, size(Wc, 1), size(Wc, 2), size(Y, 2))
    return vcat(fr.W[1:(c - 1)], [root], fr.Z)
end

"""
    frame_project(fr, H, C) -> Matrix

`W' H C Z'` for an operator `H` (cores, `nothing` = identity) applied to any TT `C`: the root
matrix of the Galerkin projection, by environments -- `H C` is never formed.
"""
function frame_project(fr::QTTFrames, H, C::AbstractVector)
    c = length(fr.W)
    L = length(C)
    Hc = H === nothing ? op_identity(L; T = ComplexF64) : H
    E = _unit_env()
    for k in 1:c
        E = _env_left(E, fr.W[k], Hc[k], C[k])
    end
    F = _unit_env()
    for k in L:-1:(c + 1)
        F = _env_right(F, fr.Z[k - c], Hc[k], C[k])
    end
    return _tc(E, (2, 3), F, (2, 3))
end

"""
    frame_operator_matrix(fr, H) -> Matrix

The root-space matrix of `W' H Z` (`vec(Y)` order, `Y[a, a2]` with `a` fastest), from the frames'
own environments: `M[(a,a2),(c,c2)] = sum_b E[a,b,c] F[a2,b,c2]`. For the stage preconditioner.
"""
function frame_operator_matrix(fr::QTTFrames, H)
    c = length(fr.W)
    E = _unit_env()
    for k in 1:c
        E = _env_left(E, fr.W[k], H[k], fr.W[k])
    end
    F = _unit_env()
    for k in length(H):-1:(c + 1)
        F = _env_right(F, fr.Z[k - c], H[k], fr.Z[k - c])
    end
    M = _tc(E, (2,), F, (2,))                        # (a, c, a2, c2)
    rl, rr = size(E, 1), size(F, 1)
    return reshape(permutedims(M, (1, 3, 2, 4)), rl * rr, rl * rr)
end

"The reduced vector field `G(Y) = W' F(t, W Y Z) Z'` and `mu`."
function reduced_field(fl::QTTFlow, fr::QTTFrames, Y, t)
    A = frame_state(fr, Y)
    H, info = flow_op(fl, A, t)
    HY = frame_project(fr, H, A)
    mu = fl isa GPFlow && fl.normalized ? real(dot(Y, HY)) / max(real(dot(Y, Y)), eps()) : 0.0
    return fl.pref .* (HY .- mu .* Y), mu, info
end

"The reduced real-linear Jacobian at `Y`, as a closure `V -> DG(Y)[V]`."
function reduced_jacobian(fl::QTTFlow, fr::QTTFrames, Y, t, mu)
    A = frame_state(fr, Y)
    J1, J2 = flow_jac_ops(fl, A, t)
    return function (V)
        AV = frame_state(fr, V)
        out = frame_project(fr, J1, AV) .- mu .* V
        J2 === nothing || (out = out .+ frame_project(fr, J2, tt_conj(AV)))
        return fl.pref .* out
    end
end

# ── implicit Runge-Kutta stage equations ──

"""
    RK_TABLEAUX

The root methods of plan §2.4: `:trapezoid` (order 2, A-stable, not L-stable; stage 1 explicit),
`:radau2` (2-stage Radau IIA, order 3, L-stable), `:radau3` (3-stage Radau IIA, order 5,
L-stable). All three are stiffly accurate, so `S_{n+1}` is the last stage.
"""
const RK_TABLEAUX = let s6 = sqrt(6.0)
    Dict(:trapezoid => (A = [0.0 0.0; 0.5 0.5], c = [0.0, 1.0]),
         :radau2 => (A = [5 / 12 -1 / 12; 3 / 4 1 / 4], c = [1 / 3, 1.0]),
         :radau3 => (A = [(88 - 7s6) / 360 (296 - 169s6) / 1800 (-2 + 3s6) / 225;
                          (296 + 169s6) / 1800 (88 + 7s6) / 360 (-2 - 3s6) / 225;
                          (16 - s6) / 36 (16 + s6) / 36 1 / 9],
                     c = [(4 - s6) / 10, (4 + s6) / 10, 1.0]))
end

"The root method of BUG order `p` (plan §2.4 table)."
qtt_root_for_order(p::Int) = p == 2 ? :trapezoid : p == 3 ? :radau2 : p == 4 ? :radau3 :
    throw(ArgumentError("nonlinear BUG orders are 2, 3, 4; got $p"))

"Restarted GMRES on real vectors (the doubled-real form of a real-linear complex map)."
function _gmres_real(A, b::Vector{Float64}; tol::Float64 = 1e-10, restart::Int = 40,
                     maxiter::Int = 400)
    n = length(b)
    x = zeros(n)
    nb = norm(b)
    nb == 0 && return x, 0, 0.0
    it = 0
    while it < maxiter
        r = b .- A(x)
        beta = norm(r)
        beta <= tol * nb && return x, it, beta / nb
        m = min(restart, maxiter - it)
        V = zeros(n, m + 1)
        Hh = zeros(m + 1, m)
        cs, sn, g = zeros(m), zeros(m), zeros(m + 1)
        V[:, 1] .= r ./ beta
        g[1] = beta
        k = 0
        for j in 1:m
            it += 1
            k = j
            w = A(V[:, j])
            for i in 1:j
                Hh[i, j] = dot(view(V, :, i), w)
                w .-= Hh[i, j] .* view(V, :, i)
            end
            Hh[j + 1, j] = norm(w)
            Hh[j + 1, j] > 0 && (V[:, j + 1] .= w ./ Hh[j + 1, j])
            for i in 1:(j - 1)
                tmp = cs[i] * Hh[i, j] + sn[i] * Hh[i + 1, j]
                Hh[i + 1, j] = -sn[i] * Hh[i, j] + cs[i] * Hh[i + 1, j]
                Hh[i, j] = tmp
            end
            rho = hypot(Hh[j, j], Hh[j + 1, j])
            cs[j], sn[j] = Hh[j, j] / rho, Hh[j + 1, j] / rho
            Hh[j, j], Hh[j + 1, j] = rho, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            abs(g[j + 1]) <= tol * nb && break
        end
        y = UpperTriangular(Hh[1:k, 1:k]) \ g[1:k]
        x .+= V[:, 1:k] * y
        abs(g[k + 1]) <= tol * nb && return x, it, abs(g[k + 1]) / nb
    end
    return x, it, norm(b .- A(x)) / nb
end

_pack(Ys) = vcat((vcat(vec(real(Y)), vec(imag(Y))) for Y in Ys)...)
function _unpack(x::Vector{Float64}, dims)
    out = Matrix{ComplexF64}[]
    o = 0
    for d in dims
        n = prod(d)
        push!(out, reshape(x[(o + 1):(o + n)] .+ 1im .* x[(o + n + 1):(o + 2n)], d))
        o += 2n
    end
    return out
end

"""
Largest stacked stage dimension `s r_L r_R` for which the dense preconditioner is factorised; above
it the stage GMRES runs unpreconditioned (and says so in `stats.precond`).
"""
const _STAGE_PRECOND_MAX = 3072

"""
    rk_stage_solve(fl, fr, S, t, h, method; tol, maxnewton, gmres_tol, Y0, precond) -> (Y, stats)

The nonlinear stage equations `Y_j = S + h sum_l A_jl G(t + c_l h, Y_l)` in the root space, by
Newton-GMRES on the doubled-real unknowns with the reduced real-linear Jacobian. The RESIDUAL is
the full nonlinear one, driven to `tol` (relative to `||S||`). `precond = :auto` right-
preconditions GMRES with `I - h A (x) pref (M - mu)`, `M = W' J1 Z` the frozen LINEAR part of the
Jacobian at the step input (the kinetic/trap/density terms; `conj(d)` is left to GMRES), factorised
once when `s r_L r_R <= _STAGE_PRECOND_MAX` (plan §2.4). `stats` records Newton and GMRES
iterations, the final residual, the field evaluations, the density-build seconds and the
preconditioner used.
"""
function rk_stage_solve(fl::QTTFlow, fr::QTTFrames, S::AbstractMatrix, t::Real, h::Real,
                        method::Symbol; tol::Real = 1e-10, maxnewton::Int = 20,
                        gmres_tol::Real = 1e-3 * tol, gmres_maxiter::Int = 600, Y0 = nothing,
                        precond::Symbol = :auto)
    tab = RK_TABLEAUX[method]
    s = length(tab.c)
    A = tab.A
    Y = Y0 === nothing ? [ComplexF64.(S) for _ in 1:s] : [ComplexF64.(y) for y in Y0]
    dims = [size(S) for _ in 1:s]
    nS = max(norm(S), eps())
    nfield, ngm, trho, res = 0, 0, 0.0, Inf
    n = length(S)
    Pf = nothing
    if precond === :auto && s * n <= _STAGE_PRECOND_MAX
        A0c = frame_state(fr, S)
        J1, _ = flow_jac_ops(fl, A0c, t + h / 2)
        M = frame_operator_matrix(fr, J1)
        mu0 = fl isa GPFlow && fl.normalized ? real(dot(vec(S), M * vec(S))) / max(norm(S)^2, eps()) : 0.0
        Jl = fl.pref .* (M - mu0 * I)
        Pf = lu(Matrix{ComplexF64}(I, s * n, s * n) .- h .* kron(A, Jl))
    end
    applyP(x) = Pf === nothing ? x : begin
        z = vcat((vec(D) for D in _unpack(x, dims))...)
        w = Pf \ z
        _pack([reshape(w[((j - 1) * n + 1):(j * n)], dims[j]) for j in 1:s])
    end
    for it in 0:maxnewton
        Gs, mus = Matrix{ComplexF64}[], Float64[]
        for l in 1:s
            G, mu, info = reduced_field(fl, fr, Y[l], t + tab.c[l] * h)
            push!(Gs, G); push!(mus, mu)
            nfield += 1
            trho += info.t_rho
        end
        R = [Y[j] .- S .- h .* sum(A[j, l] .* Gs[l] for l in 1:s) for j in 1:s]
        res = sqrt(sum(norm(r)^2 for r in R)) / nS
        if res <= tol || it == maxnewton
            return Y, (newton = it, gmres = ngm, residual = res, converged = res <= tol,
                       nfield = nfield, t_rho = trho, mu = mus[end],
                       precond = Pf === nothing ? :none : :linear_part)
        end
        Js = [reduced_jacobian(fl, fr, Y[l], t + tab.c[l] * h, mus[l]) for l in 1:s]
        op = function (x)
            D = _unpack(x, dims)
            JD = [any(!iszero, A[:, l]) ? Js[l](D[l]) : zero(D[l]) for l in 1:s]
            return _pack([D[j] .- h .* sum(A[j, l] .* JD[l] for l in 1:s) for j in 1:s])
        end
        # right preconditioning: solve (op o P^-1) y = -R, then dx = P^-1 y
        dy, gi, _ = _gmres_real(x -> op(applyP(x)), -_pack(R); tol = Float64(gmres_tol),
                                maxiter = gmres_maxiter)
        dx = applyP(dy)
        ngm += gi
        Y = [Y[j] .+ D for (j, D) in enumerate(_unpack(dx, dims))]
    end
end

# ── the nonlinear BUG step ──

"Safety factor on the local-Krylov share of `tol_basis` in `qtt_bug_step!` (plan §1.3)."
const _QTT_KRYLOV_MARGIN = 0.5

"Overwrite `psi` with `src` (same length)."
function _replace_state!(psi::SymMPS, src::SymMPS)
    for i in 1:length(psi)
        psi[i] = src[i]
    end
    psi.center = src.center
    return psi
end

"""
    qtt_bug_step!(psi, fl, t, h; order = 2, root = qtt_root_for_order(order), npc = order - 1,
                  tol_stage = 1e-10, tol_basis, tol_trunc = 1e-8, maxdim = 128, kwargs...)
        -> NamedTuple

One rank-adaptive BUG step of the FULL flow `fl` from `t` to `t + h` (plan §2.3A item 3, mode B):

1. the basis: `cbe_bug_step!` at `order` on the predictor generator `H[psi_n, t_n]` (its DD-CBE,
   local Krylov and order-p frontier closure), the frames taken from `frames_report`;
2. the root: `S_n = <frames|psi_n>` -- the STEP INPUT projected, not a predictor -- and the
   nonlinear `root` stage equations solved with the density rebuilt at every stage iterate
   (`rk_stage_solve`);
3. `npc` basis corrections: the frames are rebuilt with the stage states and their vector fields
   `F(t_j, Y_j)` kept exactly (`keep_states`), and the root re-solved from the same `S_n`;
4. ONE compression of `W Y_s Z` (`truncate_recursive!` at `tol_trunc`, `maxdim`).

The order is a claim to be measured on the whole update (plan V05), never assumed from `root`.
Returns stage statistics, frame widths and timings.
"""
function qtt_bug_step!(psi::SymMPS, fl::QTTFlow, t::Real, h::Real; order::Int = 2,
                       root::Symbol = qtt_root_for_order(order), npc::Int = order - 1,
                       tol_stage::Real = 1e-10, tol_basis::Real = 1e-6, tol_trunc::Real = 1e-8,
                       maxdim::Int = 128, keep_tol::Real = 1e-10,
                       keep::AbstractVector = Any[], kwargs...)
    t0 = time_ns()
    L = length(psi)
    A0 = qtt_cores(psi)
    Hn, hinfo = flow_op(fl, A0, t)
    # The frames' generator is shifted by the state's mean `<H>` (as imaginary time does): the
    # Krylov spaces are unchanged, but a stiff step's root no longer underflows to an exactly zero
    # matrix (V08: h lambda = -1e3 gave e^-1000 = 0 and Telum's root SVD refused it).
    mu_n = real(_op_expect(Hn, A0)) / max(tt_norm(A0)^2, eps())
    Hmpo = qtt_to_mpo(Hn)
    tau = ComplexF64(fl.pref * h)
    # plan §1.3: the adaptive local-Krylov stop overshot its target in 3/54 XX cases, so its share
    # is held to `_QTT_KRYLOV_MARGIN * tol_basis` unless `tol_krylov` is passed (no depth knob)
    tk = Float64(get(kwargs, :tol_krylov, _QTT_KRYLOV_MARGIN * tol_basis))
    kw = Base.structdiff(values(kwargs), NamedTuple{(:tol_krylov,)})
    nfallback = 0
    gkry = nothing
    function frames(keep)
        fref = Ref{Any}(nothing)
        run(ord, ks) = cbe_bug_step!(copy(psi), Hmpo, tau; order = ord,
                                     tol_basis = Float64(tol_basis), tol_trunc = Float64(tol_trunc),
                                     maxdim = maxdim, hermitian = false, frames_report = fref,
                                     keep_states = ks, truncate = false, eshift = mu_n,
                                     tol_krylov = tk, kw...)
        try
            run(order, keep)
        catch err
            # The order-p frontier closure can carry an old frame wider than a near-saturated
            # site's new local space (qt1c: GP L = 12 order 4, "frame holds 36 directions ... local
            # 16", in `_frontier_right`). The equivalent order-p basis -- order-1 frames that keep
            # the global Krylov vectors (tau H)^j psi, j < p, exactly (dissipative plan U10) -- is
            # used instead and counted (`basis_fallback`).
            order > 1 && occursin("cannot exceed its local space", sprint(showerror, err)) ||
                rethrow()
            nfallback += 1
            if gkry === nothing
                gkry = global_krylov_states(Hmpo, psi, order - 1; tol_trunc = Float64(keep_tol),
                                            maxdim = 4 * maxdim)
            end
            run(1, vcat(Any[keep...], Any[gkry...]))
        end
        return QTTFrames(fref[])
    end
    tb0 = time_ns()
    fr = frames(Any[keep...])
    t_basis = (time_ns() - tb0) / 1e9
    stats = Any[]
    Y = nothing
    S = frame_project(fr, nothing, A0)
    tr0 = time_ns()
    Y, st = rk_stage_solve(fl, fr, S, t, h, root; tol = tol_stage)
    t_root = (time_ns() - tr0) / 1e9
    push!(stats, st)
    tab = RK_TABLEAUX[root]
    for _ in 1:npc
        kst = Any[keep...]
        for (j, Yj) in enumerate(Y)
            Aj = frame_state(fr, Yj)
            push!(kst, qtt_mps(first(tt_compress(Aj; tol = keep_tol, maxdim = 4 * maxdim))))
            G, _, _ = reduced_field(fl, fr, Yj, t + tab.c[j] * h)
            Fj = first(tt_compress(frame_state(fr, G); tol = keep_tol, maxdim = 4 * maxdim))
            tt_norm(Fj) > 0 && push!(kst, qtt_mps(Fj))
        end
        tb0 = time_ns()
        fr2 = frames(kst)
        t_basis += (time_ns() - tb0) / 1e9
        S2 = frame_project(fr2, nothing, A0)
        Y0 = [frame_project(fr2, nothing, frame_state(fr, Yj)) for Yj in Y]
        tr0 = time_ns()
        Y, st = rk_stage_solve(fl, fr2, S2, t, h, root; tol = tol_stage, Y0 = Y0)
        t_root += (time_ns() - tr0) / 1e9
        push!(stats, st)
        fr = fr2
    end
    out = qtt_mps(frame_state(fr, Y[end]); center = length(fr.W) + 1)
    _, _, cum, capped = truncate_recursive!(out, length(fr.W) + 1; maxdim = maxdim,
                                            weight_tol = Float64(tol_trunc))
    _replace_state!(psi, out)
    return (stats = stats, converged = all(s.converged for s in stats),
            frame_width = (size(fr.W[end], 3), size(fr.Z[1], 1)), discarded = cum,
            rank_capped = capped, t_basis = t_basis, t_root = t_root, t_rho0 = hinfo.t_rho,
            basis_fallback = nfallback,
            t_step = (time_ns() - t0) / 1e9, chi = maximum(bond_dims(psi); init = 1),
            root = root, order = order)
end

# ── local Krylov exponential on dense arrays ──

"""
    _dense_expv(apply, v, tau; tol = 1e-12, m = 30) -> (w, napply)

`exp(tau A) v` by Arnoldi with the a-posteriori estimate `|h_{m+1,m} [exp(tau H_m)]_{m,1}|`,
halving the substep until it fits `tol` (relative).
"""
function _dense_expv(apply, v::AbstractArray, tau::Number; tol::Real = 1e-12, m::Int = 30)
    nv = norm(v)
    nv == 0 && return copy(v), 0
    nap = 0
    w = ComplexF64.(v)
    rem = ComplexF64(tau)
    dt = rem
    while abs(rem) > 0
        abs(dt) > abs(rem) && (dt = rem)
        beta = norm(w)
        Vs = Array{ComplexF64}[w ./ beta]
        Hm = zeros(ComplexF64, m + 1, m)
        k = m
        happy = false
        for j in 1:m
            u = ComplexF64.(apply(Vs[j]))
            nap += 1
            for i in 1:j
                Hm[i, j] = dot(Vs[i], u)
                u .-= Hm[i, j] .* Vs[i]
            end
            for i in 1:j                                  # one reorthogonalisation pass
                c = dot(Vs[i], u)
                Hm[i, j] += c
                u .-= c .* Vs[i]
            end
            Hm[j + 1, j] = norm(u)
            if abs(Hm[j + 1, j]) <= 1e-14 * max(1.0, opnorm(Hm[1:j, 1:j], 1))
                k = j
                happy = true
                break
            end
            push!(Vs, u ./ Hm[j + 1, j])
        end
        while true
            E = exp(dt .* Hm[1:k, 1:k])
            err = happy ? 0.0 : abs(Hm[k + 1, k] * dt) * abs(E[k, 1])
            if err <= tol || abs(dt) < 1e-12 * abs(tau)
                w = beta .* sum(E[i, 1] .* Vs[i] for i in 1:k)
                rem -= dt
                break
            end
            dt /= 2
        end
    end
    return w, nap
end

# ── TDVP2 on dense cores: lagged and stage-consistent nonlinear density ──

"""
    qtt_tdvp2_step!(psi, fl, t, h; density = :lagged, maxdim = 128, tol_trunc = 1e-10,
                    tol_krylov = 1e-12, tol_sc = 1e-10, maxsc = 12) -> NamedTuple

One symmetric two-site projector-splitting TDVP step (left-to-right then right-to-left, each `h/2`)
of the flow `fl`, with the explicitly BACKWARD one-site centre substeps `exp(-h/2 F_1)`:

- `density = :lagged` (TDVP2_lagged): `H[psi_n, t_n]` held through the step;
- `density = :consistent` (TDVP2_consistent): every substep re-evaluates `H` (and the paddle at the
  substep time) from the TRIAL state, by the self-consistent midpoint iteration
  `x' = exp(dt F[(x + x')/2]) x` to `tol_sc`, environments rebuilt at every iterate.

Records `A_back = ||M_after|| / ||M_before||` for every backward substep (plan §2.5), the
self-consistency iterations and the discarded weight. "TDVP2" names the two-site update, not a
temporal order; the order is measured (V07).
"""
function qtt_tdvp2_step!(psi::SymMPS, fl::QTTFlow, t::Real, h::Real; density::Symbol = :lagged,
                         maxdim::Int = 128, tol_trunc::Real = 1e-10, tol_krylov::Real = 1e-12,
                         tol_sc::Real = 1e-10, maxsc::Int = 12)
    density in (:lagged, :consistent) || throw(ArgumentError(
        "density must be :lagged or :consistent, got :$density"))
    t0 = time_ns()
    C, _ = tt_compress(qtt_cores(psi); tol = 0.0)          # right-canonical, centre at site 1
    L = length(C)
    flow = density === :lagged ? frozen_flow(fl, C, t) : fl
    mu_of(H, X) = fl isa GPFlow && fl.normalized ?
        real(_op_expect(H, X)) / max(tt_norm(X)^2, eps()) : 0.0
    back = Float64[]
    nsc = Int[]
    disc2 = 0.0
    nap = 0
    half = h / 2
    # The whole state with the local block `x` at sites `lo..hi` (one or two sites), factorised
    # EXACTLY (no truncation): what the density of a trial state is built from.
    function assemble(x, lo, hi)
        X = copy(C)
        if lo == hi
            X[lo] = x
        else
            cl, _, _, cr = size(x)
            X[lo] = reshape(x, cl, 2, 2cr)
            X[hi] = reshape(Matrix{ComplexF64}(I, 2cr, 2cr), 2cr, 2, cr)
        end
        return X
    end
    # environments of the sites outside `lo..hi` for the operator H, bra = ket = the state
    function local_op(H, X, lo, hi)
        E = _unit_env()
        for k in 1:(lo - 1)
            E = _env_left(E, X[k], H[k], X[k])
        end
        F = _unit_env()
        for k in L:-1:(hi + 1)
            F = _env_right(F, X[k], H[k], X[k])
        end
        return lo == hi ? (z -> _eff1(E, H[lo], F, z)) : (z -> _eff2(E, H[lo], H[hi], F, z))
    end
    # one local substep `exp(dt F_local) x` at time `ts`, generator from the trial state
    function substep(x, lo, hi, dt, ts)
        X = assemble(x, lo, hi)
        H, _ = flow_op(flow, X, ts)
        mu = mu_of(H, X)
        app = local_op(H, X, lo, hi)
        y, k = _dense_expv(z -> fl.pref .* (app(z) .- mu .* z), x, dt; tol = tol_krylov)
        nap += k
        density === :lagged && (push!(nsc, 0); return y)
        it = 0
        for _ in 1:maxsc
            it += 1
            Xm = assemble((x .+ y) ./ 2, lo, hi)              # the midpoint trial state
            H, _ = flow_op(flow, Xm, ts)
            mu = mu_of(H, Xm)
            app = local_op(H, Xm, lo, hi)
            y2, k = _dense_expv(z -> fl.pref .* (app(z) .- mu .* z), x, dt; tol = tol_krylov)
            nap += k
            done = norm(y2 .- y) <= tol_sc * max(norm(y2), eps())
            y = y2
            done && break
        end
        push!(nsc, it)
        return y
    end
    # the truncating split of a two-site block (the TDVP2 rank control)
    function cut(Th, keep_left::Bool)
        cl, _, _, cr = size(Th)
        F = svd(reshape(Th, 2cl, 2cr))
        nrm = norm(F.S)
        r = _qtt_rank(F.S, tol_trunc * nrm / sqrt(max(L - 1, 1)), maxdim)
        disc2 += sum(abs2, @view(F.S[(r + 1):end]); init = 0.0) / max(nrm^2, eps())
        U, S, Vt = F.U[:, 1:r], F.S[1:r], F.Vt[1:r, :]
        return keep_left ? (reshape(U, cl, 2, r), reshape(Diagonal(S) * Vt, r, 2, cr)) :
                           (reshape(U * Diagonal(S), cl, 2, r), reshape(Vt, r, 2, cr))
    end
    theta(i) = reshape(reshape(C[i], :, size(C[i], 3)) * reshape(C[i + 1], size(C[i + 1], 1), :),
                       size(C[i], 1), 2, 2, size(C[i + 1], 3))
    tmid = t + half / 2
    for i in 1:(L - 1)                                     # left to right, h/2
        Th = substep(theta(i), i, i + 1, half, tmid)
        C[i], M = cut(Th, true)
        if i < L - 1
            C[i + 1] = M                                   # in place for the environments
            n0 = norm(M)
            M = substep(M, i + 1, i + 1, -half, tmid)      # the BACKWARD centre substep
            push!(back, norm(M) / n0)
        end
        C[i + 1] = M
    end
    tmid = t + 3half / 2
    for i in (L - 1):-1:1                                  # right to left, h/2
        Th = substep(theta(i), i, i + 1, half, tmid)
        M, C[i + 1] = cut(Th, false)
        if i > 1
            C[i] = M
            n0 = norm(M)
            M = substep(M, i, i, -half, tmid)
            push!(back, norm(M) / n0)
        end
        C[i] = M
    end
    _replace_state!(psi, qtt_mps(C; center = 1))
    return (A_back = back, A_max = maximum(back; init = 1.0), sc_iters = nsc,
            discarded = sqrt(disc2), applications = nap, density = density,
            t_step = (time_ns() - t0) / 1e9, chi = maximum(bond_dims(psi); init = 1))
end

# ── global TT Krylov exponential and the split-dissipation TDVP control ──

"""
    tt_expv(apply, C, tau; m = 16, tol = 1e-10, maxdim = 256, tol_trunc = 1e-12) -> (TT, info)

`exp(tau A) C` by Arnoldi on TT vectors (every new vector compressed at `tol_trunc`), substeps
halved until the estimate `|h_{m+1,m} tau [exp(tau H_m)]_{m,1}|` fits `tol`. A stable global
reference with no backward substep.
"""
function tt_expv(apply, C::AbstractVector, tau::Number; m::Int = 16, tol::Real = 1e-10,
                 maxdim::Int = 256, tol_trunc::Real = 1e-12)
    rem = ComplexF64(tau)
    dt = rem
    w = [ComplexF64.(c) for c in C]
    nap = 0
    while abs(rem) > 0
        abs(dt) > abs(rem) && (dt = rem)
        beta = tt_norm(w)
        beta == 0 && break
        Vs = Any[tt_scale(w, 1 / beta)]
        Hm = zeros(ComplexF64, m + 1, m)
        k = m
        happy = false
        for j in 1:m
            u = first(tt_compress(apply(Vs[j]); tol = tol_trunc, maxdim = maxdim))
            nap += 1
            for _ in 1:2, i in 1:j
                c = tt_dot(Vs[i], u)
                Hm[i, j] += c
                u = tt_lincomb([(1.0, u), (-c, Vs[i])]; tol = tol_trunc, maxdim = maxdim)
            end
            Hm[j + 1, j] = tt_norm(u)
            if abs(Hm[j + 1, j]) <= 1e-13 * max(1.0, opnorm(Hm[1:j, 1:j], 1))
                k, happy = j, true
                break
            end
            push!(Vs, tt_scale(u, 1 / Hm[j + 1, j]))
        end
        while true
            E = exp(dt .* Hm[1:k, 1:k])
            err = happy ? 0.0 : abs(Hm[k + 1, k] * dt) * abs(E[k, 1])
            if err <= tol || abs(dt) < 1e-12 * abs(tau)
                w = tt_lincomb([(beta * E[i, 1], Vs[i]) for i in 1:k]; tol = tol_trunc,
                               maxdim = maxdim)
                rem -= dt
                break
            end
            dt /= 2
        end
    end
    return w, (applications = nap,)
end

"""
    qtt_split_tdvp_step!(psi, m::GPModel, t, h; kwargs...) -> NamedTuple

The stabilized / split-dissipation TDVP control of plan §2.5: Strang splitting of
`F = -i H - gamma H` into the conservative half `-i H` (two-site TDVP, whose backward substeps are
norm-preserving) between two dissipative half-steps `exp(-gamma h/2 H)` taken by the global TT
Krylov exponential (no backward substep). `H` is `H[psi, t]` at the start of each sub-flow
(lagged within it); labelled `:split_lagged`.
"""
function qtt_split_tdvp_step!(psi::SymMPS, m::GPModel, t::Real, h::Real; maxdim::Int = 128,
                              tol_trunc::Real = 1e-10, tol_krylov::Real = 1e-12)
    t0 = time_ns()
    diss(A, ts) = begin
        H, _ = gp_hamiltonian_op(m, A, ts)
        first(tt_expv(X -> op_apply(H, X), A, -m.gamma * h / 2; maxdim = maxdim,
                      tol = tol_krylov, tol_trunc = tol_trunc))
    end
    A = diss(qtt_cores(psi), t)
    _replace_state!(psi, qtt_mps(A))
    cons = GPFlow(m, ComplexF64(-1im), false)
    r = qtt_tdvp2_step!(psi, cons, t, h; density = :lagged, maxdim = maxdim,
                        tol_trunc = tol_trunc, tol_krylov = tol_krylov)
    A = diss(qtt_cores(psi), t + h / 2)
    _replace_state!(psi, qtt_mps(first(tt_compress(A; tol = tol_trunc, maxdim = maxdim))))
    return merge(r, (density = :split_lagged, t_step = (time_ns() - t0) / 1e9))
end
