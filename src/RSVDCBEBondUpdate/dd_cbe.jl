
# ══ IMPLICIT DD-PROJECTED, MPO-STREAMED, QR-ONLY CBE (cbe_bug_high_order_rsvd_parallel_tasks.md) ══
#
# Task 2: the production basis expansion of `cbe_bug_step!` (`cbe_policy = :dd_qr_streamed`).
# At bond (i, i+1), with left frame A = U0, right frame B = V0 and X = mat[H_eff Theta]:
#
#   K-sweep (left frame grows)              L-sweep (right frame grows)
#   Om~ = P_R^D Om = Om - B'(B Om)          Om~ = P_L^D Om = Om - A(A' Om)
#   Z   = X Om~       (MPO-streamed)        Z   = X' Om~      (MPO-streamed)
#   Y   = P_L^D Z = Z - A(A' Z)             Y   = P_R^D Z = Z - B'(B Z)
#   Y   = Q R  (sectorwise thin QR)         Y   = Q R
#   A_aug = [A, Q]                          B_aug' = [B', Q]
#
# Neither X, an X_mu, G_DD = P_L^D X P_R^D nor a dense projector is formed, and Theta = A S B is
# not formed either: the probe is contracted into B (or A) and the centre core separately. ONE
# side per call, because the two QRs belong to the two half-sweep orientations (plan T2.4).
#
# Task 1.3/1.4: the operator-action interface the CBE sketch, the local Krylov and the root
# share (`apply_two_site_sketch!`, `apply_one_site!`, `apply_root!`), and the per-kernel
# counters every step reports in one typed `StepRecord`.

# ── T1.4: per-kernel counters ─────────────────────────────────────────────────────────────

"""
    KernelCounters

Operator applications, QR ranks, sector additions, seconds and allocated bytes per kernel.
Plain mutable fields so the hot loop records with no allocation. One instance per half-sweep
(they may run concurrently) and one for the root, merged after the join.

⚠ Bytes are `Base.gc_bytes()` differences: process-wide, so under `parallel = :half_sweeps`
each half-sweep's counts include the other thread's allocations in the same interval.
"""
mutable struct KernelCounters
    n_stream::Int          # MPO-streamed two-site actions (one per sketch call, all channels)
    n_stream_cols::Int     # probe columns those actions carried
    n_one_site::Int        # one-site H applications (local Krylov)
    n_root::Int            # zero-site H applications (root solve)
    n_qr::Int              # DD-QR selections
    qr_rank::Int           # directions those QRs kept (numerical rank, summed)
    n_sector_new::Int      # charge sectors a frame gained that it did not hold before
    t_input_dd::Float64    # P^D on the thin input probe
    t_stream::Float64
    t_output_dd::Float64   # P^D on the thin streamed output
    t_qr::Float64
    t_one_site::Float64
    t_root::Float64
    b_input_dd::Int
    b_stream::Int
    b_output_dd::Int
    b_qr::Int
end
KernelCounters() = KernelCounters(0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

function merge_counters!(a::KernelCounters, b::KernelCounters)
    for f in fieldnames(KernelCounters)
        setfield!(a, f, getfield(a, f) + getfield(b, f))
    end
    return a
end

"Run `f()` and return `(result, seconds, bytes)`."
@inline function _measure(f)
    b0 = Base.gc_bytes()
    t0 = time_ns()
    r = f()
    return r, (time_ns() - t0) / 1e9, Base.gc_bytes() - b0
end

# ── T1.3: one operator-action interface ───────────────────────────────────────────────────
#
# Each operator is IMMUTABLE and built per bond visit from the environments of that visit, so a
# stale operator cannot see a later gauge or environment: invalidation is by construction, and
# `version` records which visit built it. The local Krylov and the root share the interface,
# not a tolerance. Outputs go to a caller-held `Ref` because Telum contractions allocate their
# result; there is no in-place block buffer to reuse.

"""
    TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = false, version = 0, counters)

The two-site effective Hamiltonian at bond `(i, i+1)` as a THIN-PROBE action.
`adjoint = false` is `Z = X Om` for a probe in `V0`'s layout `(g, site_r, link_r)`, returning
`(link_l, site_l, g)` (the K-sweep); `adjoint = true` is `X' Om` for a probe in `U0`'s layout,
returned as rows `(g, site_r, link_r)` (the L-sweep). The MPO channels are summed inside the
contraction chain against the SAME probe.
"""
struct TwoSiteSketchOp
    mpo::MPO
    bond::Int
    lenv::MPOLink
    renv::MPOLink
    U0::Any
    V0::Any
    core::Any            # U0*S0 (adjoint = false) or S0*V0 (adjoint = true)
    adjoint::Bool
    symmetry::Symbol
    version::Int
    counters::KernelCounters
end

function TwoSiteSketchOp(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink;
                         adjoint::Bool = false, version::Int = 0,
                         counters::KernelCounters = KernelCounters())
    core = adjoint ? to_concrete(f.S0 * f.V0) : to_concrete(f.U0 * f.S0)
    return TwoSiteSketchOp(mpo, i, lenv, renv, f.U0, f.V0, core, adjoint, symmetry_mode(),
                           version, counters)
end

"""
    apply_two_site_sketch!(Y::Ref, Om, op::TwoSiteSketchOp) -> Y

`Y[] = X Om` (or `X' Om` for `op.adjoint`), unprojected, streamed through the MPO channels.
"""
function apply_two_site_sketch!(Y::Base.RefValue, Om, op::TwoSiteSketchOp)
    op.symmetry === symmetry_mode() || throw(ErrorException(
        "TwoSiteSketchOp was built under $(op.symmetry), the active symmetry is $(symmetry_mode())"))
    Z, dt, db = _measure(() -> op.adjoint ?
        _stream_right(op.U0, op.core, op.mpo, op.bond, op.lenv, op.renv, Om) :
        _stream_left(op.core, op.V0, op.mpo, op.bond, op.lenv, op.renv, Om))
    kc = op.counters
    kc.n_stream += 1
    kc.n_stream_cols += leg_dim(Om, op.adjoint ? 3 : 1)
    kc.t_stream += dt
    kc.b_stream += db
    Y[] = Z
    return Y
end

"""
    OneSiteOp(h, version, counters)

A one-site effective Hamiltonian (`MPOOneSiteH`, possibly `shift_h`-ed) that counts its
applications. `apply_one_site(op, A)` forwards, so every existing caller works unchanged.
"""
struct OneSiteOp{H}
    h::H
    version::Int
    counters::KernelCounters
end

function apply_one_site(op::OneSiteOp, A)
    y, dt, _ = _measure(() -> apply_one_site(op.h, A))
    op.counters.n_one_site += 1
    op.counters.t_one_site += dt
    return y
end

"`Y[] = H1 X` through the shared interface."
apply_one_site!(Y::Base.RefValue, X, op::OneSiteOp) = (Y[] = apply_one_site(op, X); Y)

"""
    RootOp(h, version, counters)

The root (zero-site) generator with counted applications; `apply_zero_site(op, S)` forwards.
"""
struct RootOp{H}
    h::H
    version::Int
    counters::KernelCounters
end

function apply_zero_site(op::RootOp, S)
    y, dt, _ = _measure(() -> apply_zero_site(op.h, S))
    op.counters.n_root += 1
    op.counters.t_root += dt
    return y
end

"`Y[] = H0 S` through the shared interface."
apply_root!(Y::Base.RefValue, S, op::RootOp) = (Y[] = apply_zero_site(op, S); Y)

# ── the streamed contractions ─────────────────────────────────────────────────────────────
#
# The chains of `_fold_sketch_closures` (mpo.jl), with two differences: no projection at the end
# (the caller applies P^D to the thin result), and Theta is never formed -- the probe meets the
# opposite frame first, then the centre-carrying core. Every intermediate is thin in `g`:
# O(d w g D^2 + d^2 w^2 g D) work, the McCulloch-Osborne target.

"`X Om` for `Om` in `V0`'s layout; `K0 = U0 S0`. Legs `(link_l, site_l, g)`."
function _stream_left(K0, V0, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om)
    W1, W2 = mpo[i], mpo[i + 1]
    Omd = Om'                                                 # (g, s_bra_r, bra_r)
    C = if renv.E === nothing
        X = to_concrete!(contract(Omd, (2,), W2, (3,)))       # (g, bra_r, w_mid, s_ket_r, w_r)
        to_concrete(deleteSingleton(X, 5))                   # (g, bra_r, w_mid, s_ket_r)
    else
        X = to_concrete!(contract(Omd, (3,), renv.E, (1,)))   # (g, s_bra_r, w, ket_r)
        to_concrete!(contract(X, (2, 3), W2, (3, 4)))         # (g, ket_r, w_mid, s_ket_r)
    end
    D1 = to_concrete!(contract(V0, (2, 3), C, (4, 2)))        # (bond, g, w_mid)
    D = to_concrete!(contract(K0, (3,), D1, (1,)))            # (ℓ_l, s_l, g, w_mid)
    E = to_concrete!(contract(D, (2, 4), W1, (2, 4)))         # (ℓ_l, g, w_l, s_bra_l)
    if lenv.E === nothing
        X = to_concrete(deleteSingleton(E, 3))               # (ℓ_l, g, s_bra_l)
        return to_concrete(permutedims(X, (1, 3, 2)))        # (ℓ_l, s_bra_l, g)
    end
    X = to_concrete!(contract(E, (1, 3), lenv.E, (3, 2)))     # (g, s_bra_l, bra)
    return _unprime(to_concrete(permutedims(X, (3, 2, 1))), 1)
end

"`X' Om` for `Om` in `U0`'s layout, as rows; `M0 = S0 V0`. Legs `(g, site_r, link_r)`."
function _stream_right(U0, M0, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink, Om)
    W1, W2 = mpo[i], mpo[i + 1]
    Omd = Om'                                                 # (bra_l, s_bra_l, g)
    C = if lenv.E === nothing
        X = to_concrete!(contract(Omd, (2,), W1, (3,)))       # (bra_l, g, w_l, s_ket_l, w_mid)
        to_concrete(deleteSingleton(X, 3))                   # (bra_l, g, s_ket_l, w_mid)
    else
        X = to_concrete!(contract(Omd, (1,), lenv.E, (1,)))   # (s_bra_l, g, w, ket_l)
        Z = to_concrete!(contract(X, (1, 3), W1, (3, 1)))     # (g, ket_l, s_ket_l, w_mid)
        to_concrete(permutedims(Z, (2, 1, 3, 4)))            # (ket_l, g, s_ket_l, w_mid)
    end
    D1 = to_concrete!(contract(U0, (1, 2), C, (1, 3)))        # (bond, g, w_mid)
    D = to_concrete!(contract(M0, (1,), D1, (1,)))            # (s_r, ℓ_r, g, w_mid)
    E = to_concrete!(contract(D, (1, 4), W2, (2, 1)))         # (ℓ_r, g, s_bra_r, w_r)
    if renv.E === nothing
        X = to_concrete(deleteSingleton(E, 4))               # (ℓ_r, g, s_bra_r)
        return to_concrete(permutedims(X, (2, 3, 1)))        # (g, s_bra_r, ℓ_r)
    end
    X = to_concrete!(contract(E, (1, 4), renv.E, (3, 2)))     # (g, s_bra_r, bra_r)
    return _unprime(X, 3)
end

"""
    dd_reference_sketch(f, mpo, i, lenv, renv, Om, side) -> TLArray

TEST/DEBUG ONLY (plan T2.5 "full-DD exact tiny reference"): `P_L^D X P_R^D Om` (`side = :left`)
or `P_R^D X' P_L^D Om` (`:right`) through the MATERIALISED `H Theta` and the project-first
closures of `_sketch_closures`. Never on the production path.
"""
function dd_reference_sketch(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink,
                             Om, side::Symbol)
    skl, skr = _sketch_closures(f, () -> apply_h_two_site(frame_theta(f), mpo, i, lenv, renv))
    return side === :left ? skl(perp_component_right(f.V0, Om)) :
                            skr(perp_component(f.U0, Om))
end

# ── T2.3: probes in the discarded space only ──────────────────────────────────────────────

"""
    _dd_probe(frame, side, w; rng, boost, exact, kc) -> P^D Om, or nothing

A sector-graded Gaussian probe in `frame`'s layout, drawn ONLY in the frame's discarded space:
columns are allotted in proportion to each fused sector's discarded dimension `dfull - dkept`
(so a sector with no kept multiplicity gets its full share, and a sector the frame fills gets
none), then the frame is projected out of the thin probe. `exact = true` projects the complete
fused basis instead (the full-DD reference width). `nothing` when there is no discarded space:
then `P^D = 0` and no DD direction exists through this side.

Unlike `sector_graded_sketch(...; comp_ratio = 1.0)`, a short complement is NOT topped up from
the kept space: a kept-space column has zero DD action and would only carry a DK contribution.
"""
function _dd_probe(frame, side::Symbol, w::Int; rng::AbstractRNG, boost = nothing,
                   exact::Bool = false, kc::KernelCounters = KernelCounters())
    a, b = side === :left ? (1, 2) : (2, 3)
    F = fusion_basis(frame, a, b; tag = "cbeG")
    dfull, dT = _frame_sector_dims(frame, side, F, "_dd_probe")
    dC = Dict(q => max(dfull[q] - dT[q], 0) for q in keys(dfull))
    sum(values(dC); init = 0) == 0 && return nothing
    R = if exact
        F
    else
        cols = _alloc_columns(dC, w)
        boost === nothing || _boost!(cols, boost, dC)
        _trim_per_sector(_randomize_like(F, rng), 3, cols)
    end
    R === nothing && return nothing
    Om = side === :left ? R : to_concrete(permutedims(R, (3, 1, 2)))
    perp = side === :left ? perp_component : perp_component_right
    P, dt, db = _measure(() -> perp(frame, Om))
    kc.t_input_dd += dt
    kc.b_input_dd += db
    return norm(P) > 0 ? P : nothing
end

# ── T2.4: sectorwise thin Householder QR ──────────────────────────────────────────────────

const _TAG_DD_L, _TAG_DD_R = "cbeQ,L", "cbeQ,R"

"Block `rmt` as a `(rows, d_fl)` matrix: every leg but `fl` fused into rows, `fl` the columns."
function _block_matrix(rmt::AbstractArray, perm, rows::Int, dfl::Int)
    return reshape(permutedims(rmt, perm), rows, dfl)
end

"""
    _thin_qr(Y, fl, ref, tag, thr) -> Q or nothing

Orthonormal basis of the span of `Y`'s leg-`fl` "columns" (fl = 3: columns of the left
matricisation; fl = 1: rows of the right one), by column-pivoted Householder QR per charge
sector of leg `fl`. A direction is kept when its pivot `|R_kk|` exceeds `thr` -- an absolute,
machine-precision-scaled linear-dependence floor, not a physics cutoff; nothing is ranked.

The new leg is built in `ref`'s arrow convention (`align_charge`), so `[frame | Q]` direct-sums
directly. Abelian symmetries only (the block of a sector is then a plain matrix);
callers use `_thin_orth_nonabelian` otherwise.
"""
function _thin_qr(Y, fl::Int, ref::TLIndex, tag::AbstractString, thr::Float64)
    QD = length(Y.inds)
    syms = symm(Y)
    ydir = Y.inds[fl].dir
    others = Tuple(l for l in 1:QD if l != fl)
    groups = Dict{Any, Vector{Int}}()
    order = Any[]
    for k in eachindex(Y.qlabels)
        Y.isdefined[k] || continue
        q = Y.qlabels[k][fl]
        haskey(groups, q) || (groups[q] = Int[]; push!(order, q))
        push!(groups[q], k)
    end
    newq = similar(Y.qlabels, 0)
    newR = similar(Y.RMTs, 0)
    keep = Int[]
    space = similar(Y.spaces[fl], 0)
    for q in order
        ks = groups[q]
        rmt1 = Y.RMTs[ks[1]]
        nd = ndims(rmt1)
        all(size(rmt1, l) == 1 for l in (QD + 1):nd) || throw(ArgumentError(
            "_thin_qr: sector multiplicity axes are not trivial -- non-abelian blocks"))
        perm = (others..., fl, ((QD + 1):nd)...)
        dfl = size(rmt1, fl)
        mats = [_block_matrix(Y.RMTs[k], perm, prod(size(Y.RMTs[k], l) for l in others), dfl)
                for k in ks]
        M = reduce(vcat, mats)
        size(M, 1) == 0 && continue
        F = qr(M, ColumnNorm())
        r = count(x -> abs(x) > thr, diag(F.R))
        r == 0 && continue
        Qm = Matrix(F.Q)[:, 1:r]
        qn = align_charge(syms, q, ydir, ref.dir)
        push!(space, (qn, r))
        row = 0
        for (k, Mk) in zip(ks, mats)
            nr = size(Mk, 1)
            sz = size(Y.RMTs[k])
            blk = reshape(Qm[(row + 1):(row + nr), :],
                          (map(l -> sz[l], others)..., r, sz[(QD + 1):nd]...))
            row += nr
            push!(keep, k)
            push!(newR, permutedims(blk, invperm(collect(perm))))
            lab = Y.qlabels[k]
            push!(newq, ntuple(l -> l == fl ? qn : lab[l], QD))
        end
    end
    isempty(keep) && return nothing
    wd, wi = Telum._copy_wmat_storage(Y, keep; deep = true)
    inds = ntuple(l -> l == fl ? TLIndex(tag, ref.dir, ref.plev, 0, ref.dual) : Y.inds[l], QD)
    spaces = ntuple(l -> l == fl ? space : copy(Y.spaces[l]), QD)
    return TLArray(syms, newq, wd, wi, newR, inds, spaces)
end

"""
    _dd_select(frame, Z, side, kc) -> (Q, backend)

Output DD projection and the two-pass thin QR of plan T2.4: `Y = P^D Z`, `Q1 = qr(Y)` at the
floor `100 eps max(||Z||, ||Y||)` (the cancellation noise `P^D` leaves when most of `Z` lies in
the frame), then `Q = qr(P^D Q1)` -- reorthogonalisation against the old frame, "twice is
enough". Returns `nothing` when `Y` is numerically zero. Under SU(2) the sector blocks are not
plain matrices, so the existing numerical-rank SVD orthonormalisation (`_preselect_*` at
`stol = 0`, no ranking) is used and reported as backend `:svd_orth`, never silently.
"""
function _dd_select(frame, Z, side::Symbol, kc::KernelCounters)
    left = side === :left
    if symmetry_mode() === :SU2
        (Q, _), dt, db = _measure(() -> left ? _preselect_left(frame, Z, 0.0) :
                                               _preselect_right(frame, Z, 0.0))
        kc.t_qr += dt
        kc.b_qr += db
        kc.n_qr += 1
        Q === nothing || (kc.qr_rank += leg_dim(Q, left ? 3 : 1))
        return (Q, :svd_orth)
    end
    fl = left ? 3 : 1
    perp = left ? perp_component : perp_component_right
    Y, dt, db = _measure(() -> perp(frame, Z))
    kc.t_output_dd += dt
    kc.b_output_dd += db
    nz, ny = norm(Z), norm(Y)
    ny <= _CBE_EPS * max(nz, 1.0) && return (nothing, :householder)
    ref = frame.inds[fl]
    tag = left ? _TAG_DD_L : _TAG_DD_R
    Q, dt, db = _measure(() -> begin
        Q1 = _thin_qr(Y, fl, ref, tag, _CBE_EPS * max(nz, ny))
        Q1 === nothing ? nothing : _thin_qr(perp(frame, Q1), fl, ref, tag, _CBE_EPS)
    end)
    kc.t_qr += dt
    kc.b_qr += db
    kc.n_qr += 1
    Q === nothing || (kc.qr_rank += leg_dim(Q, fl))
    return (Q, :householder)
end

"The DD-QR backend this symmetry uses (`StepRecord.qr_backend`)."
_qr_backend() = symmetry_mode() === :SU2 ? :svd_orth : :householder

# ── T2.5: one side of the bond expansion, by policy ───────────────────────────────────────

"""
The CBE selection policies of `cbe_bug_step!` (plan T2.5). `:dd_qr_streamed` is production;
`:one_sided_qr_streamed` drops ONLY the opposite-leg projector (benchmark ablation, plan P05);
`:svd_select` is the original two-sided SVD selection of `cbe_expand` (legacy path, ablation).
"""
const CBE_POLICIES = (:dd_qr_streamed, :one_sided_qr_streamed, :svd_select)

"""
    dd_cbe_expand(f, mpo, i, lenv, renv, side; policy, alloc, ledger, workcap, growth,
                  thin_search, exact, rng, kc) -> CBEExpansion

Grow ONE frame at bond `(i, i+1)` -- `U0` for `side = :left` (the K-sweep), `V0` for
`:right` (the L-sweep) -- by DD-projected, MPO-streamed, QR-only CBE; the other frame is
returned unchanged. Rank comes from the numerical rank of the QR; the probe width from the
same start width and Test B holdout loop as `cbe_expand` (`_certified_range`), here with
DD-projected probes and holdouts (`alloc === nothing`: the numerical-rank stop alone).
`policy = :one_sided_qr_streamed` draws unprojected probes and keeps everything else.
The retained directions are bounded by the local room and the work ceiling (a resource bound,
reported as `workcap_reached`).
"""
function dd_cbe_expand(f::BondFrame, mpo::MPO, i::Int, lenv::MPOLink, renv::MPOLink,
                       side::Symbol; policy::Symbol = :dd_qr_streamed,
                       alloc::Union{Nothing, BasisAllocation} = nothing,
                       ledger::Union{Nothing, AdaptiveLedger} = nothing,
                       workcap::Int = 0, growth::Float64 = ADAPTIVE_GROWTH,
                       thin_search::Bool = true, exact::Bool = false,
                       rng::AbstractRNG = MersenneTwister(0x5EED),
                       kc::KernelCounters = KernelCounters(), version::Int = 0)
    policy in (:dd_qr_streamed, :one_sided_qr_streamed) || throw(ArgumentError(
        "dd_cbe_expand runs :dd_qr_streamed or :one_sided_qr_streamed, got $policy"))
    side in (:left, :right) || throw(ArgumentError("side must be :left or :right"))
    left = side === :left
    U0, V0 = f.U0, f.V0
    r = f.old_rank
    frame, opp = left ? (U0, V0) : (V0, U0)
    oside = left ? :right : :left
    fl, wleg = left ? (3, 1) : (1, 3)
    none() = CBEExpansion(U0, V0, 0, 0, 0.0, 0.0)

    # The fused local dimension, as `cbe_expand` reads it (SU(2) counts multiplets).
    fused(t, a, b) = symmetry_mode() === :SU2 ?
        sum(d for (_, d) in reachable_sectors(t, a, b); init = 0) : leg_dim(t, a) * leg_dim(t, b)
    full_self = left ? fused(U0, 1, 2) : fused(V0, 2, 3)
    full_opp  = left ? fused(V0, 2, 3) : fused(U0, 1, 2)
    cap = full_self - r
    wbind = false
    if alloc !== nothing && workcap > 0
        wk = max(workcap - r, 0)
        wbind = wk < cap
        cap = min(cap, wk)
    end
    cap <= 0 && return none()

    dmax = max(leg_dim(U0, 1), r, leg_dim(V0, 3))
    w0 = ceil(Int, 1.2 * max(ceil(Int, growth * dmax) - r, 1))
    wcap = exact ? w0 : full_opp
    (alloc !== nothing && workcap > 0 && !exact) && (wcap = min(wcap, workcap))

    dd = policy === :dd_qr_streamed
    draw(w, b) = exact ?
        (dd ? _dd_probe(opp, oside, w; rng = rng, exact = true, kc = kc) :
              full_local_basis(opp, oside)) :
        (dd ? _dd_probe(opp, oside, w; rng = rng, boost = b, kc = kc) :
              sector_graded_sketch(opp, oside, w; comp_ratio = nothing, rng = rng, boost = b))
    op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left, version = version,
                         counters = kc)
    out = Ref{Any}(nothing)
    sketch(Om) = (apply_two_site_sketch!(out, Om, op); out[])
    project = left ? (X -> perp_component(U0, X)) : (X -> perp_component_right(V0, X))
    select(Om) = (first(_dd_select(frame, sketch(Om), side, kc)), 0.0)

    Q, _, n = _certified_range(draw, wleg, select, fl, project, sketch, side, w0, wcap, cap,
                               exact ? nothing : alloc, ledger; thin = thin_search)
    Q === nothing && return none()
    if n > cap
        Q = _trim_total(Q, fl, cap)
        Q === nothing && return none()
    end
    ledger === nothing || (wbind && n >= cap && (ledger.workcap_reached = true))
    # Insurance only: the QR floor is per sector and absolute, so roundoff in a saturated sector
    # does not become a column the way a globally relative SVD cut lets it (`_guarded_block`).
    Q = _cap_to_local_room(frame, Q, side)
    Q === nothing && return none()
    nq = leg_dim(Q, fl)
    kc.n_sector_new += count(((q, _),) -> sector_dim(frame, fl, q) == 0, Q.spaces[fl])
    ledger === nothing || (ledger.n_selected += nq)
    grown = to_concrete(oplus([frame, to_concrete(setitag(Q, fl, frame.inds[fl].itags))], (fl,)))
    return left ?
        CBEExpansion(_check_frame(grown, :left, "dd_cbe_expand U_ex"), V0, nq, 0, 0.0, 0.0) :
        CBEExpansion(U0, _check_frame(grown, :right, "dd_cbe_expand V_ex"), 0, nq, 0.0, 0.0)
end

# ── T1.4: the typed step record ───────────────────────────────────────────────────────────

"""
    StepRecord

One `cbe_bug_step!`'s account, typed and allocation-light (plan T1.4): which algorithm ran
(`policy`, `qr_backend`, `closure_policy` for the order > 1 frontier sweeps, `order`,
`parallel`), the kernel counters, the ranks `D_old` (largest bond before), `D_stage` (largest
trial-space frame), `D_final` (after compression), root iterations, total seconds and bytes,
the process `peak_rss` (`Sys.maxrss()`, a high-water mark, not per step), and `limits`: which
of `:basis_limited`, `:root_limited`, `:truncation_limited`, `:work_limited`, `:rank_limited`
held the step back (empty when the budget held; `[:legacy]` on the legacy path, which makes no
claim).
"""
struct StepRecord
    policy::Symbol
    qr_backend::Symbol
    closure_policy::Symbol
    order::Int
    parallel::Symbol
    counters::KernelCounters
    D_old::Int
    D_stage::Int
    D_final::Int
    root_iters::Int
    t_step::Float64
    bytes_step::Int
    peak_rss::Int
    limits::Vector{Symbol}
end

"Plan invariant 4: the separate limit statuses, read off the ledger's failed checks."
function _step_limits(ledger)
    ledger === nothing && return [:legacy]
    lim = Symbol[]
    any(u -> u in (:sample, :tail, :krylov, :closure, :projection), ledger.unmet) &&
        push!(lim, :basis_limited)
    :expv in ledger.unmet && push!(lim, :root_limited)
    :trunc in ledger.unmet && push!(lim, :truncation_limited)
    ledger.maxdim_reached && push!(lim, :rank_limited)
    ledger.workcap_reached && push!(lim, :work_limited)
    return lim
end

"""
    _resolve_parallel(p) -> Bool

`parallel` of the plan's API: `:serial`/`false`, `:half_sweeps`/`true`. `:full` is the separate
parallel K/L/S integrator of plan Task 6 and is refused until it exists and passes its tests.
"""
function _resolve_parallel(p)
    p isa Bool && return p
    p === :serial && return false
    p === :half_sweeps && return true
    p === :full && throw(ArgumentError(
        "parallel = :full is the parallel K/L/S integrator (plan Task 6), not implemented yet; " *
        "use :half_sweeps or :serial"))
    throw(ArgumentError("parallel must be :serial, :half_sweeps or a Bool, got $p"))
end
