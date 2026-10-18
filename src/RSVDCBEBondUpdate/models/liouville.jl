# SYMMETRY-AWARE VECTORISED DENSITY MATRICES (imaginary-time / open-system plan, Task 2).
#
# `|rho>> = vec(rho)` as an MPS over LIOUVILLE SITES: one site per physical site, local dimension
# `d^2`, basis `|a, b>> = |a><b|` with index `(a - 1) d + b` (the ket is the more significant
# index), so a one-site superoperator `rho -> A rho B` is `kron(A, transpose(B))` and the
# dissipator of a jump `J` is `kron(J, conj(J)) - kron(J'J, I)/2 - kron(I, transpose(J'J))/2`.
#
# THE CHARGE RULES ARE EXPLICIT, FOR KET AND BRA. A physical local space carries an integer charge
# per state, `q[a]` (2 S^z, a particle number, ...). The Liouville basis state `|a,b>>` is labelled
#
#   :weak     q[a] - q[b]                   U(1)        the Liouvillian symmetry of any generator
#                                                       whose H and jumps each shift the number by
#                                                       a fixed amount (boundary drive, loss)
#   :strong   (q[a], q[b])                  U(1)xU(1)   only when H and EVERY jump commute with the
#                                                       number; rho then stays in one (N, N) block
#   :none     0                             one dense sector, the same tensors without grading
#
# Every superoperator is built from its dense `d^2 x d^2` matrix, and the builder REFUSES one whose
# nonzero entries do not all shift the label by the same amount (`liouville_op`): that is the
# charge-covariance check the plan asks for before sectors are enabled, done per term.
#
# ⛔ THE STORED TENSOR IS THE TRANSPOSE OF THE APPLIED MATRIX. `pair_mpo` (and `site_apply!`-style
# contraction of an operator's leg 1 with the site leg) applies the TRANSPOSE of a local operator's
# matrix -- MEASURED on the fused d = 4 chain, where it swapped gain and loss and froze the
# boundary-driven state into an exact fixed point (`benchmarks/fused_common.jl`, the
# `jump_transpose` note). Here that is made a rule rather than a flag: `liouville_op(ls, M)` returns
# the tensor that the MPO and `mpdo_apply!` apply AS `M`. Tests OS-U2/OS-U6 pin it against dense
# `exp(t L)`.
#
# THE CONTRACTIONS (trace, local expectation values, reduced density matrices, the dense matrix of a
# small state) WALK THE ABELIAN BLOCKS DIRECTLY (`_mpdo_walk`): an environment per link label,
# extended site by site through each block's RMT. No identity superket MPS is needed -- under
# `:strong` the trace functional `<<I|` spans every `(N, N)` block and is not a single-charge MPS,
# but contracting it block by block is exact for any grading.

using LurCGT: U1
using Telum: TLArray, TLIndex, _localspace_cgt_fields

"""
    LiouvilleSpace

The `d^2`-dimensional Liouville site for a physical space of dimension `d` with charges `q`, under
grading `sym` (`:weak`, `:strong`, `:none`). `labels[g]` is the sector of Liouville basis state `g`,
`members[label]` the basis states of a sector in increasing order (their order inside the sector's
block), `pos[g]` the position of `g` inside its sector. `I` is the identity superoperator tensor.
"""
struct LiouvilleSpace
    d::Int
    sym::Symbol
    q::Vector{Int}
    symm::Tuple
    labels::Vector{Any}
    sectors::Vector{Any}            # [(label, dim)] in sorted label order -- the leg space
    members::Dict{Any, Vector{Int}}
    pos::Vector{Int}
    I::Any
end

"Liouville basis index of `|a><b|`."
liouville_index(d::Int, a::Int, b::Int) = (a - 1) * d + b

"(a, b) of Liouville basis index `g`."
liouville_pair(d::Int, g::Int) = (div(g - 1, d) + 1, mod(g - 1, d) + 1)

_lv_label(sym::Symbol, qa::Int, qb::Int) =
    sym === :weak ? ((qa - qb,),) : sym === :strong ? ((qa,), (qb,)) : ((0,),)

_lv_diff(lo, li) = Tuple((lo[k][1] - li[k][1],) for k in eachindex(lo))
_lv_add(l, dl) = Tuple((l[k][1] + dl[k][1],) for k in eachindex(l))
_lv_iszero(dl) = all(x -> x[1] == 0, dl)

"""
    liouville_space(q; sym = :weak) -> LiouvilleSpace

The Liouville site for a physical space whose state `a` carries integer charge `q[a]`
(`length(q) = d`). See the file header for the three gradings.
"""
function liouville_space(q::AbstractVector{<:Integer}; sym::Symbol = :weak)
    sym in (:weak, :strong, :none) || throw(ArgumentError(
        "liouville_space: sym must be :weak, :strong or :none, got :$sym"))
    d = length(q)
    d >= 1 || throw(ArgumentError("liouville_space needs at least one physical state"))
    labels = Any[_lv_label(sym, Int(q[a]), Int(q[b])) for a in 1:d for b in 1:d]
    members = Dict{Any, Vector{Int}}()
    for g in 1:(d^2)
        push!(get!(members, labels[g], Int[]), g)
    end
    keys_sorted = sort(collect(keys(members)))
    sectors = Any[(l, length(members[l])) for l in keys_sorted]
    pos = zeros(Int, d^2)
    for (l, gs) in members, (k, g) in pairs(gs)
        pos[g] = k
    end
    symm = sym === :strong ? (U1, U1) : (U1,)
    ls0 = LiouvilleSpace(d, sym, collect(Int, q), symm, labels, sectors, members, pos, nothing)
    Id = _liouville_tensor(ls0, Matrix{Float64}(I, d^2, d^2), ntuple(_ -> (0,), length(symm)))
    return LiouvilleSpace(d, sym, ls0.q, symm, labels, sectors, members, pos, Id)
end

"""
    liouville_charge(ls, M) -> label difference

The single label shift every nonzero entry of the superoperator matrix `M` makes
(`label(row) - label(col)`), or an `ArgumentError` naming two entries that disagree -- `M` is
then not covariant under this grading and cannot be a symmetric tensor.
"""
function liouville_charge(ls::LiouvilleSpace, M::AbstractMatrix)
    D = ls.d^2
    size(M) == (D, D) || throw(DimensionMismatch("superoperator is $(size(M)), expected ($D, $D)"))
    shift = nothing
    first_at = nothing
    for j in 1:D, i in 1:D
        iszero(M[i, j]) && continue
        s = _lv_diff(ls.labels[i], ls.labels[j])
        if shift === nothing
            shift, first_at = s, (i, j)
        elseif s != shift
            throw(ArgumentError(
                "superoperator is not covariant under :$(ls.sym): entry $(first_at) shifts the " *
                "label by $(shift) but entry $((i, j)) by $(s) -- it mixes charge sectors, so " *
                "this grading cannot be enabled for it"))
        end
    end
    return shift === nothing ? ntuple(_ -> (0,), length(ls.symm)) : shift
end

"""
The symmetric tensor whose DATA are `S` (a real `d^2 x d^2` matrix with label shift `shift`):
rank 2 when the shift is zero, rank 3 with an op leg carrying `shift` otherwise. Legs
`("s",'+'), ("s",'-')[, ("op",'-')]`, as `z2_ising.jl`'s hand-built operators.
"""
function _liouville_tensor(ls::LiouvilleSpace, S::AbstractMatrix{<:Real}, shift)
    N = length(ls.symm)
    QT = Tuple{Vararg{Int}}
    neutral = _lv_iszero(shift)
    QD = neutral ? 2 : 3
    data = Vector{Tuple{NTuple{QD, NTuple{N, QT}}, Array{Float64, QD + N}}}()
    for (lin, din) in ls.sectors
        lout = _lv_add(lin, shift)
        haskey(ls.members, lout) || continue
        blk = Float64.(S[ls.members[lout], ls.members[lin]])
        all(iszero, blk) && continue
        shp = (size(blk)..., ntuple(_ -> 1, QD - 2 + N)...)
        push!(data, (neutral ? (lout, lin) : (lout, lin, shift), reshape(blk, shp)))
    end
    sp = [(l, n) for (l, n) in ls.sectors]           # concretely typed, as Telum's spaces are
    spaces = neutral ? (sp, sp) : (sp, sp, [(shift, 1)])
    inds = neutral ? (TLIndex("s", '+'), TLIndex("s", '-')) :
                     (TLIndex("s", '+'), TLIndex("s", '-'), TLIndex("op", '-'))
    if isempty(data)
        # a zero operator: the identity's structure times zero keeps every leg's space
        neutral || throw(ArgumentError("a zero charged superoperator has no op-leg space"))
        return to_concrete(0.0 * ls.I)
    end
    f = _localspace_cgt_fields(data, ls.symm, spaces)
    return TLArray(ls.symm, f.qlabels, f.wmatdata, f.wmatinfo, f.RMTs, inds, spaces)
end

"""
    liouville_op(ls, M) -> TLArray

The tensor that the MPO (`pair_mpo`) and [`mpdo_apply!`](@ref) apply as the superoperator `M`
(`d^2 x d^2`, any number type) -- stored as `transpose(M)`, see the file header. Refuses a
non-covariant `M`. Complex `M` is assembled from its real and imaginary parts.
"""
function liouville_op(ls::LiouvilleSpace, M::AbstractMatrix{<:Number})
    S = transpose(M)
    shift = liouville_charge(ls, S)
    re, im_ = real.(S), imag.(S)
    all(iszero, im_) && return _liouville_tensor(ls, re, shift)
    Ti = to_concrete((1.0im) * _liouville_tensor(ls, im_, shift))
    all(iszero, re) && return Ti
    return to_concrete(_liouville_tensor(ls, re, shift) + Ti)
end

"`kron(A, I)`: `rho -> A rho`."
ket_super(A::AbstractMatrix) = kron(A, Matrix{eltype(A)}(I, size(A)...))
"`kron(I, transpose(B))`: `rho -> rho B`."
bra_super(B::AbstractMatrix) = kron(Matrix{eltype(B)}(I, size(B)...), transpose(B))
"`D[J] rho = J rho J' - {J'J, rho}/2`."
function dissipator_super(J::AbstractMatrix)
    JJ = J' * J
    return kron(J, conj(J)) - ket_super(JJ) / 2 - bra_super(JJ) / 2
end

"""
    lindblad_mpo(ls, L; pairs, onsite_h = nothing, jumps = []) -> MPO

The Lindbladian `L rho = -i [H, rho] + sum_mu D[J_mu] rho` on `L` Liouville sites, EXACT, as a
`pair_mpo` with the on-site terms in its `field` slot.

  - `pairs`: `[(A, B, Jmat), ...]` with `H ⊃ sum_{i<j} Jmat[i,j] A_i B_j` (dense physical `d x d`
    matrices; `Jmat` real or complex `L x L`, strict upper triangle). A hopping is two entries,
    `(c', c, J)` and `(c, c', J)`; a charged pair must carry opposite shifts.
  - `onsite_h`: one `d x d` Hermitian matrix for every site, or a vector of `L` of them.
  - `jumps`: `[(Jop, rates), ...]`, `rates` a scalar (every site) or a length-`L` vector.
  - `triples`: `[(A, B, C, w), ...]` with `H ⊃ sum_j w[j] A_j B_{j+1} C_{j+2}` (`w` length `L-2`,
    `A` charge-neutral; [`TripleTerm`](@ref)) -- the facilitated hop `n_{j-1} c'_j c_{j+1}` of the
    U(1) East model (plan OS-3) is `(n, c', c, J)` plus its conjugate `(n, c, c', J)`.

Every superoperator passes `liouville_charge` under `ls`'s grading, so a model that breaks it is
refused at build time with the offending entries named.
"""
function lindblad_mpo(ls::LiouvilleSpace, L::Int; pairs = Any[], onsite_h = nothing,
                      jumps = Any[], triples = Any[])
    L >= 2 || throw(ArgumentError("lindblad_mpo needs at least two sites"))
    isempty(pairs) && isempty(triples) && throw(ArgumentError(
        "lindblad_mpo needs at least one two- or three-body term"))
    d = ls.d
    tts = TripleTerm[]
    for (A, B, C, w) in triples
        all(size(X) == (d, d) for X in (A, B, C)) || throw(DimensionMismatch(
            "triple operators must be $d x $d"))
        length(w) == L - 2 || throw(DimensionMismatch("triple weights must have length $(L - 2)"))
        for (sup, c) in ((ket_super, -1im), (bra_super, 1im))
            TA = liouville_op(ls, sup(A))
            length(TA.inds) == 2 || throw(ArgumentError(
                "a triple's first operator must be charge-neutral under :$(ls.sym)"))
            push!(tts, TripleTerm(TA, _liouville_vertex(ls, sup(B), sup(C)),
                                  c .* ComplexF64.(w)))
        end
    end
    verts = PairVertex[]
    Js = AbstractMatrix[]
    for (A, B, Jm) in pairs
        size(A) == (d, d) && size(B) == (d, d) || throw(DimensionMismatch(
            "pair operators must be $d x $d"))
        size(Jm) == (L, L) || throw(DimensionMismatch("coupling matrix must be $L x $L"))
        # -i H rho  (ket side)  and  +i rho H  (bra side, operators transposed per site)
        for (SA, SB, c) in ((ket_super(A), ket_super(B), -1im), (bra_super(A), bra_super(B), 1im))
            push!(verts, _liouville_vertex(ls, SA, SB))
            push!(Js, c .* ComplexF64.(Jm))
        end
    end
    hs = onsite_h === nothing ? [zeros(d, d) for _ in 1:L] :
         onsite_h isa AbstractMatrix ? [onsite_h for _ in 1:L] : collect(onsite_h)
    length(hs) == L || throw(DimensionMismatch("need one on-site H per site"))
    field = Any[]
    for i in 1:L
        M = -1im .* (ket_super(ComplexF64.(hs[i])) - bra_super(ComplexF64.(hs[i])))
        for (Jop, rates) in jumps
            g = rates isa Number ? rates : rates[i]
            iszero(g) || (M = M + g .* dissipator_super(ComplexF64.(Jop)))
        end
        T = liouville_op(ls, M)
        length(T.inds) == 2 || throw(ArgumentError(
            "the on-site superoperator at site $i is charged; H and every jump must be " *
            "covariant with a ZERO net shift under :$(ls.sym)"))
        push!(field, _mpo_block(T, i, "W,$(i - 1)", "W,$i", :none))
    end
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = ls.I, field = field,
                    triples = tts)
end

"""
One `PairVertex` applying `SA_i SB_j`. Neutral halves pair directly; charged ones pair over the op
leg, the right half as the adjoint of the tensor of `SB'` (the `(Sp, Sp')` pattern of `xxz_chain`),
which requires the two shifts to cancel.
"""
function _liouville_vertex(ls::LiouvilleSpace, SA::AbstractMatrix, SB::AbstractMatrix)
    TA = liouville_op(ls, SA)
    if length(TA.inds) == 2
        TB = liouville_op(ls, SB)
        length(TB.inds) == 2 || throw(ArgumentError(
            "pair halves must both be neutral or both charged under :$(ls.sym)"))
        return PairVertex(TA, TB, 1.0)
    end
    # `liouville_op(ls, X)` stores transpose(X); the adjoint of the tensor for `SB'` stores
    # conj(transpose(SB')) = transpose(SB) as a matrix, i.e. it applies `SB`.
    TB = to_concrete(liouville_op(ls, Matrix(SB'))')
    return PairVertex(TA, TB, 1.0)
end

# ── states ─────────────────────────────────────────────────────────────────────────────────────

"""
    mpdo_apply!(rho, j, ls, M) -> rho

Apply the one-site superoperator `M` (`d^2 x d^2`) to site `j` of the MPDO, in place; `M` must be
NEUTRAL under the grading (a charged one would change the state's sector).
"""
function mpdo_apply!(rho::SymMPS, j::Int, ls::LiouvilleSpace, M::AbstractMatrix)
    T = liouville_op(ls, M)
    length(T.inds) == 2 || throw(ArgumentError(
        "mpdo_apply! needs a neutral superoperator; this one shifts the label"))
    t = rho[j].inds[2].itags
    op = to_concrete(setitag(setitag(T, 1, t), 2, t))
    rho[j] = to_concrete(permutedims(contract(op, (1,), rho[j], (2,)), (2, 1, 3)))
    return rho
end

"""
    mpdo_product_state(ls, rhos) -> SymMPS

`rho = rho_1 (x) ... (x) rho_L` from local `d x d` matrices, each of whose vectorisation must lie
in ONE sector of the grading (e.g. `|a><a|`, or `|0><0| + |1><1|`-type identities under `:weak`).
Built as the fusion recursion of `product_state` -- one basis state of the sector (the link cut to
one component), then a neutral one-site map onto `vec(rho_j)` -- so the bond dimension is 1.
"""
function mpdo_product_state(ls::LiouvilleSpace, rhos::AbstractVector)
    L = length(rhos)
    L >= 1 || throw(ArgumentError("mpdo_product_state needs at least one site"))
    d, D = ls.d, ls.d^2
    vecs = Vector{ComplexF64}[]
    secs = Any[]
    for (j, r) in pairs(rhos)
        size(r) == (d, d) || throw(DimensionMismatch("local rho $j must be $d x $d"))
        v = ComplexF64[r[a, b] for a in 1:d for b in 1:d]
        nz = findall(!iszero, v)
        isempty(nz) && throw(ArgumentError("local rho $j is zero"))
        labs = unique(ls.labels[nz])
        length(labs) == 1 || throw(ArgumentError(
            "local rho $j spans sectors $(labs) of :$(ls.sym) -- not representable as one " *
            "charge sector (the state breaks the symmetry being used)"))
        push!(vecs, v); push!(secs, labs[1])
    end
    tensors = Any[]
    prev = (getvac(ls.I, ("L,0", "L,1")), 2)
    for i in 1:L
        F = to_concrete(getIdentity(prev, (ls.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        # ⚠ Telum refuses `preserve_space = true` with a slice INSIDE a sector, so the physical leg
        # keeps the whole sector (and its full space) and the OUTGOING link is cut to one
        # component instead -- which leaves exactly one basis state of the sector, amplitude 1.
        B = to_concrete(getsub(F, 2, s -> s == secs[i] ? Colon() : nothing; preserve_space = true))
        present = Set(ql[3] for ql in B.qlabels)
        A = to_concrete(getsub(B, 3, s -> s in present ? (1:1) : nothing))
        push!(tensors, to_concrete(A * (1.0 + 0.0im)))
        prev = (tensors[end], 3)
    end
    rho = SymMPS(tensors, L)
    for i in 1:L
        # which basis state of the sector survived the link cut is Telum's fusion order, so map
        # EVERY state of the sector onto vec(rho_i): neutral, and right whichever one it was
        M = zeros(ComplexF64, D, D)
        for g in ls.members[secs[i]]
            M[:, g] = vecs[i]
        end
        mpdo_apply!(rho, i, ls, M)
    end
    return rho
end

"""
    mpdo_identity(ls, L) -> SymMPS

The trace superket `|I>> = vec(I)^{(x)L}`, a product state (`chi = 1`) under `:weak` and `:none`,
where `vec(I)` lies in the zero sector. Under `:strong` it spans every `(N, N)` block and is not a
single-charge MPS, so it is refused; use the block-walk trace there (`mpdo_trace`).
Pass it as `cbe_bug_step!(...; keep_states = [mpdo_identity(ls, L)])` to make the Galerkin step
trace-preserving before the compression.
"""
function mpdo_identity(ls::LiouvilleSpace, L::Int)
    ls.sym === :strong && throw(ArgumentError(
        "|I>> spans every (N, N) block under :strong and is not a single-charge MPS"))
    return mpdo_product_state(ls, [Matrix{ComplexF64}(I, ls.d, ls.d) for _ in 1:L])
end

# ── contractions by walking the abelian blocks ─────────────────────────────────────────────────

"""
    _mpdo_walk(rho, ls, cov) -> ComplexF64

`sum over configurations of prod_j cov[j][g_j] * rho(g_1 ... g_L)`: the state contracted with a
PRODUCT covector, one length-`d^2` vector per site. Done block by block over the abelian sectors
(an environment row vector per link label), so it needs no covector MPS and works for every
grading. The trace is `cov[j] = vec(I)`; `Tr(O_j rho)` puts `vec(transpose(O))` at site `j`.
"""
function _mpdo_walk(rho::SymMPS, ls::LiouvilleSpace, cov::AbstractVector)
    L = length(rho)
    length(cov) == L || throw(DimensionMismatch("need one covector per site"))
    env = nothing                                      # label => row vector over the link
    for i in 1:L
        A = to_concrete(rho[i])
        new = Dict{Any, Vector{ComplexF64}}()
        for (k, ql) in pairs(A.qlabels)
            cl, s, cr = ql
            R = _rmt3(A.RMTs[k])
            dl, dp, dr = size(R)
            el = if env === nothing
                ones(ComplexF64, dl)
            else
                haskey(env, cl) || continue
                env[cl]
            end
            gs = ls.members[s]
            t = zeros(ComplexF64, dl, dr)
            for p in 1:dp
                w = cov[i][gs[p]]
                iszero(w) || (t .+= w .* @view R[:, p, :])
            end
            iszero(t) && continue
            acc = get!(new, cr, zeros(ComplexF64, dr))
            acc .+= transpose(t) * el
        end
        env = new
    end
    return sum((sum(v) for v in values(env)); init = zero(ComplexF64))
end

"A block's RMT as a dense `(link_l, phys, link_r)` array: RMTs carry one trailing singleton
multiplicity dimension per abelian symmetry (rank `QD + N`), and may be a `DiagRMT`."
_rmt3(R) = (X = Array(R); reshape(X, size(X, 1), size(X, 2), size(X, 3)))

"`vec(I)` as a Liouville covector: picks the diagonal `|a><a|` entries."
_lv_trace_cov(ls::LiouvilleSpace) =
    ComplexF64[a == b ? 1 : 0 for a in 1:ls.d for b in 1:ls.d]

"`Tr(O rho_site)` as a Liouville covector: `sum_ab O[b,a] rho[a,b]`."
_lv_op_cov(ls::LiouvilleSpace, O::AbstractMatrix) =
    ComplexF64[O[b, a] for a in 1:ls.d for b in 1:ls.d]

"`Tr rho` of an MPDO."
mpdo_trace(rho::SymMPS, ls::LiouvilleSpace) =
    _mpdo_walk(rho, ls, fill(_lv_trace_cov(ls), length(rho)))

"""
    mpdo_expect(rho, ls, ops::Dict{Int, <:AbstractMatrix}) -> ComplexF64

`Tr(prod_j O_j rho) / Tr(rho)` for one-site physical operators on distinct sites (`Dict` site =>
`d x d` matrix); NOT divided when `normalise = false`.
"""
function mpdo_expect(rho::SymMPS, ls::LiouvilleSpace, ops::AbstractDict; normalise::Bool = true)
    cov = [haskey(ops, j) ? _lv_op_cov(ls, ops[j]) : _lv_trace_cov(ls) for j in 1:length(rho)]
    v = _mpdo_walk(rho, ls, cov)
    return normalise ? v / mpdo_trace(rho, ls) : v
end

"""
    mpdo_rdm(rho, ls, sites) -> Matrix{ComplexF64}

The reduced density matrix of `sites` (sorted, distinct), normalised to unit trace, from
`d^(2k)` product-covector walks: `rho_S[a, b] = Tr(|b><a|_S rho) / Tr rho`. For local
Hermiticity / positivity diagnostics on states too large for [`mpdo_dense`](@ref).
"""
function mpdo_rdm(rho::SymMPS, ls::LiouvilleSpace, sites::AbstractVector{<:Integer})
    issorted(sites) && allunique(sites) || throw(ArgumentError("sites must be sorted and distinct"))
    d, k = ls.d, length(sites)
    tr = mpdo_trace(rho, ls)
    out = zeros(ComplexF64, d^k, d^k)
    for ia in CartesianIndices(ntuple(_ -> d, k)), ib in CartesianIndices(ntuple(_ -> d, k))
        ops = Dict{Int, Matrix{Float64}}()
        for (m, j) in pairs(sites)
            E = zeros(d, d); E[ib[m], ia[m]] = 1.0         # Tr(|b><a| rho) = rho[a, b]
            ops[j] = E
        end
        a = sum((ia[m] - 1) * d^(k - m) for m in 1:k) + 1
        b = sum((ib[m] - 1) * d^(k - m) for m in 1:k) + 1
        out[a, b] = mpdo_expect(rho, ls, ops; normalise = false) / tr
    end
    return out
end

"""
    mpdo_dense(rho, ls) -> Matrix{ComplexF64}

The full `d^L x d^L` density matrix (site 1 the most significant index, ket rows, bra columns),
NOT normalised. Small `L` only: it enumerates every configuration through the same block walk.
"""
function mpdo_dense(rho::SymMPS, ls::LiouvilleSpace)
    L, d = length(rho), ls.d
    d^L <= 4096 || throw(ArgumentError("mpdo_dense is for small systems (d^L <= 4096)"))
    D = d^2
    env = nothing                                      # label => (nacc x dlink) amplitudes
    nacc = 1
    for i in 1:L
        A = to_concrete(rho[i])
        new = Dict{Any, Matrix{ComplexF64}}()
        for (k, ql) in pairs(A.qlabels)
            cl, s, cr = ql
            R = _rmt3(A.RMTs[k])
            dl, dp, dr = size(R)
            E = env === nothing ? ones(ComplexF64, 1, dl) :
                (haskey(env, cl) ? env[cl] : continue)
            acc = get!(new, cr, zeros(ComplexF64, nacc * D, dr))
            for p in 1:dp
                g = ls.members[s][p]
                rows = ((0:(nacc - 1)) .* D) .+ g
                acc[rows, :] .+= E * R[:, p, :]
            end
        end
        env = new
        nacc *= D
    end
    v = sum(sum(m; dims = 2)[:, 1] for m in values(env))
    rhoD = zeros(ComplexF64, d^L, d^L)
    for idx in 1:(D^L)
        gs = reverse(digits(idx - 1; base = D, pad = L)) .+ 1        # site 1 most significant
        ka = 1; kb = 1
        for g in gs
            a, b = liouville_pair(d, g)
            ka = (ka - 1) * d + a
            kb = (kb - 1) * d + b
        end
        rhoD[ka, kb] = v[idx]
    end
    return rhoD
end

"""
    mpdo_normalise_trace!(rho, ls) -> trace before

Rescale the orthogonality centre so `Tr rho = 1` (the physical normalisation of an MPDO; its
Frobenius norm is the purity's square root, not 1).
"""
function mpdo_normalise_trace!(rho::SymMPS, ls::LiouvilleSpace)
    t = mpdo_trace(rho, ls)
    (isfinite(abs(t)) && abs(t) > 0) || throw(ErrorException("MPDO trace is $t"))
    c = (1 <= rho.center <= length(rho)) ? rho.center : 1
    rho[c] = to_concrete((1 / t) * rho[c])
    return t
end

"""
    mpdo_checks(rho, ls; dense = length(rho) * log2(ls.d) <= 10) -> NamedTuple

Trace, Hermiticity and positivity of an MPDO. `dense = true`: from the full matrix -- `herm`
`= ||rho - rho'|| / ||rho||` and `min_eig`, the smallest eigenvalue of the Hermitian part of
`rho / Tr rho`. Otherwise the same two numbers from every nearest-neighbour two-site reduced
density matrix (`local = true`), which can only witness a violation, not certify its absence.
"""
function mpdo_checks(rho::SymMPS, ls::LiouvilleSpace; dense::Bool = length(rho) * log2(ls.d) <= 10)
    tr = mpdo_trace(rho, ls)
    if dense
        R = mpdo_dense(rho, ls) / tr
        H = (R + R') / 2
        return (trace = tr, herm = norm(R - R') / max(norm(R), eps()),
                min_eig = minimum(eigvals(Hermitian(H))), local_only = false)
    end
    herm, mn = 0.0, Inf
    for j in 1:(length(rho) - 1)
        R = mpdo_rdm(rho, ls, [j, j + 1])
        herm = max(herm, norm(R - R') / max(norm(R), eps()))
        mn = min(mn, minimum(eigvals(Hermitian((R + R') / 2))))
    end
    return (trace = tr, herm = herm, min_eig = mn, local_only = true)
end
