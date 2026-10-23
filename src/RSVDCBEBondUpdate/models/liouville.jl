# ── symmetry-aware vectorised density matrices (open-system plan, Task 2) ──
# The stored tensor is the transpose of the applied matrix: `liouville_op(ls, M)` returns the tensor
# that the MPO and `mpdo_apply!` apply as `M`.
# Charges: `:weak` uses q[a] - q[b], `:strong` uses (q[a], q[b]) per (N, N) block, `:none` is one dense
# sector. A superoperator whose entries do not shift the label uniformly is refused at build time.

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

The Lindbladian on `L` Liouville sites, exact, as a `pair_mpo` with the on-site terms in its field
slot. `pairs` are `(A, B, Jmat)` terms of H, `jumps` are `(Jop, rates)`, and `triples` are the
three-site terms of [`TripleTerm`](@ref). Every superoperator must pass `liouville_charge`, so a
model that breaks the grading is refused at build time.
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

"""
    mpdo_identity_sector(ls, L, like) -> SymMPS

The trace superket of one `(N, N)` block under `:strong`: the sum over configurations of total
charge N of diagonal projectors. Unlike `|I>>` it is a single-charge MPS, so it can be passed as
`keep_states` to `cbe_bug_step!` to keep the Galerkin step trace-exact. `<<I_N|rho>> = Tr rho`.

Built by the fusion recursion of [`mpdo_product_state`](@ref); `like` fixes the sign convention and
needs non-negative local charges.
"""
function mpdo_identity_sector(ls::LiouvilleSpace, L::Int, like::SymMPS)
    ls.sym === :strong || throw(ArgumentError(
        "mpdo_identity_sector is for :strong; under :$(ls.sym) use mpdo_identity"))
    all(>=(0), ls.q) || throw(ArgumentError("mpdo_identity_sector needs non-negative charges"))
    length(like) == L || throw(DimensionMismatch("`like` has $(length(like)) sites, not $L"))
    targets = unique(ql[3] for ql in to_concrete(like[L]).qlabels)
    length(targets) == 1 || throw(ArgumentError("`like` is not in one (N, N) block"))
    target = targets[1]
    nof(lab) = abs(lab[1][1])                     # the charge of a diagonal label, any sign convention
    N, qmax = nof(target), maximum(ls.q)
    diag = Set(ls.labels[liouville_index(ls.d, a, a)] for a in 1:ls.d)
    NS = length(ls.symm)
    QT = Tuple{Vararg{Int}}
    tensors = Any[]
    prev = (getvac(ls.I, ("L,0", "L,1")), 2)
    for i in 1:L
        F = to_concrete(getIdentity(prev, (ls.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        B = to_concrete(getsub(F, 2, s -> s in diag ? Colon() : nothing; preserve_space = true))
        reach(lab) = i == L ? lab == target : (nof(lab) <= N && N - nof(lab) <= qmax * (L - i))
        data = Vector{Tuple{NTuple{3, NTuple{NS, QT}}, Array{Float64, 3 + NS}}}()
        for (k, ql) in pairs(B.qlabels)
            reach(ql[3]) || continue
            v = sum(real, Array(B.RMTs[k]))         # the fusion isometry's one entry on this path
            iszero(v) && continue
            push!(data, ((ql[1], ql[2], ql[3]), reshape([v], ntuple(_ -> 1, 3 + NS))))
        end
        isempty(data) && throw(ErrorException("mpdo_identity_sector: no reachable sector at site $i"))
        kept = Set(d[1][3] for d in data)
        spaces = (B.spaces[1], B.spaces[2], [(l, 1) for (l, _) in B.spaces[3] if l in kept])
        f = _localspace_cgt_fields(data, symm(B), spaces)
        A = TLArray(symm(B), f.qlabels, f.wmatdata, f.wmatinfo, f.RMTs, B.inds, spaces)
        push!(tensors, to_concrete(A * (1.0 + 0.0im)))
        prev = (tensors[end], 3)
    end
    return SymMPS(tensors, L)
end
