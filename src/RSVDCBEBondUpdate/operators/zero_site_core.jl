# ── zero-site generator, frame construction and truncation sweeps ──
# The Lubich sweep that used to sit here was removed. Its re-ranking growth passes do not carry over
# to the Krylov-accumulated frame of `cbe_bug_step!`, so measurements do not port between the two.

# CBE-BUG: K and L basis sweeps in CBE frames, then ONE Galerkin step at the centre bond (only the
# centre tensor is evolved, so nothing is counted twice). The K/L ODEs are never solved:
# `cbe_expand` supplies the directions. The sweeps must not write back into `psi` (zero padding would
# collapse the rank), and no rank-4 tensor is ever allocated.

# ── writing an expansion back into the state ─────────────────────────────────

"""
    _absorb_left!(psi, i, U_ex) -> psi

Replace `psi[i]` by the expanded left frame and push the (zero-padded) remainder into
`psi[i+1]`, leaving the state unchanged and the centre at `i+1`.

`U_ex†(A_i A_{i+1})` is evaluated as `(U_ex†A_i)A_{i+1}`: the two-site block is never
formed, and the intermediate is `bond_ex x bond_old`.
"""
function _absorb_left!(psi::SymMPS, i::Int, U_ex)
    tag = psi[i].inds[3].itags
    M = contract(U_ex', (1, 2), psi[i], (1, 2))        # (bond_ex, bond_old)
    B = contract(M, (2,), psi[i + 1], (1,))            # (bond_ex, site_r, link_r)
    psi[i]     = to_concrete(setitag(U_ex, 3, tag))
    psi[i + 1] = to_concrete(setitag(B, 1, tag))
    psi.center = i + 1
    return psi
end

"""
    _absorb_right!(psi, i, V_ex) -> psi

Mirror: `psi[i+1]` becomes the expanded right frame and the remainder goes into `psi[i]`,
leaving the centre at `i`.
"""
function _absorb_right!(psi::SymMPS, i::Int, V_ex)
    tag = psi[i].inds[3].itags
    M = contract(psi[i + 1], (2, 3), V_ex', (2, 3))    # (bond_old, bond_ex)
    A = contract(psi[i], (3,), M, (1,))                # (link_l, site_l, bond_ex)
    psi[i]     = to_concrete(setitag(A, 3, tag))
    psi[i + 1] = to_concrete(setitag(V_ex, 1, tag))
    psi.center = i
    return psi
end

"""Put a frame's two bond legs back on the state's own link tags.

BOTH legs, not one: adjacent tensors in a `SymMPS` must agree on the shared link tag, and
retagging one side only leaves `psi[i].inds[3] != psi[i+1].inds[1]`, which `contract` rejects on
the next sweep.

Lives here beside [`_frame_from`](@ref) because both are shared frame plumbing for every sweep.
It previously sat in `jan_bug.jl`, and deleting that file took `cbe_bug_step!` down with it --
the helper is not specific to any one integrator.
"""
_relink(T, t1, t3) = to_concrete(setitag(setitag(T, 1, t1), 3, t3))

"""
    _frame_from(left, right) -> BondFrame

`bond_frame`'s factorisation on tensors handed in rather than read from `psi`. The basis sweeps must
not touch the state, so the tensor is a carry `C_{i-1} psi[i]`, which `bond_frame` cannot express.

Correctness needs exactly one of `left`/`right` to carry the amplitude while the other is an
isometry: the K-sweep (psi canonical at 1) carries it in `left`, the L-sweep (psi canonical at L)
in `right`.
"""
function _frame_from(left, right)
    res_l = svd(left,  (1, 2), "bU,L", "bU,R"; cutoff = 0.0)
    res_r = svd(right, (1,),   "bV,L", "bV,R"; cutoff = 0.0)
    U0 = to_concrete(res_l.U)
    V0 = to_concrete(res_r.Vd)
    S0 = to_concrete(to_concrete(res_l.S * res_l.Vd) * to_concrete(res_r.U * res_r.S))
    return BondFrame(U0, S0, V0,
                     left.inds[1], left.inds[2], left.inds[3],
                     right.inds[2], right.inds[3],
                     leg_dim(U0, 3), copy(left.spaces[3]))
end

# ── the 0-site effective Hamiltonian ─────────────────────────────────────────

"""
    ZeroSiteH

The centre bond's effective Hamiltonian, as channel environments already pushed through
the expanded frames. Acting with it never touches a site tensor:

    S_out = done_L·S + S·done_Rᵀ + Σ_t c_t Σ_op open_L[t]·S·open_R[t]

Each term is a `bond x bond` product, so a Krylov matvec costs `O(χ²)` rather than the
`O(χ²d²)` of a two-site apply. This is what `RSVDpreBE0SiQS` returning `VLex, VRex` is
for: the whole point of expanding the bond is that the update afterwards is 0-site.
"""
struct ZeroSiteH
    done_l::Any
    done_r::Any
    open_l::Vector{Any}
    open_r::Vector{Any}
    coeffs::Vector{Float64}
end

"""
    zero_site_h(h, i, lenv, renv, U_ex, V_ex) -> ZeroSiteH

Push the channel environments at links `i` and `i+2` through the expanded frames onto the
centre bond. `U_ex` carries `(link_l, site_l, bond)` and `V_ex` carries
`(bond, site_r, link_r)` -- an MPS tensor's layout in each case -- so the same one-site
recursions the environment stacks are built from apply unchanged.
"""
function zero_site_h(h::XXZChain, i::Int, lch::ChannelSet, rch::ChannelSet,
                     U_ex, V_ex)
    terms = h.terms
    nt = length(terms)

    # Left: everything completed to the left of the centre bond, plus each half-term that
    # is still open there. A channel open at link i is closed by `term.right` at site i.
    dl = lch.done === ZERO ? ZERO : _left_step(lch.done, U_ex, nothing, i)
    for t in 1:nt
        lch.open[t] === ZERO && continue
        c = _left_step(lch.open[t], U_ex, terms[t].right, i)
        dl = _accum(dl, to_concrete(terms[t].coeff * c))
    end
    ol = Any[_left_step(lch.id, U_ex, terms[t].left, i) for t in 1:nt]

    dr = rch.done === ZERO ? ZERO :
         _right_step(rch.done, V_ex, nothing, i + 1)
    for t in 1:nt
        rch.open[t] === ZERO && continue
        c = _right_step(rch.open[t], V_ex, terms[t].left, i + 1)
        dr = _accum(dr, to_concrete(terms[t].coeff * c))
    end
    or = Any[_right_step(rch.id, V_ex, terms[t].right, i + 1) for t in 1:nt]

    return ZeroSiteH(dl, dr, ol, or, [t.coeff for t in terms])
end

zero_site_h(h::XXZChain, i::Int, lenv::LeftEnvStack, renv::RightEnvStack, U_ex, V_ex) =
    zero_site_h(h, i, left_channels(lenv, i), right_channels(renv, i + 2), U_ex, V_ex)

"""
    apply_zero_site(H0, S) -> TLArray

`H_eff S` for the centre core `S`, legs `(bond_l, bond_r)` in and out.

Each environment contributes its bra leg in place of the ket leg it consumes, so the
result carries `S`'s own legs after unpriming -- the same convention `_env_on_link` uses.
"""
function apply_zero_site(H0::ZeroSiteH, S)
    acc = nothing
    add!(x) = (acc = acc === nothing ? to_concrete(x) : to_concrete(acc + x))

    if H0.done_l !== ZERO
        # (bra_l, bond_r) <- done_l's bra replaces bond_l
        add!(_unprime(to_concrete!(contract(H0.done_l, (2,), S, (1,))), 1))
    end
    if H0.done_r !== ZERO
        out = contract(S, (2,), H0.done_r, (2,))       # (bond_l, bra_r)
        add!(_unprime(to_concrete(out), 2))
    end
    for t in eachindex(H0.open_l)
        ol, or = H0.open_l[t], H0.open_r[t]
        (ol === ZERO || or === ZERO) && continue
        # A term's two halves either BOTH carry an op-leg (the U(1) XY hop, where the leg
        # carries the +/-2 charge that makes the pair allowed) or NEITHER does (SzSz, and
        # every term without symmetry). One rank test covers both channels.
        left = contract(ol, (2,), S, (1,))             # (bra_l, [op,] bond_r)
        out = if length(ol.inds) == 3
            contract(left, (2, 3), or, (3, 2))         # close the op-leg too
        else
            contract(left, (2,), or, (2,))
        end                                             # (bra_l, bra_r)
        add!(to_concrete(H0.coeffs[t] * _unprime(_unprime(to_concrete(out), 1), 2)))
    end
    acc === nothing && throw(ArgumentError(
        "apply_zero_site: the effective Hamiltonian has no contributions at this bond"))
    return acc
end

# ── truncation sweep (Sulz algorithm 1) ──────────────────────────────────────

"""
    truncate_sweep!(psi; maxdim, cutoff) -> psi

Compress every bond after the Galerkin steps. It must run only after every bond has had its step:
earlier, it would remove directions that are merely not yet populated, which pinned the rank at 2
in the single-centre version.

One left-to-right truncating pass, then one right-to-left, ending with the centre at site 1.
"""
function truncate_sweep!(psi::SymMPS; maxdim::Int = 200, cutoff::Float64 = 1e-12)
    L = length(psi)
    canonical!(psi, 1)
    for i in 1:(L - 1)
        res = svd(psi[i], (1, 2); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[i].inds[3].itags
        M = to_concrete(res.S * res.Vd)
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(contract(M, (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1
    end
    for i in L:-1:2
        res = svd(psi[i], (1,); cutoff = cutoff, Nkeep = maxdim)
        tag = psi[i - 1].inds[3].itags
        M = to_concrete(res.U * res.S)
        psi[i]     = to_concrete(setitag(res.Vd, 1, tag))
        psi[i - 1] = to_concrete(setitag(contract(psi[i - 1], (3,), M, (1,)), 3, tag))
        psi.center = i - 1
    end
    return psi
end

"The BUG's root link `c` (the step's split and compression root is `c + 1`) when none is given."
_bug_root_link(L::Int) = max(1, min(L - 1, L ÷ 2))

"""
    _cut_divisor(L, root) -> Float64

`weight_tol ||psi|| / _cut_divisor(L, root)` is the absolute allowance of EVERY Schmidt cut under
the budgeted rule: what [`truncate_recursive!`](@ref) gives each bond of the BUG's S-step result,
and -- USER 2026-10-17, the singular values discarded must be the same for both -- what
`tdvp2_step!(...; weight_tol)` gives each two-site update, at the BUG's default root
`_bug_root_link(L) + 1`. One function, so the two rules cannot drift apart.
"""
_cut_divisor(L::Int, root::Int) = sqrt(root - 1) + sqrt(L - root)

"""
    _budget_svd(T, split, maxdim, share_of) -> (res, capped)

ONE bond's budgeted cut, the rule `truncate_recursive!` applies with `weight_tol` and the
adaptive CBE-DMRG applies at every split: keep the SMALLEST rank whose discarded Schmidt norm
fits the absolute allowance `share_of(n2)` (`n2` = this tensor's total squared weight), never
more than `maxdim`. `capped` reports that `maxdim`, not the allowance, set the rank.
"""
function _budget_svd(T, split, maxdim::Int, share_of)
    res = svd(T, split; cutoff = 0.0, Nkeep = maxdim, get_lists = true)
    share = share_of(_sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    nk = length(res.kept_list)
    # `trunc_list` holds what `maxdim` (and exact zeros) already removed: it is spent.
    k, tail = _tail_rank(res.kept_list, 1.0, share, nk; base = _sq_weight(res.trunc_list))
    capped = tail > share && !isempty(res.trunc_list)
    k = max(k, 1)            # a bond never closes; a zero state has nothing to keep anyway
    k < nk && (res = svd(T, split; cutoff = 0.0, Nkeep = k, get_lists = true))
    return res, capped
end

"""
    truncate_recursive!(psi, root; maxdim, cutoff, weight_tol) -> (psi, discarded, cumulative, capped)

Recursive root-to-leaves rank truncation of the rank-adaptive tree integrator (arXiv:2201.10291
§4.2, Algorithm 7). `cutoff` applies Telum's relative cut at each bond; `discarded` is the worst
single bond. `weight_tol` is a shared budget on `||psi - psi_trunc|| / ||psi||`: each bond gets an
equal share, and `cumulative` bounds the error by the triangle inequality across the two subtrees.
`capped` is true when `maxdim` binds past the budget; `bond_weights` receives each bond's share.
"""
function truncate_recursive!(psi::SymMPS, root::Int; maxdim::Int = 200,
                             cutoff::Float64 = 1e-12,
                             weight_tol::Union{Nothing, Float64} = nothing,
                             bond_weights::Union{Nothing, Vector{Float64}} = nothing)
    L = length(psi)
    bond_weights === nothing || length(bond_weights) == L - 1 ||
        throw(DimensionMismatch("bond_weights needs $(L - 1) entries"))
    1 <= root <= L || throw(ArgumentError("root $root outside 1:$L"))
    weight_tol === nothing || weight_tol >= 0 ||
        throw(ArgumentError("weight_tol must be non-negative, got $weight_tol"))
    worst = 0.0
    cum = [0.0, 0.0]           # absolute squared weight removed, left / right subtree
    nrm2 = -1.0                # ||psi||^2, read off the first (centre) SVD
    capped = false
    sdiv = _cut_divisor(L, root)

    # One bond's SVD under whichever rule is active. `split` is the legs `svd` keeps on `U`.
    function cut(T, split)
        weight_tol === nothing &&
            return svd(T, split; cutoff = cutoff, Nkeep = maxdim, get_lists = true)
        res, c = _budget_svd(T, split, maxdim, function (n2)
            nrm2 < 0 && (nrm2 = n2)
            return sdiv > 0 ? weight_tol * sqrt(max(nrm2, 0.0)) / sdiv : 0.0
        end)
        capped |= c
        return res
    end
    function account!(res, side, bond)
        worst = max(worst, _trunc_weight(res))
        cum[side] += _sq_weight(res.trunc_list)
        bond_weights === nothing || (bond_weights[bond] = _sq_weight(res.trunc_list))
        nrm2 < 0 && (nrm2 = _sq_weight(res.kept_list) + _sq_weight(res.trunc_list))
    end

    # ── left subtree, root-to-leaf: bonds root-1, root-2, … 1 ──
    for i in root:-1:2
        res = cut(psi[i], (1,))
        account!(res, 1, i - 1)               # splits bond (i-1, i)
        tag = psi[i - 1].inds[3].itags
        # `A = U S Vd`: keep `Vd` here and hand `U S` outward, so the child receives the weight
        # and its own SVD is a genuine Schmidt decomposition.
        psi[i]     = to_concrete(setitag(res.Vd, 1, tag))
        psi[i - 1] = to_concrete(setitag(
            contract(psi[i - 1], (3,), to_concrete(res.U * res.S), (1,)), 3, tag))
        psi.center = i - 1
    end

    # With the weight carried out to site 1 the centre has to come back before the other subtree
    # can be cut. LOSSLESS: every bond crossed was just cut, and cutting it again is exactly the
    # compounding this routine exists to avoid.
    canonical!(psi, root)

    # ── right subtree, root-to-leaf: bonds root, root+1, … L-1 ──
    for i in root:(L - 1)
        res = cut(psi[i], (1, 2))
        account!(res, 2, i)                   # splits bond (i, i+1)
        tag = psi[i].inds[3].itags
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(
            contract(to_concrete(res.S * res.Vd), (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1
    end

    cumulative = nrm2 > 0 ? (sqrt(cum[1]) + sqrt(cum[2])) / sqrt(nrm2) : 0.0
    bond_weights === nothing || (bond_weights .= nrm2 > 0 ? sqrt.(bond_weights ./ nrm2) : 0.0)
    return psi, worst, cumulative, capped
end

# ── the step ─────────────────────────────────────────────────────────────────

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` lives in baselines/tdvp2.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[i]) for i in 1:length(psi); init = 0)
