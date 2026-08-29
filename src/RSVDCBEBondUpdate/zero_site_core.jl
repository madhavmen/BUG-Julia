# ── ZERO-SITE GENERATOR, FRAME CONSTRUCTION AND THE TRUNCATION SWEEPS ────────
#
# ⛔ THIS FILE NO LONGER HOLDS A SWEEP. `cbe_lubich_sweep` was REMOVED (2026-08-29) -- the
# campaign's BUG is `sweeps/cbe_bug.jl` and nothing routes through the Lubich variant any more.
# What stayed is the shared core it happened to be defined next to, and every piece of it is
# load-bearing for `cbe_bug_step!`:
#
#   zero_site_h / apply_zero_site / ZeroSiteH   the Galerkin generator at the root bond
#   _frame_from                                 factorises both sides of a bond
#   truncate_sweep! / truncate_recursive!       the closing truncation (cbe_bug.jl:919)
#   _state_stored / _absorb_* / _relink         working-set accounting and gauge helpers
#
# ⚠ THE GROWTH KNOB DID NOT COME WITH IT. `cbe_lubich_sweep`'s `grow_iters` RE-RAN THE CBE
# SELECTION each pass, re-truncating by budget and discarding what the next application of `H`
# needs -- MEASURED on OAT it bought 1.8x and then went flat however many passes it was given.
# `cbe_bug_step!` has its own `grow_iters`, which ACCUMULATES AND NEVER RE-RANKS. They are
# different algorithms with the same name; do not port measurements between them.

# CBE-BUG: a K/L basis sweep in CBE frames, then ONE Galerkin step at the centre bond.
#
# The structure of the proven chain BUG (`exploratory_bug/src/BUG/discarded_bug.jl:16-31`,
# the MPS realisation of Sulz algorithms 5-7):
#
#   1. K-sweep   left to right,  augmented LEFT frames  W_1 … W_c
#   2. L-sweep   right to left,  augmented RIGHT frames Z_c … Z_{L-1}
#   3. Galerkin  seed S = <W, Z | psi> from the sweep carries, then integrate the single
#                centre connecting tensor -- THE ONLY TIME EVOLUTION, so no over-counting
#                and no backward substep
#   4. Truncate  (Sulz algorithm 1)
#
# THE K/L ODEs ARE NEVER SOLVED. `cbe_expand` supplies the directions those solves would
# have found -- one sketched application of H -- and the centre `expv` builds the Krylov
# basis on the expanded bond and takes the step in it. No frozen projector, no discarded
# projector, no one-site `expv`.
#
# THE BASIS SWEEPS MUST NOT TOUCH THE STATE. Two earlier versions failed here, and both
# failures are worth naming because neither throws:
#
#   * Writing each frame back as it is found (`_absorb_left!`) leaves the zero-padded
#     remainder in `psi[i+1]`; the next `bond_frame` then re-derives from that padded tensor
#     with `svd(cutoff = 0.0)`, collapsing the padding and propagating the collapse back into
#     the bond just expanded. L=18 sat at maxbond 2 with an error INDEPENDENT of dt.
#   * Compensating with a Galerkin step at every bond grows the rank but applies the global
#     H L-1 times per sweep: full-rank exactness broke to 2.24e-2.
#
# Rank grows because the evolved core is DENSE and the frames are RETAINED: S(t1) propagates
# out through W_{i+1}…W_c, so every bond ends up carrying r + Dex. The rank at bond i is NOT
# `U_ex†psi[i]` -- that is its value before the step, still zero-padded.
#
# Frames are therefore accumulated in `W`/`Z`, each built from a CARRY of the original state,
# and `psi` is assembled only after the Galerkin step.
#
# NO RANK-4 TENSOR IS EVER ALLOCATED. That is the module's second design goal
# (docs/cbe_bug.md §2b) and it is a property of how each step is factorised, not an
# optimisation applied afterwards:
#
#   write-back      U_ex†(A_i A_{i+1}) = (U_ex†A_i) A_{i+1}
#   augmented core  S₀ = (U_ex†A_c)(A_{c+1}V_ex†)
#   Galerkin        0-site channel environments pushed through U_ex / V_ex, so each
#                   Krylov matvec is a handful of bond x bond products
#
# 2-site TDVP cannot do this -- it evolves the rank-4 block -- and `kls_bond_update` does
# not either, since it forms `frame_theta(f)` and applies the gate to it.

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

`bond_frame`'s factorisation, but on tensors handed in rather than read out of `psi`.

The basis sweeps must not touch the state (see `cbe_bug_step!`), so the tensor whose
left frame is wanted is a CARRY -- `C_{i-1}·psi[i]`, the left block already expressed in the
expanded basis -- and not `psi[i]` itself. `bond_frame` cannot express that: it insists the
orthogonality centre sit on site `i`, and reads the state directly.

Correctness rests on exactly one of `left`/`right` carrying the amplitude while the other is
an isometry, which is what makes `left*right` the true two-site block:

  * K-sweep, `psi` canonical at 1: `left = C_{i-1}·psi[i]` carries it, `psi[i+1]` is a right
    isometry, so `res_r.S == 1`.
  * L-sweep, `psi` canonical at L: `psi[j]` is a left isometry (`res_l.S == 1`) and
    `right = psi[j+1]·D_{j+1}` carries it.
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
        add!(_unprime(to_concrete(contract(H0.done_l, (2,), S, (1,))), 1))
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

Compress every bond back down after the sweep over-dimensioned them.

Not housekeeping: without it the rank would ratchet up by `Dex` every step regardless of
whether the dynamics wanted the room. Safe to apply only AFTER every bond has had its
Galerkin step -- run before that, it would remove directions that are merely
not-yet-populated rather than unwanted, which is what pinned the rank at 2 in the
single-centre-step version.

One left-to-right pass with a truncating SVD, then one right-to-left, ending with the
centre back at site 1.
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

"""
    truncate_recursive!(psi, root; maxdim, cutoff) -> (psi, discarded)

`Θ_τ` — the **recursive root-to-leaves rank truncation** of the rank-adaptive tree integrator
(Ceruti, Lubich & Sulz, *Rank-adaptive time integration of tree tensor networks*,
arXiv:2201.10291 §4.2, Algorithm 7), specialised to the chain. Returns the largest relative
weight any single bond threw away.

The paper's sentence is the specification: *"For the truncation of a tree tensor network we
perform a recursive root-to-leaves SVD-based truncation with a given tolerance ϑ."* At each node
it takes `Mat_i(Ĉ_τ) = P̂_{τ_i} Σ̂_{τ_i} Q̂*_{τ_i}`, keeps the smallest rank satisfying
`(Σ_{k>r} σ_k²)^{1/2} ≤ ϑ`, pushes the isometry into the subtree (`U_l = Û_l P_l` at a leaf,
`C̃_{τ_i} = Ĉ_{τ_i} ×₀ P_{τ_i}^⊤` below one) and then RECURSES into that subtree.

"""
function truncate_recursive!(psi::SymMPS, root::Int; maxdim::Int = 200,
                             cutoff::Float64 = 1e-12)
    L = length(psi)
    1 <= root <= L || throw(ArgumentError("root $root outside 1:$L"))
    worst = 0.0

    # ── left subtree, root-to-leaf: bonds root-1, root-2, … 1 ──
    for i in root:-1:2
        res = svd(psi[i], (1,); cutoff = cutoff, Nkeep = maxdim, get_lists = true)
        worst = max(worst, _trunc_weight(res))
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
        res = svd(psi[i], (1, 2); cutoff = cutoff, Nkeep = maxdim, get_lists = true)
        worst = max(worst, _trunc_weight(res))
        tag = psi[i].inds[3].itags
        psi[i]     = to_concrete(setitag(res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag(
            contract(to_concrete(res.S * res.Vd), (2,), psi[i + 1], (1,)), 1, tag))
        psi.center = i + 1
    end

    return psi, worst
end

# ── the step ─────────────────────────────────────────────────────────────────

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` lives in tdvp2/tdvp2_baseline.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[i]) for i in 1:length(psi); init = 0)
