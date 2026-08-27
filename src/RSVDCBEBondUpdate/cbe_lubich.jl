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

The basis sweeps must not touch the state (see `cbe_lubich_sweep`), so the tensor whose
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

"""
    CBEBugInfo

What one [`cbe_lubich_sweep`](@ref) step did.

  - `expanded` -- expanded bond dimension at each bond of the basis sweep, before the
    truncation. The headline rank diagnostic, and the one to compare against `aug_k`.
  - `n_new` -- new directions admitted per bond.
  - `centre_rank` -- kept rank at the centre after the Galerkin step's SVD.
  - `err_pre`, `err_fnl` -- largest preselection / final-selection weight discarded.
  - `discarded` -- relative weight the centre truncation threw away.
  - `krylov_dim` -- Krylov dimension the 0-site exponential used.
  - `peak_elements` -- WORKING SET: the largest stored-element count of
    `state + transients alive at that moment` seen anywhere inside the step.

`peak_elements` is measured, not inferred, and it is measured at the same definition
`TDVP2Info.peak_elements` uses, because it is the number the second design goal is judged
on. Sampling the state AFTER `truncate_sweep!` would flatter this method: the sweep's whole
point is that the bonds are transiently wider than what survives, and the memory claim has
to be made against the transient peak, not the tidy result.
"""
struct CBEBugInfo
    expanded::Vector{Int}
    n_new::Vector{Int}
    centre_rank::Int
    err_pre::Float64
    err_fnl::Float64
    discarded::Float64
    krylov_dim::Int
    peak_elements::Int
end

"Total stored elements of every tensor in the state -- block-sparse, so the honest size.
(`tensor_elements` lives in tdvp2/tdvp2_baseline.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[i]) for i in 1:length(psi); init = 0)

"""
    cbe_lubich_sweep(psi, h, tau; kwargs...) -> CBEBugInfo

One CBE-BUG time step of `tau = -im*dt`, in place.

Keywords: `dex`, `growth`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only` forward to
[`cbe_expand`](@ref); `maxdim` and `trunc_thresh` control the truncations; `maxiter` and
`tol` the Krylov budget of the single 0-site exponential.

`O(L)` environment work per step. Each sweep CARRIES its own side's channels (one
`push_*_channels` per bond) and reads the other side from a single prebuilt stack, so
nothing is rebuilt per bond -- the same structure the chain BUG in qtci's
`exploratory_bug/src/BUG/discarded_bug.jl` uses, where the K-sweep carries `aps`/`aph` and
the environments are computed exactly once.

The centre step needs no rebuild either: the carried left channels already sit on link `c`
and the prebuilt right stack already holds link `c+2`.

ORDERING IS LOAD-BEARING. `canonical!` rewrites tensors, so each sweep canonicalises to
its far end BEFORE building the stack it will read, and the sweep direction is then chosen
so the write-back only ever touches sites that stack does not depend on
(`_absorb_right!` at bond `j` touches sites `j, j+1`, and the left stack is read at links
`<= j`; mirror for the other sweep). Getting this backwards reads a stale environment,
which does not throw -- it silently evolves with the wrong Hamiltonian.
"""
function cbe_lubich_sweep(psi::SymMPS, h::Union{XXZChain, MPO}, tau::ComplexF64;
                             dex::Int = 0,
                             growth::Float64 = 2.0,
                             dover::Union{Nothing, Int} = nothing,
                             comp_ratio::Union{Float64, Nothing} = 0.5,
                             sulz_cap::Bool = false,
                             preselect_only::Bool = false,
                             exact::Bool = false,
                             centre_expand::Bool = true,
                             root::Union{Nothing, Int} = nothing,
                             grow_iters::Int = 0,
                             maxdim::Int = 200,
                             trunc_thresh::Float64 = 1e-12,
                             maxiter::Int = 30,
                             tol::Float64 = 1e-15,
                             # ⛔ THIS SWEEP HAD NO `hermitian` KNOB AND HARDCODED LANCZOS, so a
                             # non-Hermitian generator (an open system's Lindbladian, a
                             # non-Hermitian H_eff) was silently solved with the wrong method --
                             # Lanczos does not fail there, it returns a plausible vector. The
                             # same defect was present in `tdvp2_step!`, which accepted the
                             # keyword and then ignored it at both call sites.
                             hermitian::Bool = true,
                             # See `BondUpdateBUG/expv.jl`: a genuine convergence test (Saad's
                             # contribution estimate) in place of the breakdown-only exit, and
                             # splitting `tau` so a large step needs no larger `maxiter`.
                             conv_tol::Float64 = 0.0,
                             substeps::Int = 1,
                             rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_lubich_sweep needs at least two sites"))
    # THE ROOT BOND. `nothing` is the centre, which is what the scheme is for and the only
    # value any driver passes. It is a keyword so that the root POSITION can be varied as a
    # controlled experiment: everything else in this function -- lossless basis sweeps, one
    # Galerkin step, truncate once -- is then held fixed, which is what makes a comparison
    # against a boundary-root scheme (`jan_bug.jl`) attribute a difference to the root rather
    # than to the surrounding machinery. See `benchmarks/root_position.jl`.
    #
    # WHY THE ROOT IS NOT A FREE PARAMETER IN PRACTICE: the Galerkin step explores exactly
    # `b_L x b_R` directions, and at the centre both are the full half-chain dimension while
    # at bond `L-1` the right factor is a SINGLE SITE, so `b_R <= d`. The variational space
    # is smaller by `chi/d` there, which is the whole point of rooting at the centre.
    c = root === nothing ? max(1, min(L - 1, L ÷ 2)) : root
    1 <= c <= L - 1 || throw(ArgumentError("root must be a bond in 1:$(L - 1), got $c"))

    expanded = zeros(Int, L - 1)
    n_new    = zeros(Int, L - 1)
    err_pre  = 0.0
    err_fnl  = 0.0
    peak     = 0

    # `rmax` is the state's widest bond, so the (optional) Sulz bound is read GLOBALLY
    # rather than per bond. Captured once, before the sweeps widen anything.
    exkw = (dex = dex, growth = growth, dover = dover,
            comp_ratio = comp_ratio, sulz_cap = sulz_cap,
            rmax = maximum(bond_dims(psi); init = 0),
            preselect_only = preselect_only, exact = exact, rng = rng)


    function record!(ex)
        err_pre = max(err_pre, ex.err_pre)
        err_fnl = max(err_fnl, ex.err_fnl)
        return ex
    end

    "Working set at this instant: the state as it stands plus the transients still alive."
    peak!(transients...) = (peak = max(peak, _state_stored(psi) +
                                       sum(tensor_elements(t) for t in transients; init = 0)))

    # K-SWEEP, L-SWEEP, then ONE Galerkin step at the centre. The structure of the proven
    # chain BUG (`exploratory_bug/src/BUG/discarded_bug.jl:16-31`, the MPS realisation of
    # Sulz algorithms 5-7): the single centre evolution is THE ONLY TIME EVOLUTION, so
    # there is no over-counting and no backward substep.
    #
    # THE BASIS SWEEPS MUST NOT TOUCH THE STATE. This is the bug an earlier version had:
    # it wrote each expanded frame straight back with `_absorb_left!`, which leaves the
    # zero-padded remainder in `psi[i+1]`, and then the next iteration's `bond_frame` used
    # `svd(cutoff = 0.0)` on that padded tensor -- collapsing the padding and propagating
    # the collapse back into the bond just expanded. Frames are therefore accumulated in
    # `W`/`Z`, every one derived from the ORIGINAL `psi`, and the state is assembled only
    # after the Galerkin step.
    #
    # Rank grows because the evolved core is DENSE and the frames are RETAINED: `S(t1)`
    # propagates out through `W_{i+1}…W_c`, so every bond ends up carrying `r + Dex`. It is
    # not necessary -- and it is actively wrong -- to put a Galerkin step on each bond.
    W = Vector{Any}(undef, c)              # augmented LEFT frames,  sites 1 … c
    Z = Vector{Any}(undef, L - 1)          # augmented RIGHT frames, Z[j] absorbs site j+1

    # ---- K-sweep: augmented LEFT frames, left to right -------------------------
    # `psi` canonical at 1, so sites 2…L are right isometries and the amplitude of the left
    # block rides in the carry `C = <W_1…W_i | psi_1…psi_i>`. The frame at bond `i` is built
    # from `C·psi[i]`, i.e. the left block already in the EXPANDED basis -- which is what
    # makes the frames chain. Building each from `psi[i]` instead gives frames whose left leg
    # is the old bond, and they cannot be composed.
    canonical!(psi, 1)
    rstack = right_env_stack(psi, h; downto = 3)
    lch = boundary_channels(h)
    lenv_c = lch                           # left channels at the expanded link c
    C = nothing
    for i in 1:c
        K = C === nothing ? psi[i] : to_concrete(contract(C, (2,), psi[i], (1,)))
        f = _frame_from(K, psi[i + 1])
        i == c && (lenv_c = lch)
        ex = (centre_expand || i != c) ?
             record!(cbe_expand(f, h, i, lch, right_channels(rstack, i + 2); exkw...)) :
             CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
        W[i] = ex.U_ex
        expanded[i] = leg_dim(ex.U_ex, 3)
        n_new[i] = ex.n_new_l
        peak!(ex.U_ex)
        C = to_concrete(contract(ex.U_ex', (1, 2), K, (1, 2)))    # (b_{i+1}, link_{i+1})
        lch = push_left_channels(lch, h, ex.U_ex, i)
    end

    # ---- L-sweep: augmented RIGHT frames, right to left ------------------------
    # The mirror, and it needs the opposite gauge: canonical at L makes sites 1…L-1 left
    # isometries, so the amplitude of the right block rides in `D` instead. Re-gauging does
    # not change the state, and `W` stays valid -- the seed below is a full contraction and
    # so is gauge independent.
    canonical!(psi, L)
    lstack = left_env_stack(psi, h; upto = L - 2)
    rch = boundary_channels(h)
    renv_c = rch                           # right channels at the expanded link c+2
    D = nothing
    for j in (L - 1):-1:c
        B = D === nothing ? psi[j + 1] : to_concrete(contract(psi[j + 1], (3,), D, (1,)))
        f = _frame_from(psi[j], B)
        j == c && (renv_c = rch)
        ex = (centre_expand || j != c) ?
             record!(cbe_expand(f, h, j, left_channels(lstack, j), rch; exkw...)) :
             CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
        Z[j] = ex.V_ex
        if j != c
            expanded[j] = leg_dim(ex.V_ex, 1)
            n_new[j] = ex.n_new_r
        end
        peak!(ex.V_ex)
        D = to_concrete(contract(B, (2, 3), ex.V_ex', (2, 3)))    # (link_{j+1}, b)
        rch = push_right_channels(rch, h, ex.V_ex, j + 1)
    end

    # ---- seed and the single Galerkin step -------------------------------------
    # S_start = <W, Z | psi>, by full contraction from each end. No M/N overlap matrices,
    # and no gauge requirement on `psi` -- this is a complete contraction of the state
    # against the frames.
    AL = nothing
    for i in 1:c
        T = AL === nothing ? psi[i] : to_concrete(contract(AL, (2,), psi[i], (1,)))
        AL = to_concrete(contract(W[i]', (1, 2), T, (1, 2)))       # (b_L, link_{i+1})
    end
    AR = nothing
    for j in (L - 1):-1:c
        T = AR === nothing ? psi[j + 1] : to_concrete(contract(psi[j + 1], (3,), AR, (1,)))
        AR = to_concrete(contract(T, (2, 3), Z[j]', (2, 3)))       # (link_{j+1}, b_R)
    end
    S0 = to_concrete(contract(AL, (2,), AR, (1,)))                 # (b_L, b_R)
    peak!(W[c], Z[c], S0)

    # `krylov_dim` is COUNTED here rather than reported by `expv` (which returns only the
    # vector, and belongs to the validated module): one increment per operator application is
    # the Lanczos dimension. It separates "more work per step" from "more steps of work" when
    # comparing expansion budgets -- see the note in `cbe_bond_update`.
    nmv = Ref(0)

    # THE ROOT SOLVE, in one of two subspaces.
    #
    # `grow_iters = 0` (default) is the 0-site Galerkin step in the frames the two basis
    # sweeps left: fixed subspace, `O(chi^2 w)` per matvec, never forms a rank-4 block.
    #
    # `grow_iters > 0` runs `_expanding_krylov` instead, which RE-EXPANDS THE FRAMES AROUND
    # EACH KRYLOV VECTOR before applying `H`. The distinction is not the number of vectors --
    # `maxiter` already saturates -- but the SUBSPACE they are built in. Measured motivation
    # (`benchmarks/root_position.jl`, L=6, full rank): at root 4 the sweeps reach `b_L = 8`
    # against a ceiling of `min(d*chi_3, 2^4) = 16`, and at root 5 they reach 4 against 8. The
    # Galerkin space is half the size the geometry allows, and a fixed-frame solve cannot
    # notice -- `v_k` carries `H^k`, so only a probe built from it asks for the directions the
    # later Krylov vectors need.
    #
    # WHAT IT COSTS, stated because it is the whole trade: `_expanding_krylov` forms
    # `theta = (U v_k) V` every iteration, so the 0-site bargain is gone -- `O(chi^2 d^2)` per
    # matvec plus one `cbe_expand` per growth pass.
    #
    # WHAT IT CANNOT DO: lift a root whose CEILING is already below the space. At root 5 the
    # ceiling is `8 x 2 = 16` inside a 20-dimensional sector, so growth improves the answer but
    # exactness is unreachable there at any budget.
    Wc, Zc, S1 = if grow_iters > 0
        f_root = BondFrame(W[c], S0, Z[c],
                           W[c].inds[1], W[c].inds[2], W[c].inds[3],
                           Z[c].inds[2], Z[c].inds[3],
                           max(leg_dim(W[c], 3), leg_dim(Z[c], 1)), copy(W[c].spaces[3]))
        theta0 = to_concrete(to_concrete(W[c] * S0) * Z[c])
        peak!(theta0)
        U_aug, V_aug, S_new, _ = _expanding_krylov(
            f_root,
            th -> apply_h_two_site(th, h, c, lenv_c, renv_c),
            fr -> record!(cbe_expand(fr, h, c, lenv_c, renv_c; exkw...)),
            Propagate(tau), theta0, nmv;
            maxiter = maxiter, tol = tol, grow_iters = grow_iters)
        (U_aug, V_aug, S_new)
    else
        H0 = zero_site_h(h, c, lenv_c, renv_c, W[c], Z[c])
        (W[c], Z[c],
         expv(x -> (nmv[] += 1; apply_zero_site(H0, x)), tau, S0;
              hermitian = hermitian, maxiter = maxiter, tol = tol,
              conv_tol = conv_tol, substeps = substeps))
    end
    expanded[c] = leg_dim(Wc, 3)
    res = svd(S1, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)
    discarded = _trunc_weight(res)

    # ---- assemble psi = W_1 … W_{c-1} (W_c U) (S Vd Z_c) Z_{c+1} … Z_{L-1} ------
    # BOTH bond legs of every frame have to be retagged, not just one. The frames carry the
    # `_frame_from` / `cbe_expand` bond tags ("bU,L" and friends), and adjacent tensors in a
    # `SymMPS` must agree on the shared link tag -- retagging one side only leaves
    # `psi[i].inds[3] != psi[i+1].inds[1]`, which `contract` rejects on the next sweep.
    "Put a frame's two bond legs back on the state's own link tags."
    relink(T, t1, t3) = to_concrete(setitag(setitag(T, 1, t1), 3, t3))

    for i in 1:(c - 1)
        psi[i] = relink(W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)
    end
    t1c, t3c = psi[c].inds[1].itags, psi[c].inds[3].itags
    t3r = psi[c + 1].inds[3].itags
    psi[c]     = relink(to_concrete(Wc * res.U), t1c, t3c)
    psi[c + 1] = relink(to_concrete((res.S * res.Vd) * Zc), t3c, t3r)
    for j in (c + 1):(L - 1)
        psi[j + 1] = relink(Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)
    end
    psi.center = c + 1
    centre_rank = leg_dim(psi[c], 3)

    # Compress once at the end. The evolved centre core has now propagated out through every
    # retained frame, so a direction carrying no weight is genuinely unwanted rather than
    # merely not-yet-populated. (It is NOT that every bond took its own Galerkin step -- there
    # is exactly one, at the centre; a per-bond Galerkin over-counts `tau` and was abandoned,
    # see docs/current_work.md section 5.)
    truncate_sweep!(psi; maxdim = maxdim, cutoff = max(trunc_thresh, 1e-14))

    return CBEBugInfo(expanded, n_new, centre_rank, err_pre, err_fnl, discarded, nmv[], peak)
end

cbe_lubich_sweep(psi::SymMPS, h::Union{XXZChain, MPO}, tau::Number; kwargs...) =
    cbe_lubich_sweep(psi, h, ComplexF64(tau); kwargs...)

# ── driver ───────────────────────────────────────────────────────────────────


