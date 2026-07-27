# CBE-BUG: CBE expansion plus Algorithm 7, at every bond of one sweep.
#
# Structure:
#
#   for each bond i = 1 … L-1
#       expand the bond (`cbe_expand`: RSVD sketch of HΘ, sector-graded, no expv)
#       Algorithm 7 on that bond: project into the expanded space, Galerkin step, split
#   truncation sweep (Sulz algorithm 1)
#
# THE K/L ODEs ARE NEVER SOLVED. `cbe_expand` supplies the directions those solves would
# have found -- one application of H, sketched -- and the per-bond Galerkin `expv` builds
# the Krylov basis on the expanded bond and takes the step in it. So there is no frozen
# projector and no discarded projector, and no one-site `expv` per bond either.
#
# A GALERKIN STEP AT EVERY BOND IS NOT OPTIONAL. An earlier version did the basis sweeps
# and then a single Galerkin step at the centre. That cannot grow a chain's ranks: an
# expansion is LOSSLESS, so it cannot change the state's Schmidt rank, and only the bond
# that gets a Galerkin step gains any. The centre was then capped at
# `room_l = d·χ_{c-1} − r` because χ_{c-1} never moved -- L=18 sat at maxbond 2 with an
# error independent of dt. The reference recursion (`exploratory_bug/src/TTN/ttn_bug.jl`,
# algorithms 5-7) updates the connecting tensor at EVERY node too; the root's update is
# merely the last one, not the only one.
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

# ── the step ─────────────────────────────────────────────────────────────────

"""
    CBEBugInfo

What one [`cbe_bug_bond_update`](@ref) step did.

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
(`tensor_elements` lives in tdvp2_baseline.jl; both callers resolve it at run time.)"
_state_stored(psi::SymMPS) = sum(tensor_elements(psi[i]) for i in 1:length(psi); init = 0)

"""
    cbe_bug_bond_update(psi, h, tau; kwargs...) -> CBEBugInfo

One CBE-BUG time step of `tau = -im*dt`, in place.

Keywords: `dex`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only` forward to
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
function cbe_bug_bond_update(psi::SymMPS, h::XXZChain, tau::ComplexF64;
                             dex::Int = 0,
                             dover::Union{Nothing, Int} = nothing,
                             comp_ratio::Float64 = 0.5,
                             sulz_cap::Bool = true,
                             preselect_only::Bool = false,
                             centre_expand::Bool = true,
                             maxdim::Int = 200,
                             trunc_thresh::Float64 = 1e-12,
                             maxiter::Int = 30,
                             tol::Float64 = 1e-15,
                             rng::AbstractRNG = MersenneTwister(0x5EED))
    L = length(psi)
    length(h) == L || throw(DimensionMismatch(
        "chain has $(length(h)) sites, state has $L"))
    L >= 2 || throw(ArgumentError("cbe_bug_bond_update needs at least two sites"))
    c = max(1, min(L - 1, L ÷ 2))                     # the centre bond

    expanded = zeros(Int, L - 1)
    n_new    = zeros(Int, L - 1)
    err_pre  = 0.0
    err_fnl  = 0.0
    peak     = 0

    exkw = (dex = dex, dover = dover, comp_ratio = comp_ratio,
            sulz_cap = sulz_cap, preselect_only = preselect_only, rng = rng)

    """
    ALGORITHM 7 at one bond, in the CBE-expanded frames: project the block into the
    expanded space, take the Galerkin step there, split back.

    This is what makes the rank grow, and nothing else does. An expansion is LOSSLESS, so
    widening a bond cannot change the state's Schmidt rank -- `Ŝ₀` is zero-padded on the
    new directions and the state is still exactly what it was. It is this `expv` that puts
    weight on them; a bond that never gets one collapses at the next gauge move, which is
    correct reporting and not a lost expansion. Measured before this was per-bond: L=18 sat
    at maxbond 2 with an error INDEPENDENT of dt, because only the centre bond had a
    Galerkin step and `room_l = d·χ_{c-1} − r` then pinned even that at 2.

    The K/L ODEs are NOT solved. `expv` builds the Krylov basis on the expanded bond and
    steps in it, with `apply_zero_site` costing `O(χ²)` per matvec and never touching a
    site tensor -- so the K/L directions come from CBE, the step from this Krylov space.
    """
    function alg7_bond!(i::Int, lenv::ChannelSet, renv::ChannelSet, ex::CBEExpansion)
        ML = contract(ex.U_ex', (1, 2), psi[i], (1, 2))          # (bond_ex_l, bond_old)
        MR = contract(psi[i + 1], (2, 3), ex.V_ex', (2, 3))      # (bond_old, bond_ex_r)
        S0 = to_concrete(contract(ML, (2,), MR, (1,)))           # zero-padded on the new dirs
        peak!(ex.U_ex, ex.V_ex, S0)
        H0 = zero_site_h(h, i, lenv, renv, ex.U_ex, ex.V_ex)
        S1 = expv(x -> apply_zero_site(H0, x), tau, S0;
                  hermitian = true, maxiter = maxiter, tol = tol)
        res = svd(S1, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim,
                  get_lists = true)
        tag = psi[i].inds[3].itags
        psi[i]     = to_concrete(setitag(ex.U_ex * res.U, 3, tag))
        psi[i + 1] = to_concrete(setitag((res.S * res.Vd) * ex.V_ex, 1, tag))
        psi.center = i + 1
        return _trunc_weight(res)
    end

    function record!(ex)
        err_pre = max(err_pre, ex.err_pre)
        err_fnl = max(err_fnl, ex.err_fnl)
        return ex
    end

    "Working set at this instant: the state as it stands plus the transients still alive."
    peak!(transients...) = (peak = max(peak, _state_stored(psi) +
                                       sum(tensor_elements(t) for t in transients; init = 0)))

    # ONE SWEEP, EVERY BOND: CBE-expand, then Algorithm 7 on that bond.
    #
    # This replaces the earlier "K/L basis sweeps + a single Galerkin step at the centre".
    # That structure cannot grow a chain's ranks: the expansion is lossless, so only the
    # bond that gets a Galerkin step gains rank, and the centre is then capped at
    # `room_l = d·χ_{c-1} − r` because χ_{c-1} never moves. The reference recursion
    # (`exploratory_bug/src/TTN/ttn_bug.jl`, algorithms 5-7) likewise updates the connecting
    # tensor at EVERY node and not only at the root -- the root's update is merely the last.
    #
    # ORDERING IS LOAD-BEARING. The left channels are carried (one `push_left_channels` per
    # bond) and the right ones are read from a single prebuilt stack. `alg7_bond!` writes
    # only sites `i` and `i+1`, so `rstack`'s links `>= i+2` stay valid for the whole sweep,
    # and it leaves the centre at `i+1`, which is exactly where bond `i+1` needs it -- no
    # `canonical!` inside the loop at all. Getting this backwards reads a stale environment,
    # which does not throw; it silently evolves with the wrong Hamiltonian.
    canonical!(psi, 1)
    rstack = right_env_stack(psi, h; downto = 3)
    lch = boundary_channels(h)                        # link 1, what bond 1 reads
    discarded = 0.0
    for i in 1:(L - 1)
        renv = right_channels(rstack, i + 2)
        f = bond_frame(psi, i)
        # `centre_expand = false` keeps the centre bond in its UNexpanded frames; with
        # every bond now getting a Galerkin step this is a pure diagnostic A/B on how much
        # the expansion at that one cut is worth.
        ex = (centre_expand || i != c) ?
             record!(cbe_expand(f, h, i, lch, renv; exkw...)) :
             CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)
        expanded[i] = leg_dim(ex.U_ex, 3)
        n_new[i] = ex.n_new_l
        discarded = max(discarded, alg7_bond!(i, lch, renv, ex))
        lch = push_left_channels(lch, h, psi[i], i)
    end
    centre_rank = leg_dim(psi[c], 3)

    # Compress once at the end. Every bond has now had its Galerkin step, so a direction
    # with no weight left is genuinely unwanted rather than merely not-yet-populated.
    truncate_sweep!(psi; maxdim = maxdim, cutoff = max(trunc_thresh, 1e-14))

    return CBEBugInfo(expanded, n_new, centre_rank, err_pre, err_fnl, discarded, 0, peak)
end

cbe_bug_bond_update(psi::SymMPS, h::XXZChain, tau::Number; kwargs...) =
    cbe_bug_bond_update(psi, h, ComplexF64(tau); kwargs...)

# ── driver ───────────────────────────────────────────────────────────────────

"""
    CBEBugOptions

Controls for one `cbe_bug!` run. Mirrors `BondUpdateOptions` where the meaning is the
same, so the two integrators can be driven from the same campaign code.

  - `dt`, `n_steps` -- real time step and how many to take.
  - `dex`, `dover`, `comp_ratio`, `sulz_cap`, `preselect_only` -- the expansion, forwarded
    to [`cbe_expand`](@ref). `dex = 0` is the full Sulz budget `r`.
  - `maxdim`, `trunc_thresh` -- truncation, applied at the centre and in the final sweep.
  - `normalize` -- rescale after each step; the norm is recorded BEFORE rescaling.
  - `maxiter`, `tol` -- Krylov budget for the single 0-site exponential.
  - `seed` -- one RNG for the whole run, so a run is reproducible while consecutive steps
    still draw different sketches.

NO `order` FIELD. `bond_update_bug!` offers `:lie` / `:strang` because it Trotterises the
even and odd bond groups and has a splitting error to compose away. CBE-BUG takes one
projected exponential per step over a globally expanded subspace -- there is no operator
splitting to symmetrise, so a composition parameter here would be meaningless.

REAL TIME ONLY, as for `bond_update_bug!`: the driver forms `tau = -im*dt` itself.
"""
Base.@kwdef struct CBEBugOptions
    dt::Float64 = 0.05
    n_steps::Int = 10
    dex::Int = 0
    dover::Union{Nothing, Int} = nothing
    comp_ratio::Float64 = 0.5
    sulz_cap::Bool = true
    preselect_only::Bool = false
    centre_expand::Bool = true
    maxdim::Int = 200
    trunc_thresh::Float64 = 1e-12
    normalize::Bool = true
    maxiter::Int = 30
    tol::Float64 = 1e-15
    seed::UInt = 0x5EED
    record_magnetisation::Bool = false
    observe::Union{Nothing, Function} = nothing
end

"""
    CBEBugRunInfo

Per-step record of a `cbe_bug!` run; every field has length `n_steps`.

`max_expanded` is the headline rank diagnostic and the one to compare against
`BondUpdateInfo.aug_k_dims`: it is the largest bond the basis sweep proposed BEFORE the
truncation, i.e. `r + Dex`, against the K/L augmentation's `2r`.
"""
struct CBEBugRunInfo
    times::Vector{Float64}
    norms::Vector{Float64}
    bond_dims::Vector{Vector{Int}}
    max_bond_dims::Vector{Int}
    expanded::Vector{Vector{Int}}
    max_expanded::Vector{Int}
    centre_ranks::Vector{Int}
    err_pre::Vector{Float64}
    err_fnl::Vector{Float64}
    discarded::Vector{Float64}
    magnetisations::Vector{Vector{Float64}}
    observations::Vector{Any}
end

Base.length(info::CBEBugRunInfo) = length(info.times)

"""
    cbe_bug!(psi, h; opts=CBEBugOptions()) -> CBEBugRunInfo

Evolve `psi` in place for `opts.n_steps` real-time steps of `opts.dt` with CBE-BUG.

One RNG is threaded through the whole run rather than re-seeded per step, so the sketches
are reproducible without every step drawing the same directions.
"""
function cbe_bug!(psi::SymMPS, h::XXZChain; opts::CBEBugOptions = CBEBugOptions())
    rng = MersenneTwister(opts.seed)
    times = Float64[]; norms = Float64[]
    bdims = Vector{Int}[]; maxb = Int[]
    expanded = Vector{Int}[]; maxex = Int[]
    cranks = Int[]; epre = Float64[]; efnl = Float64[]; disc = Float64[]
    mags = Vector{Float64}[]; obs = Any[]

    for step in 1:opts.n_steps
        info = cbe_bug_bond_update(psi, h, ComplexF64(-im * opts.dt);
                                   dex = opts.dex, dover = opts.dover,
                                   comp_ratio = opts.comp_ratio,
                                   sulz_cap = opts.sulz_cap,
                                   preselect_only = opts.preselect_only,
                                   centre_expand = opts.centre_expand,
                                   maxdim = opts.maxdim,
                                   trunc_thresh = opts.trunc_thresh,
                                   maxiter = opts.maxiter, tol = opts.tol, rng = rng)

        n = norm(psi)                                # recorded BEFORE renormalising
        if opts.normalize && n > 0
            psi[psi.center] = to_concrete((1.0 / n) * psi[psi.center])
        end

        bd = bond_dims(psi)
        push!(times, step * opts.dt); push!(norms, n)
        push!(bdims, bd); push!(maxb, maximum(bd; init = 0))
        push!(expanded, info.expanded); push!(maxex, maximum(info.expanded; init = 0))
        push!(cranks, info.centre_rank)
        push!(epre, info.err_pre); push!(efnl, info.err_fnl)
        push!(disc, info.discarded)

        # Diagnostics read a copy, so moving the centre can never disturb the live state.
        opts.record_magnetisation && push!(mags, magnetisation(copy(psi)))
        opts.observe === nothing || push!(obs, opts.observe(copy(psi), step, step * opts.dt))
    end

    return CBEBugRunInfo(times, norms, bdims, maxb, expanded, maxex, cranks,
                         epre, efnl, disc, mags, obs)
end
