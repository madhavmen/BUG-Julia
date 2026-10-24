# THE SPINFUL AUBRY-ANDRE-HUBBARD CHAIN ON FUSED d = 4 SITES (arXiv:1501.05661), graded against
# hubbard_aa_exact.jl. The interleaved layout (up1 down1 up2 down2 ...) makes every hop range 2, which
# a two-site sweep cannot see (H_eff vanishes on a product state). One physical site per MPS site
# keeps every hop range 1.

# ── the local space ──
# Telum's own spinful-fermion space (`FermionSOptions(1, charge, spin)`, dimension 4). It carries the
# fermionic signs: the intra-site Jordan-Wigner string sits in the annihilators, and the site parity Z
# carries the inter-site part. Mode order hub_mode(i,s) = 2(i-1)+s matches hubbard_aa_exact.jl.
const _HUB4_SPACES = Dict{Symbol, Any}()

"""
    hub4_space(sym = symmetry_mode()) -> NamedTuple

Fused d = 4 spinful-fermion local space, built once per mode and cached. Returns
`(; Fu, Fd, Sp, Sz, Sm, Z, I)` with `Z = (-1)^(n_up + n_down)`.
  * `:none`: dense 4x4 blocks, `Fu`/`Fd` rank 2. `:U1`: charge x spin, `Fu`/`Fd` rank 3.
`:SU2` is refused: a definite-Sz CDW product state is not representable in a total-spin basis.
"""
function hub4_space(sym::Symbol = symmetry_mode())
    haskey(_HUB4_SPACES, sym) && return _HUB4_SPACES[sym]
    sym === :SU2 && throw(ArgumentError(
        "hub4_space has no :SU2 form -- the experiment's spin-resolved CDW start state is not " *
        "representable in a total-spin basis. Use :U1 (charge x spin) or :none."))
    sym === :Z2 && throw(ArgumentError("hub4_space has no :Z2 form"))
    opts = sym === :U1   ? FermionSOptions(1, :U1, :U1, nothing) :
           sym === :none ? FermionSOptions(1, nothing, nothing, nothing) :
           throw(ArgumentError("symmetry must be :U1 or :none, got $sym"))
    _HUB4_SPACES[sym] = getLocalSpace(opts, ("s", "s", "op"))
    return _HUB4_SPACES[sym]
end

"""
    hub4_basis_index(q) -> NamedTuple

Dense indices of |0>, |up>, |down> and |up down> in the d = 4 local space, read off the diagonals of
n_up and n_down rather than hardcoded, since the ordering from `reduce` can come out mirrored.
Returns 1-based indices. `:none` only; under `:U1` the four states are separate one-dimensional
sectors and are selected by label.
"""
function hub4_basis_index(q)
    nu = diag(Matrix(to_concrete(lock(q.Fu', 2) * q.Fu).RMTs[1]))
    nd = diag(Matrix(to_concrete(lock(q.Fd', 2) * q.Fd).RMTs[1]))
    length(nu) == 4 || error("hub4_basis_index: local dimension $(length(nu)), expected 4")
    find(a, b) = begin
        hits = findall(k -> isapprox(nu[k], a; atol = 1e-10) &&
                            isapprox(nd[k], b; atol = 1e-10), 1:4)
        length(hits) == 1 || error(
            "hub4_basis_index: (n↑,n↓)=($a,$b) matched $(length(hits)) of 4 basis states; " *
            "n↑ = $nu, n↓ = $nd -- the local space is not the expected Hubbard site")
        only(hits)
    end
    return (; empty = find(0, 0), up = find(1, 0), down = find(0, 1), both = find(1, 1))
end

# ── U(1) charge x U(1) spin labels ──
# Telum appends charge first (`n_up + n_down - nchannels`, particle-hole centred), then spin
# (`n_up - n_down`), so the four states are four distinct one-dimensional sectors. ONE TUPLE PER
# SYMMETRY (`charge_symmops`/`spin_symmops` weights are `(w,)`): `((-1, 0),)` matches no leg.
const HUB4_EMPTY = ((-1,), (0,))
const HUB4_UP    = ((0,), (1,))
const HUB4_DOWN  = ((0,), (-1,))
const HUB4_BOTH  = ((1,), (0,))

hub4_sector(occ::Symbol) =
    occ === :empty ? HUB4_EMPTY :
    occ === :up    ? HUB4_UP    :
    occ === :down  ? HUB4_DOWN  :
    occ === :both  ? HUB4_BOTH  :
    throw(ArgumentError("occupation must be :empty, :up, :down or :both, got $occ"))

"""
    hub4_product_state(occ::Vector{Symbol}) -> SymMPS

A product state on fused d = 4 sites, one entry of `occ` per PHYSICAL site, each `:empty`,
`:up`, `:down` or `:both`.

Same fusion-isometry construction as `product_state`: fuse the incoming link with a fresh
physical leg, keep only the requested local state, then trim the outgoing link to what survives
-- so the reported bond dimension is 1. The physical leg keeps its full space
(`preserve_space = true`) so local operators stay contractible against it.
"""
function hub4_product_state(occ::Vector{Symbol})
    L = length(occ)
    L >= 1 || throw(ArgumentError("hub4_product_state needs at least one site"))
    sym = symmetry_mode()
    q = hub4_space(sym)
    idx = sym === :none ? hub4_basis_index(q) : nothing

    tensors = Any[]
    prev = (getvac(q.I, ("L,0", "L,1")), 2)
    for i in 1:L
        F = to_concrete(getIdentity(prev, (q.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = if sym === :none
            k = getfield(idx, occ[i] === :empty ? :empty : occ[i] === :up ? :up :
                              occ[i] === :down ? :down : occ[i] === :both ? :both :
                              throw(ArgumentError("bad occupation $(occ[i]) at site $i")))
            to_concrete(getsub(F, 3, s -> k))
        else
            target = hub4_sector(occ[i])
            B = getsub(F, 2, s -> s == target ? Colon() : nothing; preserve_space = true)
            B = to_concrete(B)
            isempty(B.qlabels) && error("hub4_product_state: site $i sector $target matches " *
                "none of the site leg's labels $(first.(F.spaces[2]))")
            present = Set(ql[3] for ql in B.qlabels)
            to_concrete(getsub(B, 3, s -> s in present ? Colon() : nothing))
        end
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

# ── the MPO ─────────────────────────────────────────────────────────────────────────────────

"""
    _hub4_channels(q) -> Vector{Tuple{Any, Any}}

The four hopping channels as `(opening, closing)` local-operator pairs, for both spins and both
hop directions.

For i < j, c^dag_{s,j} c_{s,i} = f^dag_j (prod_{i<=k<j} Z_k) f_i. At j = i+1 the string is the
single Z_i, to the left of f_i. The string is the same for both spins because the up/down
distinction is intra-site. Order matters: Z and f anticommute, so swapping them flips the sign of a
channel and gives a smooth but wrong imbalance decay.
"""
function _hub4_channels(q)
    chans = Any[]
    for A in (q.Fu, q.Fd)
        ZA  = to_concrete(lock(q.Z, 1) * A)          # Z · f4σ   (opens channel A)
        Ad  = to_concrete(A')                        # f4σ†      (closes channel A)
        AdZ = to_concrete(A' * lock(q.Z, 2))         # f4σ† · Z  (opens channel B)
        push!(chans, (_hub4_half(ZA, :left), _hub4_half(Ad, :right)))
        push!(chans, (_hub4_half(AdZ, :left), _hub4_half(to_concrete(A), :right)))
    end
    return chans
end

"""
    _hub4_half(O, side) -> TLArray

`O` as one half of a hop under `:U1`, where it carries an op-leg (rank 3; `:none` is returned as
is). The op-leg goes LAST: a product keeps its factors' leg order, so `f' * Z` is `(s, op, s)`
and `_mpo_retag` would tag its op-leg as a site leg (dt2a: "Duplicate TLIndex"). And it must
point out of an opening half and into a closing one (`_mpo_block`). Telum's fermion space has
annihilators only, so channel B opens on `f'` and closes on `f`, both pointing the other way;
`legflip` takes each to the dual, exact for U(1), which leaves their contraction unchanged.
"""
function _hub4_half(O, side::Symbol)
    length(O.inds) == 3 || return O
    k = findfirst(l -> occursin("op", O.inds[l].itags.value), 1:3)
    k === nothing && error("_hub4_half: no op-leg among the legs $(O.inds)")
    O = to_concrete(permutedims(O, (filter(!=(k), 1:3)..., k)))
    O.inds[3].dir == (side === :left ? '-' : '+') || (O = to_concrete(legflip(O, 3)))
    return O
end

"""
    hubbard_aa_mpo_d4(L; J, Delta, U, phi, beta) -> MPO

The Aubry-Andre-Hubbard chain of arXiv:1501.05661 on `L` fused d = 4 sites. `L` counts physical
sites, not modes.

Virtual dimension 6 (id, four hopping channels, done), independent of `L`, `U` and `Delta`.
`beta = 532/738` is the experiment's wavelength ratio; `phi` is the realisation label.

The field sits in the id -> done slot, so it must survive both boundary reductions. That corner is
zeroed by a field-free MPO at i = 1 and i = L, which would silently remove the potential at the ends.
"""
function hubbard_aa_mpo_d4(L::Int; J::Float64 = 1.0, Delta::Float64 = 0.0,
                           U::Float64 = 0.0, phi::Float64 = 0.0,
                           beta::Float64 = 532 / 738)
    L >= 2 || throw(ArgumentError("hubbard_aa_mpo_d4 needs L >= 2, got $L"))
    sym = symmetry_mode()
    q = hub4_space(sym)

    nu = to_concrete(lock(q.Fu', 2) * q.Fu)
    nd = to_concrete(lock(q.Fd', 2) * q.Fd)
    ntot = to_concrete(nu + nd)
    nupdn = to_concrete(lock(nu, 1) * nd)              # n↑ n↓ -- the Hubbard interaction

    chans = _hub4_channels(q)
    n = length(chans)                                  # 4
    ncol = n + 2                                        # 6

    # The quasiperiodic potential. `i` is the PHYSICAL site index, and both species see the same
    # `h_i` -- the potential is spin-independent, which is what makes the doublon fraction a
    # meaningful initial-state parameter rather than a field effect.
    h = [Delta * cos(2π * beta * i + phi) for i in 1:L]

    Ws = Any[]
    for i in 1:L
        tl, tr = "W,$(i - 1)", "W,$i"
        Iid  = _mpo_block(q.I, i, tl, tr, :none)
        Znil = to_concrete(0.0 * Iid)

        # ── every slot filled, then overwritten ──
        # `oplus` infers an omitted entry's spaces from its row and column, which fails on a row that is
        # empty at some site. Pre-filling with shape-correct zeros avoids that.
        mat = Matrix{Any}(undef, ncol, ncol)
        fill!(mat, Znil)

        i < L && (mat[1, 1] = Iid)                     # stay `id`
        i > 1 && (mat[ncol, ncol] = Iid)               # stay `done`

        onsite = to_concrete(h[i] * ntot)
        U == 0.0 || (onsite = to_concrete(onsite + U * nupdn))
        mat[1, ncol] = _mpo_block(onsite, i, tl, tr, :none)

        for (t, (open_op, close_op)) in enumerate(chans)
            i < L && (mat[1, 1 + t] = _mpo_block(open_op, i, tl, tr,
                                                 _side(open_op, :left)))
            i > 1 && (mat[1 + t, ncol] = to_concrete(-J *
                          _mpo_block(close_op, i, tl, tr, _side(close_op, :right))))
        end

        if i == 1
            mat = reshape(mat[1, :], 1, ncol)          # the `id` row only
        elseif i == L
            mat = reshape(mat[:, ncol], ncol, 1)       # the `done` column only
        end
        push!(Ws, to_concrete(oplus(mat, (1, 4))))
    end
    return MPO(Ws)
end

# ── observables ─────────────────────────────────────────────────────────────────────────────

"""
    hub4_density(psi, i) -> Float64

`⟨n_i↑ + n_i↓⟩` on physical site `i`, in `0 … 2`.

⛔ THE IMBALANCE IS A CHARGE OBSERVABLE, NOT A SPIN ONE, which is what makes it safe here. The
recorded trap is that `apply_sz!` silently returned `½·ψ` under `:SU2`, so no `S^z` quantity may
be read in a total-spin basis; the density is invariant under spin rotations and has no such
issue in any mode this file supports.
"""
function hub4_density(psi::SymMPS, i::Int)
    q = hub4_space()
    nt = to_concrete(lock(q.Fu', 2) * q.Fu + lock(q.Fd', 2) * q.Fd)
    return real(site_expval(psi, i, nt))
end

"""
    hub4_imbalance(psi) -> Float64

`I = (N_even - N_odd) / (N_even + N_odd)`, the experiment's observable, from the density profile.

The CDW start state puts every atom on EVEN sites, so `I(0) = +1` exactly -- which is the cheap
first check that the state was built as intended, and the one that `⟨H⟩` cannot provide.
"""
function hub4_imbalance(psi::SymMPS)
    L = length(psi)
    ns = [hub4_density(psi, i) for i in 1:L]
    ne = sum(ns[i] for i in 2:2:L; init = 0.0)
    no = sum(ns[i] for i in 1:2:L; init = 0.0)
    tot = ne + no
    return tot <= 0 ? NaN : (ne - no) / tot
end

"""
    hub4_cdw_occupations(L; doublon_frac, seed) -> Vector{Symbol}

The experiment's initial charge-density wave on `L` physical sites: even sites occupied, odd
sites empty, with a fraction `doublon_frac` of the occupied sites holding a DOUBLON.

Same ensemble member selection as `cdw_species` in the driver -- deterministic in `seed`, and the
singly-occupied sites alternate `:up`/`:down` rather than being random, so the two species are
balanced to within one atom instead of O(√L). See `benchmarks/mbl_scars.jl`.
"""
function hub4_cdw_occupations(L::Int; doublon_frac::Float64 = 0.34, seed::Int = 1)
    0.0 <= doublon_frac <= 1.0 || throw(ArgumentError(
        "doublon_frac must be in [0,1], got $doublon_frac"))
    iseven(L) || throw(ArgumentError("hub4_cdw_occupations needs even L, got $L"))
    occupied = collect(2:2:L)
    ndbl = round(Int, doublon_frac * length(occupied))
    rng = MersenneTwister(seed)
    dbl = Set(occupied[randperm(rng, length(occupied))[1:ndbl]])

    occ = fill(:empty, L)
    nsingle = 0
    for s in occupied
        if s in dbl
            occ[s] = :both
        else
            nsingle += 1
            occ[s] = isodd(nsingle) ? :up : :down
        end
    end
    return occ
end
