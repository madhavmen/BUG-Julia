# THE SPINFUL AUBRY-ANDRÉ-HUBBARD CHAIN ON FUSED d = 4 SITES.
#
#     H = -J Σ_{i,σ} (c†_{i+1,σ} c_{i,σ} + h.c.)
#         + Δ Σ_i cos(2πβi + φ) (n_{i↑} + n_{i↓})
#         + U Σ_i n_{i↑} n_{i↓}
#
# arXiv:1501.05661 -- the actual experiment, and the model Figs. 2, 4 and 5 measure. The spinless
# XXZ proxy in `mbl_scars.jl` reaches only Fig. 3 (`U = 0`, "only a single spin component was
# used"). Graded against `benchmarks/hubbard_aa_exact.jl`, which is dependency-free and computes
# every fermionic sign by counting occupied modes.
#
# ⛔⛔ THIS FILE EXISTS BECAUSE THE INTERLEAVED LAYOUT CANNOT BE EVOLVED AT ALL, AND THE REASON IS
# NOT A BUG. Mapping two species onto one spin chain of `2L` sites in the order ↑₁ ↓₁ ↑₂ ↓₂ …
# makes every hop a RANGE-2 term, since ↑ᵢ → ↑ᵢ₊₁ steps over ↓ᵢ. A range-2 Hamiltonian has NO
# term that lies wholly inside a two-site block, so the two-site effective Hamiltonian sees the
# far operator only through the environment, which on a product state contributes
# ⟨φ|S∓|φ⟩ = 0. Therefore `H_eff ≡ 0` on every bond and the CDW start state is an EXACT FIXED
# POINT of every two-site sweep -- bug, tdvp2 and cbe1s alike.
#
# MEASURED (job 16339405, identical in every other respect): distance-1 coupling gives
# `bond = 5` and evolves; distance-2 gives `bond = 1` and is frozen forever.
#
# ⛔ AND THE TELLS ALL POINT AT THE MPO INSTEAD. `‖ψ‖ = 1`, `energy_drift = 0`, `dt-ratio = 1.00`,
# identical under `:none` and `:U1`, and -- worst -- `⟨H⟩ = 0` EXACTLY as it should be, because
# hopping has zero expectation on ANY product state. Every energy-style check passes on a
# kinetically empty MPO. `maxbond` is the only witness that separates the two.
#
# ONE PHYSICAL SITE = ONE MPS SITE, with the local basis |0⟩, |↑⟩, |↓⟩, |↑↓⟩. Both species' hops
# are then RANGE-1 and a two-site update has something to do. This is also simply the standard
# Hubbard MPS layout; the interleaved chain was the deviation.

# ── the local space ─────────────────────────────────────────────────────────────────────────
#
# Telum's OWN spinful-fermion space, `FermionSOptions(1, charge, spin)`: one channel, so the
# local dimension is `4^1 = 4`. Using the library's space rather than a hand-built one matters
# for exactly one reason -- IT ALREADY CARRIES THE FERMIONIC SIGNS:
#
#     FF[1] = f2 ⊗ I2      (annihilate ↑; nothing lies below ↑ within the site)
#     FF[2] = z2 ⊗ f2      (annihilate ↓; the string over ↑ᵢ IS the leading z2)
#
# so the INTRA-site Jordan-Wigner string is inside the operators, and the site-parity operator
# `Z = (-1)^(n↑+n↓)` carries the INTER-site part. Writing `c_{σ,i}` out in the global mode order
# ↑₁ ↓₁ ↑₂ ↓₂ … gives, for BOTH species,
#
#     c_{σ,i} = (Π_{k<i} Z_k) · f4σ_i                                                     (*)
#
# which is what makes the two-channel construction below a literal duplication of Telum's
# SU(2)-fused `HubbardMPO` rather than a new derivation. See `_hub4_channels`.
#
# ⚠ THE MODE ORDER MATCHES `hubbard_aa_exact.jl` EXACTLY (`hub_mode(i,σ) = 2(i-1) + σ`, σ=0 for
# ↑), which is the only reason the two can be compared row by row. `f4up = f2 ⊗ I2` has no z2 and
# `f4down = z2 ⊗ f2` has one, i.e. ↑ precedes ↓ within a site -- the same ascending convention.
const _HUB4_SPACES = Dict{Symbol, Any}()

"""
    hub4_space(sym = symmetry_mode()) -> NamedTuple

The fused d = 4 spinful-fermion local space, built once per mode and cached.

Exposes `(; Fu, Fd, Sp, Sz, Sm, Z, I)`. `Fu`/`Fd` are the ↑/↓ ANNIHILATION operators and `Z` is
the site parity `(-1)^(n↑+n↓)`.

  * `:none` -- one dense 4x4 block per operator. `Fu`/`Fd` are rank 2.
  * `:U1`   -- U(1) charge x U(1) spin. `Fu`/`Fd` are rank 3: they change both charges, so they
               carry an op-leg exactly as `Sp`/`Sm` do in the spin-1/2 space.

⛔ `:SU2` IS REFUSED, AND NOT FOR LACK OF A TELUM OPTION -- `FermionSOptions(1, :U1, :SU2)` is
what Telum's own `HubbardMPO` uses. The obstruction is the INITIAL STATE: the experiment's CDW
assigns a definite ↑ or ↓ to each singly-occupied site, and a definite-Sz product state has
weight in many total-spin sectors, so it is not representable at any bond dimension (the same
reason `product_state` refuses under `:SU2`). A charge-only `FermionSOptions(1, :U1, nothing)`
would be representable, but then the ↑/↓ annihilators are not separately available.
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

Which dense index of the d = 4 local space is |0⟩, |↑⟩, |↓⟩ and |↑↓⟩, READ OFF THE OPERATORS
rather than assumed.

⛔⛔ DO NOT HARDCODE THIS. The ordering follows from `reduce(⊗, ...)` inside Telum, and the
recorded landmine in this project is exactly a pair of conventions that turned out to be MIRRORED
(`dense_state` vs `exact_sparse`) -- a transposition that is invisible on any symmetric test
state and wrong on every asymmetric one. Here it is cheap to be certain: `n↑` and `n↓` are
DIAGONAL, so their diagonals label the basis directly and the answer cannot be stale.

Returns 1-based indices into the dense 4-dim local space. `:none` only -- under `:U1` the four
states are four distinct one-dimensional charge sectors and are selected by label instead.
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

# U(1) charge x U(1) spin sector labels. Telum appends CHARGE first, then SPIN, with
# `charge = n↑ + n↓ - nchannels` (so it is particle-hole centred) and `spin = n↑ - n↓`:
#
#     |0⟩ -> (-1, 0)     |↑⟩ -> (0, +1)     |↓⟩ -> (0, -1)     |↑↓⟩ -> (+1, 0)
#
# All four are distinct, so the d = 4 space is four ONE-dimensional sectors under (:U1, :U1) --
# which is why a spin-resolved CDW product state has bond dimension 1 here.
const HUB4_EMPTY = ((-1, 0),)
const HUB4_UP    = ((0, 1),)
const HUB4_DOWN  = ((0, -1),)
const HUB4_BOTH  = ((1, 0),)

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

The four hopping channels as `(opening, closing)` local-operator pairs.

FROM `(*)` IN THE HEADER, with no free choices left. For `i < j`,

    c†_{σ,j} c_{σ,i} = f4σ†_j · (Π_{i≤k<j} Z_k) · f4σ_i

and at `j = i+1` there is no intervening site, so the whole string is the single `Z_i`, which
sits to the LEFT of `f4σ_i` in that product. Hence

    c†_{i+1} c_i  =  (Z·f4σ)_i  ⊗  (f4σ†)_{i+1}          -- channel A
    c†_i c_{i+1}  =  (f4σ†·Z)_i ⊗  (f4σ)_{i+1}           -- channel B   (the h.c.)

These are Telum's own `ZF = lock(q.Z,1)*q.F` and `FcZ = q.F' * lock(q.Z,2)` from its
DMRG-validated `HubbardMPO` (`deps/Telum/test/support/dmrg.jl`), duplicated per species. The
duplication is legitimate precisely because `(*)` holds for BOTH σ with the same `Z`: the part of
the string that distinguishes ↑ from ↓ is intra-site and already inside `f4up`/`f4down`.

⛔ ORDER IS NOT COSMETIC: `Z` and `f4σ` ANTICOMMUTE (`Z f4up = -f4up Z`, since `z2 f2 = f2` but
`f2 z2 = -f2`), so `lock(q.Z,1)*q.Fu` and `q.Fu*lock(q.Z,2)` differ by a global sign on that
channel. Getting it backwards leaves `H` Hermitian and particle-number conserving and produces a
perfectly smooth, entirely wrong imbalance decay -- which is why this is graded against
`hubbard_aa_exact.jl` and not against its own plausibility.
"""
function _hub4_channels(q)
    chans = Any[]
    for A in (q.Fu, q.Fd)
        ZA  = to_concrete(lock(q.Z, 1) * A)          # Z · f4σ   (opens channel A)
        Ad  = to_concrete(A')                        # f4σ†      (closes channel A)
        AdZ = to_concrete(A' * lock(q.Z, 2))         # f4σ† · Z  (opens channel B)
        push!(chans, (ZA, Ad))
        push!(chans, (AdZ, to_concrete(A)))
    end
    return chans
end

"""
    hubbard_aa_mpo_d4(L; J, Delta, U, phi, beta) -> MPO

The Aubry-André-Hubbard chain of arXiv:1501.05661 on `L` FUSED d = 4 sites.

`L` is the number of PHYSICAL sites, not modes -- the MPO has `L` tensors and the state has `L`
sites, in contrast to the interleaved `hubbard_aa_mpo`, which needed `2L` of each.

Virtual dimension 6: `id`, four half-open hopping channels (↑ and ↓, each with its h.c.
partner), and `done`. Independent of `L`, `U` and `Delta`.

`beta = 532/738` is the experiment's laser wavelength ratio; `phi` is the realisation label.

⚠ THE FIELD SITS IN THE `id -> done` SLOT, so it must SURVIVE both boundary reductions. At `i = 1`
only the `id` row is kept and at `i = L` only the `done` column, and in both cases that slot is
the corner that a field-free MPO zeroes out. Zeroing it here would silently delete the entire
quasiperiodic potential at the two ends -- an `L`-independent O(1/L) error that looks like a
boundary effect.
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

        # ⛔ EVERY SLOT FILLED, THEN OVERWRITTEN. `oplus` infers an omitted entry's spaces from
        # the other entries in its row and column, and a row that is entirely empty at some site
        # has nothing to infer from: "cannot infer zero TLArray at (3,1): missing space
        # information on leg 1" aborted job 16335193 for exactly that. Under `:none` every block
        # here shares one space, so pre-filling with a shape-correct zero removes the failure mode
        # outright instead of reasoning per-slot about which states are reachable yet.
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
