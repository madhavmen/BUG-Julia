# two_site_bug.jl
#
# Two-site BUG integrator: odd/even (checkerboard) Strang sweep of faithful-KLS
# local bond updates.
#
# Idea
# ----
# Write the Hamiltonian as a Trotter (checkerboard) split into two parity groups
#
#       H  =  H_odd + H_even ,
#       H_odd  = Σ_{b odd}  h_{b,b+1} ,    H_even = Σ_{b even} h_{b,b+1} ,
#
# where each h_{b,b+1} is a two-site (nearest-neighbour) term. The bonds inside
# one parity touch DISJOINT site pairs (odd: 1-2, 3-4, 5-6 …; even: 2-3, 4-5 …),
# so the terms inside H_odd (resp. H_even) mutually commute and
#   exp(-i τ H_par) = Π_{b∈par} exp(-i τ h_{b,b+1}) .
#
# One symmetric (Strang) step approximates exp(-i dt H) by
#
#       exp(-i dt H) ≈ U_odd(dt/2) U_even(dt) U_odd(dt/2) + O(dt^3) ,
#
# and each factor exp(-i τ h_{b,b+1}) is realised by the **BUG KLS local update**
# (K-augment, L-augment, S-evolve) at bond b.
#
# Applying the Hamiltonian, not its exponential
# ---------------------------------------------
# We feed the KLS step the *Hamiltonian term* h_{b,b+1} (the slice of the parity
# MPO living on that bond), NOT a precomputed gate exp(-i τ h). The KLS step is
# the integrator: its K/L/S substeps exponentiate the projected effective
# Hamiltonian internally (Krylov / Lanczos). We never form exp(-i τ H_par) or a
# per-bond gate matrix.
#
# Why per-bond term and a trivial environment (no double counting)
# ----------------------------------------------------------------
# Within a parity the other terms act on DISJOINT sites. If we contracted the
# whole parity MPO into the effective Hamiltonian at bond b, its environment
# would re-introduce the other parity terms (e.g. h_{34}, h_{56} while updating
# bond 1) and they would be applied a SECOND time at their own bonds → double
# counting (a flat error floor that does not shrink with dt). Instead each bond
# sees only its own term h_{b,b+1}; the canonical environment is the identity,
# which is why the local update reproduces exp(-i τ h_{b,b+1}) exactly at full
# rank.
#
# Gauge (no inverse)
# ------------------
# A SINGLE orthogonality centre is moved along the chain with QR/LQ (via
# `orthogonalize!`). At the active bond that gives exactly the left-isometry /
# right-isometry condition the KLS step needs — for that one bond only. No
# simultaneous-centre canonical form (which would require a diagonal inverse on
# every bond) is used, so no diagonal inverse appears here.
#
# No backward correction
# ----------------------
# The two-site BUG has no backward (single-site, backward-in-time) sub-step; the
# disjoint parity structure removes the shared-site double counting that the
# 2-site TDVP backward step exists to cancel. `info.backward_correction_calls`
# stays 0.
#
# Order and error
# ---------------
# At full rank each bond update is the exact local evolution, so the only error
# is the odd/even Trotter splitting:
#   * `:lie`    — first order,  O(dt) global error.
#   * `:strang` — second order, O(dt^2) global error.

# ── Hamiltonian decomposition ────────────────────────────────────────────────

"""
    two_site_xx_bond_gates(sites; J = 1.0) -> Vector{ITensor}

Per-bond two-site term generators `h_{b,b+1}` for the nearest-neighbour XX
Hamiltonian `H = (J/2) Σ_b (S⁺_b S⁻_{b+1} + S⁻_b S⁺_{b+1})`, built directly with
`op`. Entry `b` is a 2-site operator on `(sites[b]', sites[b+1]', sites[b],
sites[b+1])`. These are the local terms the KLS step exponentiates; their sum
(lifted with identities on the other sites) equals the full Hamiltonian.
"""
function two_site_xx_bond_gates(sites; J::Real = 1.0)
    N = length(sites)
    gates = Vector{ITensor}(undef, N - 1)
    for b in 1:(N - 1)
        gates[b] = (J / 2) * (op("S+", sites[b]) * op("S-", sites[b + 1]) +
                              op("S-", sites[b]) * op("S+", sites[b + 1]))
    end
    return gates
end

"""
    two_site_xx_parity_mpos(sites; J = 1.0) -> (W_odd, W_even, W_full)

Reference parity MPOs for the nearest-neighbour XX Hamiltonian: `W_odd` carries
the odd-bond terms {1,3,5,…}, `W_even` the even-bond terms {2,4,…}. By
construction `matrix(W_odd) + matrix(W_even) == matrix(W_full)`. Provided for
verifying the Hamiltonian decomposition; the integrator itself consumes the
per-bond gate generators from `two_site_xx_bond_gates`.
"""
function two_site_xx_parity_mpos(sites; J::Real = 1.0)
    N = length(sites)
    os_odd  = OpSum()
    os_even = OpSum()
    os_full = OpSum()
    c = J / 2
    for b in 1:(N - 1)
        if isodd(b)
            os_odd += c, "S+", b, "S-", b + 1
            os_odd += c, "S-", b, "S+", b + 1
        else
            os_even += c, "S+", b, "S-", b + 1
            os_even += c, "S-", b, "S+", b + 1
        end
        os_full += c, "S+", b, "S-", b + 1
        os_full += c, "S-", b, "S+", b + 1
    end
    W_odd  = TensorTrainOperator(MPO(os_odd,  sites))
    W_even = TensorTrainOperator(MPO(os_even, sites))
    W_full = TensorTrainOperator(MPO(os_full, sites))
    return W_odd, W_even, W_full
end
# ── Local bond snapshot (canonical, no MPO needed) ───────────────────────────

"""
    _two_site_bond_snapshot(psi, bond) -> NamedTuple

Two-site canonical snapshot at `bond`: QR of `psi[bond]` gives the left isometry
`U0`, LQ of `psi[bond+1]` the right isometry `V0`, and `S0 = R_left · L_right`
the bond centre. Mirrors `_canonical_quantum_bond_snapshot` but carries no MPO
environment (the two-site local solve uses a per-bond gate with a trivial
environment, so no `L_mpo`/`R_mpo`/`W` fields are needed).
"""
function _two_site_bond_snapshot(psi::TensorTrain, bond::Int)
    bond_inds = _tensortrain_bond_indices(psi, bond)
    link_l    = bond_inds.link_l
    link_mid  = bond_inds.link_mid
    link_r    = bond_inds.link_r
    site_l    = bond_inds.site_l
    site_r    = bond_inds.site_r

    U0_tens, S_left_tens = qr(psi[bond], link_l, site_l;
        tags = join(string.(tags(link_mid)), ","), positive = false)
    canon_u0 = commonind(U0_tens, S_left_tens)

    S_right_tens, V0_tens = lq(psi[bond + 1], site_r, link_r)
    canon_v0 = commonind(S_right_tens, V0_tens)
    if tags(canon_v0) != tags(link_mid)
        new_canon_v0 = settags(canon_v0, join(string.(tags(link_mid)), ","))
        S_right_tens = replaceind(S_right_tens, canon_v0, new_canon_v0)
        V0_tens      = replaceind(V0_tens,      canon_v0, new_canon_v0)
        canon_v0     = new_canon_v0
    end

    S0_tens = S_left_tens * S_right_tens
    return (
        link_l   = link_l,
        link_mid = link_mid,
        link_r   = link_r,
        site_l   = site_l,
        site_r   = site_r,
        U0_tens  = U0_tens,
        V0_tens  = V0_tens,
        S0_tens  = S0_tens,
        canon_u0 = canon_u0,
        canon_v0 = canon_v0,
    )
end

"""
    _two_site_local_effective_hamiltonian(gate, bond_data) -> ITensor

Lift the bare two-site term `gate` (on the physical site indices) into a 2-site
effective-Hamiltonian ITensor by dressing it with the identity on the bond's
link legs. This is the environment-dressed local operator the KLS step expects,
with the canonical environment being the identity.
"""
function _two_site_local_effective_hamiltonian(gate::ITensor, bond_data)
    return gate *
           delta(bond_data.link_l, prime(bond_data.link_l)) *
           delta(bond_data.link_r, prime(bond_data.link_r))
end

# ── Parity sweep ─────────────────────────────────────────────────────────────

"""
    _two_site_parity_sweep!(psi, gates, dt, info; parity, ...) -> psi

Evolve `psi` by `exp(-i dt H_par)` = Π_{b∈par} exp(-i dt h_{b,b+1}) using the
BUG KLS local update on every bond of the requested parity. The orthogonality
centre is moved onto each bond with `orthogonalize!`; the local KLS step uses
only that bond's term `gates[b]` (trivial canonical environment) and
exponentiates it internally. No backward correction.
"""
function _two_site_parity_sweep!(
    psi              :: TensorTrain,
    gates            :: AbstractVector{ITensor},
    dt               :: Number,
    info             :: BUGInfo;
    parity           :: Symbol,
    maxdim           :: Int,
    trunc_thresh     :: Float64 = 0.0,
    augment          :: Bool,
    aug_krylov_depth :: Int,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    substep_method   :: Symbol,
    matrixfree_sstep :: Bool,
)
    N = length(psi)
    bonds = parity === :odd  ? collect(1:2:(N - 1)) :
            parity === :even ? collect(2:2:(N - 1)) :
            error("_two_site_parity_sweep!: parity must be :odd or :even")
    isempty(bonds) && return psi
    sweep_label = parity === :odd ? :odd : :even

    for b in bonds
        # Single moving centre: bring it onto bond b (QR/LQ, no inverse).
        orthogonalize!(psi, b)
        bond_data = _two_site_bond_snapshot(psi, b)
        HW_env = _two_site_local_effective_hamiltonian(gates[b], bond_data)

        candidate = _faithful_kls_local_bond_candidate(
            bond_data;
            dt = dt,
            s_dt = dt,
            augment = augment,
            aug_krylov_depth = aug_krylov_depth,
            lanczos_tol = lanczos_tol,
            lanczos_maxiter = lanczos_maxiter,
            substep_method = substep_method,
            matrixfree_sstep = matrixfree_sstep,
            checkerboard_threads = 1,
            HW_env_override = HW_env,
        )
        s_inds = inds(candidate.S_new)
        U_s, SV_tens, keep, svals = _truncate_quantum_s_step(
            candidate.S_new, s_inds[1], s_inds[2]; maxdim = maxdim, cutoff = trunc_thresh,
        )

        _record_kl_augmentation!(info, bond_data, candidate)
        push!(info.lanczos_numops, candidate.numops_s)
        _record_s_step_rank!(info, sweep_label, b, keep, svals)

        psi[b]     = candidate.U_aug_tens * U_s
        psi[b + 1] = SV_tens * candidate.V_aug_tens
    end
    return psi
end

# ── Single (Lie / Strang) step ───────────────────────────────────────────────

"""
    _two_site_strang_step!(psi, gates, dt, info; order, ...) -> psi

Apply one Lie (`order = :lie`) or Strang (`order = :strang`) odd/even sweep of
the faithful-KLS local bond update to `psi`, in place.
"""
function _two_site_strang_step!(
    psi              :: TensorTrain,
    gates            :: AbstractVector{ITensor},
    dt               :: Number,
    info             :: BUGInfo;
    order            :: Symbol,
    maxdim           :: Int,
    trunc_thresh     :: Float64 = 0.0,
    augment          :: Bool,
    aug_krylov_depth :: Int,
    lanczos_tol      :: Float64,
    lanczos_maxiter  :: Int,
    substep_method   :: Symbol,
    matrixfree_sstep :: Bool,
)
    sweep(τ, par) = _two_site_parity_sweep!(psi, gates, τ, info;
        parity = par, maxdim = maxdim, trunc_thresh = trunc_thresh,
        augment = augment, aug_krylov_depth = aug_krylov_depth,
        lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
        substep_method = substep_method, matrixfree_sstep = matrixfree_sstep,
    )
    if order === :strang
        sweep(dt / 2, :odd)
        sweep(dt,     :even)
        sweep(dt / 2, :odd)
    else  # :lie
        sweep(dt, :odd)
        sweep(dt, :even)
    end
    return psi
end

# ── Public API ───────────────────────────────────────────────────────────────

"""
    bug_two_site!(psi, gates; dt, kwargs...) -> BUGInfo

Advance `psi` by one two-site BUG step (the faithful-KLS odd/even method).
`gates` is the vector of per-bond two-site Hamiltonian terms `h_{b,b+1}`
(e.g. from `two_site_xx_bond_gates`); they must act on the physical site indices
of `psi`.

This is the faithful-KLS local bond variant: each bond is updated via
`_faithful_kls_local_bond_candidate`, giving rank-adaptive evolution with
machine-precision local steps.

For real-time evolution pass a real `dt` (the default prefactor `-im` realises
exp(-i·dt·H)). Pass `time_prefactor = ComplexF64(1)` for imaginary time.

Keyword arguments
- `order`            : `:strang` (default, 2nd order: odd(dt/2) even(dt) odd(dt/2))
                       or `:lie` (1st order: odd(dt) even(dt)).
- `maxdim`           : hard bond-dimension cap (rank-adaptive S-step truncation).
                       `typemax(Int)` keeps full rank (Trotter error only).
- `trunc_thresh`     : relative singular-value cutoff of the post-S-step SVD,
                       measured against the largest singular value. Each bond
                       keeps only the directions whose weight exceeds it, so the
                       rank grows only as far as the entanglement requires
                       (the discarded-weight control). `0.0` (default) disables
                       the threshold, leaving the pure `maxdim` cap.
- `augment`          : enable KLS basis augmentation (default `true`).
- `aug_krylov_depth` : augmentation Krylov-chain depth (default 1).
- `substep_method`   : `:expv`, `:euler`, or `:rk4` for the K/L/S substeps.
- `matrixfree_sstep` : matrix-free S-step Krylov (default `false`).
- `lanczos_tol`, `lanczos_maxiter` : Krylov controls.
- `expv_backend`     : `:auto`, `:krylovkit`, or `:native_hermitian_lanczos`.
- `time_prefactor`   : `-im` (default, real time) or `1` (imaginary time).

Returns a `BUGInfo`; `backward_correction_calls` is always 0.
"""
function bug_two_site!(
    psi   :: TensorTrain,
    gates :: AbstractVector{ITensor};
    dt    :: Number,
    order            :: Symbol  = :strang,
    maxdim           :: Int     = 200,
    trunc_thresh     :: Float64 = 0.0,
    augment          :: Bool    = true,
    aug_krylov_depth :: Int     = 1,
    lanczos_tol      :: Float64 = 1e-15,
    lanczos_maxiter  :: Int     = 30,
    substep_method   :: Symbol  = :expv,
    matrixfree_sstep :: Bool    = BUG_DEFAULT_MATRIXFREE_SSTEP,
    expv_backend     :: Symbol  = :auto,
    time_prefactor   :: ComplexF64 = ComplexF64(-im),
)
    N = length(psi)
    N < 2 && error("bug_two_site! requires at least 2 sites")
    length(gates) == N - 1 ||
        error("bug_two_site!: expected $(N - 1) bond gates, got $(length(gates))")
    order in (:strang, :lie) ||
        error("bug_two_site!: order must be :strang or :lie")

    allowed_backends = (:krylovkit, :native_hermitian_lanczos)
    effective_backend = expv_backend === :auto ? :native_hermitian_lanczos : expv_backend
    effective_backend in allowed_backends ||
        error("Unknown two-site BUG expv_backend: $expv_backend.")

    info = BUGInfo()
    info.bond_dims_before = [dim(linkind(psi, k)) for k in 1:(N - 1)]

    info.elapsed = @elapsed begin
        _with_bug_expv_backend(effective_backend) do
            _with_bug_time_prefactor(time_prefactor) do
                _two_site_strang_step!(psi, gates, dt, info;
                    order = order, maxdim = maxdim, trunc_thresh = trunc_thresh,
                    augment = augment, aug_krylov_depth = aug_krylov_depth,
                    lanczos_tol = lanczos_tol, lanczos_maxiter = lanczos_maxiter,
                    substep_method = substep_method, matrixfree_sstep = matrixfree_sstep,
                )
            end
        end
    end

    info.bond_dims_after = [dim(linkind(psi, k)) for k in 1:(N - 1)]
    return info
end
