# THE XX CHAIN RELEASED FROM A DOMAIN WALL -- the growing-entanglement counterpart to OAT.
#
#     H = J Σ_ℓ (S^x_ℓ S^x_{ℓ+1} + S^y_ℓ S^y_{ℓ+1}) ,   |Ψ(0)⟩ = |↑…↑↓…↓⟩
#
#     observable  M_L(t) = Σ_{ℓ ≤ L/2} ⟨S^z_ℓ⟩   -- the magnetisation still in the left half
#
# WHY THIS MODEL, GIVEN THAT OAT IS ALREADY EXACTLY SOLVABLE. OAT's Schmidt rank is CONSTANT in
# time (`min(n,L-n)+1`), so `maxdim` there is either exact or a fixed truncation and the bond
# dimension plot is a flat line. Here the wall spreads, entanglement grows without bound, and
# `maxbond` climbing into the cap is the expected behaviour -- which makes this the example
# where rank adaptivity is actually under load and the bond-dimension figure is the interesting
# one. The two models together cover both regimes; either alone is misleading.
#
# THE EXACT REFERENCE IS FREE FERMIONS, NOT DIAGONALISATION. Jordan-Wigner maps the XX chain to
# non-interacting fermions with hopping `J/2`, so a Slater determinant stays a Slater determinant
# and the whole solution lives in an `L x L` matrix:
#
#     ⟨S^z_ℓ(t)⟩ = Σ_{k occupied} |[exp(-i h t)]_{ℓk}|²  −  ½
#
# This is EXACT at finite `L` with OPEN boundaries and at any time -- including after the front
# reflects off the ends -- and it costs `O(L³)`. Nothing here ever forms a `2^L` object, so the
# reference stays available at sizes where exact diagonalisation does not.
#
# U(1) OR NOT. `S^z_tot` is conserved and the domain wall is a single charge sector
# (`S^z_tot = 0` for even `L`), so `symmetry=U1` is the natural run and `symmetry=none` is the
# dense control. Same physics, same numbers to ~1e-12; only the block structure differs, so any
# gap between them is a symmetry-bookkeeping bug rather than physics.
#
#   julia --project=. examples/xx_domain_wall.jl
#   julia --project=. examples/xx_domain_wall.jl L=24 maxdim=128 symmetry=none
#   python3 examples/plot_examples.py

include(joinpath(@__DIR__, "common.jl"))

# ── PARAMETERS -- EDIT THESE (or override as name=value on the command line) ──────────────
const P = parse_params((
    L            = 16,      # sites. Even, so the wall sits on a bond and S^z_tot = 0.
    t_over_pi    = 2.0,     # total time in units of π (see the front-speed note below)
    dt           = 0.05,    # time step
    J            = 1.0,     # hopping
    maxdim       = 64,      # bond-dimension cap. UNLIKE OAT there is no exact ceiling.
    trunc_thresh = 1e-10,   # CBE root-to-leaves truncation threshold at the end of each sweep
    maxiter      = 0,       # 0 = derive the Krylov depth from ‖H‖·dt/2
    dover        = 0,       # RSVD oversampling for cbe_bug. 0 = the sweep's own default.
    sample_every = 0.2,     # record a row every this much physical time
    symmetry     = "U1",    # "U1" (S^z_tot conserved) or "none" (the dense control)
))
# ──────────────────────────────────────────────────────────────────────────────────────────

const T_MAX = P.t_over_pi * π

# The single-particle dispersion is `ε(k) = J cos k`, so the maximum group velocity is `J` and
# the front travels `≈ J·t` sites from the wall. It reaches the boundary at `t ≈ L/(2J)`; the
# default `2π ≈ 6.3` at `L = 16` stops just before `8`, so the run stays in the clean spreading
# regime. Going past it is not an error -- the reference handles reflection exactly -- but the
# physics changes, so the header says which side of it this run is on.
const T_FRONT = P.L / (2 * P.J)

# ‖H_XX‖ = Σ of the positive single-particle energies ≈ J·L/π, LINEAR in L rather than OAT's
# quadratic. `J*L/2` is a safe upper bound and is what the depth is derived from.
const HNORM   = P.J * P.L / 2
const MAXITER = P.maxiter == 0 ? krylov_depth(HNORM, P.dt) : P.maxiter

# ── the exact solution, inlined so this example stands alone ──────────────────────────────
# Same content as `tests/common/free_fermion.jl`; a downloaded copy of the package should not
# need the test tree to run an example.

"The `L x L` hopping matrix. `J(SxSx + SySy) = (J/2)(S⁺S⁻ + S⁻S⁺)` maps to `(J/2)(c†_ℓ c_{ℓ+1} +
h.c.)` -- the Jordan-Wigner string cancels between adjacent sites -- so the off-diagonal is `J/2`
and the diagonal is zero."
function xx_hopping(L::Int; J::Real = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    return h
end

"⟨S^z_ℓ(t)⟩ for every site. Up spin is an occupied fermion (`S^z = n − ½`), so `occupied` is
`1:(L÷2)` for `domain_wall_state`."
function xx_sz_profile_exact(L::Int, t::Real; J::Real = 1.0, occupied = 1:(L ÷ 2))
    U = exp(-im * t * xx_hopping(L; J = J))
    return [sum(abs2(U[j, k]) for k in occupied) - 0.5 for j in 1:L]
end

"The scalar signal: magnetisation remaining in the left half. Starts at `L/4` and decays as the
wall spreads. A SUM rather than a single site, so it is not accidentally sensitive to which site
the front happens to have reached at a sample time."
xx_left_magnetisation(L::Int, t::Real; J::Real = 1.0) =
    sum(xx_sz_profile_exact(L, t; J = J)[1:(L ÷ 2)])

# ── the run ───────────────────────────────────────────────────────────────────────────────

set_symmetry!(Symbol(P.symmetry))

const MPO_XX = xxz_mpo(P.L; J = P.J, delta = 0.0)     # delta = 0 is exactly the XX chain
const PSI0   = domain_wall_state(P.L)

# ⛔ MEASURE THE SAME HALF THE REFERENCE SUMS. `sz_expectation` is per site, and summing
# `1:(L÷2)` here mirrors `xx_left_magnetisation` exactly; taking `1:(L÷2)` in one and
# `1:cld(L,2)` in the other differs only for odd `L` and would look like a small integrator
# error rather than an indexing one.
left_magnetisation(psi) = sum(sz_expectation(psi, j) for j in 1:(P.L ÷ 2))

@printf("XX domain wall  L=%d  symmetry=%s  J=%g  t=%gπ (%.3f)  dt=%g  maxdim=%d  trunc=%g\n",
        P.L, P.symmetry, P.J, P.t_over_pi, T_MAX, P.dt, P.maxdim, P.trunc_thresh)
@printf("krylov depth = %d  (arg %.3f, Lanczos bound %.1e -- must sit under %g)\n",
        MAXITER, HNORM * P.dt / 2, krylov_bound(HNORM, P.dt, MAXITER), P.trunc_thresh)
@printf("front reaches the boundary at t ≈ %.2f -- this run ends at %.2f (%s)\n",
        T_FRONT, T_MAX, T_MAX < T_FRONT ? "clean spreading" : "INCLUDES reflection")
@printf("initial chi = %d   M_L(0) = %.6f (exact %.6f)\n\n",
        maximum(bond_dims(PSI0)), left_magnetisation(copy(PSI0)),
        xx_left_magnetisation(P.L, 0.0; J = P.J))

const OUT = joinpath(RESULTS,
                     @sprintf("xx_L%d_%s_D%d_dt%g_T%gpi%s.csv",
                              P.L, P.symmetry, P.maxdim, P.dt, P.t_over_pi,
                              P.dover == 0 ? "" : "_dover$(P.dover)"))
io = open_csv(OUT)
try
    run_model(io, "XX", P.symmetry, PSI0, MPO_XX,
              psi -> left_magnetisation(psi),                          # observable
              t   -> xx_left_magnetisation(P.L, t; J = P.J),           # its exact value
              (; t_max = T_MAX, dt = P.dt, maxdim = P.maxdim,
                 trunc_thresh = P.trunc_thresh, maxiter = MAXITER,
                 dover = P.dover, sample_every = P.sample_every))
finally
    close(io)
end
println("\nwrote ", OUT)
