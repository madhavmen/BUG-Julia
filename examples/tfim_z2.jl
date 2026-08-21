# THE TRANSVERSE-FIELD ISING CHAIN QUENCHED FROM A DOMAIN WALL, under ℤ₂ spin symmetry.
#
#     H = -J Σ_ℓ Z_ℓ Z_{ℓ+1} - h Σ_ℓ X_ℓ ,   |Ψ(0)⟩ = |+…+−…−⟩   (a wall in the X basis)
#
#     observable  M_L(t) = Σ_{ℓ ≤ L/2} ⟨X_ℓ⟩   -- transverse magnetisation left of the wall
#
# ⛔ THE BASIS IS σˣ, SO `X` IS DIAGONAL AND `Z` IS THE FLIP -- the OPPOSITE of the usual
# convention, and reading it the usual way silently describes a different model. `z2_ising.jl`
# works in the eigenbasis of the ℤ₂ charge (the global spin flip `P = Π 2S^x`), which is what
# makes the charge diagonal and the symmetric run possible at all. So `X` is charge-neutral and
# measurable site by site, while `Z` is the charge-1 operator and `Z_ℓ Z_{ℓ+1}` is neutral
# overall. Same trade as OAT, for the same reason.
#
# THE MPO IS EXPLICIT, NOT GATES. `tfim_mpo` (src/RSVDCBEBondUpdate/models/tfim.jl) builds the
# virtual-dimension-3 automaton, so this model drives the SAME environment machinery, effective
# Hamiltonians and Krylov solves as OAT and XX. A TFIM run differs from an OAT run only in the
# operator -- which is the point of comparing integrators on it.
#
# THE EXACT REFERENCE IS FREE FERMIONS AT ANY `L`, NOT A `2^L` DIAGONALISATION. Jordan-Wigner
# maps the TFIM to Majoranas with a quadratic Hamiltonian, a product state in the X basis is
# Gaussian, and Gaussian is preserved -- so the entire solution lives in a `2L x 2L` real
# orthogonal evolution. See `tfim_x_profile_exact` below for the derivation and the two pins it
# is checked against.
#
# `symmetry=Z2` is the symmetric run; `symmetry=none` is the dense control. The two agree to
# ~1e-12, so a gap between them is a charge-bookkeeping bug rather than physics.
#
#   julia --project=. examples/tfim_z2.jl
#   julia --project=. examples/tfim_z2.jl L=20 h=0.5 maxdim=128 symmetry=none
#   python3 examples/plot_examples.py

include(joinpath(@__DIR__, "common.jl"))

# ── PARAMETERS -- EDIT THESE (or override as name=value on the command line) ──────────────
const P = parse_params((
    L            = 16,      # sites. Even, so the wall sits on a bond.
    # ⛔ PLAIN TIME, NOT UNITS OF π -- same reason as `xx_domain_wall.jl`. Only OAT has a
    # period (revivals every 4π); the TFIM's scales are set by `J` and `h`, so π would hide them.
    t_max        = 3.0,     # total time, in units of 1/J
    dt           = 0.05,    # time step
    J            = 1.0,     # Ising coupling (ferromagnetic for J > 0)
    h            = 1.0,     # transverse field. h = J is the critical point.
    maxdim       = 64,      # bond-dimension cap. Like XX there is no exact ceiling.
    trunc_thresh = 1e-10,   # CBE root-to-leaves truncation threshold at the end of each sweep
    maxiter      = 0,       # 0 = derive the Krylov depth from ‖H‖·dt/2
    sample_every = 0.2,     # record a row every this much physical time
    symmetry     = "Z2",    # "Z2" (the spin flip) or "none" (the dense control)
))
# ──────────────────────────────────────────────────────────────────────────────────────────

const T_MAX = P.t_max

# ‖H‖ ≤ J(L-1) + hL by the triangle inequality -- LINEAR in L, like XX and unlike OAT.
const HNORM   = P.J * (P.L - 1) + P.h * P.L
const MAXITER = P.maxiter == 0 ? krylov_depth(HNORM, P.dt) : P.maxiter

# ── the exact solution, inlined so this example stands alone ──────────────────────────────
#
# JORDAN-WIGNER IN MAJORANAS. With the string taken along the chain,
#
#     a_ℓ = (Π_{k<ℓ} X_k) Z_ℓ ,   b_ℓ = (Π_{k<ℓ} X_k) Y_ℓ
#
# the strings cancel and leave two purely local identities (`ZY = -iX`, `YX = -iZ`):
#
#     X_ℓ = i a_ℓ b_ℓ ,            Z_ℓ Z_{ℓ+1} = i b_ℓ a_{ℓ+1}
#
# so `H = -h Σ i a_ℓ b_ℓ - J Σ i b_ℓ a_{ℓ+1}` is QUADRATIC. Writing `H = (i/4) Σ A_mn c_m c_n`
# over `c = (a_1, b_1, …, a_L, b_L)`, a term `α i c_m c_n` contributes `A_mn = 2α = -A_nm`.
#
# The Heisenberg equation gives `dc/dt = A c`, so `c(t) = e^{At} c(0)` with `A` real
# antisymmetric and `e^{At}` therefore REAL ORTHOGONAL. The covariance matrix
# `Γ_mn = (i/2)⟨[c_m, c_n]⟩` evolves as `Γ(t) = R Γ(0) Rᵀ`, `R = e^{At}` -- real arithmetic
# end to end, `O(L³)`, exact at finite `L` with open boundaries and at any time.
#
# A product state in the X basis has `Γ(0)` block diagonal with `[0 s_ℓ; -s_ℓ 0]`, `s_ℓ = ⟨X_ℓ⟩`,
# and the observable comes straight back out as `⟨X_ℓ(t)⟩ = Γ(t)_{2ℓ-1, 2ℓ}`.
#
# TWO PINS, both checked in the probe that accompanied this file:
#   * `⟨+…+|H|+…+⟩ = -h·L` for ANY `J`, because `Z_ℓ Z_{ℓ+1}` is purely off-diagonal here. A
#     wrong sign or index in `A` cannot reproduce it.
#   * against the dense `2^L` TFIM at `L = 8`, where exact diagonalisation is cheap.

"The real antisymmetric `2L x 2L` generator `A`. Majorana `a_ℓ` is index `2ℓ-1`, `b_ℓ` is `2ℓ`."
function tfim_majorana_generator(L::Int; J::Real = 1.0, h::Real = 1.0)
    A = zeros(Float64, 2L, 2L)
    for j in 1:L                                   # -h · i a_j b_j
        A[2j - 1, 2j] = -2h
        A[2j, 2j - 1] = +2h
    end
    for j in 1:(L - 1)                             # -J · i b_j a_{j+1}
        A[2j, 2j + 1] = -2J
        A[2j + 1, 2j] = +2J
    end
    return A
end

"⟨X_ℓ(t)⟩ for every site, from an X-basis product state `dirs` (`:plus` / `:minus`)."
function tfim_x_profile_exact(L::Int, t::Real, dirs::Vector{Symbol}; J::Real = 1.0, h::Real = 1.0)
    G = zeros(Float64, 2L, 2L)
    for (j, d) in enumerate(dirs)
        s = d === :plus ? 1.0 : -1.0
        G[2j - 1, 2j] = s
        G[2j, 2j - 1] = -s
    end
    R = exp(tfim_majorana_generator(L; J = J, h = h) * t)
    Gt = R * G * transpose(R)
    return [Gt[2j - 1, 2j] for j in 1:L]
end

# ── the run ───────────────────────────────────────────────────────────────────────────────

set_symmetry!(Symbol(P.symmetry))

const DIRS   = [i <= P.L ÷ 2 ? :plus : :minus for i in 1:P.L]   # what `ising_kink_state` builds
const MPO_TF = tfim_mpo(P.L; J = P.J, h = P.h)
const PSI0   = ising_kink_state(P.L)

"The scalar signal: transverse magnetisation left of the wall. Starts at `L/2` and decays. A SUM
rather than a single site, so it is not sensitive to which site the front happens to have reached
at a sample time. Indexed `1:(L÷2)` in BOTH the measured and the exact version -- see the same
note in `xx_domain_wall.jl`."
left_x_magnetisation(psi) = sum(x_profile(psi)[1:(P.L ÷ 2)])
tfim_left_x_exact(t) = sum(tfim_x_profile_exact(P.L, t, DIRS; J = P.J, h = P.h)[1:(P.L ÷ 2)])

@printf("TFIM  L=%d  symmetry=%s  J=%g  h=%g%s  t=%g (1/J)  dt=%g  maxdim=%d  trunc=%g\n",
        P.L, P.symmetry, P.J, P.h, P.J == P.h ? " (critical)" : "",
        T_MAX, P.dt, P.maxdim, P.trunc_thresh)
@printf("krylov depth = %d  (arg %.3f, Lanczos bound %.1e -- must sit under %g)\n",
        MAXITER, HNORM * P.dt / 2, krylov_bound(HNORM, P.dt, MAXITER), P.trunc_thresh)
@printf("mpo virtual dims = %s   initial chi = %d   M_L(0) = %.6f (exact %.6f)\n\n",
        string(mpo_virtual_dims(MPO_TF)), maximum(bond_dims(PSI0)),
        left_x_magnetisation(copy(PSI0)), tfim_left_x_exact(0.0))

const OUT = joinpath(RESULTS,
                     @sprintf("tfim_L%d_%s_D%d_dt%g_T%g.csv",
                              P.L, P.symmetry, P.maxdim, P.dt, T_MAX))
io = open_csv(OUT)
try
    run_model(io, "TFIM", P.symmetry, PSI0, MPO_TF,
              psi -> left_x_magnetisation(psi),     # observable
              t   -> tfim_left_x_exact(t),          # its exact value
              (; t_max = T_MAX, dt = P.dt, maxdim = P.maxdim,
                 trunc_thresh = P.trunc_thresh, maxiter = MAXITER,
                 sample_every = P.sample_every))
finally
    close(io)
end
println("\nwrote ", OUT)
