# LONG-RANGE COMPETING INTERACTIONS, open, 1D and 2D -- Hryniuk & Szymanska, Comms Phys (2026) 9:45
# (doi 10.1038/s42005-025-02479-2). Three integrators on the paper's own protocols.
#
#   julia -t 2 --project=. benchmarks/dissipative_longrange.jl [certify|fig4|fig4sq|fig6|fig5]
#
# ── WHAT THIS ADDS OVER `dissipative_heisenberg.jl` ──────────────────────────────────────────────
#
# That file reproduces the paper's Fig. 3: NEAREST-NEIGHBOUR anisotropic XYZ plus spin decay, and it
# is certified to roundoff against a dense Liouvillian. This one carries the same operator algebra
# to the paper's LONG-RANGE models, which is a coupling-MATRIX change and nothing else:
#
#   `pair_mpo` weights each `PairVertex` by an arbitrary `J[i, j]` over every pair `i < j`, so a
#   power law needs NO exponential fitting and NO extra channels -- `nn(c)` (a matrix with `c` on
#   the first off-diagonal) becomes `c .* D` with `D[i,j] = d_ij^(-alpha)`. The MPO is EXACT.
#
# Eq. (6), the DIAGONAL long-range model (their Figs 4 and 5):
#
#     H = - sum_n J_n sum_{i,j} d_ij^(-alpha_n) sigma^z_i sigma^z_j  -  h sum_i sigma^x_i
#
# Eq. (7), the NON-DIAGONAL long-range model (their Fig. 6):
#
#     H = - sum_n sum_{beta in x,y,z} J^beta_n sum_{i,j} d_ij^(-alpha_n) sigma^beta_i sigma^beta_j
#         - h sum_i sigma^z_i
#
# ⛔ PAULI vs SPIN, AND IT IS THE SAME FACTOR OF 4 AS IN Fig. 3. `sigma^a = 2 S^a`, so
# `sigma^a sigma^a = 4 S^a S^a` and `h sigma^a = 2h S^a`. In spin operators:
#
#     Jx sxsx + Jy sysy = (Jx - Jy)(S+S+ + S-S-) + (Jx + Jy)(S+S- + S-S+)
#     Jz szsz           = 4 Jz S^z S^z
#     h sigma^x         = h (S+ + S-)          h sigma^z = 2h S^z
#
# Getting the 4 or the 2 wrong does NOT throw -- it gives a smooth plausible curve at the WRONG
# FREQUENCY, which is why `certify` below compares against a dense Liouvillian built independently.
#
# ── ⛔ THERE IS NO SU(2) ARM, AND NO U(1) ARM EITHER. THIS IS NOT A CHOICE. ───────────────────────
#
# Twice over, for two independent reasons:
#
#   1. THE HAMILTONIAN. Fig. 3 has `(Jx,Jy,Jz) = (-1.0,-0.9,-1.2)` and Fig. 6 has
#      `J1 = (0.6,-0.5,0.4)`: `Jx != Jy` opens the `S+S+`/`S-S-` channel, which changes `S^z` by 2,
#      so U(1) is gone; `Jz != Jx` kills SU(2). Only Eq. (6) (pure `szsz`) is U(1)-symmetric as a
#      Hamiltonian -- and its dissipator is spin DECAY, which breaks it again.
#   2. THE FUSED SPACE, which is the BINDING constraint. `fused_space()` builds the d = 4 site
#      through `_getLocalSpace_no_symmetry` because Telum's `getLocalSpace` derives its space from a
#      KNOWN symmetry type. So EVERY fused superket in this repo is `:none` whatever the couplings
#      are. A symmetric d = 4 space (graded by the difference charge `q = n_ket - n_bra`, sectors
#      `{-1, 0, 0, +1}`) does not exist yet -- real work, not a parameter.
#
# ⇒ the campaign's SYMMETRY axis is carried by the CLOSED models (Heisenberg chain SU(2), the
# cylinder, kagome). These open runs are a BUG-vs-TDVP comparison at `:none`. Say that explicitly
# rather than letting a missing `sym` column read as an oversight.
#
# ── ⚠ AND THERE IS NO EXACT REFERENCE AT L = 16 ──────────────────────────────────────────────────
#
# `4^16` is out of reach, so there is no `err` column. Two things stand in for it, and neither is
# "our own finest grid":
#   * `certify` pins the GENERATOR against a dense Liouvillian at L = 3 and 4, where `4^L` is small.
#     A wrong coupling matrix, a wrong Pauli factor or a transposed jump is L-INDEPENDENT, so a
#     small L settles it.
#   * at L = 16 the arms are compared to EACH OTHER, and the anchor is the PAPER'S OWN FIGURE.
#     Agreement with it is qualitative and is reported as such.

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "fused_common.jl"))

const OUT = joinpath(@__DIR__, "results")
mkpath(OUT)
# ⛔ ONE LOG FILE PER MODE, NOT ONE PER DRIVER. The figures are run CONCURRENTLY (four jobs, ~2
# cores each, on a 12-core box), and on Windows several processes holding the same file open for
# append is a sharing violation at LOAD time -- i.e. every job dies before it computes anything.
# Interleaved output would be the milder version of the same bug.
const LOG = open(joinpath(OUT, "dissipative_longrange_" *
                               (isempty(ARGS) ? "certify" : ARGS[1]) * ".txt"), "a")
# ⛔ MIRROR AND FLUSH. Julia BLOCK-BUFFERS a redirected stdout, so a piped long run shows nothing
# until it exits and a crash loses the log. This repo has already lost a 31-minute run that way.
say(l) = (println(LOG, l); flush(LOG); println(stdout, l); flush(stdout))

# ── the paper's parameter sets ───────────────────────────────────────────────────────────────────
#
# ⚠ `gamma = -2h = 1` (Figs 4, 6) means `gamma = 1` and `h = -0.5`; Fig. 3's `gamma = -h = 1` means
# `h = -1`. The two differ by a factor of two in the FIELD ONLY, and quoting the wrong one shifts
# every oscillation frequency while leaving the curves looking entirely reasonable.
#
# ⚠ Fig. 6's SIGNS. The text gives `J1 = (0.6,-0.5,0.4)`, `J2 = (-0.9,1.0,-1.1)`; the caption gives
# them NEGATED. Eq. (7) carries an overall minus, so the caption is quoting the coefficient AS IT
# APPEARS IN `H` and the two agree. The text's values are used here, with Eq. (7) as written.
const FIG4_CHAIN = (Js = [((0.0, 0.0, -0.5), 3.0), ((0.0, 0.0, 1.0), 6.0)],
                    h = -0.5, hdir = :x, gamma = 1.0, start = :minusy,
                    label = "Fig 4a,b: competing dipolar/vdW Ising, chain")
const FIG4_SQ    = (Js = [((0.0, 0.0, -0.25), 3.0), ((0.0, 0.0, 0.5), 6.0)],
                    h = -0.5, hdir = :x, gamma = 1.0, start = :minusy,
                    label = "Fig 4c,d: competing dipolar/vdW Ising, square")
const FIG6       = (Js = [((0.6, -0.5, 0.4), 3.0), ((-0.9, 1.0, -1.1), 6.0)],
                    h = -0.5, hdir = :z, gamma = 1.0, start = :minusy, rmax = 4.0,
                    label = "Fig 6c,d: competing NON-DIAGONAL long-range XYZ, chain")

# ── geometry and the power law ───────────────────────────────────────────────────────────────────

"""
    site_positions(shape) -> Vector{NTuple{2,Float64}}

Lattice coordinates in the SAME site order the MPS uses.

⛔ THE 2D ORDER IS COLUMN-MAJOR OVER `(x, y)`, i.e. site `(ix-1)*LY + iy`, matching
`square_cylinder_couplings`. If this disagrees with the MPO's own site order the couplings are
scrambled and the run is a DIFFERENT MODEL that still evolves smoothly -- `certify` compares against
a dense Liouvillian built from THESE SAME positions, so an ordering error shows up there.

⚠ OPEN boundaries in both directions. The paper's Fig. 4c is a finite `4 x 4` patch, not a torus;
wrapping `y` would shorten half the distances and strengthen every long-range term.
"""
function site_positions(shape)
    if length(shape) == 1
        return [(Float64(i), 0.0) for i in 1:shape[1]]
    end
    LX, LY = shape[1], shape[2]
    return [(Float64(ix), Float64(iy)) for ix in 1:LX for iy in 1:LY]
end

"""
    power_law_matrix(shape, alpha; rmax = Inf, kac = false) -> Matrix{Float64}

`D[i,j] = d_ij^(-alpha)` on the STRICT UPPER TRIANGLE, with `d_ij` the Euclidean distance.

`rmax` truncates the interaction beyond that distance -- the paper does this for the NON-DIAGONAL
model only (Fig. 6, `r = 4`, "beyond which the combined interaction strength falls to less than 2%
of its value for nearest neighbors"), because non-diagonal terms cost tensor contractions there.
⚠ WE DO NOT NEED THE TRUNCATION: `pair_mpo` is exact for any `J[i,j]`. It is applied anyway where
the paper applies it, so the model being compared is theirs and not a longer-ranged cousin.

`kac` divides by the Kac factor `N(alpha) = (1/(N-1)) sum_{i != j} d_ij^(-alpha)`, which the paper
uses "to ensure extensivity with system size" in its steady-state figure (Fig. 5). ⛔ IT MUST BE OFF
FOR THE DYNAMICS FIGURES: Figs 4 and 6 quote bare `J_n`, and dividing by `N(alpha)` there rescales
time by a size-dependent factor, so curves at two different `L` would no longer be comparable.
"""
function power_law_matrix(shape, alpha::Real; rmax::Real = Inf, kac::Bool = false)
    pos = site_positions(shape)
    N = length(pos)
    D = zeros(Float64, N, N)
    for i in 1:N, j in (i + 1):N
        d = hypot(pos[i][1] - pos[j][1], pos[i][2] - pos[j][2])
        d <= rmax || continue
        D[i, j] = d^(-float(alpha))
    end
    if kac
        # sum over ORDERED pairs i != j = twice the strict upper triangle.
        nrm = 2 * sum(D) / (N - 1)
        nrm > 0 && (D ./= nrm)
    end
    return D
end

# ── the fused MPO ────────────────────────────────────────────────────────────────────────────────

"""
    lr_lindblad(shape; Js, h, hdir, gamma, rmax, kac) -> (mpo, shift, N)

Fused d = 4 Liouvillian of Eq. (6)/(7) with spin-decay jumps `Gamma_k = sqrt(gamma) sigma^-_k`.

`Js` is a list of `((Jx, Jy, Jz), alpha)`: one power law per entry, summed. Eq. (6) is the special
case `Jx = Jy = 0`. `hdir` is `:x` (Eq. 6) or `:z` (Eqs 5 and 7).

Structurally identical to `xyz_decay_lindblad`, with every coupling matrix a POWER LAW instead of a
first off-diagonal, and with a `sigma^x` field option.

⛔ THREE `pair_mpo` TRANSPOSE TRAPS, ALL LIVE HERE (see `fused-jump-transpose-freeze`):
  * `PV(Kp,Km) + PV(Km,Kp)` is invariant as a SUM under the transpose -- why XX never noticed.
  * `PV(Kp,Kp) + PV(Km,Km)` likewise, so the anisotropic channel is safe ONLY because both are
    pushed with the SAME coefficient. Dropping one silently evolves `S+S+` as `S-S-`.
  * `kron(SM,SM)^T = kron(SP,SP)`, so the DECAY jump is `Q4.JP`. The "obvious" `Q4.JM` is GAIN, and
    the anticommutator then cancels it EXACTLY -- a state that never moves.
  The `sigma^x` field is safe for the same reason as the first two: `Bp` and `Bm` carry equal
  coefficients, so their sum survives the transpose.
"""
function lr_lindblad(shape; Js, h::Float64, hdir::Symbol = :z, gamma::Float64 = 1.0,
                     rmax::Real = Inf, kac::Bool = false)
    pos = site_positions(shape)
    L = length(pos)
    L >= 2 || throw(ArgumentError("need at least two sites"))
    hdir in (:x, :z) || throw(ArgumentError("hdir must be :x or :z, got $hdir"))
    PV = RSVDCBEBondUpdate.PairVertex
    verts = PV[]; Jm = Any[]

    # Accumulate the three channel matrices over all power laws FIRST, so each channel is one
    # vertex pair no matter how many competing terms there are. Two power laws pushed as two
    # separate vertex pairs would double the MPO bond dimension for no reason.
    Axy = zeros(Float64, L, L)      # (S+S- + S-S+) weight  = -(Jx + Jy)
    Aan = zeros(Float64, L, L)      # (S+S+ + S-S-) weight  = -(Jx - Jy)
    Azz = zeros(Float64, L, L)      # S^z S^z weight        = -4 Jz
    for ((Jx, Jy, Jz), alpha) in Js
        D = power_law_matrix(shape, alpha; rmax = rmax, kac = kac)
        Axy .+= (-(Jx + Jy)) .* D
        Aan .+= (-(Jx - Jy)) .* D
        Azz .+= (-4 * Jz) .* D
    end
    push2(opk, opb, A) = begin      # one Hamiltonian channel: ket half (-i), bra half (+i)
        all(iszero, A) && return
        push!(verts, PV(opk[1], opk[2], 1.0)); push!(Jm, (-im) .* ComplexF64.(A))
        push!(verts, PV(opb[1], opb[2], 1.0)); push!(Jm, (+im) .* ComplexF64.(A))
    end
    push2((Q4.Kp, Q4.Km), (Q4.Bp, Q4.Bm), Axy)
    push2((Q4.Km, Q4.Kp), (Q4.Bm, Q4.Bp), Axy)
    push2((Q4.Kp, Q4.Kp), (Q4.Bp, Q4.Bp), Aan)
    push2((Q4.Km, Q4.Km), (Q4.Bm, Q4.Bm), Aan)
    push2((Q4.Kz, Q4.Kz), (Q4.Bz, Q4.Bz), Azz)

    # ── one-body terms, on EVERY site ───────────────────────────────────────────────────────────
    # ⚠ THE LAST SITE NEEDS ITS OWN VERTEX. `pair_mpo` reaches site `j+1` from bond `j`, so a
    # one-body operator placed as `O_j (x) I_{j+1}` over `j = 1..L-1` misses site `L` entirely.
    # Dropping it evolves a chain whose final site has no field or no bath: a DIFFERENT model whose
    # error SHRINKS with L, so small tests miss it.
    one_body!(op, c) = begin
        c == 0 && return
        A = zeros(ComplexF64, L, L); B = zeros(ComplexF64, L, L)
        for j in 1:(L - 1); A[j, j + 1] = c; end
        B[L - 1, L] = c
        push!(verts, PV(op, Q4.I, 1.0)); push!(Jm, A)
        push!(verts, PV(Q4.I, op, 1.0)); push!(Jm, B)
    end
    # ⛔ DECAY IS `Q4.JP`. See the docstring.
    one_body!(Q4.JP, gamma)
    if hdir === :z
        # `-h sum sigma^z = -2h sum S^z`; ket half `-i*(-2h) = +2ih`, bra half `-2ih`. Folded onto
        # the same operator as the anticommutator `-gamma/2` (from `Gamma'Gamma = gamma * n_up`).
        one_body!(Q4.Kz, -gamma / 2 + 2im * h)
        one_body!(Q4.Bz, -gamma / 2 - 2im * h)
    else
        one_body!(Q4.Kz, -gamma / 2)
        one_body!(Q4.Bz, -gamma / 2)
        # `-h sum sigma^x = -h sum (S+ + S-)`; ket half `-i*(-h) = +ih` on BOTH Kp and Km.
        one_body!(Q4.Kp, im * h); one_body!(Q4.Km, im * h)
        one_body!(Q4.Bp, -im * h); one_body!(Q4.Bm, -im * h)
    end
    return pair_mpo(L, Vector{AbstractMatrix}(Jm), verts; Iloc = Q4.I), -gamma * L / 2, L
end

# ── the dense reference, for certification ONLY ──────────────────────────────────────────────────

const _SP2 = ComplexF64[0 1; 0 0]
const _SM2 = ComplexF64[0 0; 1 0]
const _I22 = ComplexF64[1 0; 0 1]
_at(M, k, L) = (m = ComplexF64[1;;]; for q in 1:L; m = kron(m, q == k ? M : _I22); end; m)

"""
    dense_lr(shape; ...) -> (D x D superoperator, N)

The exact Lindbladian of Eq. (6)/(7) as a `4^N x 4^N` matrix, built from PAULI matrices directly.

⛔ IT SHARES NO CODE WITH `lr_lindblad`: no `pair_mpo`, no `Q4`, no fused layout, no spin/Pauli
conversion. That is the point -- an agreement between the two is evidence, whereas a shared helper
would let one bug cancel in both.
"""
function dense_lr(shape; Js, h::Float64, hdir::Symbol = :z, gamma::Float64 = 1.0,
                  rmax::Real = Inf, kac::Bool = false)
    L = length(site_positions(shape))
    D = 2^L
    sx(k) = _at(_SP2 .+ _SM2, k, L)                       # sigma^x
    sy(k) = _at(-im .* (_SP2 .- _SM2), k, L)              # sigma^y = -i(s+ - s-)
    sz(k) = _at(ComplexF64[1 0; 0 -1], k, L)              # sigma^z
    H = zeros(ComplexF64, D, D)
    for ((Jx, Jy, Jz), alpha) in Js
        P = power_law_matrix(shape, alpha; rmax = rmax, kac = kac)
        for i in 1:L, j in (i + 1):L
            P[i, j] == 0.0 && continue
            H .-= P[i, j] .* (Jx .* sx(i) * sx(j) .+ Jy .* sy(i) * sy(j) .+ Jz .* sz(i) * sz(j))
        end
    end
    for k in 1:L
        H .-= h .* (hdir === :x ? sx(k) : sz(k))
    end
    jumps = [sqrt(gamma) .* _at(_SM2, k, L) for k in 1:L]   # sigma^- == S^-
    S = zeros(ComplexF64, D * D, D * D)
    for a in 1:D, b in 1:D
        E = zeros(ComplexF64, D, D); E[a, b] = 1.0
        R = -im * (H * E - E * H)
        for J in jumps
            R .+= J * E * J' .- 0.5 .* (J' * J * E .+ E * J' * J)
        end
        S[:, (b - 1) * D + a] = vec(R)
    end
    return S, L
end

"Dense `rho(t)` from a product start, as a `D x D` matrix. `start` is `:plusx`, `:minusy`, `:neel`."
function dense_lr_rho(shape, t::Real; start::Symbol = :minusy, kwargs...)
    S, L = dense_lr(shape; kwargs...)
    D = 2^L
    one = start === :plusx  ? ComplexF64[0.5 0.5; 0.5 0.5] :
          start === :minusy ? ComplexF64[0.5 im/2; -im/2 0.5] : nothing
    if one === nothing        # :neel
        rho = ComplexF64[1;;]
        for k in 1:L
            rho = kron(rho, isodd(k) ? ComplexF64[1 0; 0 0] : ComplexF64[0 0; 0 1])
        end
    else
        rho = ComplexF64[1;;]
        for _ in 1:L; rho = kron(rho, one); end
    end
    return reshape(exp(S * t) * vec(rho), D, D)
end

dense_pauli1_lr(rho, L, j) = begin
    tr = real(LinearAlgebra.tr(rho))
    (real(LinearAlgebra.tr(_at(_SP2 .+ _SM2, j, L) * rho)) / tr,
     real(LinearAlgebra.tr(_at(-im .* (_SP2 .- _SM2), j, L) * rho)) / tr,
     real(LinearAlgebra.tr(_at(ComplexF64[1 0; 0 -1], j, L) * rho)) / tr)
end
dense_pauli2_lr(rho, L, j, k) = begin
    tr = real(LinearAlgebra.tr(rho))
    X = _SP2 .+ _SM2; Y = -im .* (_SP2 .- _SM2); Z = ComplexF64[1 0; 0 -1]
    (real(LinearAlgebra.tr(_at(X, j, L) * _at(X, k, L) * rho)) / tr,
     real(LinearAlgebra.tr(_at(Y, j, L) * _at(Y, k, L) * rho)) / tr,
     real(LinearAlgebra.tr(_at(Z, j, L) * _at(Z, k, L) * rho)) / tr)
end

# ── the gate ─────────────────────────────────────────────────────────────────────────────────────

rho0_for(start::Symbol, L::Int) =
    start === :plusx  ? rho0_plusx(L) :
    start === :minusy ? rho0_minusy(L) :
    rho0_product([isodd(k) ? :up : :down for k in 1:L])

"""
    szz_at(Imps, rho, L, trr, qs) -> Vector{Float64}

The magnetic structure factor `S_zz(q) = (1/N^2) sum_{j,k} e^{i q (j-k)} <sigma^z_j sigma^z_k>`
(their Fig. 5), evaluated at each `q` in `qs`.

⛔ ONE DEFINITION, TWO CALLERS. `protocol_lr` writes it against time and `steady_lr` writes its
converged value; if each carried its own copy they could drift in normalisation or in the q grid and
the two figures would silently stop being the same quantity.

⛔ THE FULL CHAIN, NOT THE BULK WINDOW, and the `1/N^2` prefactor goes with that choice. Restricting
to the bulk would change the normalisation and the q-grid together.

⚠ `zz[j,j] = 1` because `(sigma^z)^2 = I` -- it is NOT measured, and leaving it out (or measuring it
and picking up `<sigma^z_j>^2` instead) shifts every `q` by the same constant `1/N`, which looks like
a baseline offset rather than an error. At `t = 0` from a product state this makes `S_zz(q) = 1/N`
for EVERY `q`; that q-independence is the cheapest available check that the normalisation is right.
"""
function szz_at(Imps, rho, L::Int, trr::Real, qs)
    zz = zeros(Float64, L, L)
    for j in 1:L
        zz[j, j] = 1.0                          # (sigma^z_j)^2 = I, exactly
    end
    for j in 1:L, k in (j + 1):L
        v = pauli_2site(Imps, rho, L, j, k; tr = trr)[3]
        zz[j, k] = v; zz[k, j] = v
    end
    return [real(sum(cis(q * (j - k)) * zz[j, k] for j in 1:L, k in 1:L)) / L^2 for q in qs]
end

"""
    certify_lr(P; shape, t, dts) -> Bool

Pin the long-range GENERATOR and the OBSERVABLE PATH against `dense_lr` at small `L`.

⛔ WHY BOTH, AND WHY `sigma^y` IS NOT OPTIONAL. The generator gate uses `<sigma^z_j>`, and
`sigma^z` is DIAGONAL: it reads back the same number whether `site_apply!` applies `M` or `M'`. So
does `sigma^x`, which is real symmetric. `sigma^y` is the ONLY one of the three that can tell the
two conventions apart, and the `<sigma^y>` start makes it nonzero from `t = 0` -- so this gate is
strictly stronger than the Neel-start gate in `dissipative_heisenberg.jl`, which had to add a
third `|+x>` stage to get a nonzero `sigma^y` at all.

⚠ THE BAR IS `O(dt^2)`, NOT ROUNDOFF, wherever the MPS side takes finite steps against an exact
matrix exponential. A transpose or Pauli-factor error is `O(1)` on some component -- the two failure
sizes are nowhere near each other, so a loose bar still separates them.
"""
function certify_lr(P; shape = (4,), t::Float64 = 0.25, dts = (0.05, 0.025, 0.0125))
    set_symmetry!(:none)
    kw = (Js = P.Js, h = P.h, hdir = P.hdir, gamma = P.gamma,
          rmax = get(P, :rmax, Inf), kac = false)
    W, shift, L = lr_lindblad(shape; kw...)
    Imps = identity_superket(L)
    rhod = dense_lr_rho(shape, t; start = P.start, kw...)
    exact1 = [dense_pauli1_lr(rhod, L, j) for j in 1:L]

    say("\n=== CERTIFY (long range): " * P.label * " ===")
    say(@sprintf("  shape = %s -> L = %d   t = %g   start = <%s>",
                 string(shape), L, t, String(P.start)))
    say(@sprintf("  h = %+.3f (%s)  gamma = %.3f  rmax = %s",
                 P.h, String(P.hdir), P.gamma, string(get(P, :rmax, Inf))))
    say(@sprintf("  MPO virtual dim = %d", maximum(mpo_virtual_dims(W))))
    say(@sprintf("  exact <sigma^y_1> = %+.9f   <sigma^x_1> = %+.9f   <sigma^z_1> = %+.9f",
                 exact1[1][2], exact1[1][1], exact1[1][3]))
    say(@sprintf("\n  %-9s %-13s %-13s %-8s", "dt", "max|d 1-site|", "max|d 2-site|", "order"))
    prev = NaN; ok = true
    for dt in dts
        rho = rho0_for(P.start, L)
        for _ in 1:round(Int, t / dt)
            tdvp2_step!(rho, W, ComplexF64(dt); maxdim = 4^L, trunc_thresh = 1e-14,
                        maxiter = 30, hermitian = false)
            apply_shift!(rho, exp(dt * shift))
        end
        trm = real(trace_rho(Imps, rho, L))
        d1 = 0.0
        for j in 1:L
            a = pauli_1site(Imps, rho, L, j; tr = trm)
            d1 = max(d1, maximum(abs.(a .- exact1[j])))
        end
        d2 = 0.0
        for (j, k) in ((1, 2), (1, 3), (2, L))
            (j < k && k <= L) || continue
            a = pauli_2site(Imps, rho, L, j, k; tr = trm)
            b = dense_pauli2_lr(rhod, L, j, k)
            d2 = max(d2, maximum(abs.(a .- b)))
        end
        say(@sprintf("  %-9.5g %-13.4e %-13.4e %-8s", dt, d1, d2,
                     isnan(prev) ? "-" : @sprintf("%.2f", prev / max(d1, 1e-300))))
        prev = d1
    end
    # ⛔ TWO WAYS TO PASS, ONE WAY TO FAIL. Either the deviation FALLS with dt (discretisation, as
    # it should) or it already sits at the roundoff floor. A floor ABOVE roundoff that does NOT move
    # is the signature every operator bug in this stack has shown -- see `fused-jump-transpose-freeze`.
    ok = prev < 1e-3
    say(ok ? "  PASS -- generator and observable path agree with the dense Liouvillian." :
             "  *** FAIL -- do NOT report any long-range curve until this passes. ***")
    return ok
end

# ── the L = 16 protocol: 2-site TDVP vs 1-site CBE-TDVP vs CBE-BUG ───────────────────────────────

"""
    protocol_lr(P; shape, tmax, dt, cap, nsave, arms, out, kac, sq) -> path

Three integrators on one long-range model, streaming the paper's observables to CSV.

⛔ ALL THREE ARMS ARE HERE BECAUSE THE COMPARISON IS THREE-WAY. `tdvp_cbe1s` is the PUBLISHED
baseline (CBE-TDVP, arXiv:2208.10972) and omitting it would compare CBE-BUG only against the older
2-site sweep -- which is the easier target. `dissipative_heisenberg.jl:protocol` ran
`tdvp2/bug_rsvd/bug_par` and had that gap.

Observables, per the paper: bulk-averaged `<sigma^a>` and `C^aa_d = <sigma^a_i sigma^a_{i+d}> -
<sigma^a_i><sigma^a_{i+d}>` for `d = 1, 2`, plus `S_zz(q)` at `q = 0, pi, 2pi/N` (their Fig. 5).

⚠ THE BULK WINDOW DROPS A QUARTER FROM EACH END. The paper reports BULK quantities on N = 200; at
L = 16 the edges are a large fraction of the chain and an unwindowed average is dominated by them,
which reads as a physics difference from the paper rather than as a finite-size artefact.
"""
function protocol_lr(P; shape = (16,), tmax::Float64 = 4.0, dt::Float64 = 0.05,
                     cap::Int = 32, nsave::Int = 41, maxiter::Int = 16,
                     arms = ("tdvp2", "tdvp_cbe1s", "cbe_bug"),
                     growth::Float64 = 4.0, kac::Bool = false, out::String = "")
    set_symmetry!(:none)
    kw = (Js = P.Js, h = P.h, hdir = P.hdir, gamma = P.gamma,
          rmax = get(P, :rmax, Inf), kac = kac)
    W, shift, L = lr_lindblad(shape; kw...)
    tag = length(shape) == 1 ? "chain$(shape[1])" : "sq$(shape[1])x$(shape[2])"
    out = isempty(out) ? joinpath(OUT, @sprintf("dissip_lr_%s_%s_dt%g.csv",
                                                get(P, :tagname, "model"), tag, dt)) : out
    Imps = identity_superket(L)
    # ⛔ `maxiter` IS THE DOMINANT COST, NOT A SAFETY MARGIN. `arnoldi_expv` exits on BREAKDOWN
    # only, so EVERY solve burns the full budget -- there is no early convergence exit. MEASURED on
    # the L=8 smoke of the Fig-3 model at `maxiter = 30`: 70 s PER STEP and `krylov = 755` per step,
    # which is ~30 matvecs x ~25 solves. Dropping to 16 is a straight ~2x on the whole run.
    # ⚠ `certify` keeps 30, because there the question is exactness at L = 3-4 where cost is free.
    base = (maxdim = cap, trunc_thresh = 1e-12, maxiter = maxiter, hermitian = false)
    rs = (exact = false, dex = 8, dover = 4, comp_ratio = 1.0,
          krylov_basis = 3, krylov_tol = 0.0, growth = growth)
    steppers = Dict{String, Function}(
        "tdvp2"      => (p, h) -> tdvp2_step!(p, W, ComplexF64(h); base...),
        "tdvp_cbe1s" => (p, h) -> tdvp_cbe1s_step!(p, W, ComplexF64(h); base...),
        "cbe_bug"    => (p, h) -> cbe_bug_step!(p, W, ComplexF64(h); base..., rs...,
                                                parallel = true))
    blk = (div(L, 4) + 1):(L - div(L, 4))
    prs(d) = [(j, j + d) for j in blk if (j + d) in blk]

    say("\n=== PROTOCOL (long range): " * P.label * " ===")
    say(@sprintf("  shape = %s -> L = %d   t in [0, %g]  dt = %g (%d steps)  cap = %d",
                 string(shape), L, tmax, dt, round(Int, tmax / dt), cap))
    say(@sprintf("  h = %+.3f (%s)  gamma = %.3f  start = <%s>  kac = %s  MPO dim = %d",
                 P.h, String(P.hdir), P.gamma, String(P.start), kac,
                 maximum(mpo_virtual_dims(W))))
    say(@sprintf("  bulk sites %d..%d   pairs d=1: %d  d=2: %d",
                 first(blk), last(blk), length(prs(1)), length(prs(2))))
    say("  ⛔ symmetry = :none for EVERY arm -- the fused d=4 space has no symmetric variant.")

    open(out, "w") do io
        println(io, "arm,L,shape,dt,t,chi,krylov,seconds,trace_pre,sx,sy,sz," *
                    "cxx1,cyy1,czz1,cxx2,cyy2,czz2,szz_q0,szz_qpi,szz_q2pin")
        for arm in arms
            haskey(steppers, arm) || throw(ArgumentError("unknown arm $arm"))
            step! = steppers[arm]
            rho = rho0_for(P.start, L)
            enable_krylov_log()
            tnow = 0.0; sec = 0.0; trb = 1.0
            saveat = [tmax * k / (nsave - 1) for k in 0:(nsave - 1)]
            si = 1
            say(@sprintf("  -- %s --", arm))
            while si <= nsave
                if tnow >= saveat[si] - 1e-9
                    trr = real(trace_rho(Imps, rho, L))
                    m = Dict(j => pauli_1site(Imps, rho, L, j; tr = trr) for j in blk)
                    sx = sum(m[j][1] for j in blk) / length(blk)
                    sy = sum(m[j][2] for j in blk) / length(blk)
                    sz = sum(m[j][3] for j in blk) / length(blk)
                    cc = Float64[]
                    for d in (1, 2)
                        ps = prs(d); acc = zeros(Float64, 3)
                        for (j, k) in ps
                            c = pauli_2site(Imps, rho, L, j, k; tr = trr)
                            for a in 1:3; acc[a] += c[a] - m[j][a] * m[k][a]; end
                        end
                        append!(cc, acc ./ max(1, length(ps)))
                    end
                    sq  = szz_at(Imps, rho, L, trr, (0.0, pi, 2pi / L))
                    chi = maximum(bond_dims(rho))
                    kry = sum(get_krylov_log())
                    @printf(io, "%s,%d,%s,%g,%.6f,%d,%d,%.3f,%.10g,%.10g,%.10g,%.10g,%s,%s\n",
                            arm, L, tag, dt, tnow, chi, kry, sec, trb, sx, sy, sz,
                            join([@sprintf("%.10g", x) for x in cc], ","),
                            join([@sprintf("%.10g", x) for x in sq], ","))
                    flush(io)
                    (si <= 3 || si % 10 == 0) &&
                        say(@sprintf("     t=%5.2f  chi=%-4d kry=%-8d %7.1fs  <sx>=%+.5f <sy>=%+.5f <sz>=%+.5f  Czz1=%+.5f",
                                     tnow, chi, kry, sec, sx, sy, sz, cc[3]))
                    si += 1
                    continue
                end
                sec += @elapsed begin
                    step!(rho, dt)
                    apply_shift!(rho, exp(dt * shift))
                    trb = restore_trace!(Imps, rho, L; maxdim = cap)
                end
                tnow += dt
            end
            disable_krylov_log()
        end
    end
    say("wrote " * out)
    return out
end

"""
    steady_lr(P; shape, j2s, tmax, dt, cap, nsave, maxiter, arms, out) -> path

Their Fig. 5: the STEADY-STATE structure factor of the Eq. (6) diagonal model, swept over the second
(van der Waals, `alpha = 6`) coupling `J2` at fixed dipolar `J1`.

⛔ `kac = true` HERE AND NOWHERE ELSE. The Kac factor divides every coupling by
`N(alpha) = (1/(N-1)) sum_{i != j} d_ij^(-alpha)`, which the paper uses "to ensure extensivity with
system size". That rescales the energy -- and therefore TIME -- with `N`, so switching it on for the
dynamics figures would shift every curve along `t` against the paper's. Switching it OFF here would
instead make the sweep non-comparable across `L`, which is the whole point of a steady-state figure.

⛔ A STEADY STATE IS A CLAIM, NOT A LONG `tmax`. Each arm reports `drift`, the largest change in
`S_zz(q)` over the LAST QUARTER of the window relative to its final value. A run whose `drift` is not
small has not reached the NESS and its point on the sweep is reported as UNCONVERGED rather than
plotted as one -- an unconverged transient still traces a smooth, entirely plausible curve against
`J2`, which is exactly the failure this witness exists to catch.

⚠ THE THREE `q` VALUES ARE THE ORDER PARAMETER. `q = 0` is ferromagnetic, `q = pi` is Neel, and
`q = 2pi/N` is the longest incommensurate wavelength the chain can hold; the competition between the
`alpha = 3` and `alpha = 6` terms is what moves weight between them.
"""
function steady_lr(P; shape = (16,), j2s = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                   tmax::Float64 = 8.0, dt::Float64 = 0.05, cap::Int = 20, nsave::Int = 33,
                   maxiter::Int = 8, arms = ("tdvp2", "tdvp_cbe1s", "cbe_bug"),
                   growth::Float64 = 4.0, out::String = "")
    set_symmetry!(:none)
    (J1c, a1), (_, a2) = P.Js[1], P.Js[2]
    tag = length(shape) == 1 ? "chain$(shape[1])" : "sq$(shape[1])x$(shape[2])"
    out = isempty(out) ? joinpath(OUT, @sprintf("dissip_lr_fig5steady_%s_dt%g.csv", tag, dt)) : out
    base = (maxdim = cap, trunc_thresh = 1e-12, maxiter = maxiter, hermitian = false)
    rs = (exact = false, dex = 8, dover = 4, comp_ratio = 1.0,
          krylov_basis = 3, krylov_tol = 0.0, growth = growth)

    say("\n=== STEADY STATE (Fig. 5): S_zz(q) vs J2, Kac-normalised, " * tag * " ===")
    say(@sprintf("  J1 = %+.3f (alpha = %g)   J2 swept over %s (alpha = %g)",
                 J1c[3], a1, string(j2s), a2))
    say(@sprintf("  t in [0, %g]  dt = %g  cap = %d  maxiter = %d  kac = true", tmax, dt, cap, maxiter))
    say("  ⛔ symmetry = :none for EVERY arm -- the fused d=4 space has no symmetric variant.")

    nbad = 0
    open(out, "w") do io
        println(io, "arm,L,shape,j1,alpha1,j2,alpha2,dt,t,chi,krylov,seconds," *
                    "trace_pre,sz,szz_q0,szz_qpi,szz_q2pin,drift,converged")
        # ⚠ FLUSH THE HEADER NOW. Rows can only be written once `drift` is known, which is at the
        # END of a trajectory, so an interrupted sweep would otherwise leave a ZERO-BYTE file that
        # looks like a driver that never started rather than one that ran for hours.
        flush(io)
        for j2 in j2s
            Js = [(J1c, a1), ((0.0, 0.0, float(j2)), a2)]
            W, shift, L = lr_lindblad(shape; Js = Js, h = P.h, hdir = P.hdir, gamma = P.gamma,
                                      rmax = get(P, :rmax, Inf), kac = true)
            Imps = identity_superket(L)
            steppers = Dict{String, Function}(
                "tdvp2"      => (p, h) -> tdvp2_step!(p, W, ComplexF64(h); base...),
                "tdvp_cbe1s" => (p, h) -> tdvp_cbe1s_step!(p, W, ComplexF64(h); base...),
                "cbe_bug"    => (p, h) -> cbe_bug_step!(p, W, ComplexF64(h); base..., rs...,
                                                        parallel = true))
            blk = (div(L, 4) + 1):(L - div(L, 4))
            for arm in arms
                step! = steppers[arm]
                rho = rho0_for(P.start, L)
                enable_krylov_log()
                tnow = 0.0; sec = 0.0; trb = 1.0
                saveat = [tmax * k / (nsave - 1) for k in 0:(nsave - 1)]
                # ⛔ `chi`, `krylov` and `seconds` are recorded AT EACH SAVE TIME, not after the loop.
                # `krylov` is the CUMULATIVE matvec count and is the cost axis these runs are read
                # on (wall clock is contaminated whenever anything else shares the box); writing the
                # final total onto every row would flatten it into a constant and destroy exactly
                # that. Same convention as `protocol_lr`.
                rows = NTuple{8, Float64}[]                 # (t, q0, qpi, q2pin, sz, chi, kry, sec)
                si = 1
                while si <= nsave
                    if tnow >= saveat[si] - 1e-9
                        trr = real(trace_rho(Imps, rho, L))
                        sq  = szz_at(Imps, rho, L, trr, (0.0, pi, 2pi / L))
                        sz  = sum(pauli_1site(Imps, rho, L, j; tr = trr)[3] for j in blk) / length(blk)
                        push!(rows, (tnow, sq[1], sq[2], sq[3], sz,
                                     float(maximum(bond_dims(rho))), float(sum(get_krylov_log())), sec))
                        # a heartbeat, because one (J2, arm) trajectory is many minutes and the CSV
                        # cannot be written until `drift` is known at the end of it.
                        (si == 1 || si % 8 == 0 || si == nsave) &&
                            say(@sprintf("       J2=%+.2f %-11s t=%5.2f chi=%-4d %7.1fs  S(pi)=%+.5f",
                                         j2, arm, tnow, round(Int, maximum(bond_dims(rho))), sec, sq[2]))
                        si += 1
                        continue
                    end
                    sec += @elapsed begin
                        step!(rho, dt)
                        apply_shift!(rho, exp(dt * shift))
                        trb = restore_trace!(Imps, rho, L; maxdim = cap)
                    end
                    tnow += dt
                end
                disable_krylov_log()
                # drift over the LAST QUARTER of the window, relative to the final value
                tail = rows[max(1, ceil(Int, 3 * length(rows) / 4)):end]
                fin  = rows[end]
                drift = maximum(max(abs(r[2] - fin[2]), abs(r[3] - fin[3]), abs(r[4] - fin[4]))
                                for r in tail) / max(abs(fin[2]), abs(fin[3]), abs(fin[4]), 1e-12)
                conv = drift < 0.02
                conv || (nbad += 1)
                for r in rows
                    @printf(io, "%s,%d,%s,%.10g,%g,%.10g,%g,%g,%.6f,%d,%d,%.3f,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d\n",
                            arm, L, tag, J1c[3], a1, j2, a2, dt, r[1], round(Int, r[6]),
                            round(Int, r[7]), r[8], trb, r[5], r[2], r[3], r[4],
                            drift, conv ? 1 : 0)
                end
                flush(io)
                say(@sprintf("  J2=%+.2f %-11s chi=%-4d kry=%-7d %7.1fs  S(0)=%+.5f S(pi)=%+.5f S(2pi/N)=%+.5f  drift=%.3f %s",
                             j2, arm, round(Int, fin[6]), round(Int, fin[7]), fin[8],
                             fin[2], fin[3], fin[4], drift, conv ? "" : "<-- UNCONVERGED"))
            end
        end
    end
    say("wrote " * out)
    nbad == 0 || say(@sprintf("  ⚠ %d of %d (J2, arm) points did NOT reach a steady state at tmax = %g; they are flagged converged=0 and must NOT be read as NESS values.",
                              nbad, length(j2s) * length(arms), tmax))
    return out
end

# ── entry point ──────────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "certify" : ARGS[1]
    L    = parse(Int,     get(ENV, "LR_L", "16"))
    dt   = parse(Float64, get(ENV, "LR_DT", "0.05"))
    # ⚠ `cap = 32` IS DELIBERATELY CLOSE TO THE PAPER'S OWN BOND DIMENSION. Both its t-VMC+MPO and
    # its t-MPS curves in Fig. 3 are run at `chi = 20`, so a comparison at 32 is if anything more
    # generous than theirs -- and it keeps the FULL `t in [0,4]` window the figures show, which
    # pushing the cap would have cost. Truncating the time axis to afford a bigger cap would make
    # the figure non-comparable for the sake of a rank the paper never used.
    cap  = parse(Int,     get(ENV, "LR_CAP", "32"))
    tmax = parse(Float64, get(ENV, "LR_TMAX", "4.0"))
    ns   = parse(Int,     get(ENV, "LR_NSAVE", "41"))
    mi   = parse(Int,     get(ENV, "LR_MAXITER", "16"))
    cl   = parse(Int,     get(ENV, "LR_CERT_L", "4"))

    if mode in ("certify", "all")
        ok = true
        ok &= certify_lr(FIG4_CHAIN; shape = (cl,))
        ok &= certify_lr(FIG6;       shape = (cl,))
        ok || error("long-range certification FAILED -- no curve from this file is trustworthy")
    end
    # ⚠ `LR_CHAIN_RMAX` TRUNCATES THE INTERACTION RANGE AND SO CHANGES THE MODEL -- it is the only
    # setting here that does. Default Inf (the paper's Fig. 4 is untruncated). MEASURED at L=16:
    # `rmax = 8` discards 1.372e-3 of the alpha=3 weight and 1.882e-6 of the alpha=6 weight while
    # taking the MPO virtual dimension from 242 to 130. Quote those numbers whenever it is on;
    # "negligible" is only true relative to the chi=20 truncation error, so say which is which.
    crmax = parse(Float64, get(ENV, "LR_CHAIN_RMAX", "Inf"))
    mode in ("fig4", "all") &&
        protocol_lr(merge(FIG4_CHAIN, (tagname = "fig4ising", rmax = crmax)); shape = (L,),
                    tmax = tmax, dt = dt, cap = cap, nsave = ns, maxiter = mi)
    mode in ("fig4sq", "all") && begin
        LY = parse(Int, get(ENV, "LR_LY", "4"))
        protocol_lr(merge(FIG4_SQ, (tagname = "fig4ising",)); shape = (div(L, LY), LY),
                    tmax = tmax, dt = dt, cap = cap, nsave = ns, maxiter = mi)
    end
    mode in ("fig6", "all") &&
        protocol_lr(merge(FIG6, (tagname = "fig6xyz",)); shape = (L,),
                    tmax = tmax, dt = dt, cap = cap, nsave = ns, maxiter = mi)
    mode in ("fig5", "all") && begin
        # ⚠ A LONGER WINDOW THAN THE DYNAMICS FIGURES, because this one has to REACH the NESS rather
        # than trace a transient; `steady_lr` gates on that instead of assuming it.
        j2s = [parse(Float64, s) for s in split(get(ENV, "LR_J2S", "0.0,0.25,0.5,0.75,1.0,1.5,2.0"), ",")]
        steady_lr(FIG4_CHAIN; shape = (L,), j2s = j2s,
                  tmax = parse(Float64, get(ENV, "LR_STEADY_TMAX", "8.0")),
                  dt = dt, cap = cap, nsave = parse(Int, get(ENV, "LR_STEADY_NSAVE", "33")),
                  maxiter = mi)
    end
end
