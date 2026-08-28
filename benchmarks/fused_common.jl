# FUSED d = 4 DOUBLED CHAIN -- shared machinery for every open-system driver.
#
# One chain site per SYSTEM site, carrying the pair `(ket, bra)` with basis index `2*k + b + 1`
# and `0 = up`, `1 = down`:  1 = uu, 2 = ud, 3 = du, 4 = dd.
#
# ⛔ THE LAYOUT IS THE WHOLE DESIGN, AND THE OTHER TWO WERE MEASURED AND REJECTED.
#
#   INTERLEAVED (ket 2j-1, ancilla 2j) puts every system-system term at RANGE 2, so no two-site
#   block contains both of a term's operators. The block sees such a term only through an MPO
#   environment channel, which for a half-open term is a ONE-SITE EXPECTATION of a charge-changing
#   operator -- and `<S^-> = 0` on a product state. MEASURED: the superket NEVER MOVES,
#   `max|rho_mps(T) - rho(0)| = 0.0000e+00` exactly, error flat in dt (order 0.00 over a 16x
#   range) and identical for `tdvp2` and `cbe_bug`. ⚠ The MPO is NOT at fault -- with ENTANGLED
#   probes it reproduces the dense superoperator to 1.3e-15, and the failing response is `0`
#   EXACTLY rather than a wrong nonzero value.
#
#   BLOCKED (ket 1..L, ancilla L+1..2L) fixes the dynamics -- 7.5e-14 (tdvp2) / 2.6e-15
#   (cbe_bug) at gamma = 0, order 2.99 at gamma = 0.5 -- but `|I>>` is MAXIMALLY ENTANGLED across
#   the ket/ancilla cut (chi = 2^L) and a SPATIAL cut needs TWO MPS cuts, so operator entanglement
#   is not readable from one bond.
#
#   FUSED gets all three: `H` is RANGE 1 (inside a two-site update, so the dynamics starts), the
#   jump term is ON-SITE, `|I>>` is a PRODUCT state, and a spatial cut IS one MPS bond.
#
# ⚠ NORMALISATION IS WRITTEN OUT, NOT INHERITED. Telum's `q.Sp` carries a 1/sqrt(2) (`gates.jl`
# says so), which silently rescaled a first attempt at `|I>>`: `Tr rho0` came back -0.5, not +1.
# Every matrix here is an explicit `kron`.

using LinearAlgebra, SparseArrays
using LurCGT, Telum
using Telum: _getLocalSpace_no_symmetry
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const _I2  = Matrix{Float64}(LinearAlgebra.I, 2, 2)
const _SP  = Float64[0 1; 0 0]
const _SM  = Float64[0 0; 1 0]
const _SZ  = Float64[0.5 0; 0 -0.5]

"""
The d = 4 local space for one fused `(ket, bra)` pair, under `:none`.

⚠ `kron(A, B)` puts A on the KET (the more significant index), matching `2*k + b + 1`.
`CUP` maps `|uu>` to `(|uu> + |dd>)/sqrt2` and is written SYMMETRIC on purpose, so it acts the
same whether the contraction convention applies it or its transpose.
"""
function fused_space()
    c = 1 / sqrt(2)
    CUP = zeros(Float64, 4, 4)
    CUP[1, 1] = c; CUP[4, 1] = c; CUP[1, 4] = c; CUP[4, 4] = c
    # ⛔ THIS DICT IS Float64 AND CANNOT BE PROMOTED. MEASURED 2026-08-28: changing it to
    # `ComplexF64` to hold a `(I - sigma^y)/2` projector throws
    # `InexactError: Float64(0.0 + 0.5im)` from inside `_getLocalSpace_no_symmetry` -- Telum builds
    # the local space real, and no caller can opt out. A complex one-site operator is therefore
    # built as a LINEAR COMBINATION of the real ones already here (see `py_projector`), which needs
    # no new local space and so keeps `Q4`'s index identities.
    ops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}(
        :I   => sparse(kron(_I2, _I2)),
        :Kp  => sparse(kron(_SP, _I2)), :Km => sparse(kron(_SM, _I2)),
        :Kz  => sparse(kron(_SZ, _I2)),
        :Bp  => sparse(kron(_I2, _SP)), :Bm => sparse(kron(_I2, _SM)),
        :Bz  => sparse(kron(_I2, _SZ)),
        :JMP => sparse(kron(_SZ, _SZ)),
        # ── BOUNDARY jump terms (arXiv:2104.11479). ADDITIVE: nothing existing reads them. ────
        #
        # ⛔ THESE ARE THE FIRST OPERATORS HERE THAT TEST THE BRA-LEG TRANSPOSE, and `JMP` above
        # CANNOT. `rho B` is `kron(I, transpose(B))`, but `JMP`'s `B = S^z` is real SYMMETRIC, so
        # `transpose(S^z) = S^z` and the correct and incorrect conventions COINCIDE EXACTLY.
        # Dephasing therefore validates the layout, `|I>>`, the zero-body constant and the on-site
        # trick (§3.5) while leaving this one thing unpinned. For `sigma^+ rho sigma^-` they
        # differ, and the wrong choice is a generator that still looks like a Lindbladian but does
        # not preserve the trace:
        #
        #     sigma^+ rho sigma^-  ->  kron(_SP, transpose(_SM)) = kron(_SP, _SP)
        #     sigma^- rho sigma^+  ->  kron(_SM, transpose(_SP)) = kron(_SM, _SM)
        #
        # `benchmarks/dissipative_xx.jl`'s `calibrate()` pins it against a dense Liouvillian.
        :JP  => sparse(kron(_SP, _SP)),      # sigma^+ rho sigma^-   (gain: pumps UP)
        :JM  => sparse(kron(_SM, _SM)),      # sigma^- rho sigma^+   (loss: pumps DOWN)
        # ⛔ `PX` EXISTS BECAUSE `<sigma^x> = +1` IS NOT A FUSED BASIS STATE. The dissipative
        # Heisenberg protocol (Hryniuk & Szymanska, Comms Phys 2026 9:45) starts from the product
        # state `|+x><+x|`, whose local superket is `vec((I + sigma^x)/2) = (1,1,1,1)/2` -- a
        # uniform superposition over all four fused indices, so `fused_product_state` (which picks
        # ONE index per site) cannot build it.
        #
        # It is reachable instead as a KET-SIDE multiplication applied to `|I>>`:
        # `kron((I+sx)/2, I) * vec(I) = vec((I+sx)/2)`. That keeps the state a PRODUCT (chi = 1),
        # which the fused layout is chosen for in the first place.
        #
        # ⚠ REAL SYMMETRIC ON PURPOSE, like `CUP`. `site_apply!` contracts leg 1 and therefore
        # applies the TRANSPOSE -- the convention that silently swapped `JP`/`JM` and froze the
        # boundary-driven model. `(I + sigma^x)/2` is its own transpose, so this operator acts the
        # same either way and cannot be caught by that trap.
        :PX  => sparse(kron((_I2 .+ _SP .+ _SM) ./ 2, _I2)),
        # ⛔ `Kx` IS SAFE TO MEASURE WITH AND `Ky` CANNOT EXIST HERE. `site_apply!` contracts leg 1,
        # i.e. applies the TRANSPOSE. `sigma^x` is real SYMMETRIC, so `kron(sigma^x, I)` is its own
        # transpose and reads back the observable it names. `sigma^y` is ANTIsymmetric
        # (`sigma^y' = -sigma^y`) AND imaginary, so it would (a) come back sign-flipped and (b) not
        # fit this `Float64` dict at all. Get `<sigma^y>` from `-i(<sigma^+> - <sigma^->)` instead,
        # and VERIFY it on the start state where `<sx>=1, <sy>=<sz>=0` are known exactly.
        :Kx  => sparse(kron(_SP .+ _SM, _I2)),
        :CUP => sparse(CUP))
    return _getLocalSpace_no_symmetry(ops, ("s", "s", "op"))
end
const Q4 = fused_space()

# ⛔ ONE LOCAL SPACE FOR THE WHOLE FUSED STACK. `Q4` is a single `_getLocalSpace_no_symmetry`
# object and tensors built from a SECOND such call do not share its index identities, so a driver
# that made its own d=4 space could not contract its MPO against `identity_superket`'s state. The
# jump operators are therefore added HERE rather than in the driver that needs them.

"""
    fused_boundary_lindblad(L; J, eps_L, eps_R, mu_L, mu_R) -> (mpo, shift)

The Liouvillian of the BOUNDARY-DISSIPATED XX chain (arXiv:2104.11479) as a fused-site MPO:

    H     = J sum_k (sigma^+_k sigma^-_{k+1} + h.c.)
    L_1,2 = sqrt(eps_L (1 +- mu_L)/2) sigma^{+,-}_1 ,   L_3,4 = the same at site N

⛔ WHY THIS MODEL IS WORTH A SECOND LINDBLADIAN BUILDER. `PLAN-open-systems.md` §0 lists the
`exact ref` column for every OPEN model as "—": XX open, Heisenberg square open and the LGT are all
validated against published TRENDS (diffusion, Zeno slowing), never against a number. This one is
EXACTLY SOLVABLE by third quantization -- an `N x N` non-Hermitian matrix, `O(N^3)`, nothing of
size `4^N` -- so it turns that column into a real error. See `xx_boundary_*` in
`tests/common/free_fermion.jl`, whose gap reproduces the paper's `O(N^-3)` closed form to a ratio
of 0.9994 at N=60.

It is also the COMPLEX-FREQUENCY model: `lambda = B + 2J cos(theta)` with `theta` complex, so every
mode is an oscillation TIMES a decay -- which neither a Hermitian real-time quench (real spectrum)
nor imaginary time (completely monotone) can supply.

⛔ THE HOPPING IS `J`, FIXED BY THE REFERENCE AND NOT BY THE PAPER'S PROSE. The paper writes
`H = J sum (sx sx + sy sy)`, and `sx sx + sy sy = 2(sigma^+ sigma^- + h.c.)` in PAULI conventions;
what the reference actually requires is that the Jordan-Wigner single-particle matrix have
off-diagonal `J`, matching `xx_boundary_structure_matrix`. `PairVertex(Kp, Km, 0.5)` carries a
built-in `1/2`, so the coupling entry is `2J`.

⛔ THE ANTICOMMUTATOR NEEDS NO NEW OPERATOR. With `G+- = eps(1 +- mu)/2`,
`(L^+)'L^+ = n_dn` and `(L^-)'L^- = n_up`, and `n_dn = I/2 - S^z`, `n_up = I/2 + S^z`:

    G+ n_dn + G- n_up = (eps/2) I - eps*mu*S^z

so `-1/2{.,rho}` is `+(eps*mu/2) S^z` on EACH of the ket and bra legs plus an IDENTITY part
`-eps/4` on each. The identity part is a zero-body term `pair_mpo` cannot hold and is returned as
`shift`, exactly as `fused_lindblad` does for dephasing -- §3.2's FAULT 1, which cost a whole
diagnostic ladder the first time it was dropped.

⛔ `jump_transpose` EXISTS BECAUSE THE LOCAL-OPERATOR CONVENTION IS NOT OBVIOUS AND GETTING IT
BACKWARDS FREEZES THE STATE COMPLETELY RATHER THAN TILTING IT.

`site_apply!` contracts the operator's LEG 1, which applies the TRANSPOSE -- MEASURED: applying
`Q4.JP` (a raising operator) to `|I>>` gives `<S^z_1> = -0.5`, and `Q4.JM` gives `+0.5`. (`Q4.CUP`
is written symmetric on purpose to dodge exactly this, which is the tell that the convention was
already known to be ambiguous.) If `pair_mpo` shares that convention then `JP`/`JM` are exchanged
inside the MPO, and the consequence is NOT a sign error in a profile:

    correct   G_1 u = e1/sqrt2 + (0.5,0,0,-0.5)/sqrt2 = (1.5,0,0,-0.5)/sqrt2  -> d<S^z_1>/dt = +0.5
    swapped   G_1 u = e4/sqrt2 + (0.5,0,0,-0.5)/sqrt2 = (0.5,0,0,+0.5)/sqrt2  -> d<S^z_1>/dt =  0

The jump term becomes LOSS while the anticommutator stays GAIN's, so the two ends balance exactly,
`|I>>` becomes a genuine FIXED POINT, and the run reports a profile of EXACTLY zero with `chi = 1`
and `Tr rho = 1.0000000000` -- identically for every integrator and at every `dt`. That reads as a
dead sweep, not as a swapped operator, which is where the diagnosis went first.

⚠ THE FLAG IS SETTLED BY MEASUREMENT, NOT BY READING THE CONTRACTION -- AND THE DEFAULT IS NOW
`true`, WHICH IS THE SETTING THAT REPRODUCES THE ANALYTIC REFERENCE.

⛔ DO NOT SETTLE IT WITH `fused_freeze_probe.jl` GATE 1.5 / GATE 5. Those form the perturbed state
`p = rho0 + eta*Kz_1*rho0` and read `2^L <p|W|p>/eta`, and they return the SAME constant
(`-149999.5`) for BOTH settings -- a value that scales as `1/eta`, i.e. `eta` never entered. A gate
whose answer does not move when its scanned variable moves has measured nothing, and both of those
rungs were quoted as evidence against the swap before anyone checked that they varied.

✅ SETTLED INSTEAD BY STEPPING, in `benchmarks/fused_jump_convention.jl` GATE 10a/10b, against the
`xx_boundary_sz` Lyapunov profile (`J = eps_L = eps_R = 1`, `mu = +-1`, `dt = 0.01`):

    jump_transpose   tdvp2 L=2      tdvp2 L=3      bug_rsvd L=3    dt-halving at T=0.1
    false            4.975e-03      4.975e-03      4.975e-03       4.742485e-02 at EVERY dt
    true             1.067e-16      2.541e-16      1.656e-07       at the 1e-14 floor

The `false` column is not "less accurate" -- it is the profile of a state that never moved, and its
error is IDENTICAL at every step size, which is the fingerprint. `true` reproduces the reference to
roundoff for `tdvp2` and to the rSVD sampling level for `bug_rsvd`.

⚠ GATE 10b's ratios are NOT an order measurement. At L=3 with `D=128` nothing truncates and the
horizon `T=0.1` is short, so the error sits on the roundoff floor and the "ratios" are noise --
see the standing note that a dt-order test at full rank measures roundoff, not the scheme. It is
conclusive for the question it was asked (moving vs frozen) and for nothing else.
"""
function fused_boundary_lindblad(L::Int; J::Float64 = 1.0,
                                 eps_L::Float64 = 1.0, eps_R::Float64 = 1.0,
                                 mu_L::Float64 = 1.0, mu_R::Float64 = -1.0,
                                 jump_transpose::Bool = true)
    L >= 2 || throw(ArgumentError("fused_boundary_lindblad needs at least two sites"))
    PV = RSVDCBEBondUpdate.PairVertex
    Hk = zeros(ComplexF64, L, L); Hb = zeros(ComplexF64, L, L)
    for k in 1:(L - 1)
        Hk[k, k + 1] = -im * 2J          # the 2 cancels PairVertex's built-in 0.5
        Hb[k, k + 1] = +im * 2J
    end
    verts = PV[PV(Q4.Kp, Q4.Km, 0.5), PV(Q4.Km, Q4.Kp, 0.5),
               PV(Q4.Bp, Q4.Bm, 0.5), PV(Q4.Bm, Q4.Bp, 0.5)]
    Js = Any[Hk, Hk, Hb, Hb]
    # ⚠ ONE-BODY TERMS RIDE THE END BONDS: `O_1 (x) I_2` on bond (1,2) and `I_{L-1} (x) O_L` on
    # bond (L-1, L) -- the distribute-over-bonds trick §3.6 records, and the LAST SITE NEEDS ITS
    # OWN vertex because bond `j` only ever reaches site `j+1`.
    # ⛔ NEVER PUSH A VERTEX WHOSE COUPLING MATRIX IS IDENTICALLY ZERO. At the fully-polarised
    # drive `mu_L = +1, mu_R = -1` -- the default, and the one that makes the profile asymmetric
    # enough to be diagnostic -- `gp(eps_R, mu_R)` and `gm(eps_L, mu_L)` are EXACTLY 0, so two of
    # the twelve coupling matrices are all-zero. Pushing them anyway asks `pair_mpo` to open and
    # carry channels that contribute nothing, and the channel bookkeeping (`alive`, `rowof`,
    # `colof`) is indexed by vertex.
    #
    # ⚠ MEASURED: each one-body operator works ALONE. `benchmarks/fused_freeze_probe.jl` GATE 7
    # builds a single-vertex MPO per operator and steps it: `Kz` moves the state (6.2e-04), `JP`
    # and `JM` move it (5.9e-04). (`JMP` does not, and that is CORRECT physics rather than a
    # defect: `kron(Sz,Sz)*u = 0.25*u`, so pure dephasing leaves the maximally mixed state
    # invariant.) It is only the ASSEMBLED operator that freezes, which is what points at the
    # zero-weight vertices rather than at any single term.
    one_body!(op, wl, wr) = begin
        if wl != 0
            A = zeros(ComplexF64, L, L); A[1, 2] = wl
            push!(verts, PV(op, Q4.I, 1.0)); push!(Js, A)
        end
        if wr != 0
            B = zeros(ComplexF64, L, L); B[L - 1, L] = wr
            push!(verts, PV(Q4.I, op, 1.0)); push!(Js, B)
        end
    end
    gp(e, m) = e * (1 + m) / 2
    gm(e, m) = e * (1 - m) / 2
    # ⚠ `jump_transpose` EXCHANGES THE TWO JUMP OPERATORS, because `kron(Sp,Sp)' = kron(Sm,Sm)`
    # exactly -- so "apply the transpose" and "swap JP with JM" are the SAME operation here, and
    # the swap is expressible without introducing a transposed tensor. See the docstring.
    Jgain = jump_transpose ? Q4.JM : Q4.JP
    Jloss = jump_transpose ? Q4.JP : Q4.JM
    one_body!(Jgain, gp(eps_L, mu_L), gp(eps_R, mu_R))
    one_body!(Jloss, gm(eps_L, mu_L), gm(eps_R, mu_R))
    one_body!(Q4.Kz, eps_L * mu_L / 2, eps_R * mu_R / 2)
    one_body!(Q4.Bz, eps_L * mu_L / 2, eps_R * mu_R / 2)
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I),
           -(eps_L + eps_R) / 2
end

"A product state over fused sites; `idxs[j]` in 1..4. Mirrors `_product_state_none`."
function fused_product_state(idxs::Vector{Int})
    L = length(idxs)
    tensors = Any[]
    prev = (getvac(Q4.I, ("L,0", "L,1")), 2)
    for i in 1:L
        (1 <= idxs[i] <= 4) || throw(ArgumentError("fused index must be 1..4, got $(idxs[i])"))
        F = to_concrete(getIdentity(prev, (Q4.I, 1); itag = "L,$(i + 1)"))
        F = to_concrete(setitag(F, 2, "S,$i"))
        A = to_concrete(getsub(F, 3, s -> idxs[i]))
        A = to_concrete(A * (1.0 + 0.0im))
        push!(tensors, A)
        prev = (A, 3)
    end
    return SymMPS(tensors, L)
end

"""
Apply a one-site operator to site `k`, tagged from the tensor itself.

⛔ CONTRACT THE OPERATOR LEG 1. Operators carry `(ket '+', bra '-')` and an MPS site leg is `'-'`,
so pairing leg 2 with the site leg is two `'-'` legs and Telum refuses it outright. `apply_gate` --
the path the whole Trotter machinery is validated on -- contracts the gate's KET legs against
theta's site legs, so the one-site form is leg 1.
"""
function site_apply!(p, k::Int, M)
    t = p[k].inds[2].itags
    op = to_concrete(setitag(setitag(M, 1, t), 2, t))
    p[k] = to_concrete(permutedims(contract(op, (1,), p[k], (2,)), (2, 1, 3)))
    return p
end

"`rho0 = |psi0><psi0|` for a PRODUCT `psi0`: site j in `|s,s>` -- index 1 for up, 4 for down."
rho0_product(spins::Vector{Symbol}) =
    fused_product_state([s === :up ? 1 : 4 for s in spins])

"`|I>>` NORMALISED (the true identity superket is `2^(L/2)` times it). A PRODUCT state, chi = 1."
function identity_superket(L::Int)
    p = rho0_product([:up for _ in 1:L])
    for j in 1:L
        site_apply!(p, j, Q4.CUP)
    end
    return p
end

"""
    rho0_plusx(L) -> SymMPS

`rho = (|+x><+x|)^{(x)L}`, the start state of the dissipative Heisenberg protocol, with `Tr rho = 1`.

⛔ IT IS NOT A FUSED BASIS STATE, so `fused_product_state` cannot build it -- see `Q4.PX`. It is
built by applying the ket-side `(I + sigma^x)/2` to every site of `|I>>`, which keeps `chi = 1`.

⚠ THE SCALE IS THE PART THAT BITES. `identity_superket` is NORMALISED, i.e. it represents
`vec(I)/2^(L/2)`, so applying `PX` per site gives `rho/2^(L/2)` and the result must be scaled BACK
by `2^(L/2)`. Getting this wrong does not look like a scale error: `restore_trace!` sees a huge
deficit and projects the state onto `|I>>`, which reads as a DEAD INTEGRATOR -- exactly the failure
that cost a full diagnostic round on the maximally-mixed start. The assertion below is the guard.
"""
function rho0_plusx(L::Int)
    p = identity_superket(L)
    for j in 1:L
        site_apply!(p, j, Q4.PX)
    end
    apply_shift!(p, 2.0^(L / 2))
    tr = real(trace_rho(identity_superket(L), p, L))
    abs(tr - 1) < 1e-10 || error("rho0_plusx: Tr rho = $tr, expected 1")
    return p
end

"""
    py_projector() -> TLArray

`transpose((I - sigma^y)/2)` as a fused ket-side operator, built by COMBINING the real operators
already in `Q4`.

⛔ IT CANNOT LIVE IN `Q4`. That dict is `Float64` and Telum's `_getLocalSpace_no_symmetry` builds
the local space real; asking it for a complex entry throws `InexactError: Float64(0.0 + 0.5im)`
(measured). And a SECOND local-space call is not a workaround -- its tensors would not share `Q4`'s
index identities, so they could not contract against `identity_superket`'s state. Combining existing
`Q4` members sidesteps both: same space, same indices, complex coefficients.

The identity is checked by construction: `Q4.PX == (Q4.I + Q4.Kp + Q4.Km)/2` holds for the real
case, so the same recipe with `-i`/`+i` gives the `sigma^y` pole.

⛔ THE TRANSPOSE IS DELIBERATE. `site_apply!` contracts leg 1 and so applies `M'`. `PX` and `CUP`
are real SYMMETRIC, so there the distinction is invisible -- but `P_y = (I + i s^+ - i s^-)/2` is
HERMITIAN AND NOT SYMMETRIC, and `transpose(P_y)` is the projector onto the OPPOSITE pole. Storing
the naive `P_y` would prepare `<sigma^y> = +1` while every label says `-1`: not a crash, the whole
figure reflected. `rho0_minusy` ASSERTS the sign rather than trusting this comment.
"""
py_projector() = to_concrete(0.5 * (Q4.I - im * Q4.Kp + im * Q4.Km))

"""
    rho0_minusy(L) -> SymMPS

`rho = (|-y><-y|)^{(x)L}`, the start state of the paper's LONG-RANGE figures (Figs 4 and 6 of
Hryniuk & Szymanska: "starting from the product state `<sigma^y> = -1`").

Same construction as [`rho0_plusx`](@ref) -- apply the single-site projector to every site of
`|I>>`, then undo `identity_superket`'s `2^(L/2)` normalisation -- but with [`py_projector`](@ref).

⛔ THE TWO ASSERTIONS ARE THE WHOLE POINT AND MUST NOT BE RELAXED. `<sigma^y>` is the ONE Pauli
component that can detect a transpose slip here: `sigma^x` is real symmetric and `sigma^z` diagonal,
so both read back the same number under either convention.
"""
function rho0_minusy(L::Int)
    p = identity_superket(L)
    PY = py_projector()
    for j in 1:L
        site_apply!(p, j, PY)
    end
    apply_shift!(p, 2.0^(L / 2))
    Imps = identity_superket(L)
    tr = real(trace_rho(Imps, p, L))
    abs(tr - 1) < 1e-10 || error("rho0_minusy: Tr rho = $tr, expected 1")
    sy = pauli_1site(Imps, p, L, 1; tr = tr)[2]
    abs(sy + 1) < 1e-10 || error("rho0_minusy: <sigma^y_1> = $sy, expected -1 " *
                                 "(a transposed Q4.PY prepares the +y pole)")
    return p
end

"`Tr(rho)`, and with `at = j` the `Tr(S^z_j rho)`. `S^z` is Hermitian so it may sit on `<<I|`."
function trace_rho(Imps, rho, L::Int; at::Int = 0)
    o = copy(Imps)
    at > 0 && site_apply!(o, at, Q4.Kz)
    return 2.0^(L / 2) * overlap(o, rho)
end

"""
    trace_op(Imps, rho, L, ops) -> ComplexF64

`Tr(O_{j1} O_{j2} ... rho)` for `ops = [(j1, M1), (j2, M2), ...]`, each `M` a KET-side fused
operator from `Q4` and each site DISTINCT. Generalises [`trace_rho`](@ref) to the two-site
correlators the dissipative-Heisenberg protocol reports.

⚠ THE TWO TRANSPOSES CANCEL HERE, AND THE REASON IS WORTH KEEPING. `site_apply!` contracts leg 1
and so applies `M'` (transpose) -- the convention that silently swapped `JP`/`JM` and froze the
boundary-driven model. But this path then takes an OVERLAP, which contributes a second dagger:

    site_apply!(|I>>, j, kron(A, I))  ->  superket of A^T
    overlap(that, rho)                 =  Tr((A^T)^dag rho)  =  Tr(conj(A) rho)

and every `S^+`/`S^-`/`S^z` in `Q4` is REAL, so `conj(A) = A` and `trace_op` returns the operator
it names. ⛔ THAT IS A DERIVATION, NOT A MEASUREMENT -- and this is precisely the class of trap
that has twice produced smooth, plausible, wrong curves in this stack. `certify_obs` in
`dissipative_heisenberg.jl` pins it against a dense Liouvillian, and it does so with `sigma^y`
INCLUDED, because `sigma^y` is the only one of the three that can detect the error: `sigma^x` is
real symmetric and `sigma^z` diagonal, so both give the same answer under either convention.
"""
function trace_op(Imps, rho, L::Int, ops)
    js = [j for (j, _) in ops]
    length(unique(js)) == length(js) || throw(ArgumentError(
        "trace_op needs DISTINCT sites (two ops on one site would compose as (M1*M2)'), got $js"))
    o = copy(Imps)
    for (j, M) in ops
        1 <= j <= L || throw(ArgumentError("site $j out of range 1..$L"))
        site_apply!(o, j, M)
    end
    return 2.0^(L / 2) * overlap(o, rho)
end

"""
    pauli_1site(Imps, rho, L, j; tr) -> (sx, sy, sz)

Pauli expectations `<sigma^a_j>` at one site, all three real, normalised by `tr = Tr(rho)`.

⛔ `sigma^y` COMES FROM `-i(<sigma^+> - <sigma^->)` AND CANNOT COME FROM A `Ky` OPERATOR: `Q4` is a
`Float64` dict and `sigma^y` is imaginary, so it does not fit -- see `fused_space`.
⚠ `2 *` ON `sigma^z` ONLY. `Kz` is `S^z = sigma^z/2`, while `Kx = S^+ + S^- = sigma^x` and
`Kp/Km = sigma^{+,-}` already carry Pauli normalisation. Applying one blanket factor to all three
is the easy way to get two curves that disagree by exactly 2 and look merely "off by a constant".
"""
function pauli_1site(Imps, rho, L::Int, j::Int; tr::Float64)
    sp = trace_op(Imps, rho, L, [(j, Q4.Kp)]) / tr
    sm = trace_op(Imps, rho, L, [(j, Q4.Km)]) / tr
    return (real(trace_op(Imps, rho, L, [(j, Q4.Kx)]) / tr),
            real(-im * (sp - sm)),
            2 * real(trace_op(Imps, rho, L, [(j, Q4.Kz)]) / tr))
end

"""
    pauli_2site(Imps, rho, L, j, k; tr) -> (xx, yy, zz)

`<sigma^a_j sigma^a_k>` for `a = x, y, z`, sites distinct.

`sigma^y sigma^y` expands into FOUR two-site terms, since
`sigma^y_j sigma^y_k = -(s^+_j s^+_k - s^+_j s^-_k - s^-_j s^+_k + s^-_j s^-_k)`.
"""
function pauli_2site(Imps, rho, L::Int, j::Int, k::Int; tr::Float64)
    o(a, b) = trace_op(Imps, rho, L, [(j, a), (k, b)]) / tr
    yy = -(o(Q4.Kp, Q4.Kp) - o(Q4.Kp, Q4.Km) - o(Q4.Km, Q4.Kp) + o(Q4.Km, Q4.Km))
    return (real(o(Q4.Kx, Q4.Kx)), real(yy), 4 * real(o(Q4.Kz, Q4.Kz)))
end

"""
    fused_lindblad(L, Jm, gammas; delta = 0.0) -> (mpo, shift)

`Jm` is the SYSTEM coupling matrix (strict upper triangle), `gammas[j]` the dephasing rate on
system site `j`, `delta` the `S^zS^z` anisotropy of the system Hamiltonian.

⚠ `Jm` MUST BE REAL: the ancilla half carries `+i*H^T` and is written as `+i*Jm`, which is right
only where `H^T = H`.

⛔ THE ZERO-BODY TERM COMES BACK SEPARATELY. `L†L = 1/4` for `L = S^z`, so the anticommutator is
`-(sum_j gamma_j / 4)` times the IDENTITY, and `pair_mpo` holds two-body vertices only. It is
returned as `shift` and applied as an EXACT scalar `exp(tau*shift)` on the state, since
`exp(tau*(W + c*I)) = e^(tau*c)*exp(tau*W)`. Omitting it does not merely rescale -- `Tr rho`
inflates (MEASURED 1.1618 at gamma = 0.5).

⚠ THE ON-SITE JUMP TERM IS DISTRIBUTED OVER BONDS, `O_j (x) I_{j+1}` for `j < L` plus
`I_{L-1} (x) O_L`, because `pair_mpo` is two-body only -- the same trick `ising_bond_gates` uses
for its field. Per-vertex coupling matrices are what let the two supports coexist.
"""
function fused_lindblad(L::Int, Jm::AbstractMatrix, gammas::Vector{Float64};
                        delta::Float64 = 0.0, hz::Union{Nothing, Vector{Float64}} = nothing,
                        Jzz::Union{Nothing, AbstractMatrix} = nothing)
    all(isreal, Jm) || throw(ArgumentError(
        "fused_lindblad needs a REAL coupling matrix: the ancilla half is written +i*Jm, " *
        "which equals +i*H^T only when H^T = H"))
    length(gammas) == L || throw(ArgumentError("need one gamma per system site"))
    L >= 2 || throw(ArgumentError("fused_lindblad needs at least two sites"))
    hz === nothing || length(hz) == L || throw(ArgumentError("need one hz per system site"))
    Jzz === nothing || all(isreal, Jzz) || throw(ArgumentError("Jzz must be REAL, same reason"))
    PV = RSVDCBEBondUpdate.PairVertex
    Jk = zeros(ComplexF64, L, L); Jb = zeros(ComplexF64, L, L)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        Jk[i, j] = -im * Jm[i, j]
        Jb[i, j] = +im * Jm[i, j]
    end
    # ⛔ THE S^zS^z SUPPORT NEED NOT MATCH THE XY SUPPORT. `delta * Jm` is right for XXZ, where one
    # matrix scales all three vertices, and WRONG for the Schwinger chain, whose hopping is
    # NEAREST-NEIGHBOUR while its Coulomb term is ALL-TO-ALL. Passing `Jzz` uses it directly and
    # leaves `delta` out of it. Per-vertex coupling matrices cost no extra bond dimension: the
    # channel count is set by how many openings are still closable.
    Kzz = Jzz === nothing ? nothing : (-im) .* ComplexF64.(Jzz)
    Bzz = Jzz === nothing ? nothing : (+im) .* ComplexF64.(Jzz)
    # ⚠ ONE-BODY TERMS GO THROUGH A TWO-BODY BUILDER, `O_j (x) I_{j+1}` on bonds 1..L-1 plus
    # `I_{L-1} (x) O_L` on the last -- the trick `ising_bond_gates` uses for its field. Two
    # vertices per one-body operator, and the LAST SITE NEEDS ITS OWN because bond `j` only ever
    # reaches site `j+1`: dropping it silently evolves a chain whose final site has no field,
    # which is a different Hamiltonian whose error SHRINKS with L, so small tests miss it.
    Jj1 = zeros(ComplexF64, L, L); Jj2 = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        Jj1[j, j + 1] = gammas[j]
    end
    Jj2[L - 1, L] = gammas[L]
    zcoeff = Jzz === nothing ? delta : 1.0        # `Jzz` carries its own scale
    verts = PV[PV(Q4.Kp, Q4.Km, 0.5), PV(Q4.Km, Q4.Kp, 0.5), PV(Q4.Kz, Q4.Kz, zcoeff),
               PV(Q4.Bp, Q4.Bm, 0.5), PV(Q4.Bm, Q4.Bp, 0.5), PV(Q4.Bz, Q4.Bz, zcoeff),
               PV(Q4.JMP, Q4.I, 1.0), PV(Q4.I, Q4.JMP, 1.0)]
    Js = Any[Jk, Jk, Kzz === nothing ? Jk : Kzz,
             Jb, Jb, Bzz === nothing ? Jb : Bzz, Jj1, Jj2]
    if hz !== nothing
        Hk1 = zeros(ComplexF64, L, L); Hk2 = zeros(ComplexF64, L, L)
        Hb1 = zeros(ComplexF64, L, L); Hb2 = zeros(ComplexF64, L, L)
        for j in 1:(L - 1)
            Hk1[j, j + 1] = -im * hz[j]
            Hb1[j, j + 1] = +im * hz[j]
        end
        Hk2[L - 1, L] = -im * hz[L]
        Hb2[L - 1, L] = +im * hz[L]
        append!(verts, PV[PV(Q4.Kz, Q4.I, 1.0), PV(Q4.I, Q4.Kz, 1.0),
                          PV(Q4.Bz, Q4.I, 1.0), PV(Q4.I, Q4.Bz, 1.0)])
        append!(Js, Any[Hk1, Hk2, Hb1, Hb2])
    end
    return pair_mpo(L, Vector{AbstractMatrix}(Js), verts; Iloc = Q4.I), -sum(gammas) / 4
end

"""
The omitted zero-body term, as an exact scalar on the state.

⛔ THE SCALAR GOES ON THE ORTHOGONALITY CENTRE, NOT ON SITE 1, AND THAT IS A CORRECTNESS FIX RATHER
THAN A STYLE ONE. Scaling ANY single tensor scales the whole product, so the STATE is right either
way -- `fused_jump_convention.jl` GATE 10c confirms it by the trace, which is linear and contracts
every site: `Tr(0.001*a)/Tr(a) = 0.00100000` exactly. What site 1 breaks is the CANONICAL FORM.

`norm(psi::SymMPS) = norm(psi[psi.center])` (`symmetric_mps.jl`) is the Frobenius norm of the
centre tensor ALONE, which equals the state norm only while every other tensor is an isometry.
Scaling site 1 while the centre sits elsewhere leaves site 1 non-isometric and the centre
untouched, so `norm` silently returns the OLD value: GATE 10c measured `norm(0.001*a)/norm(a) = 1.0`
against a true ratio of 0.001. Putting the scalar on the centre keeps the invariant, so `norm`
stays honest and nothing downstream has to know.

⚠ THIS IS WHAT `fused_freeze_probe.jl` GATE 8 ACTUALLY FOUND. It reported "add_mps normalises its
arguments" from `|b|/|a| = 1` -- but that ratio came from `norm`, before `add_mps` was ever called.
`add_mps` is fine (GATE 10c: `Tr(a+b) = 1.001000000000`, exact); the misreading was the norm.

⚠ THE FALLBACK TO SITE 1 IS DELIBERATE. `SymMPS.center` is a plain field with no invariant enforced
by the type, so a path that leaves it out of range would turn a scalar multiply into a `BoundsError`
in every caller. Falling back reproduces the OLD behaviour exactly, which makes this change unable
to be worse than what it replaces.
"""
apply_shift!(p, z::Number) = begin
    c = (1 <= p.center <= length(p)) ? p.center : 1
    p[c] = to_concrete(ComplexF64(z) * p[c])
    p
end

"""
    normalise_trace!(Imps, rho, L) -> tr_before

Rescale `|rho>>` so `Tr rho = 1`, returning the trace it had BEFORE rescaling.

⛔ WHY THIS IS NEEDED AT ALL, AND WHAT IT IS NOT. `Tr rho` is the conservation law of a Lindbladian
-- `<<I|L = 0` exactly, since every term is a commutator or a jump-plus-anticommutator and `<<I|`
annihilates each. `tdvp2` inherits that exactly: its sweep is a product of local exponentials and
each local generator separately annihilates `<<I|`. A GALERKIN scheme does not, because it evolves
`exp(tau * P L P)` and

    Tr(P rho) = <<I|P|rho>> = <<P I|rho>>

so the trace survives the projection IF AND ONLY IF `P|I>> = |I>>` -- the variational basis must
CONTAIN the identity superket, and nothing in CBE/BUG puts it there. Truncation compounds it.

⚠ RESCALING IS LEGITIMATE BUT IT IS NOT A CURE. Every observable is `Tr(O rho)/Tr(rho)`, so the
physical state is `rho/Tr(rho)` and dividing is exactly what one does. It also stops the defect
COMPOUNDING: an un-normalised state whose trace decays each step shrinks multiplicatively, and the
error rides that decay. What it cannot do is repair a state whose DIRECTION is wrong -- so the
pre-normalisation trace is RETURNED, and the drivers report it. Quoting `Tr = 1` after this as
evidence of accuracy would be circular.

⚠ THE STRUCTURAL FIX, if this ever needs to be exact rather than corrected: put `|I>>` in the
basis. In the FUSED layout it is a PRODUCT state (chi = 1, measured bond dims `[1]`), so
augmenting every frame with its local vector costs ONE extra direction per bond -- and because it
is a product, each local frame containing the local vector is enough for the tensor product to
contain `|I>>`. The closing truncation would have to preserve that direction too.
"""
function normalise_trace!(Imps, rho, L::Int)
    tr = real(trace_rho(Imps, rho, L))
    abs(tr) > 1e-12 && apply_shift!(rho, 1.0 / tr)
    return tr
end

"""
    add_mps(a, b) -> SymMPS

`|a> + |b>` by DIRECT SUM on the bonds: block-diagonal in the interior, a row at the left boundary
and a column at the right. Bond dimensions add; the physical legs are shared.

⚠ `b`'s BOND TAGS ARE RETAGGED TO `a`'s BEFORE EACH `oplus`. The two states come from different
constructors (`rho` carries the sweep's `cbeW`/`cbeZ` tags, `|I>>` carries `L,i`), and Telum
refuses a direct sum across mismatched itags. Retagging the bonds is safe -- they are internal
labels, summed over -- while retagging a PHYSICAL leg would silently pair the wrong sites, which is
why only legs 1 and 3 are touched.

⛔ AND RETAGGING IS NOT ENOUGH: `Telum._qindex_match_for_oplus` requires `itags`, **`dir`**, `plev`
AND `lock` to agree, and `setitag` fixes only the first. The BOND ARROW is what differs, and it
differs BY SWEEP -- measured on the fused d = 4 chain, `psi[1]` leg 3 after one step:

    tdvp2        dir = '+'      add_mps THREW
    tdvp_cbe1s   dir = '+'      add_mps THREW
    cbe_bug      dir = '-'      add_mps OK        <- matches the identity superket

So this function only ever worked on `cbe_bug` output, which is exactly why every previous
open-system trajectory passed: `rsvd_parallel.jl` calls `post!` only in its warmup, on a `cbe_bug`
step. The first `tdvp2` trajectory on a fused generator found it immediately.

⛔ THE FIX IS TO GAUGE BOTH INPUTS THE SAME WAY, NOT TO FLIP ARROWS. `dag` would flip every leg
including the PHYSICAL one and conjugate the data; there is no `setdir` in Telum's API. But the
arrow on a bond is set by which side of the centre it lies on, so canonicalising both arguments to
the same site makes the conventions agree by construction -- and `restore_trace!` re-canonicalises
after the sum anyway, so this costs one extra gauge pass and changes no result.

⚠ THE ERROR MESSAGE IS WIDENED DELIBERATELY. Telum's own message says "no leg matching reference
leg 3" without saying WHICH attribute, and that cost an hour to localise. On failure this now prints
the four attributes side by side.
"""
function add_mps(a, b)
    n = length(a)
    n == length(b) || throw(DimensionMismatch("add_mps: $(n) vs $(length(b)) sites"))
    # Gauge both to site 1 so the bond arrows agree; see the note above.
    canonical!(a, 1)
    canonical!(b, 1)
    out = Any[]
    for i in 1:n
        A = a[i]
        B = to_concrete(setitag(setitag(b[i], 1, A.inds[1].itags), 3, A.inds[3].itags))
        legs = i == 1 ? (3,) : i == n ? (1,) : (1, 3)
        try
            push!(out, to_concrete(oplus([A, B], legs)))
        catch e
            e isa ArgumentError || rethrow()
            io = IOBuffer()
            println(io, "add_mps: oplus failed at site $i (legs $legs). Telum needs itags, dir, ",
                        "plev and lock to all agree. Side by side:")
            for k in 1:length(A.inds)
                println(io, "   leg $k  a: itags=", repr(A.inds[k].itags),
                            " dir=", repr(A.inds[k].dir), " plev=", A.inds[k].plev,
                            " lock=", repr(A.inds[k].lock))
                println(io, "          b: itags=", repr(B.inds[k].itags),
                            " dir=", repr(B.inds[k].dir), " plev=", B.inds[k].plev,
                            " lock=", repr(B.inds[k].lock))
            end
            print(io, "original: ", sprint(showerror, e))
            throw(ArgumentError(String(take!(io))))
        end
    end
    return SymMPS(out, n)
end

"""
    restore_trace!(Imps, rho, L; maxdim, cutoff) -> tr_before

Restore `Tr rho = 1` by adding the DEFICIT BACK ALONG THE IDENTITY DIRECTION:

    rho <- rho + (eps / 2^(L/2)) * Imps,    eps = 1 - Tr(rho)

⛔ WHY THIS AND NOT `rho / Tr(rho)`. Rescaling multiplies the WHOLE state, traceless part included,
and the BUG error is not a pure trace deficit -- so it OVERCORRECTS. MEASURED on the fused L=3
chain at full rank, rescaling pins `Tr = 1` and makes the state error CONSISTENTLY 1.91x WORSE
(2.42e-03 -> 4.66e-03, 1.53e-04 -> 2.92e-04, 9.56e-06 -> 1.83e-05). This correction instead leaves
the traceless part EXACTLY alone and moves only along `|I>>`, which is the minimal change that
fixes the conservation law.

⚠ `Imps` is the NORMALISED identity superket, so `Tr(Imps) = 2^(L/2)` -- hence the factor. Getting
that wrong overshoots by `2^(L/2)`, which at L = 18 is a factor of 512 and would be obvious; at
L = 2 it is 2 and would not.

⚠ THE ADDITION COSTS ONE BOND DIMENSION (`|I>>` is a PRODUCT state, chi = 1), so the state is
re-truncated afterwards. That truncation can itself shed trace -- which is exactly the shared,
cap-driven mechanism `tdvp2` suffers too -- so the CALLER still logs the trace it measured.
"""
function restore_trace!(Imps, rho, L::Int; maxdim::Int = 0, cutoff::Float64 = 1e-12,
                        max_deficit::Float64 = 0.5)
    tr = real(trace_rho(Imps, rho, L))
    eps = 1.0 - tr
    abs(eps) < 1e-15 && return tr
    # ⛔ A LARGE DEFICIT IS NOT A TRUNCATION ERROR -- IT IS A WRONG STATE, AND CORRECTING IT
    # SILENTLY DESTROYS THE RUN. A Lindbladian conserves the trace EXACTLY, so what this function
    # repairs is the O(truncation) leak a Galerkin projection plus an SVD leaves behind: small by
    # construction. When `eps` is O(1) the "minimal correction along |I>>" is no longer minimal --
    # it OVERWHELMS the physical state and projects it back onto the identity superket.
    #
    # MEASURED, and it cost a full diagnostic round: starting from `rho = identity_superket(L)`
    # instead of `identity_superket(L)/2^(L/2)` puts `Tr rho = 2^(L/2) = 16` at L = 8, so
    # `eps = -15` and `rho <- rho + (-15/16) rho = rho/16` -- a state that is STILL exactly `|I>>`
    # and now has `Tr = 1`. The run then reports `<S^z_j>` EXACTLY zero at every site and time,
    # `chi = 1`, `Tr = 1.0000000000`, IDENTICALLY for tdvp2 / bug_rsvd / bug_par, and the same
    # error at every `dt` (observed order 0.00). Every one of those symptoms reads as a dead
    # INTEGRATOR, which is where the diagnosis went first and why this guard now exists.
    abs(eps) <= max_deficit || error(
        "restore_trace!: Tr rho = $tr, deficit $eps exceeds max_deficit=$max_deficit. This is " *
        "NOT a truncation leak -- correcting it would project the state onto |I>>. Check the " *
        "INITIAL state: the maximally mixed state is `identity_superket(L)/2^(L/2)`, not " *
        "`identity_superket(L)` (whose trace is 2^(L/2)); a pure `rho0_product` already has " *
        "Tr = 1. Also check that the zero-body `shift` from the Lindbladian builder is being " *
        "applied each step.")
    corr = copy(Imps)
    apply_shift!(corr, eps / 2.0^(L / 2))
    out = add_mps(rho, corr)
    for i in eachindex(rho)
        rho[i] = out[i]
    end
    # ⚠ THE DIRECT SUM DESTROYS THE CANONICAL FORM. `SymMPS.center` is an `Int`, so it cannot be
    # marked invalid -- it is set to a site and then a full `canonical!` re-gauges from scratch.
    # Leaving a stale centre here is the failure this repo has already paid for once: downstream
    # code that assumes a true canonical snapshot reads isometries that are no longer isometric.
    rho.center = 1
    canonical!(rho, 1)
    maxdim > 0 && truncate_recursive!(rho, 1; maxdim = maxdim, cutoff = cutoff)
    return tr
end

"""
OPERATOR ENTANGLEMENT at the central cut: the von Neumann entropy of `|rho>>` treated as a state.

⚠ `|rho>>` IS NOT NORMALISED (`Tr rho = 1` is the conservation law, not the 2-norm), so the
Schmidt values are renormalised here before the entropy is taken. Returns `(S, n_states)`.

⛔ `base` IS PART OF THE OBSERVABLE, NOT A DISPLAY CHOICE. The default `e` (nats) is what every
caller before 2026-08-27 measured and is kept so their numbers do not move. The dissipative
operator-entanglement literature quotes **`log_2`** — arXiv:2410.18468 Eq. (3),
`S_op = -sum_a lambda_a^2 log_2 lambda_a^2`, and its late-time law
`S_op(t -> inf) = eta log_2(tJ) + S_0` — so a run being compared against that paper must pass
`base = 2`. Quoting nats against a `log_2` figure understates the entropy by a factor
`ln 2 = 0.693` and, worse, rescales the FITTED PREFACTOR `eta` by the same factor while leaving
the logarithmic SHAPE intact, so the plot still looks right.
"""
function operator_entanglement(psi; base::Real = ℯ)
    p = copy(psi); c = max(1, length(p) ÷ 2)
    canonical!(p, c)
    res = svd(p[c], (1, 2); cutoff = 0.0, get_lists = true)
    s = Float64[]
    for (v, deg, _, _) in res.kept_list, _ in 1:deg
        push!(s, Float64(v))
    end
    isempty(s) && return 0.0, 0
    s ./= sqrt(sum(abs2, s))
    p2 = abs2.(s)
    S = -sum(x -> x > 0 ? x * log(x) : 0.0, p2)
    return S / log(base), length(s)
end
