#!/usr/bin/env python3
"""
Generate the three RSVD-CBE tutorial notebooks.

    python3 build_notebooks.py        -> 01_rsvd_lubich.ipynb
                                         02_rsvd_gate.ipynb
                                         03_rsvd_ground_state.ipynb
                                         *.jl  (the same code, sbatch-able)

Kept as a generator rather than hand-edited JSON so the three stay in step: they
share the setup, the projector section and the parameter study, and those are
written once here.

Everything runs at L = 6 with U(1) symmetry, small enough that a 2^6 = 64 dense
reference is available for every claim.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = {"display_name": "Julia 1.11", "language": "julia", "name": "julia-1.11"}


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.rstrip("\n").split("\n")}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.rstrip("\n").split("\n")}


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": KERNEL,
                       "language_info": {"file_extension": ".jl", "mimetype":
                                         "application/julia", "name": "julia",
                                         "version": "1.11.9"}},
          "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(HERE, name + ".ipynb")
    with open(p, "w") as f:
        json.dump(nb, f, indent=1)
    # the same code as a plain script, so it can be sbatch-ed on a compute node
    jl = os.path.join(HERE, name + ".jl")
    with open(jl, "w") as f:
        f.write(f"# Generated from {name}.ipynb by build_notebooks.py\n")
        f.write("# Run on a COMPUTE NODE:  sbatch notebooks/submit_notebook.slurm "
                + name + "\n\n")
        for c in cells:
            if c["cell_type"] == "code":
                f.write("\n".join(c["source"]) + "\n\n")
            else:
                f.write("\n".join("# " + ln for ln in c["source"]) + "\n\n")
    return p, jl


# =========================================================== shared sections ==

SETUP = [
    md("""# Setup

`BUGJulia` is loaded from the repo root. `set_symmetry!(:U1)` makes every tensor
block-sparse in total $S^z$ — which is what makes the sector bookkeeping below
visible rather than implicit.

The dense reference in `tests/common/` is test-side only: nothing in `src/` may
densify. At $L=6$ the full space is $2^6 = 64$, so every claim on this page can be
checked against an exact $64\\times64$ matrix."""),
    code("""using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Random, Printf
using LurCGT, Telum
const R = RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))

set_symmetry!(:U1)
const L = 6
const DELTA = 1.0
println("symmetry = ", symmetry_mode(), "   L = ", L)"""),
]

MPS_SECTION = [
    md("""## 1. The MPS

`domain_wall_state(L)` is $|\\uparrow\\uparrow\\uparrow\\downarrow\\downarrow\\downarrow\\rangle$:
a product state, so every bond has dimension 1. That is deliberate — it is the
hardest possible start for a rank-preserving method, and the reason CBE has to
exist.

Each tensor is rank 3 with legs `(link_l, sigma, link_r)`. Under U(1) it is stored
as a list of charge blocks, not a dense array."""),
    code("""psi = domain_wall_state(L)
@printf("bond dimensions: %s\\n", string(bond_dims(psi)))
for i in 1:L
    T = psi[i]
    @printf("psi[%d]  legs %-18s  blocks: %d\\n", i,
            string(Tuple(leg_dim(T, k) for k in 1:3)), length(T.qlabels))
end"""),
    code("""# The charge sectors on one bond. A U(1) MPS carries a charge on every link;
# these labels are what the sketch has to respect.
canonical!(psi, 3)
T = psi[3]
println("psi[3] has ", length(T.qlabels), " charge blocks; sector dims per leg:")
for leg in 1:3
    dims = [(q, sector_dim(T, leg, q)) for (q, _) in T.spaces[leg]]
    @printf("  leg %d: %s\\n", leg, string(dims))
end"""),
    code("""# Densify one tensor, purely to look at it. `dense_state` is the test-side helper.
v = dense_state(psi)
@printf("state vector: %d amplitudes, norm %.6f, nonzero %d\\n",
        length(v), norm(v), count(!iszero, v))
@printf("total Sz = %.1f (U(1) pins this for the whole run)\\n", total_sz(psi))"""),
]

MPO_SECTION = [
    md("""## 2. The Hamiltonian

The XXZ chain is stored as a **term list**, not as an explicit rank-4 $W$ tensor:

$$H = \\sum_i \\sum_t c_t\\, O^L_t(i)\\, O^R_t(i+1)$$

Under U(1) the $S^+/S^-$ halves are rank 3 — the third leg carries the $\\pm 2$
charge that makes the pair symmetry-allowed. Contracting that leg *is* charge
conservation.

The factored form exists because the CBE sketch has to **split a bond term across
the two halves of the bond**. A fused gate has already joined them."""),
    code("""h = xxz_chain(L; delta = DELTA)
@printf("%d terms, J = %.2f, delta = %.2f\\n", length(h.terms), h.J, h.delta)
for (t, term) in enumerate(h.terms)
    @printf("  term %d: coeff %+.3f   left rank %d   right rank %d\\n",
            t, term.coeff, length(term.left.inds), length(term.right.inds))
end"""),
    code("""# The two-site gate is the SAME terms, fused. One-way: this is why the term list
# is what gets stored.
gates = bond_gates(psi)
g = gates[3]
@printf("gate on bond 3: rank %d, legs %s\\n",
        length(g.inds), string(Tuple(leg_dim(g, k) for k in 1:length(g.inds))))
@printf("energy via gates        = %.12f\\n", real(energy(psi, gates)))
@printf("energy via environments = %.12f\\n", real(env_energy(psi, h)))
@printf("  (these agreeing to machine precision is what pins the term list's\\n")
@printf("   normalisation against the validated gate path)\\n")"""),
]

ENV_SECTION = [
    md("""## 3. The environments

`henv.jl` stores the MPO's virtual index as **named channels** rather than a fused
leg. Reading the virtual index as an automaton scanning the chain:

| state | meaning |
|---|---|
| `id` | nothing has happened yet; the identity has been transported |
| `open[t]` | term `t` placed its LEFT operator, waiting for its partner |
| `done` | one complete term lies entirely behind us |

`ZERO` is not `nothing`: `done` starts at `ZERO` (no completed terms), `id` starts
at `nothing` (the chain boundary). Conflating them is a silent factor-of-one bug."""),
    code("""psi2 = domain_wall_state(L)
# warm it up so the bonds are not all 1 -- a product state makes every environment
# trivial and hides the structure.
cbe_bond_update_bug!(psi2, bond_gates(psi2);
                     opts = CBEBugOptions(dt = 0.05, n_steps = 3, maxdim = 16))
@printf("warmed bond dimensions: %s\\n", string(bond_dims(psi2)))

canonical!(psi2, 3)
lenv = left_env_stack(psi2, h; upto = 2)
renv = right_env_stack(psi2, h; downto = 5)
lch = left_channels(lenv, 3)
rch = right_channels(renv, 5)

println("left channels on link 3:")
@printf("  id   : %s\\n", lch.id === nothing ? "nothing (boundary)" :
        string(Tuple(leg_dim(lch.id, k) for k in 1:length(lch.id.inds))))
@printf("  done : %s\\n", lch.done === R.ZERO ? "ZERO (no completed terms)" :
        string(Tuple(leg_dim(lch.done, k) for k in 1:length(lch.done.inds))))
for (t, o) in enumerate(lch.open)
    @printf("  open[%d]: %s\\n", t, o === R.ZERO ? "ZERO" :
            string(Tuple(leg_dim(o, k) for k in 1:length(o.inds))))
end"""),
    code("""# What a contraction RESULTS IN: the two-site effective operator, applied to Theta.
f = bond_frame(psi2, 3)
theta = frame_theta(f)
HT = apply_h_two_site(theta, h, 3, lenv, renv)
@printf("Theta : rank %d  legs %s\\n", length(theta.inds),
        string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("H*Theta: rank %d  legs %s   (same legs -- H maps the block to itself)\\n",
        length(HT.inds), string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("<Theta|H|Theta> = %.12f   vs env_energy = %.12f\\n",
        real(tensor_inner(theta, HT)), real(env_energy(psi2, h)))"""),
]

PROJ_SECTION = [
    md("""## 4. The projectors

The bond frame factorises the two-site block, $\\Theta = U_0 S_0 V_0$. $U_0$ and
$V_0$ are **isometries**: orthonormal bases for the $r$-dimensional subspaces the
state currently uses.

The fused local space $(\\ell_{i-1}\\otimes\\sigma_i)$ has dimension $d\\cdot\\chi$
and splits in two:

$$P_K = U_0 U_0^\\dagger \\quad\\text{(kept)}, \\qquad
  P_D = I - U_0 U_0^\\dagger \\quad\\text{(discarded)}$$

with $P_K + P_D = I$, $P_K^2 = P_K$, $P_K P_D = 0$. **Everything CBE does happens
inside $P_D$.**"""),
    code("""f = bond_frame(psi2, 3)
@printf("U0 legs %s   V0 legs %s   S0 legs %s\\n",
        string(Tuple(leg_dim(f.U0, k) for k in 1:3)),
        string(Tuple(leg_dim(f.V0, k) for k in 1:3)),
        string(Tuple(leg_dim(f.S0, k) for k in 1:2)))
@printf("rank r = %d\\n", f.old_rank)
@printf("left  isometry defect ||U0'U0 - I|| = %.2e\\n", left_isometry_defect(f.U0))
@printf("right isometry defect ||V0 V0' - I|| = %.2e\\n", right_isometry_defect(f.V0))"""),
    code("""# The kept and discarded projectors, applied to the two-site block.
# perp_component(U0, X) = X - U0 (U0' X)  =  P_D X.
HT   = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
PD_HT = to_concrete(perp_component(f.U0, HT))          # discarded part
PK_HT = to_concrete(HT - PD_HT)                        # kept part

@printf("||H*Theta||      = %.6e\\n", norm(HT))
@printf("||P_K H*Theta||  = %.6e   (inside the current frame)\\n", norm(PK_HT))
@printf("||P_D H*Theta||  = %.6e   (OUTSIDE it -- what CBE is hunting)\\n", norm(PD_HT))
@printf("completeness ||H*Theta - (P_K + P_D)H*Theta|| = %.2e\\n",
        norm(to_concrete(HT - to_concrete(PK_HT + PD_HT))))
@printf("idempotence  ||P_D(P_D X) - P_D X||           = %.2e\\n",
        norm(to_concrete(to_concrete(perp_component(f.U0, PD_HT)) - PD_HT)))
@printf("orthogonality <P_K X | P_D X>                 = %.2e\\n",
        abs(tensor_inner(PK_HT, PD_HT)))"""),
    md("""### The residual is the whole story

$\\|P_D (H\\Theta)\\| / \\|H\\Theta\\|$ is the fraction of the Hamiltonian's action at
this bond that the **current** frame cannot represent. If it is zero the frame is
already complete and no expansion can help. If it is large, a rank-preserving
method (1-site TDVP without CBE) is throwing that fraction away every step."""),
    code("""for i in 1:(L - 1)
    canonical!(psi2, i)
    fi = bond_frame(psi2, i)
    le = left_env_stack(psi2, h; upto = i - 1)
    re = right_env_stack(psi2, h; downto = i + 2)
    Hi = apply_h_two_site(frame_theta(fi), h, i, le, re)
    resid = norm(to_concrete(perp_component(fi.U0, Hi))) / norm(Hi)
    @printf("bond %d: r = %2d   ||P_D H*Theta|| / ||H*Theta|| = %.3e\\n",
            i, fi.old_rank, resid)
end"""),
]

SKETCH_SECTION = [
    md("""## 5. The sketch — and it probes the PROJECTOR

`sketch_h_left(f, h, i, lch, rch, Om)` returns

$$Y = \\big(P_\\perp\\, H\\Theta\\big)\\,\\Omega^\\dagger, \\qquad
  P_\\perp = I - U_0U_0^\\dagger$$

**The projector is applied first.** This is the only sketching path; the
fold-$\\Omega$-first variant was removed on 2026-08-06.

The two orderings agree *exactly* — $P_\\perp$ acts on $(\\ell_{i-1},\\sigma_i)$ and
$\\Omega$ on $(\\sigma_{i+1},\\ell_{i+1})$, disjoint legs, so they commute. What
project-first changes is cost: $H\\Theta$ is now formed once per bond.

Its signature is checkable in one line: **$Y$ is orthogonal to the frame for any
probe.**"""),
    code("""rng = MersenneTwister(0xC0FFEE)
canonical!(psi2, 3)
f = bond_frame(psi2, 3)
lenv = left_env_stack(psi2, h; upto = 2); renv = right_env_stack(psi2, h; downto = 5)
lch = left_channels(lenv, 3); rch = right_channels(renv, 5)

Om = sector_graded_sketch(f.V0, :right, 6; rng = rng)
@printf("probe Omega: legs %s\\n", string(Tuple(leg_dim(Om, k) for k in 1:3)))
Y = sketch_h_left(f, h, 3, lch, rch, Om)
@printf("sketch Y   : legs %s\\n", string(Tuple(leg_dim(Y, k) for k in 1:3)))

K0 = to_concrete(f.U0 * f.S0)
ovl = to_concrete(contract(K0', (1, 2), Y, (1, 2)))     # (bond, g)
@printf("\\nPROJECT-FIRST SIGNATURE  ||<U0 S0 | Y>|| = %.3e   (must vanish)\\n", norm(ovl))

# and it equals the explicitly projected action, folded afterwards
HT = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
want = to_concrete(contract(to_concrete(perp_component(f.U0, HT)), (3, 4), Om', (2, 3)))
@printf("||Y - (P_perp H*Theta) Om'||        = %.3e\\n", norm(to_concrete(Y - want)))

# the orderings commute -- the identity that makes this change free of consequence
want2 = to_concrete(perp_component(f.U0, to_concrete(contract(HT, (3, 4), Om', (2, 3)))))
@printf("||Y - P_perp((H*Theta) Om')||       = %.3e   <- the commuting identity\\n",
        norm(to_concrete(Y - want2)))"""),
    md("""### The probe is sector-graded, not Gaussian

Under a symmetry a plain Gaussian is wrong: columns must be allocated **per charge
sector**, or most sectors get drawn with zero weight. `comp_ratio` splits the
budget between the orthogonal complement and the isometry space (the reference's
`CompRatio`, default 0.5).

The reference is explicit about keeping both halves
(`RSVDpreBE0SiQS.m:337`): *"do not project into complement space / 1-site
components are also important for bond update"*."""),
    code("""for cr in (0.0, 0.5, 1.0, nothing)
    Om = sector_graded_sketch(f.V0, :right, 6; comp_ratio = cr,
                              rng = MersenneTwister(7))
    if Om === nothing
        @printf("comp_ratio = %-6s -> no probe\\n", string(cr))
        continue
    end
    Y = sketch_h_left(f, h, 3, lch, rch, Om)
    @printf("comp_ratio = %-6s  cols = %d  ||Y|| = %.4e  ||<U0S0|Y>|| = %.1e\\n",
            string(cr), leg_dim(Om, 1), norm(Y),
            norm(to_concrete(contract(K0', (1, 2), Y, (1, 2)))))
end"""),
]

EXPAND_SECTION = [
    md("""## 6. The expansion

`cbe_expand` ranks the candidates and returns

$$U_{ex} = [\\,U_0 \\mid C_L\\,], \\qquad V_{ex} = \\begin{bmatrix}V_0\\\\ C_R\\end{bmatrix}$$

as **direct sums**, so $U_0$ survives as the first block. The expansion is
therefore lossless — and by itself **cannot change a Schmidt rank**. It is a change
of basis; a `canonical!` would collapse the widened bond straight back. The rank
grows only because the core is then *evolved* in the widened space."""),
    code("""ex = cbe_expand(f, h, 3, lch, rch; growth = 2.0)
@printf("r = %d  ->  U_ex bond %d,  V_ex bond %d\\n",
        f.old_rank, leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1))
@printf("new directions: left %d, right %d\\n", ex.n_new_l, ex.n_new_r)
@printf("err_pre = %.3e   (weight the PRESELECTION discarded)\\n", ex.err_pre)
@printf("err_fnl = %.3e   (weight the FINAL ranking discarded)\\n", ex.err_fnl)
@printf("U_ex still an isometry: defect %.2e\\n", left_isometry_defect(ex.U_ex))"""),
    code("""# Does the expanded frame actually capture more of H*Theta? This is the number
# that says whether the expansion was worth anything.
HT = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
# `ex.U_ex` carries cbe_expand's own bond tag, which trips contract's itag guard.
# Renaming leg 3 onto the frame's tag is pure bookkeeping -- no numbers move.
retag(U, ref) = to_concrete(setitag(U, 3, ref.inds[3].itags))
before = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
after  = norm(to_concrete(perp_component(retag(ex.U_ex, f.U0), HT))) / norm(HT)
@printf("residual outside the frame: %.3e  ->  %.3e   (%.1fx smaller)\\n",
        before, after, before / max(after, eps()))"""),
]


def param_study(kind):
    """The parameter study, specialised per notebook."""
    if kind == "lubich":
        run = """function run_lubich(; growth, dex, comp_ratio, exact, maxdim = 32,
                     nsteps = 20, dt = 0.05)
    p = domain_wall_state(L)
    info = nothing
    for _ in 1:nsteps
        info = cbe_lubich_sweep(p, h, ComplexF64(-im * dt);
                                growth = growth, dex = dex, comp_ratio = comp_ratio,
                                exact = exact, maxdim = maxdim)
    end
    return p, info
end"""
        head = "cbe_lubich_sweep"
    elif kind == "gate":
        run = """function run_gate(; growth, dex, comp_ratio, exact, s_iters = 1,
                   maxdim = 32, nsteps = 20, dt = 0.05)
    p = domain_wall_state(L)
    gs = bond_gates(p)
    info = cbe_bond_update_bug!(p, gs;
              opts = CBEBugOptions(dt = dt, n_steps = nsteps, maxdim = maxdim,
                                   growth = growth, dex = dex,
                                   comp_ratio = comp_ratio, exact = exact,
                                   s_iters = s_iters))
    return p, info
end"""
        head = "cbe_gate_evolve!"
    elif kind == "imag":
        run = """function run_imag(; growth, dex, comp_ratio, exact,
                   dt = 0.05, nsteps = 60, maxdim = 32)
    p = neel_state(L)
    Es = Float64[]
    for _ in 1:nsteps
        tdvp1_cbe_step!(p, h, ComplexF64(-dt); exact = exact, growth = growth,
                        dex = dex, comp_ratio = comp_ratio, maxdim = maxdim)
        n = norm(p); n > 0 && (p[p.center] = to_concrete((1.0 / n) * p[p.center]))
        push!(Es, real(env_energy(p, h)))
    end
    return p, Es
end"""
        head = "tdvp1_cbe_step!  (imaginary time)"
    else:
        run = """function run_gs(; growth, dex, comp_ratio, exact, grow_iters = 1,
                 maxdim = 32, n_sweeps = 10)
    p = neel_state(L)
    info = R.GroundState.solve!(p, h;
              opts = CBEBugOptions(maxdim = maxdim, growth = growth, dex = dex,
                                   comp_ratio = comp_ratio, exact = exact),
              n_sweeps = n_sweeps, grow_iters = grow_iters)
    return p, info
end"""
        head = "GroundState.solve!"

    return [
        md(f"""## 7. Playing with the RSVD parameters

The knobs that decide what the expansion looks at:

| knob | what it controls |
|---|---|
| `growth` | budget $=\\lceil \\text{{growth}}\\cdot d_{{max}}\\rceil - r$; how many directions to propose |
| `dex` | a fixed budget, overriding `growth` when > 0 |
| `comp_ratio` | complement / isometry split of the probe (`nothing` = no projector in the probe) |
| `exact` | swap the thin random probe for the complete fused basis — plain CBE, the reference the sketch approximates |
| `maxdim` | the truncation ceiling, applied *after* the update |

Each row below runs `{head}` from the same start state and reports the error
against the dense reference."""),
        code(run),
    ]


REFERENCE = [
    md("""### The reference

At $L=6$ the full Hilbert space is 64-dimensional, so the exact propagator is a
$64\\times64$ matrix exponential. That is a genuinely independent check — it shares
no code with the integrator."""),
    code("""function dense_reference(dt, nsteps; delta = DELTA)
    Hd = sum(dense_bond_term(L, i; delta = delta) for i in 1:(L - 1))
    v0 = dense_state(domain_wall_state(L))
    return exp(-im * dt * nsteps * Matrix(Hd)) * v0
end

err_vs_dense(p, vref) = begin
    v = dense_state(p)
    ph = dot(vref, v); ph = abs(ph) > 1e-14 ? ph / abs(ph) : 1.0 + 0im
    norm(v / ph - vref) / norm(vref)   # global phase is not physical
end"""),
]



# ─────────────────────────── one sweep, step by step ─────────────────────────

ONE_SWEEP = [
    md("""## 5. ONE SWEEP, STEP BY STEP

Everything above was static. This section walks a **single bond of a single
sweep** and prints every object as it is built: what is constructed, what each
contraction produces, and what the local solve actually solves.

The bond update is the same five moves in every integrator on this page:

| # | move | produces |
|---|---|---|
| 1 | `canonical!(psi, i)` | the gauge the frame needs |
| 2 | `bond_frame(psi, i)` | `Theta = U0 S0 V0` |
| 3 | `sketch_h_left/right` | `Y = (P_perp H Theta) Om'` |
| 4 | `cbe_expand` | `U_ex = [U0 | C_L]`, `V_ex` |
| 5 | local solve + `svd` | the new `psi[i]`, `psi[i+1]` |"""),
    code("""p = domain_wall_state(L)
for _ in 1:3      # warm up so the bond is not trivially rank 1
    tdvp1_rsvd_cbe_step!(p, h, ComplexF64(-0.05); maxdim = 32)
    p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center])
end
i = 3
@printf("STATE  bond_dims %s   centre %d\\n", string(bond_dims(p)), p.center)

# ---- move 1: the gauge -----------------------------------------------------
canonical!(p, i)
@printf("\\n1. canonical!(psi, %d)  -> centre now %d\\n", i, p.center)
@printf("   sites 1..%d are LEFT isometries, %d..%d RIGHT isometries\\n",
        i - 1, i + 1, L)
@printf("   left  isometry defect on psi[1]: %.2e\\n", left_isometry_defect(p[1]))
@printf("   right isometry defect on psi[%d]: %.2e\\n", L, right_isometry_defect(p[L]))"""),
    code("""# ---- move 2: the bond frame -------------------------------------------------
f = bond_frame(p, i)
theta = frame_theta(f)
@printf("2. bond_frame(psi, %d)\\n", i)
@printf("   Theta  legs %-20s  = the two-site block\\n",
        string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("   U0     legs %-20s  left isometry,  defect %.1e\\n",
        string(Tuple(leg_dim(f.U0, k) for k in 1:3)), left_isometry_defect(f.U0))
@printf("   S0     legs %-20s  the core: ALL the amplitude\\n",
        string(Tuple(leg_dim(f.S0, k) for k in 1:2)))
@printf("   V0     legs %-20s  right isometry, defect %.1e\\n",
        string(Tuple(leg_dim(f.V0, k) for k in 1:3)), right_isometry_defect(f.V0))
@printf("   rank r = %d\\n", f.old_rank)
@printf("   reassembly check ||U0 S0 V0 - Theta|| = %.2e\\n",
        norm(to_concrete(to_concrete(to_concrete(f.U0 * f.S0) * f.V0) - theta)))"""),
    code("""# ---- the local operator, and what its contraction produces -----------------
lenv = left_env_stack(p, h; upto = i - 1)
renv = right_env_stack(p, h; downto = i + 2)
lch = left_channels(lenv, i); rch = right_channels(renv, i + 2)

HT = apply_h_two_site(theta, h, i, lenv, renv)
@printf("\\nLOCAL OPERATOR  apply_h_two_site(Theta, h, %d, lenv, renv)\\n", i)
@printf("   in  : Theta   %s\\n", string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("   out : H*Theta %s   (same legs -- H maps the block to itself)\\n",
        string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("   <Theta|H|Theta> = %.12f   vs env_energy = %.12f\\n",
        real(tensor_inner(theta, HT)), real(env_energy(p, h)))

println("\\n   the operator is assembled from THREE kinds of term, by where they")
println("   sit relative to the bond:")
@printf("     entirely LEFT  : lch.done  %s\\n",
        lch.done === R.ZERO ? "ZERO" : string(Tuple(leg_dim(lch.done, k) for k in 1:2)))
@printf("     entirely RIGHT : rch.done  %s\\n",
        rch.done === R.ZERO ? "ZERO" : string(Tuple(leg_dim(rch.done, k) for k in 1:2)))
for (t, o) in enumerate(lch.open)
    @printf("     STRADDLING t=%d: lch.open[%d] %s  (waiting for its partner)\\n",
            t, t, o === R.ZERO ? "ZERO" :
            string(Tuple(leg_dim(o, k) for k in 1:length(o.inds))))
end"""),
    code("""# ---- move 3: the sketch probes the PROJECTOR --------------------------------
PD = to_concrete(perp_component(f.U0, HT))
@printf("\\n3. the projector, then the sketch\\n")
@printf("   ||H*Theta||      = %.6e\\n", norm(HT))
@printf("   ||P_K H*Theta||  = %.6e   inside the frame (1-site TDVP sees only this)\\n",
        norm(to_concrete(HT - PD)))
@printf("   ||P_D H*Theta||  = %.6e   OUTSIDE it -- what CBE is hunting\\n", norm(PD))

Om = sector_graded_sketch(f.V0, :right, 6; rng = MersenneTwister(5))
Y  = sketch_h_left(f, h, i, lch, rch, Om)
@printf("   Omega  legs %-18s  %d random columns, sector-graded\\n",
        string(Tuple(leg_dim(Om, k) for k in 1:3)), leg_dim(Om, 1))
@printf("   Y      legs %-18s  = (P_perp H Theta) Omega'\\n",
        string(Tuple(leg_dim(Y, k) for k in 1:3)))
ovl = to_concrete(contract(to_concrete(f.U0 * f.S0)', (1, 2), Y, (1, 2)))
@printf("   ||<U0 S0 | Y>|| = %.2e  <- vanishes: the sketch lives in the COMPLEMENT\\n",
        norm(ovl))"""),
    code("""# ---- move 4: the expansion --------------------------------------------------
ex = cbe_expand(f, h, i, lch, rch; growth = 2.0)
retag(U, ref) = to_concrete(setitag(U, 3, ref.inds[3].itags))
@printf("\\n4. cbe_expand -> U_ex = [U0 | C_L],  V_ex = [V0 ; C_R]\\n")
@printf("   U_ex legs %-18s  bond %d -> %d   (+%d new directions)\\n",
        string(Tuple(leg_dim(ex.U_ex, k) for k in 1:3)),
        f.old_rank, leg_dim(ex.U_ex, 3), ex.n_new_l)
@printf("   V_ex legs %-18s  bond %d -> %d   (+%d)\\n",
        string(Tuple(leg_dim(ex.V_ex, k) for k in 1:3)),
        f.old_rank, leg_dim(ex.V_ex, 1), ex.n_new_r)
@printf("   still isometries: defect %.1e / %.1e\\n",
        left_isometry_defect(ex.U_ex), right_isometry_defect(ex.V_ex))
b4 = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
af = norm(to_concrete(perp_component(retag(ex.U_ex, f.U0), HT))) / norm(HT)
@printf("   residual outside the frame: %.3e -> %.3e\\n", b4, af)
@printf("   err_pre %.2e   err_fnl %.2e\\n", ex.err_pre, ex.err_fnl)"""),
    code("""# ---- move 5: the local solve, and its contraction count ---------------------
# The Galerkin problem lives in the EXPANDED basis: seed by projecting Theta in,
# apply the operator, project back. That round trip is one matvec.
S_start = to_concrete(contract(contract(ex.U_ex', (1, 2), theta, (1, 2)),
                               (2, 3), ex.V_ex', (2, 3)))
@printf("\\n5. the local solve\\n")
@printf("   S_start legs %-16s  <- U_ex' Theta V_ex'  (seed, no inverse anywhere)\\n",
        string(Tuple(leg_dim(S_start, k) for k in 1:2)))
@printf("   lossless? ||U_ex S_start V_ex - Theta|| = %.2e\\n",
        norm(to_concrete(to_concrete(to_concrete(ex.U_ex * S_start) * ex.V_ex) - theta)))

nmv = Ref(0)
function apply_s(x)
    nmv[] += 1
    th  = to_concrete((ex.U_ex * x) * ex.V_ex)          # expand out  (bond -> block)
    ev  = apply_h_two_site(th, h, i, lenv, renv)        # apply H
    pr  = contract(ex.U_ex', (1, 2), ev, (1, 2))        # project back
    return to_concrete(contract(pr, (2, 3), ex.V_ex', (2, 3)))
end
S1 = expv(apply_s, ComplexF64(-0.05), S_start; hermitian = true, maxiter = 30, tol = 1e-15)
@printf("   %d matvecs; each is expand-out -> apply H -> project back\\n", nmv[])
@printf("   S1 legs %s   (still just two bond indices: NO physical leg)\\n",
        string(Tuple(leg_dim(S1, k) for k in 1:2)))"""),
    code("""# ---- and the split that writes it back --------------------------------------
res = svd(S1, (1,); cutoff = 1e-14, Nkeep = 32, get_lists = true)
sv = [s for (s, _, _, _) in res.kept_list]
@printf("\\n   svd(S1) keeps %d of %d singular values\\n", length(sv), leg_dim(S1, 1))
@printf("   spectrum: %s\\n", string(round.(sv[1:min(6, end)], sigdigits = 3)))
@printf("   -> psi[%d] = U_ex * U  (a left isometry again)\\n", i)
@printf("   -> psi[%d] = (S Vd) * V_ex  (the new centre)\\n", i + 1)
println("\\n   TRUNCATION COMES LAST. Run before the solve it would delete the")
println("   directions the expansion just bought -- they carry no weight YET.")"""),
]

# ================================================================ notebook 1 ==

def nb_lubich():
    c = [md("""# RSVD-CBE, notebook 1 of 3: the **Lubich BUG sweep**

`cbe_lubich_sweep` (`src/RSVDCBEBondUpdate/cbe_lubich.jl`).

Two basis sweeps that touch nothing, then **exactly one** time evolution at the
centre bond. No backward substep, no inverse anywhere.

This notebook walks the objects the sweep builds — the MPS, the MPO term list,
the environments, the bond frame, the projectors, the sketch, the expansion — and
then runs the sweep and varies the RSVD knobs.

> **The sketch probes the projector.** Since 2026-08-06 `sketch_h_left/right`
> apply $P_\\perp$ *before* folding in $\\Omega$, and the alternative ordering has
> been removed. Section 5 checks both that this is what happens and that it changes
> no numbers.

Companion diagrams: `docs/tndiag/out/rsvdcbe.html`.""")]
    c += SETUP + MPS_SECTION + MPO_SECTION + ENV_SECTION + PROJ_SECTION
    c += SKETCH_SECTION + EXPAND_SECTION
    c += [md("""## 6b. The sweep itself

The structure, in one cell: canonical at 1, K-sweep building augmented LEFT
frames; canonical at L, L-sweep building augmented RIGHT frames; seed
$S_0 = \\langle W,Z|\\psi\\rangle$; one 0-site Galerkin step at the centre; reassemble;
truncate once.

**The basis sweeps must not touch the state.** Writing an expanded frame back pads
the neighbour with zeros, the next frame is re-derived from the padded tensor, and
the padding collapses — pinning the rank. The measured symptom was $L=18$ stuck at
`maxbond 2` with an error *independent of dt*."""),
             code("""p = domain_wall_state(L)
@printf("start          : bond_dims %s\\n", string(bond_dims(p)))
for step in 1:6
    info = cbe_lubich_sweep(p, h, ComplexF64(-im * 0.05); growth = 2.0, maxdim = 32)
    @printf("after step %d   : bond_dims %-22s centre rank %2d  krylov %2d  err_pre %.1e\\n",
            step, string(bond_dims(p)), info.centre_rank, info.krylov_dim, info.err_pre)
end""")]
    c += REFERENCE
    c += param_study("lubich")
    c += [code("""vref = dense_reference(0.05, 20)
@printf("%-34s  %-9s  %-7s  %s\\n", "setting", "err", "maxbond", "err_fnl")
for (lab, kw) in (
        ("growth=2.0, comp_ratio=0.5 (default)", (; growth=2.0, dex=0, comp_ratio=0.5, exact=false)),
        ("growth=1.1  (the DMRG schedule)",      (; growth=1.1, dex=0, comp_ratio=0.5, exact=false)),
        ("growth=3.0",                           (; growth=3.0, dex=0, comp_ratio=0.5, exact=false)),
        ("dex=4   (fixed small budget)",         (; growth=2.0, dex=4, comp_ratio=0.5, exact=false)),
        ("dex=32  (fixed large budget)",         (; growth=2.0, dex=32, comp_ratio=0.5, exact=false)),
        ("comp_ratio=0.0 (isometry only)",       (; growth=2.0, dex=0, comp_ratio=0.0, exact=false)),
        ("comp_ratio=1.0 (complement only)",     (; growth=2.0, dex=0, comp_ratio=1.0, exact=false)),
        ("comp_ratio=nothing (no projector)",    (; growth=2.0, dex=0, comp_ratio=nothing, exact=false)),
        ("exact=true (full basis, no sketch)",   (; growth=2.0, dex=0, comp_ratio=0.5, exact=true)),
    )
    p, info = run_lubich(; kw...)
    @printf("%-34s  %.3e  %-7d  %.2e\\n", lab, err_vs_dense(p, vref),
            maximum(bond_dims(p)), info.err_fnl)
end"""),
          md("""### Reading that table

- **`exact = true` vs the default** is the A/B that says whether the *sketch* costs
  accuracy. If they agree, the randomised probe is doing its job and the thin
  Gaussian is free.
- **`err_fnl` > 0** means the final ranking discarded candidate weight for want of
  budget — the cap, not the selection, is the accuracy limiter. Raise `growth` or
  `dex`.
- **`comp_ratio = 1.0`** is the recorded failure mode: near a chain edge the
  opposite frame's complement is 0- or 1-dimensional, so the probe collapses to
  0–1 columns and the expansion silently declines *with room to spare*. One blocked
  narrow bond then caps every bond inward.
- **`growth = 1.1`** is the DMRG schedule. DMRG re-converges the same state every
  sweep, so a 10% ratchet only costs iterations; a time integrator gets one shot
  per step, and a direction it does not admit is gone rather than deferred.""")]
    return c


# ================================================================ notebook 2 ==

def nb_gate():
    c = [md("""# RSVD-CBE, notebook 2 of 3: the **gate sweep**

`cbe_bond_update` / `cbe_bond_update_sweep!` / `cbe_gate_evolve!`
(`src/RSVDCBEBondUpdate/cbe_core.jl`).

The Trotter path. The operator at a bond is the **bare two-site gate** — the
environments are the identity by canonical form — so no channel machinery appears
at all. Everything else is the same expansion as notebook 1.""")]
    c += SETUP + MPS_SECTION + MPO_SECTION
    c += [md("""## 3. Gates instead of environments

`bond_gates(psi)` fuses each bond's terms into one rank-4 object. That fusion is
one-way, and it is why the *Hamiltonian* is stored as a term list: the sketch has
to split a bond term across the two halves of the bond, and a fused gate has
already joined them.

It is also why the ground-state search (notebook 3) cannot run on this path:
minimising $\\langle h_{i,i+1}\\rangle$ bond by bond does not have the global ground
state as its fixed point."""),
          code("""psi2 = domain_wall_state(L)
gates = bond_gates(psi2)
cbe_bond_update_bug!(psi2, gates;
                     opts = CBEBugOptions(dt = 0.05, n_steps = 3, maxdim = 16))
@printf("warmed bond dimensions: %s\\n", string(bond_dims(psi2)))

canonical!(psi2, 3)
f = bond_frame(psi2, 3)
g = gates[3]
@printf("gate legs %s\\n", string(Tuple(leg_dim(g, k) for k in 1:4)))
theta = frame_theta(f)
HT = apply_gate(g, theta, f.site_l.itags, f.site_r.itags)
@printf("Theta  legs %s\\n", string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("g*Theta legs %s\\n", string(Tuple(leg_dim(HT, k) for k in 1:4)))""")]
    c += [md("""## 4. Projectors and the project-first sketch, on the gate path

Identical construction to notebook 1, with `sketch_bond_left/right` in place of
`sketch_h_left/right`. Same signature check: the sketch output is orthogonal to
the frame."""),
          code("""PD_HT = to_concrete(perp_component(f.U0, HT))
@printf("||g*Theta||     = %.6e\\n", norm(HT))
@printf("||P_D g*Theta|| = %.6e   (what the expansion can reach)\\n", norm(PD_HT))

Om = sector_graded_sketch(f.V0, :right, 6; rng = MersenneTwister(11))
Y = sketch_bond_left(f, g, Om)
K0 = to_concrete(f.U0 * f.S0)
ovl = to_concrete(contract(K0', (1, 2), Y, (1, 2)))
@printf("\\n||<U0 S0 | Y>|| = %.3e   (project-first signature)\\n", norm(ovl))
want = to_concrete(contract(PD_HT, (3, 4), Om', (2, 3)))
@printf("||Y - (P_D g*Theta) Om'|| = %.3e\\n", norm(to_concrete(Y - want)))"""),
          code("""ex = cbe_expand_bond(f, g; growth = 2.0)
@printf("r = %d -> U_ex %d, V_ex %d   new: L %d, R %d\\n",
        f.old_rank, leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1),
        ex.n_new_l, ex.n_new_r)
retag(U, ref) = to_concrete(setitag(U, 3, ref.inds[3].itags))
before = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
after  = norm(to_concrete(perp_component(retag(ex.U_ex, f.U0), HT))) / norm(HT)
@printf("residual outside the frame: %.3e -> %.3e\\n", before, after)""")]
    c += [md("""## 5. One bond update, and what it returns

`cbe_bond_update` expands, solves the Galerkin problem in the enlarged basis, then
splits. `left_core` comes back **already a left isometry** with the centre in
`right_core`, which is why the sweep needs no re-canonicalisation between bonds.

At full rank the expanded frames span the whole local space, the Galerkin generator
IS the gate, and the step is the *exact* two-site propagator. That is the first
thing the tests check."""),
          code("""res = cbe_bond_update(f, g, ComplexF64(-im * 0.05); maxdim = 32)
@printf("aug_k %d  aug_l %d  keep %d  discarded %.2e  matvecs %d  passes %d\\n",
        res.aug_k, res.aug_l, res.keep, res.discarded, res.n_matvec, res.n_iter)
@printf("left_core is a left isometry: defect %.2e\\n",
        left_isometry_defect(res.left_core))
@printf("singular values kept: %s\\n",
        string(round.(res.svals[1:min(6, end)], sigdigits = 3)))"""),
          md("""### The Strang composition

Bonds of one parity act on disjoint site pairs, so a parity group is an **exact**
factor of the Trotter step. The step is `even tau/2, odd tau, even tau/2`."""),
          code("""p = domain_wall_state(L)
gs = bond_gates(p)
@printf("even bonds: %s   odd bonds: %s\\n",
        string(collect(parity_bonds(L, :even))), string(collect(parity_bonds(L, :odd))))
for step in 1:6
    a = cbe_bond_update_sweep!(p, gs, :even, ComplexF64(-im * 0.025); maxdim = 32)
    b = cbe_bond_update_sweep!(p, gs, :odd,  ComplexF64(-im * 0.05);  maxdim = 32)
    c = cbe_bond_update_sweep!(p, gs, :even, ComplexF64(-im * 0.025); maxdim = 32)
    @printf("step %d: bond_dims %-22s matvecs %3d  err_fnl %.1e\\n", step,
            string(bond_dims(p)), a.n_matvec + b.n_matvec + c.n_matvec,
            max(a.err_fnl, b.err_fnl, c.err_fnl))
end""")]
    c += REFERENCE
    c += param_study("gate")
    c += [code("""vref = dense_reference(0.05, 20)
@printf("%-36s  %-9s  %-7s  %s\\n", "setting", "err", "maxbond", "matvecs")
for (lab, kw) in (
        ("growth=2.0, s_iters=1 (default)",   (; growth=2.0, dex=0, comp_ratio=0.5, exact=false, s_iters=1)),
        ("growth=1.25, s_iters=1",            (; growth=1.25, dex=0, comp_ratio=0.5, exact=false, s_iters=1)),
        ("growth=1.25, s_iters=3 (ladder)",   (; growth=1.25, dex=0, comp_ratio=0.5, exact=false, s_iters=3)),
        ("growth=2.0,  s_iters=3",            (; growth=2.0, dex=0, comp_ratio=0.5, exact=false, s_iters=3)),
        ("comp_ratio=1.0 (complement only)",  (; growth=2.0, dex=0, comp_ratio=1.0, exact=false, s_iters=1)),
        ("comp_ratio=nothing",                (; growth=2.0, dex=0, comp_ratio=nothing, exact=false, s_iters=1)),
        ("exact=true (no sketch)",            (; growth=2.0, dex=0, comp_ratio=0.5, exact=true, s_iters=1)),
    )
    p, info = run_gate(; kw...)
    @printf("%-36s  %.3e  %-7d  %d\\n", lab, err_vs_dense(p, vref),
            maximum(bond_dims(p)), sum(info.krylov_dims))
end"""),
          md("""### The growth ladder

`s_iters > 1` does **not** build a Krylov space — with $\\tau$ small every pass
re-derives $H\\Theta_0$, never $H^2\\Theta_0$. What it compounds is the *growth
schedule*: the budget is recomputed from the current rank each pass, so
$r \\to g r \\to g^2 r$, and each rung is probed by a sketch matched to the rank it
is probing. A sequence of well-conditioned narrow probes beats one wide draw.

Measured at $L=8$, bond (4,5), `growth = 1.25`: one-shot 1.449e-11 at rank 9,
three passes 1.978e-15 at rank 12. It is flat at `growth = 2.0` only because the
budget already saturates the room there.""")]
    return c


# ================================================================ notebook 3 ==

def nb_imag():
    c = [md("""# RSVD-CBE, notebook 3 of 4: **imaginary time**

Three integrators, all at $\\tau = -\\mathrm{d}t$:

| integrator | entry point |
|---|---|
| 1-site CBE (exact basis) | `tdvp1_cbe_exact_step!` |
| 1-site CBE + RSVD (sketched) | `tdvp1_rsvd_cbe_step!` |
| Lubich BUG sweep + RSVD-CBE | `cbe_lubich_sweep(psi, h, -dt)` |

**Real vs imaginary time is one line.** $\\exp(-i\\,\\mathrm{d}t\\,H)$ is unitary;
$\\exp(-\\mathrm{d}t\\,H)$ is a **contraction** — it damps every eigencomponent by
$e^{-\\mathrm{d}t E}$, so the norm decays and renormalising is *load-bearing* rather
than hygiene. The log of the decay factor is an energy estimate, and the fixed
point is the ground state.

> Note: the standing "real time only" rule in this project is about
> `bond_update_bug`, the validated gate integrator. This notebook uses the
> `RSVDCBEBondUpdate` module, which has an explicit `ImaginaryTime` mode."""),
         ]
    c += SETUP
    c += [code("""h = xxz_chain(L; delta = DELTA)
psi = neel_state(L)
@printf("start: bond_dims %s   Sz = %.1f   E = %.12f\\n",
        string(bond_dims(psi)), total_sz(psi), real(env_energy(psi, h)))

# the exact answer in this U(1) sector, for reference
Hd = Matrix(sum(dense_bond_term(L, i; delta = DELTA) for i in 1:(L - 1)))
vals, vecs = eigen(Hermitian(Hd))
szop = Diagonal([sum(((k >> (L - j)) & 1) == 0 ? 0.5 : -0.5 for j in 1:L)
                 for k in 0:(2^L - 1)])
sec = [k for k in 1:length(vals)
       if abs(real(dot(vecs[:, k], szop * vecs[:, k])) - total_sz(psi)) < 1e-8]
const E_exact = minimum(vals[sec])
@printf("exact ground energy in this sector: %.12f\\n", E_exact)""")]
    c += MPS_SECTION[:1] + MPO_SECTION + ENV_SECTION + PROJ_SECTION + SKETCH_SECTION
    c += ONE_SWEEP
    c += [md("""## 6. The three integrators, in imaginary time

Each drives the same start state toward the ground state. `normalize` after every
step is what keeps it from underflowing."""),
          code("""function imag_run(stepper; dt = 0.05, nsteps = 60, maxdim = 32)
    p = neel_state(L)
    Es = Float64[]
    for _ in 1:nsteps
        stepper(p, dt, maxdim)
        n = norm(p)
        n > 0 && (p[p.center] = to_concrete((1.0 / n) * p[p.center]))
        push!(Es, real(env_energy(p, h)))
    end
    return p, Es
end

step_1site_cbe(p, dt, md_) = tdvp1_cbe_exact_step!(p, h, ComplexF64(-dt); maxdim = md_)
step_1site_rsvd(p, dt, md_) = tdvp1_rsvd_cbe_step!(p, h, ComplexF64(-dt); maxdim = md_)
step_lubich(p, dt, md_) = cbe_lubich_sweep(p, h, ComplexF64(-dt);
                                           growth = 2.0, maxdim = md_)"""),
          code("""@printf("%-28s  %-16s  %-12s  %s\\n", "integrator", "E_final", "E - E_exact", "maxbond")
for (lab, st) in (("1-site CBE (exact basis)", step_1site_cbe),
                  ("1-site CBE + RSVD",        step_1site_rsvd),
                  ("Lubich BUG + RSVD-CBE",    step_lubich))
    p, Es = imag_run(st)
    @printf("%-28s  %+.12f  %+.3e  %d\\n", lab, Es[end], Es[end] - E_exact,
            maximum(bond_dims(p)))
end"""),
          md("""### Convergence trace

The 1-site CBE and 1-site CBE+RSVD arms should agree closely: that A/B is exactly
the question of whether the *sketch* costs accuracy. If they diverge, the probe is
too narrow — raise `growth` or `dex`, and watch `err_fnl`."""),
          code("""p1, E1 = imag_run(step_1site_cbe)
p2, E2 = imag_run(step_1site_rsvd)
p3, E3 = imag_run(step_lubich)
@printf("%-6s  %-18s  %-18s  %s\\n", "step", "1site CBE", "1site CBE+RSVD", "Lubich RSVD-CBE")
for k in (1, 5, 10, 20, 40, 60)
    k <= length(E1) || continue
    @printf("%-6d  %+.12f  %+.12f  %+.12f\\n", k, E1[k], E2[k], E3[k])
end
@printf("\\nexact                       %+.12f\\n", E_exact)
@printf("|1site CBE - 1site RSVD| at the end: %.3e   <- the sketch's cost\\n",
        abs(E1[end] - E2[end]))""")]
    c += param_study("imag")
    c += [code("""@printf("%-36s  %-13s  %-7s  %s\\n", "setting", "E - E_exact", "maxbond", "steps")
for (lab, kw) in (
        ("growth=2.0 (default)",              (; growth=2.0, dex=0, comp_ratio=0.5, exact=false)),
        ("growth=1.1",                        (; growth=1.1, dex=0, comp_ratio=0.5, exact=false)),
        ("dex=4",                             (; growth=2.0, dex=4, comp_ratio=0.5, exact=false)),
        ("comp_ratio=1.0 (complement only)",  (; growth=2.0, dex=0, comp_ratio=1.0, exact=false)),
        ("comp_ratio=nothing",                (; growth=2.0, dex=0, comp_ratio=nothing, exact=false)),
        ("exact=true (no sketch)",            (; growth=2.0, dex=0, comp_ratio=0.5, exact=true)),
    )
    p, Es = run_imag(; kw...)
    @printf("%-36s  %+.4e  %-7d  %d\\n", lab, Es[end] - E_exact,
            maximum(bond_dims(p)), length(Es))
end""")]
    return c


def nb_dmrg():
    c = [md("""# RSVD-CBE, notebook 4 of 4: **ground-state search (DMRG)**

`GroundState.solve!` — two-site DMRG whose bond expansion is CBE, in two arms:

| arm | setting |
|---|---|
| CBE-DMRG | `exact = true` — the complete fused basis |
| RSVD-CBE-DMRG | `exact = false` — the sketched probe |

Not imaginary time: this is a **variational** minimisation, so every reported
energy is an upper bound on the true ground energy. A number below `E_exact` is a
bug, not a lucky sweep.

Two things are swapped relative to the time modes: the **operator**
(`apply_h_two_site`, the environment-dressed $L+h+R$, not a gate) and the
**reduction** (lowest Ritz vector, not $\\exp(\\tau H)$)."""),
         ]
    c += SETUP
    c += [code("""h = xxz_chain(L; delta = DELTA)
psi = neel_state(L)
Hd = Matrix(sum(dense_bond_term(L, i; delta = DELTA) for i in 1:(L - 1)))
vals, vecs = eigen(Hermitian(Hd))
szop = Diagonal([sum(((k >> (L - j)) & 1) == 0 ? 0.5 : -0.5 for j in 1:L)
                 for k in 0:(2^L - 1)])
sec = [k for k in 1:length(vals)
       if abs(real(dot(vecs[:, k], szop * vecs[:, k])) - total_sz(psi)) < 1e-8]
const E_exact = minimum(vals[sec])
@printf("Neel start E = %.12f   exact ground E = %.12f\\n",
        real(env_energy(psi, h)), E_exact)"""),
          md("""### Why this mode is MPO-path only

The gate path's environments are the identity by canonical form, so its local
operator is the **bare** two-site term. Minimising $\\langle h_{i,i+1}\\rangle$ at
each bond independently does not have the global ground state as its fixed point —
each bond drives its own pair toward a local singlet and the next bond undoes it.
DMRG converges precisely because $L + h + R$ makes a *local* minimisation lower the
*total* energy."""),
          code("""p = neel_state(L)
R.GroundState.solve!(p, h; opts = CBEBugOptions(maxdim = 16), n_sweeps = 2)
canonical!(p, 3)
f = bond_frame(p, 3)
lenv = left_env_stack(p, h; upto = 2); renv = right_env_stack(p, h; downto = 5)
HT = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
@printf("Theta   legs %s\\n", string(Tuple(leg_dim(frame_theta(f), k) for k in 1:4)))
@printf("H*Theta legs %s\\n", string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("Rayleigh quotient = %.12f   env_energy = %.12f\\n",
        real(tensor_inner(frame_theta(f), HT)), real(env_energy(p, h)))
PD = to_concrete(perp_component(f.U0, HT))
@printf("||P_D H*Theta|| / ||H*Theta|| = %.3f%%  (energy still on the table)\\n",
        100 * norm(PD) / norm(HT))""")]
    c += [md("""## The two-phase local solve

`_expanding_krylov`'s tridiagonal is a Galerkin approximation across a **nested
sequence** of subspaces: $\\alpha_k$ is exact, but $\\beta_k$ is measured against the
projector in force at step $k$ and misses whatever part of $Hv_k$ fell outside the
then-current basis.

For a short-time exponential that is a small perturbation of an already-small
correction. For an eigensolve it is not acceptable — the Ritz value **is** the
answer, and one from an inconsistent tridiagonal is not variational, so it can sit
*below* the true ground energy. Hence phase A grows the frames, phase B is a clean
Lanczos in the now-fixed frames."""),
          code("""@printf("%-30s  %-13s  %-7s  %-8s  %s\\n",
        "arm", "E - E_exact", "maxbond", "matvecs", "converged")
for (lab, ex) in (("CBE-DMRG      (exact basis)", true),
                  ("RSVD-CBE-DMRG (sketched)",    false))
    for gi in (0, 1, 2)
        p = neel_state(L)
        info = R.GroundState.solve!(p, h;
                  opts = CBEBugOptions(maxdim = 32, growth = 2.0, exact = ex),
                  n_sweeps = 12, grow_iters = gi)
        E = minimum(info.energies)
        @printf("%-24s g=%d  %+.4e  %-7d  %-8d  %s\\n", lab, gi, E - E_exact,
                maximum(info.max_bond_dims), sum(info.krylov_dims),
                string(info.converged))
    end
end"""),
          md("""**`grow_iters = 0` freezes the state.** The frames never widen, the local
eigensolve minimises over a one-dimensional space, and the sweep reports the Néel
product-state energy $(L-1)\\times(-0.25) = -1.75$ at bond dimension 1 — and
"converges" in two sweeps. The CBE growth pass is doing **all** of the rank
generation and the eigensolve none of it, which also means `info.converged` is not
evidence of anything by itself."""),
          code("""p = neel_state(L)
info = R.GroundState.solve!(p, h;
          opts = CBEBugOptions(maxdim = 32, growth = 2.0), n_sweeps = 15,
          grow_iters = 1)
for (k, E) in enumerate(info.energies)
    @printf("sweep %2d: E = %+.12f   E - E_exact = %+.3e   maxbd %2d\\n",
            k, E, E - E_exact, info.max_bond_dims[k])
end
@printf("\\nVARIATIONAL? every energy >= E_exact: %s\\n",
        string(all(E -> E >= E_exact - 1e-10, info.energies)))""")]
    return c


def main():
    for name, cells in (("01_rsvd_lubich", nb_lubich()),
                        ("02_rsvd_gate", nb_gate()),
                        ("03_imaginary_time", nb_imag()),
                        ("04_ground_state_dmrg", nb_dmrg())):
        p, jl = write(name, cells)
        print("wrote", os.path.basename(p), "and", os.path.basename(jl),
              f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
