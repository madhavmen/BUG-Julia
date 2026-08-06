# Generated from 02_rsvd_gate.ipynb by build_notebooks.py
# Run on a COMPUTE NODE:  sbatch notebooks/submit_notebook.slurm 02_rsvd_gate

# # RSVD-CBE, notebook 2 of 3: the **gate sweep**
# 
# `cbe_bond_update` / `cbe_bond_update_sweep!` / `cbe_gate_evolve!`
# (`src/RSVDCBEBondUpdate/cbe_core.jl`).
# 
# The Trotter path. The operator at a bond is the **bare two-site gate** — the
# environments are the identity by canonical form — so no channel machinery appears
# at all. Everything else is the same expansion as notebook 1.

# # Setup
# 
# `BUGJulia` is loaded from the repo root. `set_symmetry!(:U1)` makes every tensor
# block-sparse in total $S^z$ — which is what makes the sector bookkeeping below
# visible rather than implicit.
# 
# The dense reference in `tests/common/` is test-side only: nothing in `src/` may
# densify. At $L=6$ the full space is $2^6 = 64$, so every claim on this page can be
# checked against an exact $64\times64$ matrix.

using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Random, Printf
using LurCGT, Telum
const R = RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))

set_symmetry!(:U1)
const L = 6
const DELTA = 1.0
println("symmetry = ", symmetry_mode(), "   L = ", L)

# ## 1. The MPS
# 
# `domain_wall_state(L)` is $|\uparrow\uparrow\uparrow\downarrow\downarrow\downarrow\rangle$:
# a product state, so every bond has dimension 1. That is deliberate — it is the
# hardest possible start for a rank-preserving method, and the reason CBE has to
# exist.
# 
# Each tensor is rank 3 with legs `(link_l, sigma, link_r)`. Under U(1) it is stored
# as a list of charge blocks, not a dense array.

psi = domain_wall_state(L)
@printf("bond dimensions: %s\n", string(bond_dims(psi)))
for i in 1:L
    T = psi[i]
    @printf("psi[%d]  legs %-18s  blocks: %d\n", i,
            string(Tuple(leg_dim(T, k) for k in 1:3)), length(T.qlabels))
end

# The charge sectors on one bond. A U(1) MPS carries a charge on every link;
# these labels are what the sketch has to respect.
canonical!(psi, 3)
T = psi[3]
println("psi[3] has ", length(T.qlabels), " charge blocks; sector dims per leg:")
for leg in 1:3
    dims = [(q, sector_dim(T, leg, q)) for (q, _) in T.spaces[leg]]
    @printf("  leg %d: %s\n", leg, string(dims))
end

# Densify one tensor, purely to look at it. `dense_state` is the test-side helper.
v = dense_state(psi)
@printf("state vector: %d amplitudes, norm %.6f, nonzero %d\n",
        length(v), norm(v), count(!iszero, v))
@printf("total Sz = %.1f (U(1) pins this for the whole run)\n", total_sz(psi))

# ## 2. The Hamiltonian
# 
# The XXZ chain is stored as a **term list**, not as an explicit rank-4 $W$ tensor:
# 
# $$H = \sum_i \sum_t c_t\, O^L_t(i)\, O^R_t(i+1)$$
# 
# Under U(1) the $S^+/S^-$ halves are rank 3 — the third leg carries the $\pm 2$
# charge that makes the pair symmetry-allowed. Contracting that leg *is* charge
# conservation.
# 
# The factored form exists because the CBE sketch has to **split a bond term across
# the two halves of the bond**. A fused gate has already joined them.

h = xxz_chain(L; delta = DELTA)
@printf("%d terms, J = %.2f, delta = %.2f\n", length(h.terms), h.J, h.delta)
for (t, term) in enumerate(h.terms)
    @printf("  term %d: coeff %+.3f   left rank %d   right rank %d\n",
            t, term.coeff, length(term.left.inds), length(term.right.inds))
end

# The two-site gate is the SAME terms, fused. One-way: this is why the term list
# is what gets stored.
gates = bond_gates(psi)
g = gates[3]
@printf("gate on bond 3: rank %d, legs %s\n",
        length(g.inds), string(Tuple(leg_dim(g, k) for k in 1:length(g.inds))))
@printf("energy via gates        = %.12f\n", real(energy(psi, gates)))
@printf("energy via environments = %.12f\n", real(env_energy(psi, h)))
@printf("  (these agreeing to machine precision is what pins the term list's\n")
@printf("   normalisation against the validated gate path)\n")

# ## 3. Gates instead of environments
# 
# `bond_gates(psi)` fuses each bond's terms into one rank-4 object. That fusion is
# one-way, and it is why the *Hamiltonian* is stored as a term list: the sketch has
# to split a bond term across the two halves of the bond, and a fused gate has
# already joined them.
# 
# It is also why the ground-state search (notebook 3) cannot run on this path:
# minimising $\langle h_{i,i+1}\rangle$ bond by bond does not have the global ground
# state as its fixed point.

psi2 = domain_wall_state(L)
gates = bond_gates(psi2)
cbe_bond_update_bug!(psi2, gates;
                     opts = CBEBugOptions(dt = 0.05, n_steps = 3, maxdim = 16))
@printf("warmed bond dimensions: %s\n", string(bond_dims(psi2)))

canonical!(psi2, 3)
f = bond_frame(psi2, 3)
g = gates[3]
@printf("gate legs %s\n", string(Tuple(leg_dim(g, k) for k in 1:4)))
theta = frame_theta(f)
HT = apply_gate(g, theta, f.site_l.itags, f.site_r.itags)
@printf("Theta  legs %s\n", string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("g*Theta legs %s\n", string(Tuple(leg_dim(HT, k) for k in 1:4)))

# ## 4. Projectors and the project-first sketch, on the gate path
# 
# Identical construction to notebook 1, with `sketch_bond_left/right` in place of
# `sketch_h_left/right`. Same signature check: the sketch output is orthogonal to
# the frame.

PD_HT = to_concrete(perp_component(f.U0, HT))
@printf("||g*Theta||     = %.6e\n", norm(HT))
@printf("||P_D g*Theta|| = %.6e   (what the expansion can reach)\n", norm(PD_HT))

Om = sector_graded_sketch(f.V0, :right, 6; rng = MersenneTwister(11))
Y = sketch_bond_left(f, g, Om)
K0 = to_concrete(f.U0 * f.S0)
ovl = to_concrete(contract(K0', (1, 2), Y, (1, 2)))
@printf("\n||<U0 S0 | Y>|| = %.3e   (project-first signature)\n", norm(ovl))
want = to_concrete(contract(PD_HT, (3, 4), Om', (2, 3)))
@printf("||Y - (P_D g*Theta) Om'|| = %.3e\n", norm(to_concrete(Y - want)))

ex = cbe_expand_bond(f, g; growth = 2.0)
@printf("r = %d -> U_ex %d, V_ex %d   new: L %d, R %d\n",
        f.old_rank, leg_dim(ex.U_ex, 3), leg_dim(ex.V_ex, 1),
        ex.n_new_l, ex.n_new_r)
retag(U, ref) = to_concrete(setitag(U, 3, ref.inds[3].itags))
before = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
after  = norm(to_concrete(perp_component(retag(ex.U_ex, f.U0), HT))) / norm(HT)
@printf("residual outside the frame: %.3e -> %.3e\n", before, after)

# ## 5. One bond update, and what it returns
# 
# `cbe_bond_update` expands, solves the Galerkin problem in the enlarged basis, then
# splits. `left_core` comes back **already a left isometry** with the centre in
# `right_core`, which is why the sweep needs no re-canonicalisation between bonds.
# 
# At full rank the expanded frames span the whole local space, the Galerkin generator
# IS the gate, and the step is the *exact* two-site propagator. That is the first
# thing the tests check.

res = cbe_bond_update(f, g, ComplexF64(-im * 0.05); maxdim = 32)
@printf("aug_k %d  aug_l %d  keep %d  discarded %.2e  matvecs %d  passes %d\n",
        res.aug_k, res.aug_l, res.keep, res.discarded, res.n_matvec, res.n_iter)
@printf("left_core is a left isometry: defect %.2e\n",
        left_isometry_defect(res.left_core))
@printf("singular values kept: %s\n",
        string(round.(res.svals[1:min(6, end)], sigdigits = 3)))

# ### The Strang composition
# 
# Bonds of one parity act on disjoint site pairs, so a parity group is an **exact**
# factor of the Trotter step. The step is `even tau/2, odd tau, even tau/2`.

p = domain_wall_state(L)
gs = bond_gates(p)
@printf("even bonds: %s   odd bonds: %s\n",
        string(collect(parity_bonds(L, :even))), string(collect(parity_bonds(L, :odd))))
for step in 1:6
    a = cbe_bond_update_sweep!(p, gs, :even, ComplexF64(-im * 0.025); maxdim = 32)
    b = cbe_bond_update_sweep!(p, gs, :odd,  ComplexF64(-im * 0.05);  maxdim = 32)
    c = cbe_bond_update_sweep!(p, gs, :even, ComplexF64(-im * 0.025); maxdim = 32)
    @printf("step %d: bond_dims %-22s matvecs %3d  err_fnl %.1e\n", step,
            string(bond_dims(p)), a.n_matvec + b.n_matvec + c.n_matvec,
            max(a.err_fnl, b.err_fnl, c.err_fnl))
end

# ### The reference
# 
# At $L=6$ the full Hilbert space is 64-dimensional, so the exact propagator is a
# $64\times64$ matrix exponential. That is a genuinely independent check — it shares
# no code with the integrator.

function dense_reference(dt, nsteps; delta = DELTA)
    Hd = sum(dense_bond_term(L, i; delta = delta) for i in 1:(L - 1))
    v0 = dense_state(domain_wall_state(L))
    return exp(-im * dt * nsteps * Matrix(Hd)) * v0
end

err_vs_dense(p, vref) = begin
    v = dense_state(p)
    ph = dot(vref, v); ph = abs(ph) > 1e-14 ? ph / abs(ph) : 1.0 + 0im
    norm(v / ph - vref) / norm(vref)   # global phase is not physical
end

# ## 7. Playing with the RSVD parameters
# 
# The knobs that decide what the expansion looks at:
# 
# | knob | what it controls |
# |---|---|
# | `growth` | budget $=\lceil \text{growth}\cdot d_{max}\rceil - r$; how many directions to propose |
# | `dex` | a fixed budget, overriding `growth` when > 0 |
# | `comp_ratio` | complement / isometry split of the probe (`nothing` = no projector in the probe) |
# | `exact` | swap the thin random probe for the complete fused basis — plain CBE, the reference the sketch approximates |
# | `maxdim` | the truncation ceiling, applied *after* the update |
# 
# Each row below runs `cbe_gate_evolve!` from the same start state and reports the error
# against the dense reference.

function run_gate(; growth, dex, comp_ratio, exact, s_iters = 1,
                   maxdim = 32, nsteps = 20, dt = 0.05)
    p = domain_wall_state(L)
    gs = bond_gates(p)
    info = cbe_bond_update_bug!(p, gs;
              opts = CBEBugOptions(dt = dt, n_steps = nsteps, maxdim = maxdim,
                                   growth = growth, dex = dex,
                                   comp_ratio = comp_ratio, exact = exact,
                                   s_iters = s_iters))
    return p, info
end

vref = dense_reference(0.05, 20)
@printf("%-36s  %-9s  %-7s  %s\n", "setting", "err", "maxbond", "matvecs")
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
    @printf("%-36s  %.3e  %-7d  %d\n", lab, err_vs_dense(p, vref),
            maximum(bond_dims(p)), sum(info.krylov_dims))
end

# ### The growth ladder
# 
# `s_iters > 1` does **not** build a Krylov space — with $\tau$ small every pass
# re-derives $H\Theta_0$, never $H^2\Theta_0$. What it compounds is the *growth
# schedule*: the budget is recomputed from the current rank each pass, so
# $r \to g r \to g^2 r$, and each rung is probed by a sketch matched to the rank it
# is probing. A sequence of well-conditioned narrow probes beats one wide draw.
# 
# Measured at $L=8$, bond (4,5), `growth = 1.25`: one-shot 1.449e-11 at rank 9,
# three passes 1.978e-15 at rank 12. It is flat at `growth = 2.0` only because the
# budget already saturates the room there.

