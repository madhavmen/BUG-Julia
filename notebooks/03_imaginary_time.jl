# Generated from 03_imaginary_time.ipynb by build_notebooks.py
# Run on a COMPUTE NODE:  sbatch notebooks/submit_notebook.slurm 03_imaginary_time

# # RSVD-CBE, notebook 3 of 4: **imaginary time**
# 
# Three integrators, all at $\tau = -\mathrm{d}t$:
# 
# | integrator | entry point |
# |---|---|
# | 1-site CBE (exact basis) | `tdvp1_cbe_exact_step!` |
# | 1-site CBE + RSVD (sketched) | `tdvp1_rsvd_cbe_step!` |
# | Lubich BUG sweep + RSVD-CBE | `cbe_lubich_sweep(psi, h, -dt)` |
# 
# **Real vs imaginary time is one line.** $\exp(-i\,\mathrm{d}t\,H)$ is unitary;
# $\exp(-\mathrm{d}t\,H)$ is a **contraction** — it damps every eigencomponent by
# $e^{-\mathrm{d}t E}$, so the norm decays and renormalising is *load-bearing* rather
# than hygiene. The log of the decay factor is an energy estimate, and the fixed
# point is the ground state.
# 
# > Note: the standing "real time only" rule in this project is about
# > `bond_update_bug`, the validated gate integrator. This notebook uses the
# > `RSVDCBEBondUpdate` module, which has an explicit `ImaginaryTime` mode.

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

h = xxz_chain(L; delta = DELTA)
psi = neel_state(L)
@printf("start: bond_dims %s   Sz = %.1f   E = %.12f\n",
        string(bond_dims(psi)), total_sz(psi), real(env_energy(psi, h)))

# the exact answer in this U(1) sector, for reference
Hd = Matrix(sum(dense_bond_term(L, i; delta = DELTA) for i in 1:(L - 1)))
vals, vecs = eigen(Hermitian(Hd))
szop = Diagonal([sum(((k >> (L - j)) & 1) == 0 ? 0.5 : -0.5 for j in 1:L)
                 for k in 0:(2^L - 1)])
sec = [k for k in 1:length(vals)
       if abs(real(dot(vecs[:, k], szop * vecs[:, k])) - total_sz(psi)) < 1e-8]
const E_exact = minimum(vals[sec])
@printf("exact ground energy in this sector: %.12f\n", E_exact)

# ## 1. The MPS
# 
# `domain_wall_state(L)` is $|\uparrow\uparrow\uparrow\downarrow\downarrow\downarrow\rangle$:
# a product state, so every bond has dimension 1. That is deliberate — it is the
# hardest possible start for a rank-preserving method, and the reason CBE has to
# exist.
# 
# Each tensor is rank 3 with legs `(link_l, sigma, link_r)`. Under U(1) it is stored
# as a list of charge blocks, not a dense array.

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

# ## 3. The environments
# 
# `henv.jl` stores the MPO's virtual index as **named channels** rather than a fused
# leg. Reading the virtual index as an automaton scanning the chain:
# 
# | state | meaning |
# |---|---|
# | `id` | nothing has happened yet; the identity has been transported |
# | `open[t]` | term `t` placed its LEFT operator, waiting for its partner |
# | `done` | one complete term lies entirely behind us |
# 
# `ZERO` is not `nothing`: `done` starts at `ZERO` (no completed terms), `id` starts
# at `nothing` (the chain boundary). Conflating them is a silent factor-of-one bug.

psi2 = domain_wall_state(L)
# warm it up so the bonds are not all 1 -- a product state makes every environment
# trivial and hides the structure.
cbe_bond_update_bug!(psi2, bond_gates(psi2);
                     opts = CBEBugOptions(dt = 0.05, n_steps = 3, maxdim = 16))
@printf("warmed bond dimensions: %s\n", string(bond_dims(psi2)))

canonical!(psi2, 3)
lenv = left_env_stack(psi2, h; upto = 2)
renv = right_env_stack(psi2, h; downto = 5)
lch = left_channels(lenv, 3)
rch = right_channels(renv, 5)

println("left channels on link 3:")
@printf("  id   : %s\n", lch.id === nothing ? "nothing (boundary)" :
        string(Tuple(leg_dim(lch.id, k) for k in 1:length(lch.id.inds))))
@printf("  done : %s\n", lch.done === R.ZERO ? "ZERO (no completed terms)" :
        string(Tuple(leg_dim(lch.done, k) for k in 1:length(lch.done.inds))))
for (t, o) in enumerate(lch.open)
    @printf("  open[%d]: %s\n", t, o === R.ZERO ? "ZERO" :
            string(Tuple(leg_dim(o, k) for k in 1:length(o.inds))))
end

# What a contraction RESULTS IN: the two-site effective operator, applied to Theta.
f = bond_frame(psi2, 3)
theta = frame_theta(f)
HT = apply_h_two_site(theta, h, 3, lenv, renv)
@printf("Theta : rank %d  legs %s\n", length(theta.inds),
        string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("H*Theta: rank %d  legs %s   (same legs -- H maps the block to itself)\n",
        length(HT.inds), string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("<Theta|H|Theta> = %.12f   vs env_energy = %.12f\n",
        real(tensor_inner(theta, HT)), real(env_energy(psi2, h)))

# ## 4. The projectors
# 
# The bond frame factorises the two-site block, $\Theta = U_0 S_0 V_0$. $U_0$ and
# $V_0$ are **isometries**: orthonormal bases for the $r$-dimensional subspaces the
# state currently uses.
# 
# The fused local space $(\ell_{i-1}\otimes\sigma_i)$ has dimension $d\cdot\chi$
# and splits in two:
# 
# $$P_K = U_0 U_0^\dagger \quad\text{(kept)}, \qquad
#   P_D = I - U_0 U_0^\dagger \quad\text{(discarded)}$$
# 
# with $P_K + P_D = I$, $P_K^2 = P_K$, $P_K P_D = 0$. **Everything CBE does happens
# inside $P_D$.**

f = bond_frame(psi2, 3)
@printf("U0 legs %s   V0 legs %s   S0 legs %s\n",
        string(Tuple(leg_dim(f.U0, k) for k in 1:3)),
        string(Tuple(leg_dim(f.V0, k) for k in 1:3)),
        string(Tuple(leg_dim(f.S0, k) for k in 1:2)))
@printf("rank r = %d\n", f.old_rank)
@printf("left  isometry defect ||U0'U0 - I|| = %.2e\n", left_isometry_defect(f.U0))
@printf("right isometry defect ||V0 V0' - I|| = %.2e\n", right_isometry_defect(f.V0))

# The kept and discarded projectors, applied to the two-site block.
# perp_component(U0, X) = X - U0 (U0' X)  =  P_D X.
HT   = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
PD_HT = to_concrete(perp_component(f.U0, HT))          # discarded part
PK_HT = to_concrete(HT - PD_HT)                        # kept part

@printf("||H*Theta||      = %.6e\n", norm(HT))
@printf("||P_K H*Theta||  = %.6e   (inside the current frame)\n", norm(PK_HT))
@printf("||P_D H*Theta||  = %.6e   (OUTSIDE it -- what CBE is hunting)\n", norm(PD_HT))
@printf("completeness ||H*Theta - (P_K + P_D)H*Theta|| = %.2e\n",
        norm(to_concrete(HT - to_concrete(PK_HT + PD_HT))))
@printf("idempotence  ||P_D(P_D X) - P_D X||           = %.2e\n",
        norm(to_concrete(to_concrete(perp_component(f.U0, PD_HT)) - PD_HT)))
@printf("orthogonality <P_K X | P_D X>                 = %.2e\n",
        abs(tensor_inner(PK_HT, PD_HT)))

# ### The residual is the whole story
# 
# $\|P_D (H\Theta)\| / \|H\Theta\|$ is the fraction of the Hamiltonian's action at
# this bond that the **current** frame cannot represent. If it is zero the frame is
# already complete and no expansion can help. If it is large, a rank-preserving
# method (1-site TDVP without CBE) is throwing that fraction away every step.

for i in 1:(L - 1)
    canonical!(psi2, i)
    fi = bond_frame(psi2, i)
    le = left_env_stack(psi2, h; upto = i - 1)
    re = right_env_stack(psi2, h; downto = i + 2)
    Hi = apply_h_two_site(frame_theta(fi), h, i, le, re)
    resid = norm(to_concrete(perp_component(fi.U0, Hi))) / norm(Hi)
    @printf("bond %d: r = %2d   ||P_D H*Theta|| / ||H*Theta|| = %.3e\n",
            i, fi.old_rank, resid)
end

# ## 5. The sketch — and it probes the PROJECTOR
# 
# `sketch_h_left(f, h, i, lch, rch, Om)` returns
# 
# $$Y = \big(P_\perp\, H\Theta\big)\,\Omega^\dagger, \qquad
#   P_\perp = I - U_0U_0^\dagger$$
# 
# **The projector is applied first.** This is the only sketching path; the
# fold-$\Omega$-first variant was removed on 2026-08-06.
# 
# The two orderings agree *exactly* — $P_\perp$ acts on $(\ell_{i-1},\sigma_i)$ and
# $\Omega$ on $(\sigma_{i+1},\ell_{i+1})$, disjoint legs, so they commute. What
# project-first changes is cost: $H\Theta$ is now formed once per bond.
# 
# Its signature is checkable in one line: **$Y$ is orthogonal to the frame for any
# probe.**

rng = MersenneTwister(0xC0FFEE)
canonical!(psi2, 3)
f = bond_frame(psi2, 3)
lenv = left_env_stack(psi2, h; upto = 2); renv = right_env_stack(psi2, h; downto = 5)
lch = left_channels(lenv, 3); rch = right_channels(renv, 5)

Om = sector_graded_sketch(f.V0, :right, 6; rng = rng)
@printf("probe Omega: legs %s\n", string(Tuple(leg_dim(Om, k) for k in 1:3)))
Y = sketch_h_left(f, h, 3, lch, rch, Om)
@printf("sketch Y   : legs %s\n", string(Tuple(leg_dim(Y, k) for k in 1:3)))

K0 = to_concrete(f.U0 * f.S0)
ovl = to_concrete(contract(K0', (1, 2), Y, (1, 2)))     # (bond, g)
@printf("\nPROJECT-FIRST SIGNATURE  ||<U0 S0 | Y>|| = %.3e   (must vanish)\n", norm(ovl))

# and it equals the explicitly projected action, folded afterwards
HT = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
want = to_concrete(contract(to_concrete(perp_component(f.U0, HT)), (3, 4), Om', (2, 3)))
@printf("||Y - (P_perp H*Theta) Om'||        = %.3e\n", norm(to_concrete(Y - want)))

# the orderings commute -- the identity that makes this change free of consequence
want2 = to_concrete(perp_component(f.U0, to_concrete(contract(HT, (3, 4), Om', (2, 3)))))
@printf("||Y - P_perp((H*Theta) Om')||       = %.3e   <- the commuting identity\n",
        norm(to_concrete(Y - want2)))

# ### The probe is sector-graded, not Gaussian
# 
# Under a symmetry a plain Gaussian is wrong: columns must be allocated **per charge
# sector**, or most sectors get drawn with zero weight. `comp_ratio` splits the
# budget between the orthogonal complement and the isometry space (the reference's
# `CompRatio`, default 0.5).
# 
# The reference is explicit about keeping both halves
# (`RSVDpreBE0SiQS.m:337`): *"do not project into complement space / 1-site
# components are also important for bond update"*.

for cr in (0.0, 0.5, 1.0, nothing)
    Om = sector_graded_sketch(f.V0, :right, 6; comp_ratio = cr,
                              rng = MersenneTwister(7))
    if Om === nothing
        @printf("comp_ratio = %-6s -> no probe\n", string(cr))
        continue
    end
    Y = sketch_h_left(f, h, 3, lch, rch, Om)
    @printf("comp_ratio = %-6s  cols = %d  ||Y|| = %.4e  ||<U0S0|Y>|| = %.1e\n",
            string(cr), leg_dim(Om, 1), norm(Y),
            norm(to_concrete(contract(K0', (1, 2), Y, (1, 2)))))
end

# ## 5. ONE SWEEP, STEP BY STEP
# 
# Everything above was static. This section walks a **single bond of a single
# sweep** and prints every object as it is built: what is constructed, what each
# contraction produces, and what the local solve actually solves.
# 
# The bond update is the same five moves in every integrator on this page:
# 
# | # | move | produces |
# |---|---|---|
# | 1 | `canonical!(psi, i)` | the gauge the frame needs |
# | 2 | `bond_frame(psi, i)` | `Theta = U0 S0 V0` |
# | 3 | `sketch_h_left/right` | `Y = (P_perp H Theta) Om'` |
# | 4 | `cbe_expand` | `U_ex = [U0 | C_L]`, `V_ex` |
# | 5 | local solve + `svd` | the new `psi[i]`, `psi[i+1]` |

p = domain_wall_state(L)
for _ in 1:3      # warm up so the bond is not trivially rank 1
    tdvp1_rsvd_cbe_step!(p, h, ComplexF64(-0.05); maxdim = 32)
    p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center])
end
i = 3
@printf("STATE  bond_dims %s   centre %d\n", string(bond_dims(p)), p.center)

# ---- move 1: the gauge -----------------------------------------------------
canonical!(p, i)
@printf("\n1. canonical!(psi, %d)  -> centre now %d\n", i, p.center)
@printf("   sites 1..%d are LEFT isometries, %d..%d RIGHT isometries\n",
        i - 1, i + 1, L)
@printf("   left  isometry defect on psi[1]: %.2e\n", left_isometry_defect(p[1]))
@printf("   right isometry defect on psi[%d]: %.2e\n", L, right_isometry_defect(p[L]))

# ---- move 2: the bond frame -------------------------------------------------
f = bond_frame(p, i)
theta = frame_theta(f)
@printf("2. bond_frame(psi, %d)\n", i)
@printf("   Theta  legs %-20s  = the two-site block\n",
        string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("   U0     legs %-20s  left isometry,  defect %.1e\n",
        string(Tuple(leg_dim(f.U0, k) for k in 1:3)), left_isometry_defect(f.U0))
@printf("   S0     legs %-20s  the core: ALL the amplitude\n",
        string(Tuple(leg_dim(f.S0, k) for k in 1:2)))
@printf("   V0     legs %-20s  right isometry, defect %.1e\n",
        string(Tuple(leg_dim(f.V0, k) for k in 1:3)), right_isometry_defect(f.V0))
@printf("   rank r = %d\n", f.old_rank)
@printf("   reassembly check ||U0 S0 V0 - Theta|| = %.2e\n",
        norm(to_concrete(to_concrete(to_concrete(f.U0 * f.S0) * f.V0) - theta)))

# ---- the local operator, and what its contraction produces -----------------
lenv = left_env_stack(p, h; upto = i - 1)
renv = right_env_stack(p, h; downto = i + 2)
lch = left_channels(lenv, i); rch = right_channels(renv, i + 2)

HT = apply_h_two_site(theta, h, i, lenv, renv)
@printf("\nLOCAL OPERATOR  apply_h_two_site(Theta, h, %d, lenv, renv)\n", i)
@printf("   in  : Theta   %s\n", string(Tuple(leg_dim(theta, k) for k in 1:4)))
@printf("   out : H*Theta %s   (same legs -- H maps the block to itself)\n",
        string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("   <Theta|H|Theta> = %.12f   vs env_energy = %.12f\n",
        real(tensor_inner(theta, HT)), real(env_energy(p, h)))

println("\n   the operator is assembled from THREE kinds of term, by where they")
println("   sit relative to the bond:")
@printf("     entirely LEFT  : lch.done  %s\n",
        lch.done === R.ZERO ? "ZERO" : string(Tuple(leg_dim(lch.done, k) for k in 1:2)))
@printf("     entirely RIGHT : rch.done  %s\n",
        rch.done === R.ZERO ? "ZERO" : string(Tuple(leg_dim(rch.done, k) for k in 1:2)))
for (t, o) in enumerate(lch.open)
    @printf("     STRADDLING t=%d: lch.open[%d] %s  (waiting for its partner)\n",
            t, t, o === R.ZERO ? "ZERO" :
            string(Tuple(leg_dim(o, k) for k in 1:length(o.inds))))
end

# ---- move 3: the sketch probes the PROJECTOR --------------------------------
PD = to_concrete(perp_component(f.U0, HT))
@printf("\n3. the projector, then the sketch\n")
@printf("   ||H*Theta||      = %.6e\n", norm(HT))
@printf("   ||P_K H*Theta||  = %.6e   inside the frame (1-site TDVP sees only this)\n",
        norm(to_concrete(HT - PD)))
@printf("   ||P_D H*Theta||  = %.6e   OUTSIDE it -- what CBE is hunting\n", norm(PD))

Om = sector_graded_sketch(f.V0, :right, 6; rng = MersenneTwister(5))
Y  = sketch_h_left(f, h, i, lch, rch, Om)
@printf("   Omega  legs %-18s  %d random columns, sector-graded\n",
        string(Tuple(leg_dim(Om, k) for k in 1:3)), leg_dim(Om, 1))
@printf("   Y      legs %-18s  = (P_perp H Theta) Omega'\n",
        string(Tuple(leg_dim(Y, k) for k in 1:3)))
ovl = to_concrete(contract(to_concrete(f.U0 * f.S0)', (1, 2), Y, (1, 2)))
@printf("   ||<U0 S0 | Y>|| = %.2e  <- vanishes: the sketch lives in the COMPLEMENT\n",
        norm(ovl))

# ---- move 4: the expansion --------------------------------------------------
ex = cbe_expand(f, h, i, lch, rch; growth = 2.0)
retag(U, ref) = to_concrete(setitag(U, 3, ref.inds[3].itags))
@printf("\n4. cbe_expand -> U_ex = [U0 | C_L],  V_ex = [V0 ; C_R]\n")
@printf("   U_ex legs %-18s  bond %d -> %d   (+%d new directions)\n",
        string(Tuple(leg_dim(ex.U_ex, k) for k in 1:3)),
        f.old_rank, leg_dim(ex.U_ex, 3), ex.n_new_l)
@printf("   V_ex legs %-18s  bond %d -> %d   (+%d)\n",
        string(Tuple(leg_dim(ex.V_ex, k) for k in 1:3)),
        f.old_rank, leg_dim(ex.V_ex, 1), ex.n_new_r)
@printf("   still isometries: defect %.1e / %.1e\n",
        left_isometry_defect(ex.U_ex), right_isometry_defect(ex.V_ex))
b4 = norm(to_concrete(perp_component(f.U0, HT))) / norm(HT)
af = norm(to_concrete(perp_component(retag(ex.U_ex, f.U0), HT))) / norm(HT)
@printf("   residual outside the frame: %.3e -> %.3e\n", b4, af)
@printf("   err_pre %.2e   err_fnl %.2e\n", ex.err_pre, ex.err_fnl)

# ---- move 5: the local solve, and its contraction count ---------------------
# The Galerkin problem lives in the EXPANDED basis: seed by projecting Theta in,
# apply the operator, project back. That round trip is one matvec.
S_start = to_concrete(contract(contract(ex.U_ex', (1, 2), theta, (1, 2)),
                               (2, 3), ex.V_ex', (2, 3)))
@printf("\n5. the local solve\n")
@printf("   S_start legs %-16s  <- U_ex' Theta V_ex'  (seed, no inverse anywhere)\n",
        string(Tuple(leg_dim(S_start, k) for k in 1:2)))
@printf("   lossless? ||U_ex S_start V_ex - Theta|| = %.2e\n",
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
@printf("   %d matvecs; each is expand-out -> apply H -> project back\n", nmv[])
@printf("   S1 legs %s   (still just two bond indices: NO physical leg)\n",
        string(Tuple(leg_dim(S1, k) for k in 1:2)))

# ---- and the split that writes it back --------------------------------------
res = svd(S1, (1,); cutoff = 1e-14, Nkeep = 32, get_lists = true)
sv = [s for (s, _, _, _) in res.kept_list]
@printf("\n   svd(S1) keeps %d of %d singular values\n", length(sv), leg_dim(S1, 1))
@printf("   spectrum: %s\n", string(round.(sv[1:min(6, end)], sigdigits = 3)))
@printf("   -> psi[%d] = U_ex * U  (a left isometry again)\n", i)
@printf("   -> psi[%d] = (S Vd) * V_ex  (the new centre)\n", i + 1)
println("\n   TRUNCATION COMES LAST. Run before the solve it would delete the")
println("   directions the expansion just bought -- they carry no weight YET.")

# ## 6. The three integrators, in imaginary time
# 
# Each drives the same start state toward the ground state. `normalize` after every
# step is what keeps it from underflowing.

function imag_run(stepper; dt = 0.05, nsteps = 60, maxdim = 32)
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
                                           growth = 2.0, maxdim = md_)

@printf("%-28s  %-16s  %-12s  %s\n", "integrator", "E_final", "E - E_exact", "maxbond")
for (lab, st) in (("1-site CBE (exact basis)", step_1site_cbe),
                  ("1-site CBE + RSVD",        step_1site_rsvd),
                  ("Lubich BUG + RSVD-CBE",    step_lubich))
    p, Es = imag_run(st)
    @printf("%-28s  %+.12f  %+.3e  %d\n", lab, Es[end], Es[end] - E_exact,
            maximum(bond_dims(p)))
end

# ### Convergence trace
# 
# The 1-site CBE and 1-site CBE+RSVD arms should agree closely: that A/B is exactly
# the question of whether the *sketch* costs accuracy. If they diverge, the probe is
# too narrow — raise `growth` or `dex`, and watch `err_fnl`.

p1, E1 = imag_run(step_1site_cbe)
p2, E2 = imag_run(step_1site_rsvd)
p3, E3 = imag_run(step_lubich)
@printf("%-6s  %-18s  %-18s  %s\n", "step", "1site CBE", "1site CBE+RSVD", "Lubich RSVD-CBE")
for k in (1, 5, 10, 20, 40, 60)
    k <= length(E1) || continue
    @printf("%-6d  %+.12f  %+.12f  %+.12f\n", k, E1[k], E2[k], E3[k])
end
@printf("\nexact                       %+.12f\n", E_exact)
@printf("|1site CBE - 1site RSVD| at the end: %.3e   <- the sketch's cost\n",
        abs(E1[end] - E2[end]))

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
# Each row below runs `tdvp1_cbe_step!  (imaginary time)` from the same start state and reports the error
# against the dense reference.

function run_imag(; growth, dex, comp_ratio, exact,
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
end

@printf("%-36s  %-13s  %-7s  %s\n", "setting", "E - E_exact", "maxbond", "steps")
for (lab, kw) in (
        ("growth=2.0 (default)",              (; growth=2.0, dex=0, comp_ratio=0.5, exact=false)),
        ("growth=1.1",                        (; growth=1.1, dex=0, comp_ratio=0.5, exact=false)),
        ("dex=4",                             (; growth=2.0, dex=4, comp_ratio=0.5, exact=false)),
        ("comp_ratio=1.0 (complement only)",  (; growth=2.0, dex=0, comp_ratio=1.0, exact=false)),
        ("comp_ratio=nothing",                (; growth=2.0, dex=0, comp_ratio=nothing, exact=false)),
        ("exact=true (no sketch)",            (; growth=2.0, dex=0, comp_ratio=0.5, exact=true)),
    )
    p, Es = run_imag(; kw...)
    @printf("%-36s  %+.4e  %-7d  %d\n", lab, Es[end] - E_exact,
            maximum(bond_dims(p)), length(Es))
end

