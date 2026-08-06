# Generated from 04_ground_state_dmrg.ipynb by build_notebooks.py
# Run on a COMPUTE NODE:  sbatch notebooks/submit_notebook.slurm 04_ground_state_dmrg

# # RSVD-CBE, notebook 4 of 4: **ground-state search (DMRG)**
# 
# `GroundState.solve!` — two-site DMRG whose bond expansion is CBE, in two arms:
# 
# | arm | setting |
# |---|---|
# | CBE-DMRG | `exact = true` — the complete fused basis |
# | RSVD-CBE-DMRG | `exact = false` — the sketched probe |
# 
# Not imaginary time: this is a **variational** minimisation, so every reported
# energy is an upper bound on the true ground energy. A number below `E_exact` is a
# bug, not a lucky sweep.
# 
# Two things are swapped relative to the time modes: the **operator**
# (`apply_h_two_site`, the environment-dressed $L+h+R$, not a gate) and the
# **reduction** (lowest Ritz vector, not $\exp(\tau H)$).

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
Hd = Matrix(sum(dense_bond_term(L, i; delta = DELTA) for i in 1:(L - 1)))
vals, vecs = eigen(Hermitian(Hd))
szop = Diagonal([sum(((k >> (L - j)) & 1) == 0 ? 0.5 : -0.5 for j in 1:L)
                 for k in 0:(2^L - 1)])
sec = [k for k in 1:length(vals)
       if abs(real(dot(vecs[:, k], szop * vecs[:, k])) - total_sz(psi)) < 1e-8]
const E_exact = minimum(vals[sec])
@printf("Neel start E = %.12f   exact ground E = %.12f\n",
        real(env_energy(psi, h)), E_exact)

# ### Why this mode is MPO-path only
# 
# The gate path's environments are the identity by canonical form, so its local
# operator is the **bare** two-site term. Minimising $\langle h_{i,i+1}\rangle$ at
# each bond independently does not have the global ground state as its fixed point —
# each bond drives its own pair toward a local singlet and the next bond undoes it.
# DMRG converges precisely because $L + h + R$ makes a *local* minimisation lower the
# *total* energy.

p = neel_state(L)
R.GroundState.solve!(p, h; opts = CBEBugOptions(maxdim = 16), n_sweeps = 2)
canonical!(p, 3)
f = bond_frame(p, 3)
lenv = left_env_stack(p, h; upto = 2); renv = right_env_stack(p, h; downto = 5)
HT = apply_h_two_site(frame_theta(f), h, 3, lenv, renv)
@printf("Theta   legs %s\n", string(Tuple(leg_dim(frame_theta(f), k) for k in 1:4)))
@printf("H*Theta legs %s\n", string(Tuple(leg_dim(HT, k) for k in 1:4)))
@printf("Rayleigh quotient = %.12f   env_energy = %.12f\n",
        real(tensor_inner(frame_theta(f), HT)), real(env_energy(p, h)))
PD = to_concrete(perp_component(f.U0, HT))
@printf("||P_D H*Theta|| / ||H*Theta|| = %.3f%%  (energy still on the table)\n",
        100 * norm(PD) / norm(HT))

# ## The two-phase local solve
# 
# `_expanding_krylov`'s tridiagonal is a Galerkin approximation across a **nested
# sequence** of subspaces: $\alpha_k$ is exact, but $\beta_k$ is measured against the
# projector in force at step $k$ and misses whatever part of $Hv_k$ fell outside the
# then-current basis.
# 
# For a short-time exponential that is a small perturbation of an already-small
# correction. For an eigensolve it is not acceptable — the Ritz value **is** the
# answer, and one from an inconsistent tridiagonal is not variational, so it can sit
# *below* the true ground energy. Hence phase A grows the frames, phase B is a clean
# Lanczos in the now-fixed frames.

@printf("%-30s  %-13s  %-7s  %-8s  %s\n",
        "arm", "E - E_exact", "maxbond", "matvecs", "converged")
for (lab, ex) in (("CBE-DMRG      (exact basis)", true),
                  ("RSVD-CBE-DMRG (sketched)",    false))
    for gi in (0, 1, 2)
        p = neel_state(L)
        info = R.GroundState.solve!(p, h;
                  opts = CBEBugOptions(maxdim = 32, growth = 2.0, exact = ex),
                  n_sweeps = 12, grow_iters = gi)
        E = minimum(info.energies)
        @printf("%-24s g=%d  %+.4e  %-7d  %-8d  %s\n", lab, gi, E - E_exact,
                maximum(info.max_bond_dims), sum(info.krylov_dims),
                string(info.converged))
    end
end

# **`grow_iters = 0` freezes the state.** The frames never widen, the local
# eigensolve minimises over a one-dimensional space, and the sweep reports the Néel
# product-state energy $(L-1)\times(-0.25) = -1.75$ at bond dimension 1 — and
# "converges" in two sweeps. The CBE growth pass is doing **all** of the rank
# generation and the eigensolve none of it, which also means `info.converged` is not
# evidence of anything by itself.

p = neel_state(L)
info = R.GroundState.solve!(p, h;
          opts = CBEBugOptions(maxdim = 32, growth = 2.0), n_sweeps = 15,
          grow_iters = 1)
for (k, E) in enumerate(info.energies)
    @printf("sweep %2d: E = %+.12f   E - E_exact = %+.3e   maxbd %2d\n",
            k, E, E - E_exact, info.max_bond_dims[k])
end
@printf("\nVARIATIONAL? every energy >= E_exact: %s\n",
        string(all(E -> E >= E_exact - 1e-10, info.energies)))

