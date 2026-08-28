#!/bin/bash
# MODEL 3 of the campaign: the dissipative anisotropic Heisenberg chain of Hryniuk & Szymanska
# (Comms Phys 2026, 9:45), on a UNIFORM grid, BUG against 2-site TDVP.
#
#   sbatch benchmarks/submit_dissipative_heisenberg.slurm
#   python3 benchmarks/plot_dissip_heis.py benchmarks/results/dissip_heis_protocol_L16_dt*.csv
#
# ⛔ WHAT IS *NOT* HERE. `certify` (L=3, dense Liouvillian), `certify_obs` (L=4, the sigma^y /
# two-site-order gate) and `spectrum` (L<=4) are correctness gates costing seconds, and they belong
# LOCALLY where a failure is seen immediately rather than after a queue wait. Only L = 16 is here.
#
# ⛔ THIS MODEL HAS NO SYMMETRY ARM, AND THAT IS A MACHINERY LIMIT, NOT A MODELLING CHOICE.
# `fused_space()` builds the d = 4 site through `_getLocalSpace_no_symmetry` -- the same route
# `z2_ising.jl` had to take, because `getLocalSpace` derives its space from a KNOWN symmetry type
# and cannot be handed an arbitrary one. So every fused superket in this repo is `:none`, whatever
# the couplings are. Switching to XXZ (`Jx = Jy`, which WOULD restore U(1) at the model level, since
# the spin-decay dissipator preserves `n_ket - n_bra`) buys nothing on its own: a symmetry arm needs
# a NEW d = 4 local space graded by `q = n_ket - n_bra` with sectors {-1, 0, 0, +1}.
# ⇒ the campaign's symmetry axis is carried by the CHAIN and the CYLINDER (`submit_logtime_suite`).
#
# ⛔ THERE IS NO EXACT REFERENCE AT L = 16 -- `4^16` is 4.3 billion amplitudes. The `err` column
# models 1 and 2 carry has NO counterpart here, and our own finest grid must NOT be quietly
# substituted for an anchor (see the campaign's external-anchor rule). Two honest claims instead:
#   (a) CONVERGENCE -- the array below is a three-point `dt` ladder, so each arm's own order can be
#       estimated from its self-agreement.
#   (b) COMPARABILITY -- the paper's published Fig. 3a-c is the anchor, and agreement with it is
#       qualitative. Ours is N = 16, theirs N = 200; do not report a numerical deviation from it.
#
# ⚠ THE PRIMARY COST AXIS IS `krylov`, NOT `seconds`. Array tasks can share a node, so wall-clock
# carries contention that matvec counts do not. Threads are pinned to 1 below so `seconds` stays
# broadly comparable, but any headline BUG-vs-TDVP cost ratio should be quoted on `krylov`.
#
#SBATCH --job-name=dissipheis
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=/home/madhav.menon/benchmarks/dissipheis_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/dissipheis_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

# ⛔ A THREE-POINT `dt` LADDER, NOT A SINGLE PRODUCTION GRID. With no exact reference the only
# accuracy statement available is self-convergence, and two points cannot separate "converged" from
# "both wrong by the same amount" -- the same reason the cylinder bulk fit needs three lengths.
DTS=(0.02 0.01 0.005)
export DH_DT=${DTS[$SLURM_ARRAY_TASK_ID]}

export DH_L=${DH_L:-16}
export DH_TMAX=${DH_TMAX:-4.0}

# ⚠ `CAP` DECIDES WHETHER THE CORRELATORS MEAN ANYTHING. Operator entanglement is what carries the
# connected part: a superket truncated too hard keeps a smooth, plausible `<sigma^a(t)>` and a
# perfect `Tr rho = 1` while `C^aa_d` decays for a purely numerical reason. If chi SATURATES at
# `CAP` in the log, the correlators are not converged -- raise it and resubmit before drawing any
# conclusion about either integrator.
export DH_CAP=${DH_CAP:-128}

# ⛔ THREADS PINNED TO 1, matching every other submit script here.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "task ${SLURM_ARRAY_TASK_ID} -> L=$DH_L dt=$DH_DT tmax=$DH_TMAX cap=$DH_CAP"

# ⛔ ONE CSV PER TASK -- `protocol` names its output by L and dt, so the three tasks cannot collide.
julia --project=. benchmarks/dissipative_heisenberg.jl protocol
