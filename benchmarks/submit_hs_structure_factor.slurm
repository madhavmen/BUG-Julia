#!/bin/bash
# The Haldane-Shastry dynamical structure factor S(k, w), as an ARRAY over ring size.
#
# WHY THIS RUNS ON THE CLUSTER AND NOT ON THE LAPTOP. The identities in
# `benchmarks/hs_structure_factor.jl` are exact at ANY L, so a small local run fully decides
# whether the pipeline is correct -- and L=8 was run locally for exactly that. What a laptop
# cannot deliver is the PHYSICS comparison: the reference is the behaviour in Li et al.
# (arXiv:2208.10972) Fig. 3, a thermodynamic-limit object, and eight ring momenta sample the
# two-spinon continuum far too coarsely to be held against it. That needs L >= 16.
#
#   ⛔ AND L >= 16 IS GENUINELY EXPENSIVE HERE, for a reason specific to this observable.
#   `S^z_{j0}|0>` on the HS ring is close to FULL RANK the moment it is formed -- measured at
#   L=16, the centre bond is 247 against a 256 cap -- so no truncation ever helps and every step
#   pays full price. The earlier L=16 HS real-time arm cost about 5.5 hours for this reason.
#   Budget accordingly; this is not a short job.
#
#   sbatch benchmarks/submit_hs_structure_factor.slurm
#   python3 benchmarks/plot_hs_skw.py            # after the array finishes
#
#SBATCH --job-name=hs_skw
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=/home/madhav.menon/benchmarks/hs_skw_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/hs_skw_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

# ⚠ `T` SETS THE FREQUENCY RESOLUTION, `dt` ONLY THE CEILING. The record is finite, so
# `dw ~ 2 pi / T` no matter how finely it is sampled, and the whole two-spinon band tops out at
# `w_+(pi) = pi^2/4 = 2.47`. T=32 gives dw ~ 0.20, about twelve bins across the band -- enough for
# a lineshape. T=8 would give three, which is enough for the identities and useless for a figure.
# Shrinking `dt` instead raises the Nyquist ceiling `pi/dt`, already 63 at dt=0.05, which was never
# the binding constraint.
T=32.0
DT=0.05
# ⛔ ONE BUG SCHEME EVERYWHERE. `bug_decoupled` no longer exists.
SCHEME=bug_interleaved

# One ring size per task. L=8 duplicates the local run deliberately: it is the cheap task that
# proves the cluster environment reproduces the identity gate before the expensive ones are
# believed. L=12 is the intermediate that shows the trend is not an L=16 artefact.
SIZES=(8 12 16)
L=${SIZES[$SLURM_ARRAY_TASK_ID]}

echo "task ${SLURM_ARRAY_TASK_ID} -> L=$L T=$T dt=$DT scheme=$SCHEME"

# ⛔ THREADS PINNED TO 1, matching `submit_cbe_sweeps_l16.slurm`. Not a cost comparison here, but
# the runs should be reproducible and comparable with the campaign's.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# The driver ENFORCES the six exact identities and exits non-zero if any fails, AFTER writing the
# CSV -- so a failed task still leaves the data that makes the failure diagnosable, and `set -e`
# above turns that into a visibly failed slurm task rather than a quietly wrong figure.
julia --project=. benchmarks/run_hs_structure_factor.jl "$L" "$T" "$DT" "$SCHEME"
