#!/bin/bash
# RSVD parameter scan, L = 16, OAT and XX -- the run whose WALL TIME is trustworthy.
#
# ⛔ ONE JOB AT A TIME AND ONE THREAD, and neither is fussiness. The scan's whole purpose is to
# compare seconds between arms; two jobs sharing a node, or BLAS spreading across cores by a
# schedule that depends on matrix size, makes the seconds a property of the machine's load
# rather than of the parameter being scanned. An array with %1 serialises the tasks, and the
# thread pinning below removes the second source.
#
# The ERROR columns do not need any of this and can be produced locally; it is only the `secs`
# column that is invalidated by a shared machine. That is why `phase=0` is worth running
# interactively and this file exists for the rest.
#
# The array is (model x phase), serialised:
#     0 xx   phase 0    saturation gate      -- is the sketch lossy where rank GROWS?
#     1 xx   phase 1    noise floor          -- seed spread = the resolution of everything
#     2 xx   phase 2    budget scan
#     3 oat  phase 0
#     4 oat  phase 1
#     5 oat  phase 2
#
#   sbatch benchmarks/submit_rsvd_scan.slurm
#   sbatch --array=0-2%1 benchmarks/submit_rsvd_scan.slurm      # XX only

#SBATCH --job-name=rsvd_scan
#SBATCH --array=0-5%1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/results/rsvd_scan_%a.out

set -euo pipefail

# Single-threaded everywhere. Julia's own thread count and BLAS's are separate settings and BOTH
# have to be pinned -- leaving BLAS free is the usual way a "single-threaded" timing run quietly
# uses every core on large matrices and none on small ones, which is exactly the size-dependent
# bias a budget scan would misread as a parameter effect.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

MODELS=(xx  xx  xx  oat oat oat)
PHASES=(0   1   2   0   1   2)
I=${SLURM_ARRAY_TASK_ID:-0}
MODEL=${MODELS[$I]}
PHASE=${PHASES[$I]}

echo "task $I: model=$MODEL phase=$PHASE  on $(hostname)  $(date)"
echo "threads: JULIA=$JULIA_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"

# `t_max` is left OFF so each model uses its own default -- π for OAT, which has a 4π revival,
# and 6.0 (units of 1/J) for XX, which has no period and is instead bounded by the front reaching
# the wall at L/(2J) = 8. Forcing one shared time on both is what the π parametrisation used to
# do, and for XX it was simply the wrong scale.
julia --project=. benchmarks/rsvd_param_scan.jl \
      model="$MODEL" phase="$PHASE" L=16 dt=0.05

echo "done $(date)"
