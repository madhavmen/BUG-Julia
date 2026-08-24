#!/bin/bash
# The Heisenberg tolerance study at L = 20, dt = 0.05 -- docs/PLAN-tolerance-control.md
#
#   sbatch benchmarks/submit_heisenberg_tolerance.slurm             # all four phases, serialised
#   sbatch --array=2 benchmarks/submit_heisenberg_tolerance.slurm   # phase `split` (6A) alone
#
# ORDER IS DELIBERATE AND THE ARRAY IS SERIALISED (%1).
#
#   0  tau    0a/0b  err and chi vs tau_trunc      -- the knee. Run this FIRST: it sets the
#                                                     tau_trunc every later phase uses.
#   1  dt     0c/0d  err vs dt per tau_trunc       -- the calibration curve tau_trunc*(dt)
#   2  split  6A     closing truncation OFF        -- the LARGEST ranks in the study
#   3  grid   6B     root-out x half-sweep, 8x8    -- 64 arms, the longest task by count
#
# ⛔ %1 IS NOT POLITENESS. `seconds` is in the CSV, and two of these sharing a node makes that
# column a property of the machine's load. The ERROR and KRYLOV columns are contention-immune and
# the figures use those -- but a contaminated timing column is worse than a missing one, because
# it looks like data. One job at a time, one thread, both pinned below.
#
# ⚠ PHASE `split` RUNS WITH NO CLOSING TRUNCATION AND NO ROOT TRUNCATION. `split_cutoff` is the
# only rank control anywhere in the step, so at the loose end of the scan the bond dimension is
# free to ratchet up every step. It gets the biggest memory allocation and the longest wall time,
# and its loosest cutoff runs first so a `chi` blow-up is visible in the log before the tight
# cutoffs have burned hours. If it OOMs, lower `t_max` rather than capping `maxdim` -- a cap would
# put a third truncation back into the step and defeat the experiment.

#SBATCH --job-name=heis_tol
#SBATCH --array=0-3%1
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=benchmarks/results/heis_tol_%a.out
#SBATCH --error=benchmarks/results/heis_tol_%a.err

set -euo pipefail

# Julia's thread count and BLAS's are separate settings and BOTH have to be pinned. Leaving BLAS
# free is the usual way a "single-threaded" run quietly uses every core on large matrices and
# none on small ones -- a size-dependent bias that reads as a parameter effect.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

PHASES=(tau dt split grid)
I=${SLURM_ARRAY_TASK_ID:-0}
PHASE=${PHASES[$I]}

echo "task $I: phase=$PHASE  L=20  dt=0.05  on $(hostname)  $(date)"
echo "threads: JULIA=$JULIA_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"

# T = 2.5 (50 steps) is the PLAN'S STARTING VALUE, not a settled one -- Néel entanglement grows
# linearly in t, so the affordable window is decided by the chi(t) the `tau` phase measures.
# Raise it here once that number exists.
julia --project=. benchmarks/heisenberg_tolerance.jl \
      phase="$PHASE" L=20 dt=0.05 t_max=2.5

echo "done $(date)"
