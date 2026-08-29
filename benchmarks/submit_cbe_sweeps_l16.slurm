#!/bin/bash
# The L = 16 CBE-sweep campaign, as an ARRAY over (phase, scheme).
#
# WHY AN ARRAY AND NOT ONE JOB. The phases are independent and the per-scheme runs inside a phase
# are too, so one job would serialise ~13 independent runs behind each other and put the whole
# campaign on the critical path of the slowest one. It would also make every wall-clock number a
# measurement of a machine running twelve other things.
#
#   ⛔ ONE SCHEME PER TASK IS THE POINT. `elapsed` is a headline column -- the comparison is
#   cost-to-accuracy between integrators -- so two schemes must never time each other. Each task
#   gets its own node allocation and its own CSV; `plot_cbe_sweeps_l16.py` reads the union.
#
# Task 0 is the ground-state phase (which does not use the scheme list -- it runs the two DMRG-CBE
# arms, RSVD against exact). Tasks 1-6 are imaginary time, 7-12 real time.
#
#   sbatch benchmarks/submit_cbe_sweeps_l16.slurm
#   python3 benchmarks/plot_cbe_sweeps_l16.py   # after the array finishes
#
#SBATCH --job-name=cbe_l16
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-16
#SBATCH --output=/home/madhav.menon/benchmarks/cbe_l16_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/cbe_l16_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

L=16
T=5.0
DT=0.05
BETA=20.0

# ⛔ ONE BUG SCHEME FOR THE METHOD COMPARISON: `bug_interleaved`, first in the list.
# `bug_decoupled` is gone -- it and `bug_interleaved` had been the SAME RUN since the K-step /
# `kaug` / `rexpand` removal on 2026-08-24, and two entries producing one curve read as agreement
# between methods in a plot.
# The trailing `bug_interleaved_*` entries are that SAME sweep at different KRYLOV DEPTHS -- a
# separate axis, kept in the array so the depth study runs alongside, never mixed into the method
# comparison. 8 schemes => array 0-16 (1 GSE task + 8 ITE + 8 RTE).
SCHEMES=(bug_interleaved tdvp_cbe1s tdvp2 jan_bug jan_bug_centre          bug_interleaved_m0 bug_interleaved_m3 bug_interleaved_tol4)
NS=${#SCHEMES[@]}

i=${SLURM_ARRAY_TASK_ID}
if [ "$i" -eq 0 ]; then
  PHASE=gse
  SUBSET=""
elif [ "$i" -le "$NS" ]; then
  PHASE=ite
  SUBSET=${SCHEMES[$((i - 1))]}
else
  PHASE=rte
  SUBSET=${SCHEMES[$((i - NS - 1))]}
fi

echo "task $i -> phase=$PHASE scheme=${SUBSET:-<all-dmrg-arms>}"

# ⛔ THREADS PINNED TO 1. The comparison is between integrators, and BLAS threading would let a
# scheme whose tensors happen to be shaped for it look faster for a reason that is not the
# algorithm. Pinned here rather than left to the node's default so the numbers are reproducible.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ -z "$SUBSET" ]; then
  julia --project=. benchmarks/cbe_sweeps_l16.jl "$PHASE" "$L" "$T" "$DT"
else
  julia --project=. benchmarks/cbe_sweeps_l16.jl "$PHASE" "$L" "$T" "$DT" "$SUBSET" "$BETA"
fi
