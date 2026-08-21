#!/bin/bash
# The LARGER-L one-axis-twisting campaign (arXiv:2208.10972 Fig. 2), as an ARRAY over (L, scheme).
#
# WHY THIS RUNS ON THE CLUSTER AND THE L = 10 RUN DID NOT. Not cost -- OAT is cheap, its exact
# Schmidt rank is only `min(n, L-n)+1` (9 at L = 16, 11 at L = 20) so no arm ever carries a large
# bond. It is that `elapsed` and `krylov` are headline columns here: the claim being tested is
# COST-TO-ACCURACY between integrators, and a shared laptop running three arms at once measures
# the machine, not the algorithm. L = 10 and L = 12 stay local because the question there is
# CORRECTNESS against the closed form, which survives contention; these do not.
#
#   ⛔ ONE SCHEME PER TASK. Same rule as `submit_cbe_sweeps_l16.slurm` -- two schemes must never
#   share a node, or each becomes the other's noise. Each task writes its own CSV and
#   `plot_oat.py` reads the union of everything matching the glob.
#
# WHY L = 16 AND L = 20 AND NOT ONE OF THEM. The L = 10 result is that `cbe_bug` self-truncates to
# the exact rank and beats both TDVP arms by ~3 orders against the closed form. Two sizes are the
# minimum that distinguishes "true at L = 10" from "true as L grows" -- with one extra size a
# flat comparison and a widening one look identical.
#
#   sbatch benchmarks/submit_oat_large.slurm
#   python3 benchmarks/plot_oat.py            # after the array finishes
#
#SBATCH --job-name=oat_large
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-5
#SBATCH --output=/home/madhav.menon/benchmarks/oat_large_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/oat_large_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

DT=0.05
# 10π -- the paper's own scaled window, and 2.5 collapse-and-revival periods rather than one.
# ⛔ THE LONGER WINDOW IS THE POINT, NOT A COSMETIC AXIS CHANGE. OAT revives every 4π, and a run
# that STOPS at the first revival cannot distinguish an integrator whose error is bounded from one
# whose error grows per period -- both look identical over a single cycle. 2.5 periods is the
# shortest window in which "the error comes back to the same place each revival" is falsifiable.
TMAX=31.416
EVERY=4
WARMUP=1             # obsolete since `kaug`; kept at 1 so nothing is graded away silently
SYM=Z2               # the paper's own symmetry (global spin flip), not U(1)
KAUG=true            # the union fix -- see USAGE.md 6b

SCHEMES=(tdvp_cbe1s tdvp2 cbe_bug)

# MAXDIMS BRACKET THE EXACT CEILING, which is `min(n, L-n)+1` and differs per L. Below it every
# scheme is rank-starved and the errors are O(1) -- those arms are in the figure to show WHERE the
# rank becomes sufficient, not to rank anything. 16 sits above both ceilings and is the control
# that says a bigger cap buys nothing once the exact rank is reached.
i=${SLURM_ARRAY_TASK_ID}
if [ "$i" -le 2 ]; then
  L=16
  MAXDIMS=2,4,9,16   # ceiling 9
  SUBSET=${SCHEMES[$i]}
else
  L=20
  MAXDIMS=2,4,11,16  # ceiling 11
  SUBSET=${SCHEMES[$((i - 3))]}
fi

echo "task $i -> L=$L scheme=$SUBSET maxdims=$MAXDIMS"

# ⛔ THREADS PINNED TO 1, for the same reason as the L = 16 CBE campaign: BLAS threading would let
# a scheme whose tensors happen to be shaped for it look faster for a reason that is not the
# algorithm. Pinned here rather than left to the node default so the numbers are reproducible.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Arg 10 (the Krylov depth) is deliberately OMITTED so the script derives it from ‖H_OAT‖·dt/2.
# ‖H‖ grows like L², so a depth fixed at the L = 10 value of 12 would quietly become the accuracy
# limiter here (1e-10 at L = 16, 3e-08 at L = 20). Each task's log prints the depth it chose and
# the resulting Lanczos bound -- check that line before trusting the arm.
julia --project=. benchmarks/oat_l10.jl \
  "$L" "$DT" "$TMAX" all "$MAXDIMS" "$EVERY" "$WARMUP" "$SYM" "$KAUG"
