#!/bin/bash
# The L=18 XX (Delta=0) domain-wall chain campaign -- the error matrix across Krylov settings.
#
#   CHI=64  sbatch --time=12:00:00 benchmarks/submit_heisenberg_tolerance.slurm
#   CHI=128 sbatch --time=96:00:00 benchmarks/submit_heisenberg_tolerance.slurm
#   CHI=256 -- DOES NOT FIT IN ONE SLOT PER GRID. See "SPLITTING A GRID BY ROW" below.
#
#   CHI=64 sbatch --array=2 ...      # one configuration (here: grid, sweep m3) alone
#
# Array index -> configuration, 5 per cap:
#   0  tau    the THREE-SCHEME tolerance scan (tdvp2 / tdvp_cbe1s / bug) -- 5 taus x 3 = 15 arms
#   1  grid mdef   \
#   2  grid m3      |  bug only, 5 taus x 5 splits = 25 arms each. `mdef` is the package default
#   3  grid m0      |  (what a user actually gets); m0/m2/m3 pin krylov_basis.
#   4  grid m2     /
#
# ⛔ THIS ARRAY IS **NOT** SERIALISED, AND THAT IS A DELIBERATE REVERSAL OF THE OLD `%1`.
# The deliverables of the L=18 stage are `err_prof`, `maxbond` and `krylov` -- all three are
# deterministic and contention-immune (reproduced bit-identically across independent launches).
# `seconds` is NOT, and running five tasks per node contaminates it. That is an accepted loss here
# because no L=18 figure plots it.
# ⛔ THE L=50 STAGE IS THE OPPOSITE: its spec asks for WALL TIME, so it must run `--array=...%1`
#    with `--exclusive`, one job at a time, and must NOT reuse this file's parallel settings.
#
# WALL-TIME BUDGET, MEASURED not guessed. The completed chi=64 `mdef` grid ran 49 arms in 15.8 h
# at ~796 s/arm median, so a 25-arm grid is ~5.5 h of compute. Per-arm cost goes as ~chi^3 in the
# local solves, hence ~8x at chi=128 (~44 h/grid) and ~64x at chi=256 (~350 h/grid).
#
# SPLITTING A GRID BY ROW (needed at chi=256). `knobtag()` carries L, dt, T, maxdim and maxiter but
# NOT tau/split, so two tasks running different rows of the same grid would write the SAME filename
# and one would silently overwrite the other. Pass a distinct `tag` per row and concatenate:
#
#   for r in 1e-4 1e-5 1e-6 1e-7 1e-8; do
#     CHI=256 ROW=$r sbatch --array=1-4 --time=72:00:00 benchmarks/submit_heisenberg_tolerance.slurm
#   done
#   # then, keeping ONE header:
#   awk 'FNR==1 && NR!=1 {next} {print}' heis_grid_m3_L18_*_D256_mauto_r*.csv > heis_grid_m3_..._D256_mauto.csv
#
# THE OLD L=20 NEEL STUDY (docs/PLAN-tolerance-control.md) is one parameter change away and is not
# deleted, only retargeted: `delta=1.0 init=neel L=20 t_max=2.5` plus phases `dt`/`split`.
# ⚠ At delta=1 there is NO closed-form real-time reference and sparse ED caps L~20; the Delta=0
# campaign below is scored against the EXACT analytic free-fermion profile, valid at any L.

#SBATCH --job-name=heis_chain
#SBATCH --array=0-4
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/results/heis_chain_%a.out
#SBATCH --error=benchmarks/results/heis_chain_%a.err

set -euo pipefail

# Julia's thread count and BLAS's are separate settings and BOTH have to be pinned. Leaving BLAS
# free is the usual way a "single-threaded" run quietly uses every core on large matrices and
# none on small ones -- a size-dependent bias that reads as a parameter effect.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

CHI=${CHI:-64}
ROW=${ROW:-}                 # set to ONE tau to run a single grid row (see above)
I=${SLURM_ARRAY_TASK_ID:-0}
CONFIG=(tau mdef m3 m0 m2)
WHICH=${CONFIG[$I]}

# ⛔ maxiter=0 is the DRIVER'S DERIVED DEPTH and it is deliberately kept here. The depth study
# showed the derived depth over-provisions (error saturates at maxiter=4) -- but it is the SHIPPED
# behaviour and the spec asks for the error matrix at it. Changing it would make these figures
# incomparable with the chi=64 data already collected. Vary depth in `phase=krylov` instead.
COMMON="delta=0.0 init=domainwall L=18 t_max=20 dt=0.05 sample_every=0.5 maxiter=0 maxdim=$CHI"
[ -n "$ROW" ] && COMMON="$COMMON taus=$ROW tag=r$ROW"

echo "task $I: $WHICH  chi=$CHI  row=${ROW:-all}  on $(hostname)  $(date)"
echo "threads: JULIA=$JULIA_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"

if [ "$WHICH" = tau ]; then
  julia --project=. benchmarks/heisenberg_tolerance.jl phase=tau $COMMON
else
  julia --project=. benchmarks/heisenberg_tolerance.jl phase=grid $COMMON grid_sweeps="$WHICH"
fi

echo "done $(date)"
