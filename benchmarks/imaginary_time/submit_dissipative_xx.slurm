#!/bin/bash
# The boundary-driven XX chain (MPDO / fused d=4 superket), as an ARRAY over the two runs that do
# not belong on a laptop.
#
#   sbatch benchmarks/imaginary_time/submit_dissipative_xx.slurm
#   python3 benchmarks/imaginary_time/plot_logtime.py benchmarks/results/logtime_xx_dissip.csv
#
# ⛔ WHAT IS *NOT* HERE, AND WHY. `calibrate` and `order` run at L = 8 against an ANALYTIC 8x8
# Lyapunov solve and are the actual correctness gate; they belong locally, where a failure is seen
# in minutes. Only `ness` (L = 16, 2000 steps per arm) and `grids` (a wall-clock column) are
# cluster work. Running the gate here too would just delay the answer by a queue wait.
#
# ⛔ THE REFERENCE IS EXACT AND COSTS O(N^3), SO THERE IS NO DENSE OBJECT ANYWHERE IN THIS JOB.
# Third quantization gives `dC/dt = XC + CX' + G` with `X = i h - D/2`; the steady state is one
# `lyap` call and `C(t) = C_ss + e^{Xt}(C_0 - C_ss)e^{X't}` is exact at EVERY `t`. A `4^16` dense
# Liouvillian would be 4 billion amplitudes and is never built -- see `tests/common/free_fermion.jl`
# (`xx_boundary_*`) and `benchmarks/imaginary_time/certify_xx_reference.jl`, which certifies the analytic solution
# against a small dense one at L = 2..5 to 7.2e-16.
#
# ⚠ THE VERDICT IS TAKEN AT FINITE `t`, NOT AGAINST THE `t -> inf` CLOSED FORM. The Liouvillian gap
# closes as `N^-3`, so a true steady state at N = 16 needs `t ~ 1300` (~66,000 steps at dt = 0.02).
# Charging the integrator for the residual physical transient would report relaxation as an
# integrator error. `ness` prints `exact(t)` beside the closed form for exactly this reason.
#
# ⛔ AND THIS MODEL IS THE ONE WHERE A LOG GRID CANNOT WORK -- that is its purpose in the suite, not
# a defect. Its observable is QUADRATIC in the modes, so the fastest angular frequency is
# `2 max|Im mu| ~ 4J` and the Nyquist step is `pi/4`, NOT `pi/2`. MEASURED at N=30: even `n = 2000`
# log steps overshoot by 79x, and `n = 12` by 8406x, against ~12,000 UNIFORM steps to reach t_max.
#
#SBATCH --job-name=dissipxx
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=/home/madhav.menon/benchmarks/dissipxx_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/dissipxx_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

# ⛔ ONE MODE PER TASK for FAILURE ISOLATION: `ness` is the long one, and a death there must not
# cost the grid study. It also pays the SU(2)/no-symmetry JIT twice on purpose -- minutes against
# multi-hour tasks.
MODES=(ness grids)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

# ⛔ ONE CSV PER TASK. Concurrent appends to a shared file interleave into something that still
# PARSES, so the corruption would reach a figure instead of announcing itself.
export DISSIP_OUT="$ROOT/benchmarks/results/logtime_xx_dissip_${MODE}.csv"
rm -f "$DISSIP_OUT"

echo "task ${SLURM_ARRAY_TASK_ID} -> mode=$MODE out=$DISSIP_OUT"

# ⛔ THREADS PINNED TO 1, matching every other submit script here. `grids` reports wall time as a
# first-class column and a run at a different thread count is not comparable with the campaign even
# when every other knob matches.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ⚠ `CAP` IS THE KNOB MOST LIKELY TO NEED RAISING, and it is the one that decides whether the
# current means anything. The superket's operator entanglement is what carries transport: a state
# truncated too hard keeps a plausible FLAT magnetisation profile and a perfect `Tr rho = 1` while
# its current is wrong. If `ness` misses the closed form by more than its tolerance, raise `CAP`
# and resubmit BEFORE concluding anything about an integrator.
export DISSIP_CAP=${DISSIP_CAP:-256}

julia --project=. benchmarks/imaginary_time/dissipative_xx.jl "$MODE"
