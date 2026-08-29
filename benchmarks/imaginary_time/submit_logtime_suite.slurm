#!/bin/bash
# The three-model logarithmic-grid suite, as an ARRAY over models.
#
# WHY THIS RUNS ON THE CLUSTER. Locally the suite runs only in `smoke` mode -- shrunk to L <= 12,
# where the question is whether the WIRING is right (references, MPO layouts, observable
# conventions, the bulk fit) and an exact reference decides it in minutes. Everything below is
# either too large for a laptop or is a wall-clock number, and neither belongs there.
#
#   sbatch benchmarks/imaginary_time/submit_logtime_suite.slurm
#   cat .../logtime_suite_chain.csv > logtime_suite.csv          # keep ONE header
#   tail -n +2 .../logtime_suite_cylinder.csv >> logtime_suite.csv
#   tail -n +2 .../logtime_suite_neel.csv     >> logtime_suite.csv
#   python3 benchmarks/imaginary_time/plot_logtime.py
#
# ⛔ EVERY MODEL IS SCORED AGAINST A PUBLISHED NUMBER, NOT AGAINST ANOTHER ARM OF OURS:
#   chain     Bethe ansatz, exact
#   cylinder  Loop QMC, arXiv:1705.03222 Table 1   (e_bulk = -0.683282(2) at Ly = 4)
#   neel      Barmettler et al., arXiv:0810.4845   (exact J_0(2Jt)/2 at Delta = 0)
#
#SBATCH --job-name=logtime
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=/home/madhav.menon/benchmarks/logtime_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/logtime_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

# ⛔ ONE MODEL PER TASK, AND THAT COSTS A REPEATED JIT ON PURPOSE. SU(2) specialisation is roughly
# 165-350 s per distinct code path PER PROCESS -- MEASURED again in the local smoke run, where the
# first arm took 293 s and the next two took 5.2 s and 5.5 s -- so splitting three ways pays it
# three times. That is minutes against multi-hour tasks, and it buys two things worth more:
# FAILURE ISOLATION (these runs have repeatedly died part-way, and a death in the cylinder must
# not cost the chain and the quench too) and parallelism across the array.
MODELS=(chain cylinder neel)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# ⚠ 2026-08-28: BOTH `chain` AND `cylinder` NOW RUN TWO SYMMETRIES (`:SU2` then `:U1`), so each of
# those tasks is roughly TWICE the work it used to be. This became possible only now -- `cbe_bug`
# was UNUSABLE under U(1) until two itag collisions were fixed (`fusion_basis` dropping its `tag`;
# `_dimer_state_projected` not restoring the bond tag, `sweep.jl:121`), which is why every earlier
# U(1) row in this suite read `E = NaN, mv = 0, chi = 0`.
#
# ✅ THE ORDERING IS THE SAFETY NET: `:SU2` runs FIRST, so if a task hits the 48 h wall it loses
# only the U(1) rows, and the established SU(2) result is already on disk. Drop `:U1` by passing
# `syms = (:SU2,)` if a task turns out to be tight.
#
# ⛔ DO NOT COMPARE `:SU2` AND `:U1` ROWS AT EQUAL `D`. `maxdim` counts MULTIPLETS under SU(2) and
# STATES otherwise, so equal `D` is not equal variational space. MEASURED on the cylinder: `tdvp2`
# at `:U1, maxdim=128` gave `err = 2.9e-02` against `2.76e-03` at `:SU2, maxdim=70` -- the U(1) run
# was UNDER-RESOLVED, not worse-behaved.

# ⛔ ONE CSV PER TASK. Three concurrent tasks appending to a shared file interleave their writes
# into something that still PARSES -- rows from different models spliced mid-line -- so the
# corruption would survive into a figure rather than announcing itself.
export LOGTIME_OUT="$ROOT/benchmarks/results/logtime_suite_${MODEL}.csv"
rm -f "$LOGTIME_OUT"

echo "task ${SLURM_ARRAY_TASK_ID} -> model=$MODEL out=$LOGTIME_OUT"

# ⛔ THREADS PINNED TO 1, matching the other submit scripts here. The suite reports wall time as a
# first-class column, and a run whose thread count differs from the campaign's is not comparable
# with it even when every other knob matches.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ⚠ THE CYLINDER'S BOND DIMENSION IS THE KNOB MOST LIKELY TO NEED RAISING, and it is the one that
# decides whether the QMC comparison means anything. `model_cylinder` defaults to `D = 96`
# MULTIPLETS, which is generous for a chain and thin for a four-leg cylinder; if the fitted
# `e_bulk` sits above -0.683282 by more than the fit residual, the deficit is TRUNCATION, not the
# integrator, and the arms cannot be separated until D is raised. Edit the default in
# `model_cylinder` and resubmit rather than adding a flag that lets the two runs be confused.
#
# ⚠ AND THE FIT RESIDUAL IS THE GATE. `model_cylinder` prints "NOT LINEAR, do not quote" when the
# three lengths are not yet in the asymptotic regime. A row carrying that warning says the
# cylinder is too short for the extrapolation, NOT that the integrator is wrong.
julia --project=. benchmarks/imaginary_time/logtime_suite.jl "$MODEL"
