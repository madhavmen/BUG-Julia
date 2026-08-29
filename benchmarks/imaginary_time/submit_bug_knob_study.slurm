#!/bin/bash
# The rSVD knob study (A) and the truncation cross (B), as an ARRAY over (study x model).
#
#   sbatch benchmarks/imaginary_time/submit_bug_knob_study.slurm
#
# ⛔ THIS JOB EXISTS BECAUSE STUDY A IS A WALL-CLOCK STUDY AND WALL CLOCK CANNOT BE MEASURED ON THE
# LAPTOP. Its question is literally "which knob moves the CLOCK" -- `dover`, `growth`,
# `comp_ratio`, `share_ht`, `parallel`. MEASURED there: one rSVD sweep read 94.5 s against the
# exact arm's 2.87 s for IDENTICAL work (same 135 Krylov applications, same rank), purely from a
# concurrent job; and an L=16 arm took 597.8 s where an identical launch took 316.7 s, bit-for-bit
# the same `err`/`matvec`/`chi`. A contaminated timing column is WORSE than a missing one, because
# it looks like data. Accuracy, `matvec` and `chi` are contention-immune; the clock is not.
#
# ⛔ THE TWO STUDIES ARE SCORED AGAINST DIFFERENT THINGS AND MIXING THEM UP VOIDS BOTH.
#   A  asks "does the RANDOMISED sketch reproduce the DETERMINISTIC one, more cheaply?" -- a
#      question about the SKETCH, so its control is `exact = true` AT THE SAME SETTINGS, not nature.
#      A row differing from exact CBE by 1e-13 has reproduced it; whether both are 1e-6 from the
#      true answer is a different question.
#   B  asks "how much accuracy does each truncation cost?" -- a question about NATURE, so it is
#      scored against the model's EXACT reference (Bethe for the chain). Scoring it against another
#      of our runs would only measure how far two truncations sit apart.
#
# ⚠ THE CHAIN IS THE IMAGINARY-TIME TARGET AND IS ALREADY CONFIGURED FOR IT: L = 16, SU(2),
# `dimer_state` start, Bethe reference, 12 log steps to tau = 30, D = 64. No edit needed.
# ⛔ `cylinder` HAS `eref = NaN` BY CONSTRUCTION -- the QMC number is a BULK density recovered by a
# multi-length fit, far too expensive to redo inside a knob scan. So study B CANNOT score the
# cylinder against nature and only its study-A rows mean anything. That is why the array below runs
# B on the chain only.
#
#SBATCH --job-name=knobstudy
#SBATCH --partition=long
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=/home/madhav.menon/benchmarks/knob_%A_%a.log
#SBATCH --error=/home/madhav.menon/benchmarks/knob_%A_%a.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators/BUG-Julia
cd "$ROOT"

# task 0: A chain   task 1: B chain   task 2: A cylinder   task 3: A dissipative   task 4: B dissipative
#
# ⚠ `dissipative` IS MODEL 3 OF THE CAMPAIGN -- it is the BOUNDARY-DRIVEN XX CHAIN
# (`fused_boundary_lindblad`), the same generator `dissipative_xx.jl` validates and runs. The name
# predates the three-model scoping; do not read it as a second, unrelated open model.
#
# ✅ AND STUDY B *IS* QUOTABLE HERE, UNLIKE THE CYLINDER. `build("dissipative")` carries a genuine
# `eref` -- `xx_boundary_sz`, the analytic Lyapunov solution, exact at every `t` and certified to
# 7.2e-16 against a dense Liouvillian at N = 2..5. The cylinder's `eref` is `NaN` by construction
# (its QMC anchor is a bulk density from a multi-length fit, far too expensive inside a knob scan),
# which is the ONLY reason B is not run there.
#
# ⛔ THIS IS ALSO THE MODEL THE rSVD KNOBS EXIST FOR. `PLAN-open-systems.md` §7 puts rSVD last
# because the closed models' MPOs are too NARROW for a sketch to pay -- `npre/(d*chi)` never gets
# small. The fused Liouvillian is the first generator here wide enough (d = 4 plus a jump channel
# per boundary), so a knob scan that omitted it would omit the case rSVD was added for.
STUDIES=(A B A A B)
MODELS=(chain chain cylinder dissipative dissipative)
STUDY=${STUDIES[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# ⛔ ONE CSV PER TASK. Concurrent appends to a shared file interleave into rows that still PARSE,
# so the corruption reaches a figure instead of announcing itself.
export KNOB_OUT="$ROOT/benchmarks/results/bug_knob_${STUDY}_${MODEL}.csv"
rm -f "$KNOB_OUT"

echo "task ${SLURM_ARRAY_TASK_ID} -> study=$STUDY model=$MODEL out=$KNOB_OUT"

# ⛔ THREADS PINNED TO 1. `parallel` is one of the KNOBS study A scans, so the thread count must be
# fixed by the harness and not inherited from the node -- otherwise the knob and the environment
# move together and neither can be attributed.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ⚠ `share_ht = false` AND `share_ht = true` MUST AGREE TO ROUNDOFF -- it is a performance fix
# (`cbe_expand` used to rebuild `H*Theta` for each of its three sketch calls), not a knob. The
# driver checks this FIRST and reports it as a gate: if those two rows differ in the observable,
# every timing below them is uninterpretable.
julia --project=. benchmarks/imaginary_time/bug_knob_study.jl "$STUDY" "$MODEL"
