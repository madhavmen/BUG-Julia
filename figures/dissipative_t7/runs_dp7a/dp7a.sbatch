#!/bin/bash
#SBATCH --job-name=dp7a
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --time=09:00:00
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7a/log/%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7a/log/%j.err
# Task 7 parallel subset: BUG serial (-t 1), half_sweeps (-t 2), full K/L/S (-t 3) on A2 L = 48 (dr7b's
# point, tol_trunc 1e-7), dt ladder 0.2/0.1/0.05, sequentially on this node. BLAS 1 thread in every arm:
# the parallelism measured is the solver's tasks. Checkpoints off (no I/O in the walls).
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit 048eca6 pkg /home/m/Madhav.Menon/bugjl/pkg/3baec21c827b cpus $SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|^CPU\(s\)"
cd /home/m/Madhav.Menon/bugjl/snap/dp7a
for kv in E=7par DR_MODEL=A2 DR_L=48 DR_NEXC=2 DR_START=block DR_JZ=1 DR_GAMMA=1 DR_WARM=0.2 DR_MODE=no_jump DR_MAXDIM=256 DR_EVERY=1.0 DR_T=20 DR_TOLB=1e-8 DR_TOLR=1e-12 DR_INT=bug DR_TOLT=1e-7 DR_JIT_WARMUP=1 DR_CKPT_SEC=1e9; do export "$kv"; done
for dt in 0.2 0.1 0.05; do
  for arm in serial:1 half_sweeps:2 full:3; do
    p=${arm%%:*}; nt=${arm##*:}; tag=${p}_dt${dt}
    export DR_PARALLEL=$p DR_DT=$dt DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7a/out/$tag.json
    echo "$tag start $(date +%T)"
    /usr/bin/time -v -o /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7a/log/$tag.time timeout -k 60 10800 julia -t $nt --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/3baec21c827b benchmarks/dissipative_run.jl > /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7a/log/$tag.log 2>&1
    echo "$tag EXIT=$? $(date +%T)"
  done
done
