#!/bin/bash
#SBATCH --job-name=dp7b
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7b/log/%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7b/log/%j.err
# Task 7 parallel subset, large-frame point: B2 L = 32 (dr7w's point, D_stage ~550-650), tol_trunc 1e-7, to
# Jt = 2. BUG serial (-t 1), half_sweeps (-t 2), full K/L/S (-t 3) at dt 0.1, plus serial dt 0.05 as the
# time-error yardstick; sequentially on this node, BLAS 1 thread, checkpoints off.
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit c75b983 pkg /home/m/Madhav.Menon/bugjl/pkg/3baec21c827b cpus $SLURM_CPUS_PER_TASK"
lscpu | grep -E "Model name|^CPU\(s\)"
cd /home/m/Madhav.Menon/bugjl/snap/dp7b
for kv in E=7par DR_MODEL=B2 DR_L=32 DR_NEXC=4 DR_START=alt DR_JZ=1 DR_GAMMA=2 DR_MAXDIM=512 DR_TOLB=1e-8 DR_TOLR=1e-12 DR_MODE=no_jump DR_WARM=0.2 DR_EVERY=0.25 DR_INT=bug DR_T=2 DR_TOLT=1e-7 DR_JIT_WARMUP=1 DR_CKPT_SEC=1e9; do export "$kv"; done
for run in serial:1:0.1 half_sweeps:2:0.1 full:3:0.1 serial:1:0.05; do
  IFS=: read p nt dt <<< "$run"; tag=${p}_dt${dt}
  export DR_PARALLEL=$p DR_DT=$dt DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7b/out/$tag.json
  echo "$tag start $(date +%T)"
  /usr/bin/time -v -o /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7b/log/$tag.time timeout -k 60 7200 julia -t $nt --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/3baec21c827b benchmarks/dissipative_run.jl > /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dp7b/log/$tag.log 2>&1
  echo "$tag EXIT=$? $(date +%T)"
done
