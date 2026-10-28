#!/bin/bash
#SBATCH --job-name=dk7b
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=9G
#SBATCH --time=20:00:00
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-28-dk7b/log/%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-28-dk7b/log/%j.err
# dk7a (17526941) found BUG dt 0.1 at tol_krylov 1e-6..1e-4 within 1e-7 of the default dt 0.05 reference,
# the default dt 0.1 itself 4.8e-6 off. Which is nearer the truth: the default dt 0.05 again (t0 check
# against dk7a), 1e-6 at dt 0.05, and the default at dt 0.025 as a reference for the reference.
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit ef6c30d pkg /home/m/Madhav.Menon/bugjl/pkg/399af190c4f2"
lscpu | grep -E "Model name"
cd /home/m/Madhav.Menon/bugjl/snap/dk7b
for kv in E=7k DR_MODEL=A2 DR_L=24 DR_NEXC=6 DR_START=block DR_JZ=1 DR_GAMMA=5 DR_WARM=0.2 DR_MODE=no_jump DR_T=4 DR_MAXDIM=512 DR_EVERY=0.4 DR_TOLB=1e-8 DR_TOLR=1e-12 DR_TOLT=1e-9 DR_INT=bug DR_JIT_WARMUP=1 DR_CKPT_SEC=1e9; do export "$kv"; done
for run in none:0.05 1e-6:0.05 none:0.025; do
  IFS=: read k dt <<< "$run"; tag=k${k}_dt${dt}
  export DR_TOLK=$k DR_DT=$dt DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-28-dk7b/out/$tag.json
  echo "$tag start $(date +%T)"
  /usr/bin/time -v -o /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-28-dk7b/log/$tag.time timeout -k 60 32000 julia -t 1 --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/399af190c4f2 benchmarks/dissipative_run.jl > /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-28-dk7b/log/$tag.log 2>&1
  echo "$tag EXIT=$? $(date +%T)"
done
