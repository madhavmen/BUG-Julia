#!/bin/bash
#SBATCH --job-name=dk7a
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=14:00:00
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dk7a/log/%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dk7a/log/%j.err
# USER 10-27: tune the recycled local Krylov tolerance (tol_krylov). dr6ta's A2 point (L = 24, 6 exc.,
# Gamma = 5, maxdim 512, Jt = 4, tol_basis 1e-8): BUG dt 0.1 x tol_krylov {none = 1e-8, 1e-7 .. 1e-4},
# plus the default-tolerance dt 0.05 reference. One node: the warm start state is CPU-dependent.
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit ef6c30d pkg /home/m/Madhav.Menon/bugjl/pkg/399af190c4f2"
lscpu | grep -E "Model name"
cd /home/m/Madhav.Menon/bugjl/snap/dk7a
for kv in E=7k DR_MODEL=A2 DR_L=24 DR_NEXC=6 DR_START=block DR_JZ=1 DR_GAMMA=5 DR_WARM=0.2 DR_MODE=no_jump DR_T=4 DR_MAXDIM=512 DR_EVERY=0.4 DR_TOLB=1e-8 DR_TOLR=1e-12 DR_TOLT=1e-9 DR_INT=bug DR_JIT_WARMUP=1 DR_CKPT_SEC=1e9; do export "$kv"; done
for run in none:0.1 1e-7:0.1 1e-6:0.1 1e-5:0.1 1e-4:0.1 none:0.05; do
  IFS=: read k dt <<< "$run"; tag=k${k}_dt${dt}
  export DR_TOLK=$k DR_DT=$dt DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dk7a/out/$tag.json
  echo "$tag start $(date +%T)"
  /usr/bin/time -v -o /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dk7a/log/$tag.time timeout -k 60 14400 julia -t 1 --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/399af190c4f2 benchmarks/dissipative_run.jl > /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dk7a/log/$tag.log 2>&1
  echo "$tag EXIT=$? $(date +%T)"
done
