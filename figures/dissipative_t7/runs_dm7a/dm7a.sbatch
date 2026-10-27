#!/bin/bash
#SBATCH --job-name=dm7a
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dm7a/log/%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dm7a/log/%j.err
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit 3a12dca pkg /home/m/Madhav.Menon/bugjl/pkg/9ac3e1306f9a"
cd /home/m/Madhav.Menon/bugjl/snap/dm7a
for kv in E=7mem DR_MODEL=B1 DR_L=20 DR_U=-4 DR_JLINK=1 DR_KAPPA=2 DR_MODE=traj DR_SEED=1 DR_EVERY=0.25 DR_TOLB=1e-5 DR_TOLR=1e-12 DR_INT=bug DR_MAXDIM=512 DR_DT=0.05 DR_NTRAJ=2 DR_T=2.5 DR_TOLT=1e-5 DR_CKPT_SEC=1e9; do export "$kv"; done
for b in default none; do
  export DR_FUSION_BUDGET=$b DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dm7a/out/fusion_$b.json
  echo "$b start $(date +%T)"
  /usr/bin/time -v -o /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dm7a/log/fusion_$b.time timeout -k 60 18000 julia -t 1 --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/9ac3e1306f9a benchmarks/dissipative_run.jl > /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-27-dm7a/log/fusion_$b.log 2>&1
  echo "$b EXIT=$? $(date +%T)"
done
