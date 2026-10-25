#!/bin/bash
#SBATCH --job-name=dr5e
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=1-24%50
#SBATCH --constraint=x86-64-v3&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr5e/log/%a_%A.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr5e/log/%a_%A.err
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr5e/configs.txt)
echo "== node $(hostname) job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID commit bf4cf5f: $line"
for kv in $line; do export "$kv"; done
export DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr5e/out/${SLURM_ARRAY_TASK_ID}.json
cd /home/m/Madhav.Menon/bugjl/snap/dr5e
timeout -k 60 28620 julia -t 1 --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/3f019d2fe0ec benchmarks/dissipative_run.jl
echo "EXIT-RUN=$?"
