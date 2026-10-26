#!/bin/bash
#SBATCH --job-name=dr6e
#SBATCH --partition=small,cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00
#SBATCH --nodelist=th-cl-dell09
#SBATCH --constraint=x86-64-v4&intel
#SBATCH --exclude=cip-cl-compute5,cip-cl-compute6,cip-cl-compute7,cip-cl-compute8,cip-cl-compute9,cip-cl-computea,cip-cl-computeb,cip-cl-computec,th-cl-dell03
#SBATCH --output=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr6e/log/%x_%j.out
#SBATCH --error=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr6e/log/%x_%j.err
# Task 6 A1 time-to-accuracy: every arm in sequence in this one job, so all share one node and core
set -u
export JULIA_DEPOT_PATH=/scratch/m/Madhav.Menon/rsvd_adaptive/depot_v3:$HOME/.julia
export JULIA_CPU_TARGET="x86-64-v3"
export OMP_NUM_THREADS=1
echo "== node $(hostname) job $SLURM_JOB_ID commit 4a76d77 $(grep -m1 'model name' /proc/cpuinfo)"
cd /home/m/Madhav.Menon/bugjl/snap/dr6e
n=0
while read -r line; do
  n=$((n+1))
  [ -s /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr6e/out/$n.json ] && { echo "skip $n (done)"; continue; }
  echo "== arm $n: $line"
  ( for kv in $line; do export "$kv"; done
    export DR_OUT=/scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr6e/out/$n.json DR_JIT_WARMUP=2
    timeout -k 60 14000 julia -t 1 --startup-file=no --project=/home/m/Madhav.Menon/bugjl/pkg/974e02774c69 benchmarks/dissipative_run.jl < /dev/null )
  echo "EXIT-RUN $n=$?"
done < /scratch/m/Madhav.Menon/rsvd_adaptive/2026-10-25-dr6e/configs.txt
