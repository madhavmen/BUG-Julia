#!/bin/bash
#SBATCH --job-name=rsvd_nb
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/madhav.menon/BUG-Julia/notebooks/%x_%j.log
#SBATCH --error=/home/madhav.menon/BUG-Julia/notebooks/%x_%j.err

# Run one tutorial script on a compute node:
#     sbatch notebooks/submit_notebook.slurm 01_rsvd_lubich
set -uo pipefail
export PATH=/home/madhav.menon/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin:$PATH
export JULIA_NUM_THREADS=1
cd /home/madhav.menon/BUG-Julia
echo "=== $1 on $(hostname)  $(date -Is)"
julia --project=. notebooks/$1.jl
echo "=== exit $?  $(date -Is)"
