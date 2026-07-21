#!/bin/bash
#SBATCH --job-name=bugpy
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# NOTE: no --time on purpose; inherit the partition maximum.
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/madhav.menon/miniconda3/envs/aliceenv/bin/python "$@"
