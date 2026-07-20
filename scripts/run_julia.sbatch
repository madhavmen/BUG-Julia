#!/bin/bash
#SBATCH --job-name=bugjl
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# NOTE: no --time on purpose; inherit the partition maximum.
set -euo pipefail
ROOT=/home/madhav.menon/BUG-Julia
$HOME/.juliaup/bin/julia --project="$ROOT" --threads=1 "$@"
