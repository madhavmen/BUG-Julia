#!/bin/bash
#SBATCH --job-name=qtci_full_tests
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --output=/home/madhav.menon/benchmarks/qtci_full_tests_%j.log
#SBATCH --error=/home/madhav.menon/benchmarks/qtci_full_tests_%j.err

set -euo pipefail

ROOT=/home/madhav.menon/qtci_time_integrators
cd "$ROOT"

julia --project=. tests/runtests.jl
