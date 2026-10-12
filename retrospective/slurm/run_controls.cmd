#!/bin/bash
# Step 5 (control arms) alone. Usage (repository root): sbatch -A <allocation> retrospective/slurm/run_controls.cmd
#SBATCH -J cdopt_ctl
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 01:00:00
#SBATCH -p spr
#SBATCH -o retrospective/work/slurm_controls_%j.out
set -euo pipefail
module load python/3.12.11
export OMP_NUM_THREADS=112 MKL_NUM_THREADS=112 OPENBLAS_NUM_THREADS=112
python3 retrospective/05_controls.py 5
