#!/bin/bash
# 08_al_step.py on the gamma-balanced split with the single-state-point random control arm (random_5000K), saving models.
# Usage (repository root): sbatch md_audit/slurm/run_al_step_control.cmd
#SBATCH -J cdopt_alctl
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 00:30:00
#SBATCH -p skx-dev
#SBATCH -o md_audit/work/slurm_alctl_%j.out
set -euo pipefail
module load python/3.12.11
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48
python3 md_audit/08_al_step.py --split balanced --save-coefs --random-sp 5000K_2.0gcc > md_audit/work/al_step_balanced.log 2>&1
echo done
