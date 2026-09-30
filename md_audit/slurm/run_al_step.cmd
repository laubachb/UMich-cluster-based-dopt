#!/bin/bash
# One active-learning iteration (08_al_step.py) for the seed split and the gamma-balanced split (10_balanced_split.py),
# saving every refit model's coefficients for the MD stability study (11_al_md_setup.py).
# Usage (repository root): sbatch md_audit/slurm/run_al_step.cmd
#SBATCH -J cdopt_alstep
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx
#SBATCH -o md_audit/work/slurm_alstep_%j.out
set -euo pipefail
module load python/3.12.11
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24
python3 md_audit/10_balanced_split.py
python3 md_audit/08_al_step.py --split seed --save-coefs > md_audit/work/al_step_seed.log 2>&1 &
python3 md_audit/08_al_step.py --split balanced --save-coefs > md_audit/work/al_step_balanced.log 2>&1 &
wait
echo done
