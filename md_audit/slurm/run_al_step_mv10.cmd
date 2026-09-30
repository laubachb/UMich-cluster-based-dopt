#!/bin/bash
# 08_al_step.py on the gamma-balanced split with 10 MaxVol solutions per gamma rule (draws 0-4 identical to the
# 5-draw run) and the random_5000K control, saved separately (results/al_step_balanced_mv10.csv,
# work/al_models/balanced_mv10/) for the doubled MD stability runs. Usage (repository root):
#   sbatch md_audit/slurm/run_al_step_mv10.cmd
#SBATCH -J cdopt_mv10
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 00:40:00
#SBATCH -p skx-dev
#SBATCH -o md_audit/work/slurm_mv10_%j.out
set -euo pipefail
module load python/3.12.11
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48
python3 md_audit/08_al_step.py --split balanced --save-coefs --random-sp 5000K_2.0gcc --n-maxvol 10 --out-tag _mv10 \
    > md_audit/work/al_step_balanced_mv10.log 2>&1
echo done
