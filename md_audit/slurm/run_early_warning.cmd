#!/bin/bash
# Step 15 (early warning): descriptors of the pre-event frames of the base model's 40 seeds at 5000 K, gamma, analysis.
# Run the prep stage first (login node is fine):  python3 md_audit/15_early_warning.py prep
# Usage (repository root): sbatch -A <allocation> md_audit/slurm/run_early_warning.cmd
#SBATCH -J cdopt_ew
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 02:00:00
#SBATCH -p spr
#SBATCH -o md_audit/work/slurm_early_warning_%j.out
set -euo pipefail
module load intel/24.0 impi/21.11 python/3.12.11
source env.sh                         # CHIMES_LSQ (written by install/install_chimes.sh)
export MPIRUN="ibrun -n 56" OMP_NUM_THREADS=1
python3 md_audit/15_early_warning.py descriptors
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
python3 md_audit/15_early_warning.py gamma
python3 md_audit/15_early_warning.py analysis
echo done
