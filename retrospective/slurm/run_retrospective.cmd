#!/bin/bash
# Example SLURM driver (TACC Stampede3) for the retrospective study, Figs. 1-5.
# Usage: sbatch retrospective/slurm/run_retrospective.cmd   (from the repository root)
#SBATCH -J cdopt_retro
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 04:00:00
#SBATCH -p skx
#SBATCH -o retrospective/work/slurm_%j.out
set -euo pipefail
module load intel/24.0 impi/21.11 python/3.12.11
source env.sh                      # written by install/install_chimes.sh (sets CHIMES_LSQ, ...)
export MPIRUN="ibrun -n 48" OMP_NUM_THREADS=48
mkdir -p retrospective/work
python3 retrospective/01_descriptors.py
python3 retrospective/02_cluster.py
python3 retrospective/03_errors.py
python3 retrospective/04_gamma_ensemble.py
python3 retrospective/05_controls.py 5
python3 retrospective/06_figures.py
