#!/bin/bash
# Example SLURM driver (TACC Stampede3): MD-audit analysis from the tracked data (no LAMMPS or VASP needed).
# Requires the retrospective work/ products (retrospective steps 01-02). Usage (repository root):
#   sbatch md_audit/slurm/run_analysis.cmd
#SBATCH -J cdopt_mdan
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx
#SBATCH -o md_audit/work/slurm_analysis_%j.out
set -euo pipefail
module load intel/24.0 impi/21.11 python/3.12.11
source env.sh
export MPIRUN="ibrun -n 48" OMP_NUM_THREADS=48
python3 md_audit/02_descriptors.py
python3 md_audit/03_gamma.py 10
python3 md_audit/06_errors.py
python3 md_audit/07_analysis.py
python3 md_audit/08_al_step.py
python3 md_audit/09_figures.py
