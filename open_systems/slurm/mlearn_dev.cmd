#!/bin/bash
# Test V of CRITERIA.md (mlearn confirmation of RWV) on one skx-dev node. Usage (repository root):
#   sbatch -A <allocation> open_systems/slurm/mlearn_dev.cmd ; then: cd open_systems && python3 12_mlearn.py --score
#SBATCH -J cdopt_mlearn
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -o open_systems/work/slurm_%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd open_systems
python3 12_mlearn.py ${ELEMENTS:-Cu Ge Li Mo Ni Si} --jobs 24
