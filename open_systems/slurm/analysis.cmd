#!/bin/bash
# Arms A and B of CRITERIA.md on one spr node. Usage (from open_systems/):
#   sbatch -A <allocation> --export=ALL,NAME=si_gap18,ARM=shift slurm/analysis.cmd
#   sbatch -A <allocation> --export=ALL,NAME=sio2,ARM=al slurm/analysis.cmd
#SBATCH -J cdopt_open
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 08:00:00
#SBATCH -p spr
#SBATCH -o work/slurm_%x_%j.out
JOBS=${JOBS:-28}; T=$(( 112 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
if [ "${ARM:-shift}" = "shift" ]; then
    python3 02_shift.py ${NAME:-si_gap18} --jobs $JOBS
else
    OMP_NUM_THREADS=112 OPENBLAS_NUM_THREADS=112 MKL_NUM_THREADS=112 python3 03_al_retro.py ${NAME:-sio2}
fi
