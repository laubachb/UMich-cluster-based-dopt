#!/bin/bash
# Arms A/B of CRITERIA.md on one skx-dev node (48 cores, 2 h limit). Arm A is split by regime so each job fits.
# Usage (from open_systems/):
#   sbatch -A <allocation> --export=ALL,NAME=sio2,ARM=al slurm/analysis_dev.cmd
#   sbatch -A <allocation> --export=ALL,NAME=sio2,ARM=shift,REGIME="liquid quench" slurm/analysis_dev.cmd  (-> results/shift_sio2_liquid+quench.csv)
#SBATCH -J cdopt_dev
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -o work/slurm_%x_%j.out
if [ "${ARM:-shift}" = "shift" ]; then
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    python3 ${SCRIPT:-02_shift.py} ${NAME:-sio2} --jobs $JOBS --only $REGIME --tag _${REGIME// /+}
else
    export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
    python3 03_al_retro.py ${NAME:-sio2}
fi
