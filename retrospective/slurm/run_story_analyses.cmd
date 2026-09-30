#!/bin/bash
# Analyses for the revised figures (no new DFT or MD): raw/label/hybrid grades on the retrospective candidates
# (retrospective/07), hybrid grades on the MD audit (md_audit/14) and the clustering-sensitivity scan (retrospective/08).
# Usage (repository root): sbatch -A <allocation> retrospective/slurm/run_story_analyses.cmd
#SBATCH -J cdopt_story
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 02:00:00
#SBATCH -p spr
#SBATCH -o retrospective/work/slurm_story_%j.out
set -euo pipefail
module load python/3.12.11
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
python3 retrospective/07_raw_and_hybrid.py      > retrospective/work/07_raw_and_hybrid.log 2>&1 &
python3 md_audit/14_hybrid_md.py                > md_audit/work/14_hybrid_md.log 2>&1 &
python3 retrospective/08_cluster_sensitivity.py > retrospective/work/08_cluster_sensitivity.log 2>&1 &
wait
echo done
