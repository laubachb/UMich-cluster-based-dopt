#!/bin/bash
# Tests R1 and R2 of CRITERIA.md (addendum of 2026-10-12) on one skx-dev node (48 cores, 2 h limit).
# Usage (from the repository root):
#   sbatch -A <allocation> --export=ALL,MODE=nitrogen open_systems/slurm/robustness_dev.cmd
#   sbatch -A <allocation> --export=ALL,MODE=open,NAMES="si_gap18 sio2" open_systems/slurm/robustness_dev.cmd
#   sbatch -A <allocation> --export=ALL,MODE=open,NAMES=water,CLUSTER=--cluster open_systems/slurm/robustness_dev.cmd
#   sbatch -A <allocation> --export=ALL,MODE=al open_systems/slurm/robustness_dev.cmd          (sio2 arm B)
# Addenda T/X: MODE=tda,NAMES="si_gap18 sio2 water" (features); MODE=tdan (nitrogen features + test); MODE=tpart,NAMES=water (partition test); MODE=ceil
# Addendum N (advances): MODE=nadv (nitrogen); MODE=adv,NAMES=water,METHODS=soft+cal (plus-separated: --export splits on commas)
# Addendum H (information gain): MODE=nig (nitrogen); MODE=ig,NAMES="si_gap18 sio2" or NAMES=water,CLUSTER=--cluster
# Descriptive addendum (RWV with state-point groups on the nitrogen MD): MODE=nigsp
# Addendum M (minimum over groups) and the leverage refits for test S:
#   sbatch -A <allocation> --export=ALL,MODE=nmin open_systems/slurm/robustness_dev.cmd        (nitrogen)
#   sbatch -A <allocation> --export=ALL,MODE=min,NAMES=sio2,ONLY="liquid quench",TAG=_a open_systems/slurm/robustness_dev.cmd
#SBATCH -J cdopt_rob
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -o open_systems/work/slurm_%x_%j.out
set -euo pipefail
module load python/3.12.11 2>/dev/null || true
case "${MODE:-nitrogen}" in
nitrogen)
    export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
    python3 retrospective/11_label_partitions.py > retrospective/work/11_label_partitions.log 2>&1 &
    python3 md_audit/20_label_partitions.py      > md_audit/work/20_label_partitions.log 2>&1 &
    wait ;;
open)
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    for n in ${NAMES:-si_gap18}; do python3 06_leverage.py $n --jobs $JOBS ${CLUSTER:-}; done ;;
nmin)
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
    python3 retrospective/12_min_rule.py    > retrospective/work/12_min_rule.log 2>&1 &
    python3 md_audit/22_min_rule.py         > md_audit/work/22_min_rule.log 2>&1 &
    python3 md_audit/21_al_step_leverage.py > md_audit/work/21_al_step_leverage_rerun.log 2>&1 &
    wait ;;
min)
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    python3 08_min_rule.py ${NAMES:-si_gap18} --jobs $JOBS ${ONLY:+--only $ONLY} ${TAG:+--tag $TAG} ;;
nig)
    export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
    python3 retrospective/13_info_gain.py > retrospective/work/13_info_gain.log 2>&1 &
    python3 md_audit/23_info_gain.py      > md_audit/work/23_info_gain.log 2>&1 &
    wait ;;
ig)
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    for n in ${NAMES:-si_gap18}; do python3 10_info_gain.py $n --jobs $JOBS ${CLUSTER:-} ${ONLY:+--only $ONLY} ${TAG:+--tag $TAG}; done ;;
nigsp)
    export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
    python3 md_audit/25_info_gain_statepoint.py > md_audit/work/25_info_gain_statepoint.log 2>&1 ;;
nadv)
    export OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24
    python3 retrospective/14_advance.py > retrospective/work/14_advance.log 2>&1 &
    python3 md_audit/24_advance.py      > md_audit/work/24_advance.log 2>&1 &
    wait ;;
adv)
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    python3 13_advance.py ${NAMES:-si_gap18} --methods ${METHODS//+/,} --jobs $JOBS ${ONLY:+--only $ONLY} ${TAG:+--tag $TAG} ${MLEARN:+--mlearn} ;;
tda)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    cd open_systems
    for n in ${NAMES:-si_gap18}; do python3 15_tda_features.py $n --jobs 48; done ;;
tdan)
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    (cd open_systems && python3 15_tda_features.py --nitrogen --jobs 48) > retrospective/work/15_tda_features.log 2>&1
    export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
    python3 retrospective/15_tda_ceiling.py > retrospective/work/15_tda_ceiling.log 2>&1 ;;
tpart)
    JOBS=${JOBS:-12}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    python3 16_tda_partition.py ${NAMES:-si_gap18} --jobs $JOBS ${ONLY:+--only $ONLY} ${TAG:+--tag $TAG} ;;
ceil)
    JOBS=${JOBS:-24}; T=$(( 48 / JOBS )); export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
    cd open_systems
    for n in ${NAMES:-si_gap18 sio2 water}; do python3 18_ceiling.py $n --jobs $JOBS; done ;;
al)
    export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48
    cd open_systems && python3 06_leverage.py sio2 --arm al --cluster ;;
esac
echo done
