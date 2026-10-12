#!/bin/bash
# Feed skx-dev (max 4 submitted jobs per user) with the remaining open-data jobs, in order, as slots free up:
# water arm A (one job per regime, after the water descriptors), then the C7 agreement jobs for sio2 and water.
# Run from open_systems/ in the background: bash slurm/water_queue.sh > work/water_queue.log 2>&1 &
source ../env.sh
DESC=$1
until sacct -j $DESC -n -o State | head -1 | grep -q -E "COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL"; do sleep 120; done
sacct -j $DESC -n -o State | head -1 | grep -q COMPLETED || { echo "descriptor job $DESC did not complete"; exit 1; }
JOBS=(
  "ARM=shift,NAME=water,REGIME=tetrahedral"
  "ARM=shift,NAME=water,REGIME=intermediate"
  "ARM=shift,NAME=water,REGIME=dense"
  "ARM=shift,SCRIPT=05_agreement.py,NAME=sio2,REGIME=bulk_amo bulk_cryst bulk_cryst_hp"
  "ARM=shift,SCRIPT=05_agreement.py,NAME=sio2,REGIME=liquid quench surfaces vacancy"
  "ARM=shift,SCRIPT=05_agreement.py,NAME=water,REGIME=tetrahedral intermediate dense"
)
for j in "${JOBS[@]}"; do
  until [ "$(squeue -u $USER -h -p skx-dev | wc -l)" -lt 4 ]; do sleep 120; done
  sbatch -A TG-CHM250015 -J cdopt_dev_q --exclude=c455-002,c455-034 --export="ALL,$j" slurm/analysis_dev.cmd
  sleep 5
done
echo "all submitted $(date)"
