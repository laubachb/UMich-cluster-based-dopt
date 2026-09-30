#!/bin/bash
# MD stability of the active-learning refits (11_al_md_setup.py): 352 serial 64-atom LAMMPS runs.
# 64 atoms scale poorly over MPI ranks (4 ranks = 2x one), so each run is serial and every node runs 48 at once,
# one per core (pinned with taskset). Array task i of n takes every n-th job; its core c takes every 48th of those.
# Other run sets: sbatch --array=0-7 --export=ALL,SPLIT=balanced_seeds md_audit/slurm/run_al_md.cmd
# Only newly added runs:  sbatch --array=0-7 --export=ALL,SPLIT=balanced_seeds,JOBS=jobs_added.txt md_audit/slurm/run_al_md.cmd
# Sapphire Rapids nodes (112 cores): add  -p spr --ntasks-per-node 112  to the sbatch line.
# Usage (repository root): python3 md_audit/11_al_md_setup.py balanced 1 && sbatch md_audit/slurm/run_al_md.cmd
#SBATCH -J cdopt_almd
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx
#SBATCH --array=0-3
#SBATCH -o md_audit/work/al_md/slurm_%A_%a.out
module load intel/24.0 impi/21.11
export I_MPI_PIN=off OMP_NUM_THREADS=1
source env.sh                        # sets LMP_CHIMES (written by install/install_chimes.sh)
LMP=$LMP_CHIMES
SPLIT=${SPLIT:-balanced}
JOBS=${JOBS:-jobs.txt}          # jobs_added.txt = only the runs added by the last 11_al_md_setup.py call
cd md_audit/work/al_md/$SPLIT
mapfile -t ALL < <(grep -v '^$' "$JOBS")
NT=${SLURM_ARRAY_TASK_COUNT:-4}
MINE=(); for ((j = SLURM_ARRAY_TASK_ID; j < ${#ALL[@]}; j += NT)); do MINE+=("${ALL[j]}"); done
# Up to 3 attempts per run. With ~50-100 runs starting at once on /work2, LAMMPS occasionally reads a truncated
# data.in ("Unexpected end of data file") or an empty in.lammps (it then exits cleanly without running any step),
# although both files are complete. A run counts as done only if its log has "Loop time" (the MD loop finished);
# otherwise it is retried after a short pause.
worker() {
  local core=$1; shift
  for d in "$@"; do
    ( cd "$d" || exit
      for a in 1 2 3; do
        taskset -c "$core" mpirun -np 1 "$LMP" -in in.lammps -log log.lammps > stdout.txt 2>&1
        grep -q "^Loop time" log.lammps 2>/dev/null && { echo "OK $d"; exit; }
        grep -q "Unexpected end of data file\|^Total wall time" stdout.txt || break   # a real LAMMPS error: stop
        sleep $((5 * a))
      done
      echo "FAIL $d" )
  done
}
NC=${SLURM_CPUS_ON_NODE:-48}     # 48 on skx, 112 on spr: one serial run per core
for ((c = 0; c < NC; c++)); do
  list=(); for ((j = c; j < ${#MINE[@]}; j += NC)); do list+=("${MINE[j]}"); done
  [ ${#list[@]} -gt 0 ] && worker "$c" "${list[@]}" &
done
wait
date
