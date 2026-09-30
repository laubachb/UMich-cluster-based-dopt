#!/bin/bash
# Example SLURM array (TACC Stampede3): VASP single points written by md_audit/04_make_dft.py.
# Usage (repository root): sbatch --array=0-7 md_audit/slurm/run_dft_array.cmd   (finished frames are skipped)
#SBATCH -J cdopt_dft
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx
#SBATCH -o md_audit/work/dft/slurm_%A_%a.out
module load intel/24.0 impi/21.11
source env.sh
cd md_audit/work/dft
while read d; do
  [ -z "$d" ] && continue
  grep -qs "General timing" "$d/OUTCAR" && continue
  # stdin from /dev/null: ibrun/mpirun otherwise consume the rest of the frame list
  ( cd "$d" && ibrun -n 48 "$VASP_GAM" > vasp.out 2>&1 < /dev/null; rm -f WAVECAR CHG CHGCAR vasprun.xml )
  grep -qs "General timing" "$d/OUTCAR" && echo "OK $d" || echo "FAIL $d"
done < chunk_${SLURM_ARRAY_TASK_ID}.txt
