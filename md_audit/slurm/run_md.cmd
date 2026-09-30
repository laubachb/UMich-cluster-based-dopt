#!/bin/bash
# Example SLURM driver (TACC Stampede3): the 33 Langevin MD runs, 3 at a time with 16 MPI ranks each.
# Usage (repository root): python3 md_audit/00_setup_md.py && sbatch md_audit/slurm/run_md.cmd
#SBATCH -J cdopt_md
#SBATCH -N 1
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx
#SBATCH -o md_audit/work/md/slurm_%j.out
module load intel/24.0 impi/21.11
source env.sh
cd md_audit/work/md
run() { cd "$1"; mpirun -np 16 "$LMP_CHIMES" -in in.lammps > log.lammps 2>&1 && echo "OK $1" || echo "FAIL $1"; cd ..; }
export -f run
xargs -P 3 -I{} bash -c 'run {}' < jobs.txt
