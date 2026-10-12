#!/bin/bash
# Multi-node version of run_al_md.cmd for skx-dev (one job, N nodes, 2 h limit): node i runs run_al_md.cmd as if it
# were array task i of N, so the split of the job list is identical to the array version.
# mpirun inside a job step must start LAMMPS locally, not through srun: I_MPI_HYDRA_BOOTSTRAP=fork.
# Usage (repository root): sbatch -A <allocation> -N 4 --export=ALL,SPLIT=leverage_seeds md_audit/slurm/run_al_md_dev.cmd
#SBATCH -J cdopt_almd_dev
#SBATCH -N 4
#SBATCH --ntasks-per-node 48
#SBATCH -t 02:00:00
#SBATCH -p skx-dev
#SBATCH -o md_audit/work/al_md/slurm_dev_%j.out
module load intel/24.0 impi/21.11
export I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_FABRICS=shm LOCAL_MPI_HOST=1   # LOCAL_MPI_HOST: mpirun -hosts <this node>
# The build in env.sh (chimes_calculator-myLLfork) was removed on 2026-10-05; this build reproduces the manuscript runs
# exactly (first 500 steps of gamma_cluster__k10__d0__s2__5000K identical in every thermo column, checked 2026-10-12).
export LMP_OVERRIDE=${LMP_OVERRIDE:-/work2/09982/blaubach/stampede3/cdopt_install_test/external/chimes_calculator/etc/lmp/exe/lmp_mpi_chimes}
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
for i in "${!nodes[@]}"; do
  srun -N1 -n1 -c 48 --exclusive -w "${nodes[$i]}" \
       --export=ALL,SLURM_ARRAY_TASK_ID=$i,SLURM_ARRAY_TASK_COUNT=${#nodes[@]},SLURM_CPUS_ON_NODE=48,SPLIT=${SPLIT:-leverage_seeds} \
       bash md_audit/slurm/run_al_md.cmd > "md_audit/work/al_md/slurm_dev_${SLURM_JOB_ID}_node$i.out" 2>&1 &
done
wait
date
