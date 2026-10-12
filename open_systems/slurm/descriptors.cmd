#!/bin/bash
# ChIMES design matrix of an open data set (01_prepare.py output): chimes_lsq on one spr node, then A.txt -> A.npy
# (3 rows per atom, file order). Usage (from open_systems/): source ../env.sh; sbatch -A <allocation> --export=ALL,NAME=sio2 slurm/descriptors.cmd
#SBATCH -J cdopt_desc
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 03:00:00
#SBATCH -p spr
#SBATCH -o work/slurm_desc_%j.out
module load intel/24.0 impi/21.11
cd work/${NAME:-sio2}
[ -s A.txt ] || ibrun -n ${SLURM_NTASKS:-112} $CHIMES_LSQ fm_setup.in > fm_setup.log 2>&1
python3 - <<'PY'
import numpy as np, pandas as pd
n = pd.read_csv("frames.csv").n_atoms.sum()
chunks = [c.to_numpy(np.float64) for c in pd.read_csv("A.txt", sep=r"\s+", header=None, chunksize=200000)]
A = np.vstack(chunks); assert A.shape[0] == 3 * n, (A.shape, n)
np.save("A.npy", A)
nz = np.abs(A).max(0) > 0
print("A", A.shape, "nonzero columns", int(nz.sum()), flush=True)
PY
rm -f A.txt b.txt b-labeled.txt
