#!/bin/bash
# Clone and build the ChIMES tools at the exact commits used for the manuscript, then write env.sh.
#
#   chimes_lsq         design-matrix (descriptor) generation            -> CHIMES_LSQ
#   chimes_calculator  ChIMES force evaluation (python API + library)   -> CHIMES_CALCULATOR
#   LAMMPS + chimesFF  MD with ChIMES models (built by chimes_calculator) -> LMP_CHIMES
#
# Usage (from anywhere):
#   export hosttype=UT-TACC     # optional: a modfiles/<hosttype>.mod of the ChIMES repos to load compilers/MPI
#   bash install/install_chimes.sh [--skip-lammps]
#   source env.sh
#
# Requirements: git, cmake >= 3.x, a C++11 compiler and MPI (the ChIMES install scripts assume Intel oneAPI
# compilers + Intel MPI), python >= 3.9. VASP is licensed separately and is NOT installed here; set VASP_GAM and
# VASP_POTCAR_N in env.sh yourself if you want to rerun the DFT steps.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXT=${EXT:-$ROOT/external}
ENV_FILE=${ENV_FILE:-$ROOT/env.sh}
SKIP_LAMMPS=0; [[ "${1:-}" == "--skip-lammps" ]] && SKIP_LAMMPS=1

LSQ_URL=https://github.com/laubachb/chimes_lsq-myLLfork.git
LSQ_COMMIT=a07329d7d49dd99316bdde5802aeae9f1cc43ee7          # branch laubachb/gpu-acceleartion
CALC_URL=https://github.com/laubachb/chimes_calculator-myLLfork.git
CALC_COMMIT=cff1feb5bef88699336b672e18b878b9742305dd          # branch laubachb/gpu-acceleration

fetch() {  # url commit dir
    if [[ ! -d "$3/.git" ]]; then git clone "$1" "$3"; fi
    git -C "$3" fetch --quiet origin
    git -C "$3" checkout --quiet --detach "$2"
    echo "$(basename "$3") at $(git -C "$3" rev-parse --short HEAD)"
}

mkdir -p "$EXT"
fetch "$LSQ_URL"  "$LSQ_COMMIT"  "$EXT/chimes_lsq"
fetch "$CALC_URL" "$CALC_COMMIT" "$EXT/chimes_calculator"

echo "== building chimes_lsq"
( cd "$EXT/chimes_lsq" && ./install.sh )
test -x "$EXT/chimes_lsq/build/chimes_lsq"

echo "== building chimes_calculator"
( cd "$EXT/chimes_calculator" && ./install.sh )
test -f "$EXT/chimes_calculator/build/libchimescalc_dl.so"

if [[ $SKIP_LAMMPS -eq 0 ]]; then
    echo "== building LAMMPS with pair_style chimesFF (downloads LAMMPS; takes a while)"
    ( cd "$EXT/chimes_calculator/etc/lmp" && ./install.sh )
    test -x "$EXT/chimes_calculator/etc/lmp/exe/lmp_mpi_chimes"
fi

cat > "$ENV_FILE" <<ENV
# Written by install/install_chimes.sh; source before running the pipelines.
export CHIMES_LSQ=$EXT/chimes_lsq/build/chimes_lsq
export CHIMES_CALCULATOR=$EXT/chimes_calculator
export LMP_CHIMES=$EXT/chimes_calculator/etc/lmp/exe/lmp_mpi_chimes
export MPIRUN="\${MPIRUN:-mpirun -np 4}"      # e.g. "ibrun -n 48" on TACC
# Only needed to rerun the DFT steps (md_audit steps 4-5, dft_calib):
export VASP_GAM=\${VASP_GAM:-}                  # Gamma-only VASP executable (vasp_gam)
export VASP_POTCAR_N=\${VASP_POTCAR_N:-}        # PAW_PBE N POTCAR (08Apr2002)
ENV
echo "wrote $ENV_FILE"
