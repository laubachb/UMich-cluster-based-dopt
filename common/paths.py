"""Repository paths and external-tool locations.

External tools are located through environment variables so the scripts run on any machine:
  CHIMES_LSQ         chimes_lsq executable (descriptor / design-matrix generation)
  CHIMES_CALCULATOR  root of a chimes_calculator build (python API in serial_interface/api, lib in build/)
  LMP_CHIMES         LAMMPS executable built with pair_style chimesFF
  VASP_GAM           Gamma-only VASP executable
  VASP_POTCAR_N      PAW_PBE N POTCAR file
  MPIRUN             MPI launcher prefix, e.g. "ibrun -n 48" or "mpirun -np 48" (default: "mpirun -np 1")
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ChIMES-N_dataset" / "xyzf_files"
FIGURES = ROOT / "figures"
RETRO = ROOT / "retrospective"
RETRO_WORK = RETRO / "work"          # large regenerable intermediates (git-ignored)
RETRO_RESULTS = RETRO / "results"    # small tables behind the figures (tracked)
MD = ROOT / "md_audit"

DFT_FILES = ["DFT.2000K_1.0gcc.xyzf", "DFT.300K_1.0gcc.xyzf", "DFT.5000K_2.0gcc.xyzf",
             "DFT.6000K_2.5gcc.xyzf", "DFT.7000K_3.1gcc.xyzf", "DFT.8000K_4.5gcc.xyzf"]
POOLS = {"alc2": "ChIMES.fromALC-1.forALC-2.OUTCAR.xyzf", "alc3": "ChIMES.fromALC-2.forALC-3.OUTCAR.xyzf",
         "alc4": "ChIMES.fromALC-3.forALC-4.OUTCAR.xyzf", "alc5": "ChIMES.fromALC-4.forALC-5.OUTCAR.xyzf"}
MINORITY_CLUSTERS = (0, 3, 4)        # molecular 300 K / 2000 K clusters of the spectral partition (k = 6)


def tool(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"environment variable {name} is not set (see common/paths.py)")
    return v


def mpirun() -> list[str]:
    return os.environ.get("MPIRUN", "mpirun -np 1").split()
