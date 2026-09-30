#!/usr/bin/env python3
"""Step 4: VASP single-point inputs for every MD frame (t > 0) of the DFT-only model -> work/dft/f<frame>/.

Protocol (calibrated in dft_calib/, reproduces the dataset forces to ~0.006 eV/A): PBE + D2 (IVDW = 1),
ENCUT = 1000 eV, PREC = Accurate, LREAL = .FALSE., Fermi smearing (ISMEAR = -1) with SIGMA = k_B T, Gamma point,
EDIFF = 1e-6. Also writes work/dft/chunk_<i>.txt lists for slurm/run_dft_array.cmd. Needs VASP_POTCAR_N.
usage: 04_make_dft.py [n_chunks] [model]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD, tool
from common.xyzf import read_lines, frame_offsets, frame

KB = 8.617333e-5
INCAR = dict(PREC="Accurate", ENCUT=1000, IVDW=1, ISMEAR=-1, EDIFF="1E-06", NELM=200, NELMIN=4, ALGO="Normal",
             LREAL=".FALSE.", ISYM=0, IBRION=-1, NSW=0, LWAVE=".FALSE.", LCHARG=".FALSE.", NCORE=8)


def main():
    n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    model = sys.argv[2] if len(sys.argv) > 2 else "shared_t0_dft_train"
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    L = read_lines(MD / "data" / "traj.xyzf.gz"); off = frame_offsets(L)
    sel = fm[(fm.model == model) & (fm.step > 0)]
    out = MD / "work" / "dft"; out.mkdir(parents=True, exist_ok=True)
    pot = Path(tool("VASP_POTCAR_N")).read_text()
    for f, sp, n in zip(sel.frame, sel.statepoint, sel.n_atoms):
        d = out / f"f{f}"; d.mkdir(exist_ok=True)
        cell, x = frame(L, off[f])
        (d / "POSCAR").write_text(f"N frame {f} {sp}\n1.0\n" + "".join(" ".join(f"{v:.10f}" for v in r) + "\n" for r in cell)
                                  + f"N\n{n}\nCartesian\n" + "".join(" ".join(f"{v:.8f}" for v in r) + "\n" for r in x))
        T = float(sp.split("K_")[0])
        (d / "INCAR").write_text("".join(f"{k} = {v}\n" for k, v in {**INCAR, "SIGMA": f"{KB * T:.6f}"}.items()))
        (d / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
        (d / "POTCAR").write_text(pot)
    names = [f"f{f}" for f in sel.frame]
    for i in range(n_chunks):
        (out / f"chunk_{i}.txt").write_text("\n".join(names[i::n_chunks]) + "\n")
    print(f"{len(names)} frames of {model} -> {out} ({n_chunks} chunks)")


if __name__ == "__main__":
    main()
