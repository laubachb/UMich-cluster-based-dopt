#!/usr/bin/env python3
"""DFT protocol calibration: recompute frame 5 of four DFT-MD files of the nitrogen dataset under four candidate
VASP settings -> work/dft_calib/<statepoint>_f5_<setting>/. Compare with compare.py. Needs VASP_POTCAR_N.
The dataset README states only "allPBE"; PBE + D2 at ENCUT 1000 reproduces the stored forces (~0.006 eV/A)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import DATA, MD, tool
from common.xyzf import read_lines, frame_offsets, frame

KB = 8.617333e-5
FRAMES = [("300K_1.0gcc", 300, 5), ("2000K_1.0gcc", 2000, 5), ("5000K_2.0gcc", 5000, 5), ("8000K_4.5gcc", 8000, 5)]
SETTINGS = {"e700": dict(ENCUT=700), "e700_d2": dict(ENCUT=700, IVDW=1), "e1000": dict(ENCUT=1000),
            "e1000_d2": dict(ENCUT=1000, IVDW=1)}


def incar(T, **kw):
    s = dict(PREC="Accurate", ENCUT=700, ISMEAR=-1, SIGMA=f"{KB * T:.6f}", EDIFF="1E-06", NELM=200, NELMIN=4,
             ALGO="Normal", LREAL=".FALSE.", ISYM=0, IBRION=-1, NSW=0, LWAVE=".FALSE.", LCHARG=".FALSE.", NCORE=8)
    s.update(kw)
    return "".join(f"{k} = {v}\n" for k, v in s.items())


def main():
    out = MD / "work" / "dft_calib"; out.mkdir(parents=True, exist_ok=True)
    pot = Path(tool("VASP_POTCAR_N")).read_text()
    for sp, T, k in FRAMES:
        L = read_lines(DATA / f"DFT.{sp}.xyzf"); s = frame_offsets(L)[k]
        cell, x = frame(L, s); n = len(x)
        F = np.array([l.split()[4:7] for l in L[s + 2:s + 2 + n]], float) * 51.42208619   # Ha/bohr -> eV/A
        for tag, kw in SETTINGS.items():
            d = out / f"{sp}_f{k}_{tag}"; d.mkdir(exist_ok=True)
            (d / "POSCAR").write_text(f"N {sp} frame {k}\n1.0\n" + "".join(" ".join(f"{v:.10f}" for v in r) + "\n" for r in cell)
                                      + f"N\n{n}\nCartesian\n" + "".join(" ".join(f"{v:.8f}" for v in r) + "\n" for r in x))
            (d / "INCAR").write_text(incar(T, **kw))
            (d / "KPOINTS").write_text("Gamma\n0\nGamma\n1 1 1\n0 0 0\n")
            (d / "POTCAR").write_text(pot)
            np.save(d / "F_ref.npy", F)
            print(d.name, n)


if __name__ == "__main__":
    main()
