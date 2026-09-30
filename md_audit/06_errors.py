#!/usr/bin/env python3
"""Step 6: true per-atom force error of the DFT-only model on its own MD frames.

F_model : chimes_calculator with models/<model>/params.txt (kcal/mol/A -> eV/A); needs CHIMES_CALCULATOR
F_DFT   : data/dft_forces.npz (step 5); includes D2, like the training labels
err     : |F_model - F_DFT| (vector norm, eV/A)
Joined with gamma (work/gamma_md.npz) -> work/atom_errors_<model>.csv.
usage: 06_errors.py [model]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD, tool
from common.xyzf import read_lines, frame_offsets, frame, load_dft_forces

KCAL2EV = 0.0433641


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "shared_t0_dft_train"
    CC = tool("CHIMES_CALCULATOR")
    sys.path.insert(0, CC + "/serial_interface/api")
    import chimescalc_serial_py as cs
    cs.chimes_wrapper = cs.init_chimes_wrapper(CC + "/build/libchimescalc_dl.so"); cs.set_chimes()
    cs.init_chimes(str(MD / "models" / model / "params.txt"), 0)

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    L = read_lines(MD / "data" / "traj.xyzf.gz"); off = frame_offsets(L)
    FD = load_dft_forces(MD / "data" / "dft_forces.npz")
    sel = fm[(fm.model == model) & (fm.step > 0) & fm.frame.isin(list(FD))]
    g = np.load(MD / "work" / "gamma_md.npz")
    atoms = g[f"{model}__atoms"]; gb = g[f"{model}__gb"]; gc = g[f"{model}__gc"]
    atom_start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    pos = {a: i for i, a in enumerate(atoms)}
    rows = []
    for f, sp, n, rep, step in zip(sel.frame, sel.statepoint, sel.n_atoms, sel.rep, sel.step):
        cell, x = frame(L, off[f])
        out = cs.calculate_chimes(n, list(x[:, 0]), list(x[:, 1]), list(x[:, 2]), ["N"] * n, list(cell[0]), list(cell[1]),
                                  list(cell[2]), 0.0, [0.0] * n, [0.0] * n, [0.0] * n, [0.0] * 9)
        Fm = np.c_[list(out[0]), list(out[1]), list(out[2])] * KCAL2EV
        Fd = FD[f]
        d = Fm - Fd
        for a in range(n):
            k = pos[atom_start[f] + a]
            rows.append((f, sp, rep, step, a, int(g["cluster"][atom_start[f] + a]), np.linalg.norm(d[a]), np.abs(d[a]).mean(),
                         np.linalg.norm(Fd[a]), gb[:, k].mean(), gc[:, k].mean(), (gb[:, k] > 1).mean(), (gc[:, k] > 1).mean()))
    df = pd.DataFrame(rows, columns=["frame", "statepoint", "seed", "step", "atom", "cluster", "err", "mae", "fdft",
                                     "gamma_bulk", "gamma_cluster", "pb_flag", "pc_flag"])
    df.to_csv(MD / "work" / f"atom_errors_{model}.csv", index=False)
    print(f"{df.frame.nunique()} frames, {len(df)} atoms; err by statepoint:")
    print(df.groupby("statepoint").err.describe().round(3).to_string())


if __name__ == "__main__":
    main()
