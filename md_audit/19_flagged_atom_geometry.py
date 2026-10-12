#!/usr/bin/env python3
"""Step 19: what are the 5000 K MD atoms that only gamma_cluster flags, chemically?

For every atom of the 300 DFT-labelled 5000 K frames of the DFT-only model: distance to the nearest and second-nearest
atom (minimum image, cubic cell) and the number of atoms within 1.3 A (bonded). Compares the atoms flagged by
gamma_cluster only (gamma_cluster > 1 in >= 80 % of MaxVol solutions, gamma_bulk <= 1 in every solution; step 7) with
unflagged atoms and with all atoms. Also prints the flag counts behind the main-text numbers at every state point.
Writes results/flagged_atom_geometry.csv and results/flag_counts.csv.
"""
import gzip, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD

MODEL = "shared_t0_dft_train"


def main():
    d = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    cnt = []
    for sp, g in d.groupby("statepoint"):
        c80, b80 = g.pc_flag >= 0.8, g.pb_flag >= 0.8
        cnt.append(dict(statepoint=sp, atoms=len(g), mean_frac_gc_gt1=g.pc_flag.mean(), mean_frac_gb_gt1=g.pb_flag.mean(),
                        gc_robust=int(c80.sum()), gc_robust_and_gb_robust=int((c80 & b80).sum()),
                        gc_robust_gb_in_some_solutions=int((c80 & (g.pb_flag > 0) & ~b80).sum()),
                        gc_only=int((c80 & (g.pb_flag == 0)).sum()), gb_robust=int(b80.sum()),
                        unflagged=int(((g.pc_flag == 0) & (g.pb_flag == 0)).sum())))
    cnt = pd.DataFrame(cnt); cnt.to_csv(MD / "results" / "flag_counts.csv", index=False)
    g5 = d[d.statepoint == "5000K_2.0gcc"]; want = set(g5.frame.unique()); rows = []
    with gzip.open(MD / "data" / "traj.xyzf.gz", "rt") as f:
        i = 0
        while True:
            l = f.readline()
            if not l: break
            n = int(l); box = float(f.readline().split()[1])
            xyz = np.array([f.readline().split()[1:4] for _ in range(n)], float)
            if i in want:
                dv = xyz[:, None, :] - xyz[None, :, :]; dv -= box * np.round(dv / box)
                r = np.sqrt((dv ** 2).sum(-1)); np.fill_diagonal(r, 99.0); s = np.sort(r, 1)
                rows += [(i, a, s[a, 0], s[a, 1], int((r[a] < 1.3).sum())) for a in range(n)]
            i += 1
    m = g5.merge(pd.DataFrame(rows, columns=["frame", "atom", "d1", "d2", "n_bonded"]), on=["frame", "atom"])
    sets = {"all 5000 K atoms": np.ones(len(m), bool), "gamma_cluster only": ((m.pc_flag >= 0.8) & (m.pb_flag == 0)).to_numpy(),
            "unflagged": ((m.pc_flag == 0) & (m.pb_flag == 0)).to_numpy()}
    out = pd.DataFrame([dict(group=k, n=int(v.sum()), d1_median=m.d1[v].median(), d2_median=m.d2[v].median(),
                             frac_bonded=(m.n_bonded[v] >= 1).mean(), frac_one_bond=(m.n_bonded[v] == 1).mean(),
                             frac_two_or_more_within_1p3=(m.n_bonded[v] >= 2).mean(), frac_d2_below_1p8=(m.d2[v] < 1.8).mean(),
                             fdft_median=m.fdft[v].median(), err_median=m.err[v].median()) for k, v in sets.items()])
    out.to_csv(MD / "results" / "flagged_atom_geometry.csv", index=False)
    pd.set_option("display.width", 250); print(cnt.round(3).to_string(index=False)); print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
