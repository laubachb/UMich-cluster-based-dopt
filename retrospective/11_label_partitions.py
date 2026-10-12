#!/usr/bin/env python3
"""Step 11: tests R1 and R2 of open_systems/CRITERIA.md (addendum of 2026-10-12) on the retrospective candidates.

R1: grades with a basis per known simulation condition instead of per cluster.
  gamma_density  one MaxVol basis per frame density (1.0, 2.0, 2.5, 3.1, 4.5 g/cm3); a candidate is graded against
                 the basis of its frame's density (computed from cell and atom count; every candidate frame has one
                 of the five DFT-MD densities)
  gamma_phase    two bases: 1.0 g/cm3 (molecular state points) and the rest
  Same MaxVol as the manuscript (common/dopt.py), 10 solutions, seeds 300 + t, reference = the 120 DFT-MD frames.
  gamma_bulk and gamma_cluster are the manuscript's (work/gamma_retro.npz from step 7).
R2: relative ridge leverage (open_systems/core.Leverage) against all DFT-MD atoms (h_bulk), the atoms of the
  candidate's cluster (h_cluster) or of its frame's density (h_density). Deterministic.
Every score is evaluated within each spectral cluster (as the manuscript's Fig. 5) and within each density group.
Writes results/label_partitions_by_group.csv, results/label_partitions_recall.csv and prints criteria R1-1, R1-2,
R2-1, R2-2 and R2-3 (nitrogen parts).
usage: 11_label_partitions.py
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import SEED0, atom_rows_from_indices, random_restart_maxvol, gamma_for_rows, spearman, \
    auroc_high_error, recall_curve
from common.paths import DATA, RETRO_WORK as WORK, RETRO_RESULTS as RES, MINORITY_CLUSTERS
from common.xyzf import read_lines, frame_offsets
from open_systems.core import Leverage

N_TRIALS = 10
DENSITIES = np.array([1.0, 2.0, 2.5, 3.1, 4.5])
N_MASS = 14.007 / 6.02214076e23          # g per N atom


def frame_densities():
    """(source_file, frame_idx) -> density code (index into DENSITIES), from the cell of every frame."""
    out = {}
    for f in sorted(DATA.glob("*.xyzf")):
        if f.name == "training_data.xyzf":
            continue
        L = read_lines(f); off = frame_offsets(L)
        for k, s in enumerate(off[:-1]):
            n = int(L[s]); c = L[s + 1].split()
            vol = abs(np.linalg.det(np.array(c[1:10], float).reshape(3, 3)))
            rho = n * N_MASS / (vol * 1e-24)
            code = int(np.argmin(np.abs(DENSITIES - rho)))
            assert abs(DENSITIES[code] - rho) < 0.05, (f.name, k, rho)
            out[f.name, k] = code
    return out


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    dens = frame_densities()
    dcode = np.array([dens[r["source_file"], int(r["frame_idx_in_file"])] for r in man])
    A = np.load(WORK / "A_atomic.npy")
    A_ch = A[atom_rows_from_indices(ch)]
    errs = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx")
    y = errs.loc[ch, "mae"].to_numpy(float)
    ldft = np.load(WORK / "labels_dft.npy"); lch = np.load(WORK / "labels_candidates.npy")
    g = np.load(WORK / "gamma_retro.npz"); GB, GC = g["gb"], g["gc"]
    assert GB.shape == (N_TRIALS, len(ch))
    print("candidate atoms per density:", dict(zip(DENSITIES, np.bincount(dcode[ch], minlength=5))), flush=True)
    print("candidate atoms per (cluster, density):\n" + pd.crosstab(lch, DENSITIES[dcode[ch]]).to_string(), flush=True)

    parts = {"density": dcode, "phase": (dcode != 0).astype(int)}
    grades = {"gamma_bulk": GB, "gamma_cluster": GC}
    for pname, lab in parts.items():
        Gm = np.full((N_TRIALS, len(ch)), np.nan)
        for t in range(N_TRIALS):
            rng = np.random.default_rng(SEED0 + t)
            for c in np.unique(lab[dft]):
                ref, loc = dft[lab[dft] == c], np.where(lab[ch] == c)[0]
                inv = random_restart_maxvol(A[atom_rows_from_indices(ref)], rng)
                if loc.size:
                    Gm[t, loc] = gamma_for_rows(A_ch[atom_rows_from_indices(loc)], inv)
        grades[f"gamma_{pname}"] = Gm
        print(f"gamma_{pname} done", flush=True)

    def lev(ref_lab, cand_lab):
        H = np.full(len(ch), np.nan)
        for c in np.unique(cand_lab):
            loc = np.where(cand_lab == c)[0]
            H[loc] = Leverage(A[atom_rows_from_indices(dft[ref_lab == c])]).grade(A_ch[atom_rows_from_indices(loc)])
        return H[None]
    grades["h_bulk"] = lev(np.zeros(len(dft), int), np.zeros(len(ch), int))
    grades["h_cluster"] = lev(ldft, lch)
    grades["h_density"] = lev(dcode[dft], dcode[ch])

    rows, rec = [], []
    groups = [("cluster", c, lch == c) for c in sorted(np.unique(lch))] + \
             [("density", DENSITIES[c], dcode[ch] == c) for c in range(5)]
    hid = np.isin(lch, MINORITY_CLUSTERS); info = y[hid] > np.quantile(y[hid], 0.9)
    for s, Gm in grades.items():
        for t, v in enumerate(Gm):
            for kind_, gname, m in groups:
                rows.append(dict(score=s, trial=t, group_kind=kind_, group=gname, n=int(m.sum()),
                                 spearman=spearman(v[m], y[m]), auroc_q90=auroc_high_error(v[m], y[m])))
            rec.append(dict(score=s, trial=t, recall10_minority=recall_curve(v[hid], info, np.array([0.10]))[0],
                            recall20_minority=recall_curve(v[hid], info, np.array([0.20]))[0]))
    df, rc = pd.DataFrame(rows), pd.DataFrame(rec)
    df.to_csv(RES / "label_partitions_by_group.csv", index=False); rc.to_csv(RES / "label_partitions_recall.csv", index=False)

    pd.set_option("display.width", 250)
    m = df.groupby(["group_kind", "group", "score"])[["spearman", "auroc_q90"]].mean().unstack("score")
    print("\n== within-group metrics (mean over solutions)\n" + m.round(3).to_string())
    print("\n== minority-cluster recall (pooled c0, c3, c4)\n" + rc.groupby("score")[["recall10_minority", "recall20_minority"]].mean().round(3).to_string())

    a = df[df.group_kind == "cluster"].groupby(["group", "score"]).auroc_q90.mean().unstack("score")
    mi = a.loc[list(MINORITY_CLUSTERS)]
    d_den, d_cl = (mi.gamma_density - mi.gamma_bulk), (mi.gamma_cluster - mi.gamma_bulk)
    d_hcl = mi.h_cluster - mi.h_bulk
    crit = [
        ("R1-1", bool((d_den > 0).all()), "AUROC(gamma_density) - AUROC(gamma_bulk) per minority cluster: "
         + ", ".join(f"c{c} {v:+.3f}" for c, v in d_den.items())),
        ("R1-2", bool(d_den.mean() >= 0.75 * d_cl.mean()), f"mean gain density {d_den.mean():.3f} vs 0.75 x cluster {0.75 * d_cl.mean():.3f} (cluster {d_cl.mean():.3f})"),
        ("R2-1 (N)", bool((mi.h_bulk < 0.6).all()), "AUROC(h_bulk) per minority cluster: " + ", ".join(f"c{c} {v:.3f}" for c, v in mi.h_bulk.items())),
        ("R2-2 (N)", bool((d_hcl > 0).all()), "AUROC(h_cluster) - AUROC(h_bulk): " + ", ".join(f"c{c} {v:+.3f}" for c, v in d_hcl.items())),
        ("R2-3 (N)", bool(abs(d_cl.mean() - d_hcl.mean()) < 0.05), f"mean Delta gamma_cluster {d_cl.mean():.3f} vs Delta h_cluster {d_hcl.mean():.3f}"),
    ]
    cr = pd.DataFrame(crit, columns=["criterion", "passed", "detail"]); cr.to_csv(RES / "label_partitions_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
