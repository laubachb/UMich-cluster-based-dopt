#!/usr/bin/env python3
"""Step 12: test M1 of open_systems/CRITERIA.md: minimum over groups (Shuang et al., arXiv 2508.03405) vs assignment.

For the cluster partition (manuscript clustering) and the density partition (step 11), every candidate atom is graded
against the MaxVol basis of every group; gamma_min = the smallest of these, gamma_assigned = the grade against the
atom's own group. The bases are those of step 7 (cluster: same seeds and call order, so gamma_assigned reproduces the
manuscript's gamma_cluster, checked against work/gamma_retro.npz) and step 11 (density). 10 MaxVol solutions.
Writes results/min_rule_by_group.csv and results/min_rule_criteria.csv.
usage: 12_min_rule.py
"""
import csv, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import SEED0, atom_rows_from_indices, random_restart_maxvol, gamma_for_rows, spearman, \
    auroc_high_error, recall_curve
from common.paths import RETRO, RETRO_WORK as WORK, RETRO_RESULTS as RES, MINORITY_CLUSTERS

N_TRIALS = 10
spec = importlib.util.spec_from_file_location("lp", RETRO / "11_label_partitions.py")
lp = importlib.util.module_from_spec(spec); spec.loader.exec_module(lp)


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    dens = lp.frame_densities()
    dcode = np.array([dens[r["source_file"], int(r["frame_idx_in_file"])] for r in man])
    A = np.load(WORK / "A_atomic.npy"); A_dft, A_ch = A[atom_rows_from_indices(dft)], A[atom_rows_from_indices(ch)]
    y = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch, "mae"].to_numpy(float)
    ldft = np.load(WORK / "labels_dft.npy"); lch = np.load(WORK / "labels_candidates.npy")
    GC_ref = np.load(WORK / "gamma_retro.npz")["gc"]

    grades = {}
    for pname, (lref, lcand, bulk_first) in {"cluster": (ldft, lch, True), "density": (dcode[dft], dcode[ch], False)}.items():
        Ga, Gm = np.full((N_TRIALS, len(ch)), np.nan), np.full((N_TRIALS, len(ch)), np.nan)
        for t in range(N_TRIALS):
            rng = np.random.default_rng(SEED0 + t)
            if bulk_first:                                   # step 7 draws the bulk basis first from the same stream
                random_restart_maxvol(A_dft, rng)
            per = {}
            for c in sorted(int(c) for c in np.unique(lref)):
                inv = random_restart_maxvol(A[atom_rows_from_indices(dft[lref == c])], rng)
                per[c] = gamma_for_rows(A_ch, inv)
            P = np.vstack([per[c] for c in sorted(per)]); keys = np.array(sorted(per))
            Ga[t] = P[np.searchsorted(keys, lcand), np.arange(len(ch))]
            Gm[t] = P.min(0)
        if pname == "cluster":
            dev = np.nanmax(np.abs(Ga - GC_ref))
            print(f"assigned gamma_cluster vs manuscript (gamma_retro.npz): max |diff| = {dev:.2e}", flush=True)
            assert dev < 1e-8
        grades[f"gamma_{pname}"], grades[f"gamma_min_{pname}"] = Ga, Gm
        print(f"{pname} done; share of candidates whose minimum is their own group: "
              f"{np.mean(np.argmin(P, 0) == np.searchsorted(keys, lcand)):.3f} (last solution)", flush=True)

    rows, rec = [], []
    hid = np.isin(lch, MINORITY_CLUSTERS); info = y[hid] > np.quantile(y[hid], 0.9)
    for s, G in grades.items():
        for t, v in enumerate(G):
            for c in sorted(np.unique(lch)):
                m = lch == c
                rows.append(dict(score=s, trial=t, cluster=int(c), spearman=spearman(v[m], y[m]), auroc_q90=auroc_high_error(v[m], y[m])))
            rec.append(dict(score=s, trial=t, recall10_minority=recall_curve(v[hid], info, np.array([0.10]))[0]))
    df, rc = pd.DataFrame(rows), pd.DataFrame(rec)
    df.to_csv(RES / "min_rule_by_group.csv", index=False)
    pd.set_option("display.width", 250)
    a = df.groupby(["cluster", "score"]).auroc_q90.mean().unstack("score")
    print("\n== within-cluster AUROC (mean over solutions)\n" + a.round(3).to_string())
    print("\n== minority recall at 10 %\n" + rc.groupby("score").recall10_minority.mean().round(3).to_string())
    mi = a.loc[list(MINORITY_CLUSTERS)]
    d = mi.gamma_cluster.mean() - mi.gamma_min_cluster.mean()
    cr = pd.DataFrame([dict(criterion="M1", passed=bool(d >= 0.05),
                            detail=f"minority mean AUROC gamma_cluster {mi.gamma_cluster.mean():.3f} vs gamma_min_cluster "
                                   f"{mi.gamma_min_cluster.mean():.3f} (difference {d:+.3f})")])
    cr.to_csv(RES / "min_rule_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
