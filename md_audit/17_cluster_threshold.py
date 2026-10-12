#!/usr/bin/env python3
"""Step 17: does a per-cluster threshold calibrated on held-out DFT-MD frames make gamma_cluster usable as a flag?

gamma_cluster > 1 holds for about half of the atoms of the DFT-only model's 300 K and 2000 K MD (step 7). Here the
threshold of each cluster is set from the 30 held-out DFT-MD frames (not in the 90 training frames): tau_c = the q-th
percentile of gamma_cluster over the held-out atoms of cluster c, graded against the same per-cluster bases as the MD
atoms (same MaxVol seeds and call order as step 3; checked against work/gamma_md.npz). An MD atom is flagged if its
gamma_cluster (mean over the 10 MaxVol solutions) exceeds tau_c of its cluster. The same is done for gamma_bulk with
one threshold. Reports, per state point: flagged fraction, median error, share of flagged atoms in the top-10 % error
(precision), and share of the top-10 % error atoms that are flagged (recall), next to the gamma > 1 flags of step 7.
Writes results/cluster_threshold.csv and results/cluster_threshold_tau.csv.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, RETRO_WORK as RW

MODEL, N_TRIALS = "shared_t0_dft_train", 10


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft_ids = np.where(kind == "dft")[0]
    lab = np.full(len(kind), -1); lab[dft_ids] = np.load(RW / "labels_dft.npy")
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    ref = np.array(sorted(a for f in meta["files"] for i in f["indices"] for a in by_frame[f["source"], i]))
    hold = np.setdiff1d(dft_ids, ref)                       # atoms of the 30 held-out DFT-MD frames
    print(f"reference atoms {ref.size}, held-out atoms {hold.size} (per cluster {np.bincount(lab[hold], minlength=6)})")

    z = np.load(MD / "work" / "gamma_md.npz")
    md, gb_md, gc_md, cl = z[f"{MODEL}__atoms"], z[f"{MODEL}__gb"], z[f"{MODEL}__gc"], z["cluster"]
    Am = np.load(MD / "work" / "A.npy")
    gb_h = np.zeros((N_TRIALS, hold.size)); gc_h = np.zeros_like(gb_h); chk = 0.0
    for t in range(N_TRIALS):                                # same seeds and call order as 03_gamma.py
        rng = np.random.default_rng(E.SEED0 + t)
        inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(ref)], rng)
        gb_h[t] = E.gamma_for_rows(A[E.atom_rows_from_indices(hold)], inv)
        chk = max(chk, np.abs(E.gamma_for_rows(Am[E.atom_rows_from_indices(md[:2000])], inv) - gb_md[t, :2000]).max())
        for c in range(6):
            inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[lab[ref] == c])], rng)
            hc = np.where(lab[hold] == c)[0]
            gc_h[t, hc] = E.gamma_for_rows(A[E.atom_rows_from_indices(hold[hc])], inv)
            mc = np.where(cl[md[:2000]] == c)[0]
            if mc.size:
                chk = max(chk, np.abs(E.gamma_for_rows(Am[E.atom_rows_from_indices(md[mc])], inv) - gc_md[t, mc]).max())
    print(f"max |gamma - stored gamma_md| on 2000 MD atoms: {chk:.2e}")
    assert chk < 1e-3
    gbh, gch = gb_h.mean(0), gc_h.mean(0)
    print("held-out DFT-MD atoms with gamma > 1 (mean over solutions): bulk %d, cluster %d of %d" % ((gbh > 1).sum(), (gch > 1).sum(), hold.size))

    d = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    taus, rows = [], []
    for q in (95, 99):
        tb = np.percentile(gbh, q); tc = np.array([np.percentile(gch[lab[hold] == c], q) for c in range(6)])
        taus.append(dict(q=q, tau_bulk=tb, **{f"tau_c{c}": tc[c] for c in range(6)}))
        for sp, g in d.groupby("statepoint"):
            top = g.err > g.err.quantile(0.9)
            sets = {f"gamma_bulk > tau (q{q})": g.gamma_bulk > tb,
                    f"gamma_cluster > tau_c (q{q})": g.gamma_cluster > tc[g.cluster.to_numpy()]}
            if q == 95:
                sets["gamma_bulk > 1 (80 % of solutions)"] = g.pb_flag >= 0.8
                sets["gamma_cluster > 1 (80 % of solutions)"] = g.pc_flag >= 0.8
            for name, f in sets.items():
                rows.append(dict(statepoint=sp, rule=name, n_flagged=int(f.sum()), frac_flagged=f.mean(),
                                 median_err_flagged=g.err[f].median() if f.any() else np.nan,
                                 median_err_unflagged=g.err[~f].median(),
                                 precision_top10=top[f].mean() if f.any() else np.nan, recall_top10=f[top].mean()))
    pd.DataFrame(taus).to_csv(MD / "results" / "cluster_threshold_tau.csv", index=False)
    out = pd.DataFrame(rows).sort_values(["statepoint", "rule"]); out.to_csv(MD / "results" / "cluster_threshold.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(taus).round(3).to_string(index=False)); print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
