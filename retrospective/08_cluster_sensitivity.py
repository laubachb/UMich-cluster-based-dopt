#!/usr/bin/env python3
"""Step 8: sensitivity of the cluster-local grade to the clustering (number of clusters k and clustering seed).

For k in {4,...,8} and SpectralClustering seeds {42 (manuscript), 0, 1}: cluster the DFT atoms exactly as 02_cluster.py,
assign candidates (15-NN vote), and recompute gamma_cluster for 5 MaxVol solutions (seed SEED0 + t; the bulk basis is
drawn first, as in 04, so gamma_bulk is the manuscript's). Every clustering is scored on the same atoms:
  retrospective : the molecular candidates of the manuscript clustering (clusters 0, 3, 4): Spearman(gamma, error),
                  AUROC@q90 and recall of their top-10% error at a 10% budget; gamma_bulk on the same atoms as reference
  MD (5000 K)   : frame-level Spearman(frame-max gamma_cluster, frame force RMSE) on the DFT-only model's MD, with
                  cluster bases from that model's training atoms and MD atoms assigned by the same 15-NN vote
Also recorded: adjusted Rand index to the manuscript clustering, and the share of 300/2000 K DFT atoms that fall in
"molecular" clusters (> 50% of their DFT atoms from 300 or 2000 K).
Writes results/cluster_sensitivity.csv. usage: 08_cluster_sensitivity.py
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES, MD

KS, SEEDS, N_TRIALS = [4, 5, 6, 7, 8], [42, 0, 1], 5
REF_MOL = (0, 3, 4)
MODEL, SP = "shared_t0_dft_train", "5000K_2.0gcc"


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man]); src = np.array([r["source_file"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy"); X = A.reshape(len(kind), -1)
    sc = StandardScaler().fit(X[dft]); Xd, Xc = sc.transform(X[dft]), sc.transform(X[ch])
    sp_dft = np.array([re.match(r"^DFT\.(.+)\.xyzf$", s).group(1) for s in src[dft]])
    cold = np.isin(sp_dft, ["300K_1.0gcc", "2000K_1.0gcc"])
    ref_lab = np.load(WORK / "labels_dft.npy"); ref_ch = np.load(WORK / "labels_candidates.npy")
    errs = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx")
    y = errs.loc[ch, "mae"].to_numpy(float)
    mol = np.isin(ref_ch, REF_MOL); ym = y[mol]; info = ym > np.quantile(ym, 0.9)
    A_dft, A_ch = A[E.atom_rows_from_indices(dft)], A[E.atom_rows_from_indices(ch)]

    # MD side: training atoms of the DFT-only model, its 5000 K MD atoms, their descriptors and true error
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    ref = np.array(sorted(a for f in meta["files"] for i in f["indices"] for a in by_frame[f["source"], i]))
    ref_pos = np.searchsorted(dft, ref); assert (dft[ref_pos] == ref).all()     # training atoms are DFT atoms
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv"); start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    ae = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv"); ae = ae[ae.statepoint == SP]
    md_idx = np.array([start[f] + a for f, a in zip(ae.frame, ae.atom)])
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    A_md = np.asarray(Am[E.atom_rows_from_indices(md_idx)])
    X_md = sc.transform(A_md.reshape(len(md_idx), -1))
    frames = ae.frame.to_numpy(); err = ae.err.to_numpy()
    rmse = pd.Series(err ** 2).groupby(frames).mean() ** 0.5

    rows = []
    gb_ref = []
    for t in range(N_TRIALS):
        gb_ref.append(E.gamma_for_rows(A_ch, E.random_restart_maxvol(A_dft, np.random.default_rng(E.SEED0 + t))))
    gb_ref = np.mean(gb_ref, 0)
    base = dict(sp_bulk=E.spearman(gb_ref[mol], ym), auroc_bulk=E.auroc_high_error(gb_ref[mol], ym),
                recall10_bulk=E.recall_curve(gb_ref[mol], info, np.array([0.10]))[0])
    for k in KS:
        for seed in SEEDS:
            lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=20, assign_labels="kmeans",
                                     random_state=seed, n_jobs=1).fit_predict(Xd)
            knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(Xd, lab)
            lch, lmd = knn.predict(Xc), knn.predict(X_md)
            molc = [c for c in range(k) if cold[lab == c].mean() > 0.5]
            gc_r, gc_m = np.zeros((N_TRIALS, len(ch))), np.zeros((N_TRIALS, len(md_idx)))
            for t in range(N_TRIALS):
                rng = np.random.default_rng(E.SEED0 + t)
                E.random_restart_maxvol(A_dft, rng)                        # bulk basis first, as in 04
                for c in range(k):
                    loc = np.where(lch == c)[0]
                    inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(dft[lab == c])], rng)
                    gc_r[t, loc] = E.gamma_for_rows(A[E.atom_rows_from_indices(ch[loc])], inv)
                # MD: bases from the model's training atoms of each cluster; same draw order as md_audit/03_gamma.py
                # (bulk basis of the training set first, clusters without MD atoms skipped), so k = 6, seed 42
                # reproduces the manuscript's gamma_cluster at 5000 K
                rng2 = np.random.default_rng(E.SEED0 + t)
                E.random_restart_maxvol(A[E.atom_rows_from_indices(ref)], rng2)
                for c in range(k):
                    tr = ref[lab[ref_pos] == c]; loc = np.where(lmd == c)[0]
                    if len(loc) == 0:
                        continue
                    if len(tr) * 3 < E.RANK:
                        gc_m[t, loc] = np.nan; continue
                    inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(tr)], rng2)
                    gc_m[t, loc] = E.gamma_for_rows(A_md[E.atom_rows_from_indices(loc)], inv)
            gr, gm = gc_r.mean(0), gc_m.mean(0)
            fmax = pd.Series(gm).groupby(frames).max()
            r = dict(k=k, seed=seed, ari_vs_manuscript=adjusted_rand_score(ref_lab, lab),
                     n_molecular_clusters=len(molc), cold_atoms_in_molecular=cold[np.isin(lab, molc)].sum() / cold.sum(),
                     sp_cluster=E.spearman(gr[mol], ym), auroc_cluster=E.auroc_high_error(gr[mol], ym),
                     recall10_cluster=E.recall_curve(gr[mol], info, np.array([0.10]))[0],
                     md5000_frame_sp_cluster=spearmanr(fmax.to_numpy(), rmse.loc[fmax.index].to_numpy(),
                                                       nan_policy="omit")[0], **base)
            rows.append(r)
            print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(RES / "cluster_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
