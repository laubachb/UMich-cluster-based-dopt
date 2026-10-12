#!/usr/bin/env python3
"""Step 14: test N of open_systems/CRITERIA.md (addendum N) on the retrospective candidates.

Baselines: gamma_cluster (assigned; reproduces work/gamma_retro.npz) and gamma_min (minimum over the 6 cluster bases),
10 MaxVol solutions with the seeds and call order of step 7. New grades:
  gamma_soft (N-A)  exp(sum_k p_k log gamma_k), p from the 15-NN distance-weighted vote of step 2 (predict_proba; its
                    argmax reproduces labels_candidates.npy); gamma_soft_arith = sum p gamma
  gamma_ens (N-B)   exp(mean over the 15 clusterings of step 8 (k = 4-8 x seeds 42, 0, 1) of log gamma_cluster), 5 solutions
  gamma_cal (N-C)   q_k * gamma_cluster with per-cluster conformal quantiles (alpha = 0.1) of |error| / grade on the ALC-2
                    pool, tested on the ALC-3 to ALC-5 pools; gamma_cal_min for gamma_min. Mean-over-solution grades.
Within-cluster metrics for every grade (manuscript Fig. 5); pooled metrics over all clusters on ALC-3 to ALC-5 for the
cross-cluster comparability test (N3a); coverage of the 90 % bounds.
Writes results/advance_by_cluster.csv, results/advance_pooled.csv, results/advance_criteria.csv.
usage: 14_advance.py
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import SEED0, atom_rows_from_indices as R, random_restart_maxvol, gamma_for_rows, spearman, \
    auroc_high_error, recall_curve
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES, MINORITY_CLUSTERS
from open_systems.core import soft_grade, mondrian_q

N_TRIALS, N_ENS_TRIALS, ALPHA = 10, 5, 0.1
ENS = [(k, s) for k in (4, 5, 6, 7, 8) for s in (42, 0, 1)]
CAL_POOL = "ChIMES.fromALC-1.forALC-2.OUTCAR.xyzf"


def main():
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man]); src = np.array([r["source_file"] for r in man])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    A = np.load(WORK / "A_atomic.npy"); A_dft, A_ch = A[R(dft)], A[R(ch)]
    X = A.reshape(len(kind), -1)
    y = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx").loc[ch, "mae"].to_numpy(float)
    ldft = np.load(WORK / "labels_dft.npy"); lch = np.load(WORK / "labels_candidates.npy")
    g = np.load(WORK / "gamma_retro.npz"); GB, GC_ref = g["gb"], g["gc"]
    sc = StandardScaler().fit(X[dft]); Xd, Xc = sc.transform(X[dft]), sc.transform(X[ch])
    knn = KNeighborsClassifier(15, weights="distance").fit(Xd, ldft)
    proba = knn.predict_proba(Xc); assert (knn.classes_ == np.arange(6)).all() and (proba.argmax(1) == lch).all()
    print(f"membership: {np.mean(proba.max(1) > 0.99):.3f} of candidates have p_max > 0.99; mean p_max {proba.max(1).mean():.3f}", flush=True)

    P = np.zeros((N_TRIALS, 6, len(ch)))
    for t in range(N_TRIALS):
        rng = np.random.default_rng(SEED0 + t)
        random_restart_maxvol(A_dft, rng)                             # step 7 draws the bulk basis first
        for c in range(6):
            P[t, c] = gamma_for_rows(A_ch, random_restart_maxvol(A[R(dft[ldft == c])], rng))
    Ga = P[:, lch, np.arange(len(ch))]
    assert np.abs(Ga - GC_ref).max() < 1e-8
    S = {"gamma_bulk": GB, "gamma_cluster": Ga, "gamma_min": P.min(1),
         "gamma_soft": np.vstack([soft_grade(P[t], proba) for t in range(N_TRIALS)]),
         "gamma_soft_arith": np.vstack([soft_grade(P[t], proba, geometric=False) for t in range(N_TRIALS)])}
    print("soft done", flush=True)

    logs = np.zeros((N_ENS_TRIALS, len(ch)))
    for k, s in ENS:
        lab = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=20, assign_labels="kmeans",
                                 random_state=s, n_jobs=1).fit_predict(Xd)
        lc = KNeighborsClassifier(15, weights="distance").fit(Xd, lab).predict(Xc)
        for t in range(N_ENS_TRIALS):
            rng = np.random.default_rng(30_000 + 1000 * k + 10 * s + t)
            for c in np.unique(lc):
                m = np.where(lc == c)[0]
                logs[t, m] += np.log(np.maximum(gamma_for_rows(A_ch[R(m)], random_restart_maxvol(A[R(dft[lab == c])], rng)), 1e-300))
        print(f"ensemble k={k} seed={s} done", flush=True)
    S["gamma_ens"] = np.exp(logs / len(ENS))

    is_cal = src[ch] == CAL_POOL; test = ~is_cal
    pooled, cov = [], []
    for base in ("gamma_cluster", "gamma_min"):
        s_ = S[base].mean(0)
        q, qp = mondrian_q(s_[is_cal], y[is_cal], lch[is_cal], ALPHA)
        qk = np.array([q[int(c)] for c in lch])
        name = "gamma_cal" if base == "gamma_cluster" else "gamma_cal_min"
        S[name] = (qk * s_)[None]
        bound = qk * s_
        cov.append(dict(score=name, coverage_test=float(np.mean(y[test] <= bound[test])), coverage_pooled_q=float(np.mean(y[test] <= qp * s_[test])),
                        **{f"coverage_c{c}": float(np.mean(y[test & (lch == c)] <= bound[test & (lch == c)])) for c in range(6)},
                        q=";".join(f"{c}:{q[c]:.4g}" for c in sorted(q))))

    rows, rec = [], []
    hid = np.isin(lch, MINORITY_CLUSTERS); info = y[hid] > np.quantile(y[hid], 0.9)
    for s_, Gm in S.items():
        for t, v in enumerate(Gm):
            for c in range(6):
                m = lch == c
                rows.append(dict(score=s_, trial=t, cluster=c, spearman=spearman(v[m], y[m]), auroc_q90=auroc_high_error(v[m], y[m])))
            rec.append(dict(score=s_, trial=t, recall10_minority=recall_curve(v[hid], info, np.array([0.10]))[0]))
            pooled.append(dict(score=s_, trial=t, spearman_test=spearman(v[test], y[test]), auroc_test=auroc_high_error(v[test], y[test]),
                               recall10_test=recall_curve(v[test], y[test] > np.quantile(y[test], 0.9), np.array([0.10]))[0]))
    df, rc, pl = pd.DataFrame(rows), pd.DataFrame(rec), pd.DataFrame(pooled)
    df.to_csv(RES / "advance_by_cluster.csv", index=False)
    pl.groupby("score").mean(numeric_only=True).drop(columns="trial").to_csv(RES / "advance_pooled.csv")
    pd.DataFrame(cov).to_csv(RES / "advance_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    a = df.groupby(["cluster", "score"]).auroc_q90.mean().unstack("score")
    print("\n== within-cluster AUROC\n" + a.round(3).to_string())
    print("\n== minority recall at 10 %\n" + rc.groupby("score").recall10_minority.mean().round(3).to_string())
    print("\n== pooled over clusters, ALC-3 to ALC-5\n" + pl.groupby("score").mean(numeric_only=True).drop(columns="trial").round(3).to_string())
    print("\n== coverage of 90 % bounds on ALC-3 to ALC-5\n" + pd.DataFrame(cov).round(3).to_string(index=False))
    mi = a.loc[list(MINORITY_CLUSTERS)].mean()
    ps = pl.groupby("score").spearman_test.mean()
    cr = pd.DataFrame([
        dict(criterion="N1-N (soft)", passed=bool(mi.gamma_soft >= mi.gamma_cluster - 0.02 and mi.gamma_soft > mi.gamma_min),
             detail=f"minority AUROC soft {mi.gamma_soft:.3f}, cluster {mi.gamma_cluster:.3f}, min {mi.gamma_min:.3f}, soft_arith {mi.gamma_soft_arith:.3f}"),
        dict(criterion="N2-N (ensemble)", passed=bool(mi.gamma_ens >= mi.gamma_cluster), detail=f"minority AUROC ens {mi.gamma_ens:.3f} vs cluster {mi.gamma_cluster:.3f}"),
        dict(criterion="N3a-retro (calibrated, pooled)", passed=bool(ps.gamma_cal >= 0.30),
             detail=f"pooled Spearman on ALC-3..5: gamma_cal {ps.gamma_cal:.3f}, gamma_cluster {ps.gamma_cluster:.3f}, gamma_cal_min {ps.gamma_cal_min:.3f}, gamma_bulk {ps.gamma_bulk:.3f}"),
    ])
    cr.to_csv(RES / "advance_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
