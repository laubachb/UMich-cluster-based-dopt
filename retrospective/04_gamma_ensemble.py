#!/usr/bin/env python3
"""Step 4: bulk vs cluster-local gamma for every candidate atom, 10 MaxVol solutions (Figs. 2-5).

For each solution t (seed SEED0 + t):
  gamma_bulk    : MaxVol basis from all DFT atoms; every candidate atom graded against it
  gamma_cluster : one MaxVol basis per spectral cluster from that cluster's DFT atoms; each candidate atom graded
                  against the basis of its assigned cluster
Outputs (results/, tracked):
  outliers.csv    gamma > 1 counts per cluster (Fig. 2)
  aggregate.csv   pooled Spearman / AUROC@q90 of gradient-boosting regressors on gamma_bulk vs gamma_cluster + cluster
                  label; 5-fold GroupKFold by frame and temporal ALC-2 -> ALC-3 / ALC-4 (Fig. 3)
  by_cluster.csv  the GroupKFold predictions scored within each cluster (Fig. 4)
  recall.csv      recall of the top-10% error minority-cluster atoms vs labeling budget (Fig. 5)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.dopt import RANK, SEED0, THRESHOLD, atom_rows_from_indices, random_restart_maxvol, gamma_for_rows, \
    spearman, auroc_high_error, recall_curve
from common.paths import RETRO_WORK as WORK, RETRO_RESULTS as RES, POOLS as _POOLS

N_TRIALS = 10
POOLS = {k: _POOLS[k] for k in ("alc2", "alc3", "alc4")}


def fit_bulk_groupkfold(gamma_bulk, y, groups, seed):
    X = pd.DataFrame({"gamma_bulk": gamma_bulk})
    pre = ColumnTransformer([("num", StandardScaler(), ["gamma_bulk"])])
    model = Pipeline([("pre", pre), ("est", HistGradientBoostingRegressor(random_state=seed))])
    gkf = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])
        preds[te] = model.predict(X.iloc[te])
    return preds


def fit_cid_groupkfold(cluster, gamma_cluster, y, groups, seed):
    X = pd.DataFrame({"gamma_cluster": gamma_cluster, "cluster": cluster})
    pre = ColumnTransformer(
        [("num", StandardScaler(), ["gamma_cluster"]),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cluster"])]
    )
    model = Pipeline([("pre", pre), ("est", HistGradientBoostingRegressor(random_state=seed))])
    gkf = GroupKFold(n_splits=min(5, pd.Series(groups).nunique()))
    preds = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        model.fit(X.iloc[tr], y[tr])
        preds[te] = model.predict(X.iloc[te])
    return preds, model


def fit_bulk_temporal(gamma_bulk_tr, y_tr, gamma_bulk_te, seed):
    Xtr = pd.DataFrame({"gamma_bulk": gamma_bulk_tr})
    Xte = pd.DataFrame({"gamma_bulk": gamma_bulk_te})
    pre = ColumnTransformer([("num", StandardScaler(), ["gamma_bulk"])])
    model = Pipeline([("pre", pre), ("est", HistGradientBoostingRegressor(random_state=seed))])
    model.fit(Xtr, y_tr)
    return model.predict(Xte)


def fit_cid_temporal(cluster_tr, gamma_cluster_tr, y_tr, cluster_te, gamma_cluster_te, seed):
    Xtr = pd.DataFrame({"gamma_cluster": gamma_cluster_tr, "cluster": cluster_tr})
    Xte = pd.DataFrame({"gamma_cluster": gamma_cluster_te, "cluster": cluster_te})
    pre = ColumnTransformer(
        [("num", StandardScaler(), ["gamma_cluster"]),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cluster"])]
    )
    model = Pipeline([("pre", pre), ("est", HistGradientBoostingRegressor(random_state=seed))])
    model.fit(Xtr, y_tr)
    return model.predict(Xte)


def main():
    kind = np.array([r["kind"] for r in csv.DictReader(open(WORK / "atom_manifest.csv"))])
    source = np.array([r["source_file"] for r in csv.DictReader(open(WORK / "atom_manifest.csv"))])
    dft_atom_ids = np.where(kind == "dft")[0]
    ch_atom_ids = np.where(kind == "chimes")[0]
    A_atomic = np.load(WORK / "A_atomic.npy")
    A_dft = A_atomic[atom_rows_from_indices(dft_atom_ids)]
    A_ch = A_atomic[atom_rows_from_indices(ch_atom_ids)]

    errs = pd.read_csv(WORK / "atom_errors.csv").set_index("global_atom_idx")
    y_all = errs.loc[ch_atom_ids, "mae"].to_numpy(dtype=np.float64)
    global_frame = errs.loc[ch_atom_ids, "global_frame_idx"].to_numpy()
    ch_source = source[ch_atom_ids]
    pool_mask = {p: (ch_source == f) for p, f in POOLS.items()}

    dft_labels = np.load(WORK / "labels_dft.npy")
    ch_labels = np.load(WORK / "labels_candidates.npy")
    clusters = sorted(int(c) for c in np.unique(dft_labels))
    hidden = {0, 3, 4}
    hidden_mask = np.isin(ch_labels, list(hidden))

    budgets = np.concatenate([np.linspace(0.01, 0.2, 20), np.linspace(0.22, 1.0, 20)])

    agg_rows, bycluster_rows, outlier_rows, recall_rows = [], [], [], []

    for trial in range(N_TRIALS):
        rng = np.random.default_rng(SEED0 + trial)
        print(f"\n=== trial {trial} ===", flush=True)

        inv_bulk = random_restart_maxvol(A_dft, rng)
        gamma_bulk = gamma_for_rows(A_ch, inv_bulk)
        b_out = gamma_bulk > THRESHOLD

        gamma_cluster = np.full(ch_atom_ids.size, np.nan, dtype=np.float64)
        for cid in clusters:
            dft_global = dft_atom_ids[dft_labels == cid]
            ch_local = np.where(ch_labels == cid)[0]
            if dft_global.size < RANK or ch_local.size == 0:
                continue
            A_ref = A_atomic[atom_rows_from_indices(dft_global)]
            inv_c = random_restart_maxvol(A_ref, rng)
            A_cand = A_atomic[atom_rows_from_indices(ch_atom_ids[ch_local])]
            gamma_cluster[ch_local] = gamma_for_rows(A_cand, inv_c)
        c_out = gamma_cluster > THRESHOLD

        # --- outliers (fig2) ---
        c_only = int((c_out & ~b_out).sum())
        outlier_rows.append({"trial": trial, "cluster": "ALL", "n": int(ch_atom_ids.size),
                              "cluster_outliers": int(c_out.sum()), "bulk_outliers": int(b_out.sum()),
                              "cluster_only": c_only, "both": int((c_out & b_out).sum())})
        for cid in clusters:
            m = ch_labels == cid
            n = int(m.sum())
            if n == 0:
                continue
            co = int((c_out[m] & ~b_out[m]).sum())
            outlier_rows.append({"trial": trial, "cluster": cid, "n": n,
                                  "cluster_outliers": int(c_out[m].sum()), "bulk_outliers": int(b_out[m].sum()),
                                  "cluster_only": co, "both": int((c_out[m] & b_out[m]).sum())})

        # --- GroupKFold protocol (fig3, fig4) ---
        pred_bulk_gkf = fit_bulk_groupkfold(gamma_bulk, y_all, global_frame, seed=SEED0 + trial)
        pred_cid_gkf, cid_model_gkf_full = fit_cid_groupkfold(ch_labels, gamma_cluster, y_all, global_frame, seed=SEED0 + trial)
        agg_rows.append({
            "trial": trial, "protocol": "groupkfold",
            "bulk_spearman": spearman(pred_bulk_gkf, y_all), "bulk_auroc_q90": auroc_high_error(pred_bulk_gkf, y_all),
            "cid_spearman": spearman(pred_cid_gkf, y_all), "cid_auroc_q90": auroc_high_error(pred_cid_gkf, y_all),
        })
        for cid in clusters:
            m = ch_labels == cid
            if m.sum() < 20:
                continue
            bycluster_rows.append({
                "trial": trial, "cluster": cid, "niche": "hidden" if cid in hidden else "populous", "n": int(m.sum()),
                "bulk_spearman": spearman(pred_bulk_gkf[m], y_all[m]), "bulk_auroc_q90": auroc_high_error(pred_bulk_gkf[m], y_all[m]),
                "cid_spearman": spearman(pred_cid_gkf[m], y_all[m]), "cid_auroc_q90": auroc_high_error(pred_cid_gkf[m], y_all[m]),
            })

        # --- temporal protocols (fig3 only) ---
        tr_mask = pool_mask["alc2"]
        for test_pool in ("alc3", "alc4"):
            te_mask = pool_mask[test_pool]
            pb = fit_bulk_temporal(gamma_bulk[tr_mask], y_all[tr_mask], gamma_bulk[te_mask], seed=SEED0 + trial)
            pc = fit_cid_temporal(ch_labels[tr_mask], gamma_cluster[tr_mask], y_all[tr_mask],
                                   ch_labels[te_mask], gamma_cluster[te_mask], seed=SEED0 + trial)
            agg_rows.append({
                "trial": trial, "protocol": f"temporal_alc2_to_{test_pool}",
                "bulk_spearman": spearman(pb, y_all[te_mask]), "bulk_auroc_q90": auroc_high_error(pb, y_all[te_mask]),
                "cid_spearman": spearman(pc, y_all[te_mask]), "cid_auroc_q90": auroc_high_error(pc, y_all[te_mask]),
            })

        # --- fig5: recall-at-budget within hidden regime, using TRAINED cid model score ---
        y_h = y_all[hidden_mask]
        thr_h = np.quantile(y_h, 0.90)
        is_info = y_h > thr_h
        rc_bulk = recall_curve(pred_bulk_gkf[hidden_mask], is_info, budgets)
        rc_cid = recall_curve(pred_cid_gkf[hidden_mask], is_info, budgets)
        for b, rb, rc in zip(budgets, rc_bulk, rc_cid):
            recall_rows.append({"trial": trial, "budget": b, "recall_bulk": rb, "recall_cid": rc})

        print(f"  groupkfold: bulk sp={agg_rows[-3]['bulk_spearman']:.3f} cid sp={agg_rows[-3]['cid_spearman']:.3f}  "
              f"recall@10%: bulk={np.interp(0.10,budgets,rc_bulk):.3f} cid={np.interp(0.10,budgets,rc_cid):.3f}", flush=True)

    outdir = RES
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agg_rows).to_csv(outdir / "aggregate.csv", index=False)
    pd.DataFrame(bycluster_rows).to_csv(outdir / "by_cluster.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(outdir / "outliers.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(outdir / "recall.csv", index=False)
    print("\nWrote all outputs to", outdir, flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
