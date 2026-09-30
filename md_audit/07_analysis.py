#!/usr/bin/env python3
"""Step 7: does gamma rank the DFT-only model's true (DFT) force error on its own MD? (Fig. 6A tables)
Reads work/atom_errors_<model>.csv (step 6). Writes results/audit_<model>_{atom,frame,flags}.csv.
Metrics per statepoint: Spearman, AUROC for top-10% error, recall of top-10% error at a 10% labelling budget,
within-cluster Spearman, error of robust cluster-only flags (gamma_cluster>1 in >=80% of MaxVol trials,
gamma_bulk never >1) vs unflagged, and frame-level ranking (frame max/mean gamma vs frame force RMSE).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def recall_at(score, y, budget=0.10, q=0.90):
    info = y > np.quantile(y, q); k = max(1, int(round(budget * len(y))))
    return info[np.argsort(-score)[:k]].sum() / info.sum()


def auroc(score, y, q=0.90):
    return roc_auc_score(y > np.quantile(y, q), score)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "shared_t0_dft_train"
    df = pd.read_csv(MD / "work" / f"atom_errors_{model}.csv")
    rng = np.random.default_rng(0)
    atom, frame, flags = [], [], []
    for sp, d in df.groupby("statepoint"):
        y = d.err.to_numpy()
        for name, s in [("gamma_bulk", d.gamma_bulk), ("gamma_cluster", d.gamma_cluster),
                        ("cluster_id_mean_err_oracle", d.groupby("cluster").err.transform("mean")),
                        ("random", pd.Series(rng.random(len(d))))]:
            s = s.to_numpy()
            r = dict(statepoint=sp, score=name, n=len(d), spearman=spearmanr(s, y)[0], auroc_q90=auroc(s, y),
                     recall10=recall_at(s, y))
            for c, dc in d.groupby("cluster"):
                if len(dc) >= 200 and name.startswith("gamma"):
                    r[f"sp_c{c}"] = spearmanr(dc[name], dc.err)[0]
            atom.append(r)
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), gb_max=("gamma_bulk", "max"),
                                    gc_max=("gamma_cluster", "max"), gb_mean=("gamma_bulk", "mean"),
                                    gc_mean=("gamma_cluster", "mean"), step=("step", "first"))
        frame.append(dict(statepoint=sp, n_frames=len(fr), **{f"sp_{k}": spearmanr(fr[k], fr.rmse)[0]
                                                               for k in ["gb_max", "gc_max", "gb_mean", "gc_mean", "step"]}))
        rc = (d.pc_flag >= 0.8) & (d.pb_flag == 0); rb = (d.pb_flag >= 0.8); un = (d.pc_flag == 0) & (d.pb_flag == 0)
        flags.append(dict(statepoint=sp, n_cluster_only=int(rc.sum()), n_bulk=int(rb.sum()), n_unflagged=int(un.sum()),
                          err_cluster_only=d.err[rc].median(), err_bulk=d.err[rb].median(), err_unflagged=d.err[un].median(),
                          frac_top10_cluster_only=(d.err[rc] > d.err.quantile(.9)).mean(),
                          frac_top10_bulk=(d.err[rb] > d.err.quantile(.9)).mean()))
    pd.set_option("display.width", 250)
    for tag, rows in [("atom", atom), ("frame", frame), ("flags", flags)]:
        t = pd.DataFrame(rows); t.to_csv(MD / "results" / f"audit_{model}_{tag}.csv", index=False)
        print(f"\n== {tag}\n" + t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
