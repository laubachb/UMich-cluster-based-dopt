#!/usr/bin/env python3
"""Step 14: hybrid grades on the DFT-only model's own MD (companion of retrospective/07_raw_and_hybrid.py).

Per MD atom and MaxVol solution t (the 10 solutions of 03_gamma.py, work/gamma_md.npz):
  bulk        gamma_bulk
  cluster     gamma_cluster
  hyb_max     max(gamma_bulk, gamma_cluster)
  hyb_switch  gamma_cluster in clusters under-represented in that solution's bulk basis, gamma_bulk elsewhere.
              The bulk basis is recomputed with the same seed and call order as 03_gamma.py (so it is the same basis),
              and representation = (share of basis rows from cluster c's training atoms) / (share of training rows in c)
  hyb_rank    max of the two grades' percentile ranks over all MD atoms of the model
Scores are averaged over solutions, then scored against the true force error (work/atom_errors_<model>.csv):
atom-level Spearman and AUROC@q90 per state point, and frame-level Spearman (frame-max score vs frame force RMSE) with
a 95% frame-bootstrap interval. Writes results/hybrid_md_atom.csv, hybrid_md_frame.csv, hybrid_md_representation.csv.
Also flags (gamma > 1 in >= 80% of solutions) per grade in under- and well-represented clusters
-> results/hybrid_md_flags.csv.
usage: 14_hybrid_md.py [model]
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, RETRO_WORK as RW

N_BOOT = 1000


def basis_rows(A_ref, inv):
    B = np.abs(A_ref @ inv)
    rows = np.where((np.abs(B.max(1) - 1) < 1e-8) & (np.abs(B.sum(1) - 1) < 1e-6))[0]
    assert len(rows) == inv.shape[0]
    return rows


def pct_rank(x):
    r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(len(x)); return r / (len(x) - 1)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "shared_t0_dft_train"
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    lab = np.full(len(kind), -1)
    lab[kind == "dft"] = np.load(RW / "labels_dft.npy"); lab[kind == "chimes"] = np.load(RW / "labels_candidates.npy")
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / model / "train_meta.json"))
    ref = np.array(sorted(a for f in meta["files"] for i in f["indices"] for a in by_frame[f["source"], i]))
    A_ref = A[E.atom_rows_from_indices(ref)]; row_lab = np.repeat(lab[ref], 3)

    g = np.load(MD / "work" / "gamma_md.npz")
    atoms, gb, gc, cl = g[f"{model}__atoms"], g[f"{model}__gb"], g[f"{model}__gc"], g["cluster"][g[f"{model}__atoms"]]
    T = gb.shape[0]
    rep, sw = [], np.empty_like(gb)
    for t in range(T):
        inv_b = E.random_restart_maxvol(A_ref, np.random.default_rng(E.SEED0 + t))   # first draw of 03's rng
        brow = row_lab[basis_rows(A_ref, inv_b)]
        under = []
        for c in range(6):
            r = (brow == c).mean() / (row_lab == c).mean()
            rep.append(dict(trial=t, cluster=c, basis_rows=int((brow == c).sum()), ratio=r)); under += [c] if r < 1 else []
        sw[t] = np.where(np.isin(cl, under), gc[t], gb[t])
        print(f"solution {t}: under-represented {under}", flush=True)
    rk = np.array([np.maximum(pct_rank(gb[t]), pct_rank(gc[t])) for t in range(T)])
    S = {"bulk": gb.mean(0), "cluster": gc.mean(0), "hyb_max": np.maximum(gb, gc).mean(0),
         "hyb_switch": sw.mean(0), "hyb_rank": rk.mean(0)}

    # join to the true error, atom by atom (same order as 06_errors.py)
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    pos = {a: i for i, a in enumerate(atoms)}
    df = pd.read_csv(MD / "work" / f"atom_errors_{model}.csv")
    k = np.array([pos[start[f] + a] for f, a in zip(df.frame, df.atom)])
    for s, v in S.items():
        df[s] = v[k]
    rng = np.random.default_rng(0)
    arows, frows = [], []
    for sp, d in df.groupby("statepoint"):
        y = d.err.to_numpy()
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in S})
        idx = rng.integers(0, len(fr), (N_BOOT, len(fr)))
        for s in S:
            arows.append(dict(statepoint=sp, score=s, spearman=E.spearman(d[s].to_numpy(), y),
                              auroc_q90=E.auroc_high_error(d[s].to_numpy(), y)))
            x, r = fr[s].to_numpy(), fr.rmse.to_numpy()
            bs = [spearmanr(x[i], r[i])[0] for i in idx]
            frows.append(dict(statepoint=sp, score=s, spearman=spearmanr(x, r)[0],
                              lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
    # flags (gamma > 1 in >= 80% of solutions, as 07_analysis.py): bulk, cluster, switch; precision = share of flagged
    # atoms in their state point's top-10% error; per cluster group (under-represented vs well-represented)
    under_any = sorted({r["cluster"] for r in rep if r["ratio"] < 1})
    fb, fc, fs = [((x > 1).mean(0) >= 0.8)[k] for x in (gb, gc, sw)]
    flg = []
    for sp, d in df.groupby("statepoint"):
        m = (df.statepoint == sp).to_numpy(); top = (df.err > df.err[m].quantile(0.9)).to_numpy()
        lab_atom = cl[k]                                  # df.cluster was overwritten by the gamma_cluster score
        for grp, cm in (("under-represented", np.isin(lab_atom, under_any)), ("well-represented", ~np.isin(lab_atom, under_any))):
            mm = m & cm
            for name, f in (("bulk", fb), ("cluster", fc), ("switch", fs)):
                n = int((f & mm).sum())
                flg.append(dict(statepoint=sp, clusters=grp, grade=name, atoms=int(mm.sum()), flagged=n,
                                precision_top10=(f & mm & top).sum() / n if n else np.nan,
                                median_err_flagged=float(df.err[f & mm].median()) if n else np.nan))
    R = MD / "results"
    pd.DataFrame(flg).to_csv(R / "hybrid_md_flags.csv", index=False)
    print(pd.DataFrame(flg).round(3).to_string(index=False))
    pd.DataFrame(arows).to_csv(R / "hybrid_md_atom.csv", index=False)
    pd.DataFrame(frows).to_csv(R / "hybrid_md_frame.csv", index=False)
    pd.DataFrame(rep).to_csv(R / "hybrid_md_representation.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(arows).pivot(index="score", columns="statepoint", values="spearman").round(3))
    print(pd.DataFrame(frows).round(3).to_string(index=False))
    print(pd.DataFrame(rep).groupby("cluster")[["basis_rows", "ratio"]].mean().round(3))


if __name__ == "__main__":
    main()
