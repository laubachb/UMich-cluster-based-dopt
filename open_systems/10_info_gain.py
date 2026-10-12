#!/usr/bin/env python3
"""Step 10: test H3 of CRITERIA.md (addendum H): information-gain grades on the open data, arm A.

For every arm-A scenario (split and error model exactly as 02_shift.py) and the partitions regime (element|regime) and,
with --cluster, the manuscript's clustering: regime noise sigma_k = RMS force-component residual of group k under the
ridge fit of the training rows (the error model's own fit); IG_hom (one sigma), IG_het (sigma_k), IG_hier (partial
pooling, tau by marginal likelihood) via core.BayesGrade; atom score = information gain of its 3 rows.
Writes results/infogain_NAME[TAG].csv (one row per scenario, partition and score, with the chosen t).
usage: 10_info_gain.py NAME [--cluster] [--jobs N] [--only REGIME ...] [--tag TAG]
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

MIN_FRAMES = 60
G = {}


def split(R, f, d):
    """identical to the training/candidate split of 02_shift.scenario."""
    D, tab = G["D"], G["tab"]
    base = tab[~tab.al]
    regimes = sorted(base.regime.unique())
    rng = np.random.default_rng(10_000 + 1000 * d + 10 * regimes.index(R) + int(round(100 * f)))
    tr = []
    for reg, g in base.groupby("regime"):
        fr = g.frame.to_numpy()
        n = max(2, int(round(f * len(fr)))) if reg == R else int(round(0.75 * len(fr)))
        tr.append(rng.choice(fr, n, replace=False))
    tr = np.concatenate(tr)
    train = np.isin(D["frame"], tr)
    cand = (D["regime"] == R) & ~train & np.isin(D["frame"], base.frame)
    return train, cand


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time()
    D = G["D"]; X, F = D["X"], D["F"]
    train, cand = split(R, f, d)
    err = C.error_model(D, train, cand)
    tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    Xr, yr, Xc = X[C.rows_of(tr_idx)], F[train].ravel(), X[C.rows_of(ca_idx)]
    coef = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(X.shape[1]), Xr.T @ yr)        # = the error model's fit
    parts = C.partitions(D, train, G["multi"], cluster=G["cluster"])
    rows = []
    for pname in [p for p in ("regime", "cluster") if p in parts]:
        lab = parts[pname]; gr = np.repeat(lab[tr_idx], 3)
        sigma, s0 = C.group_sigma(Xr, yr, gr, coef)
        sc = {"IG_het": C.BayesGrade(Xr, yr, gr, sigma, s0).info_gain(Xc, lab[ca_idx])}
        mh, t, ev = C.fit_hier(Xr, yr, gr, sigma, s0)
        sc["IG_hier"] = mh.info_gain(Xc, lab[ca_idx])
        if pname == "regime":
            sc["IG_hom"] = C.BayesGrade(Xr, yr, gr, {k: s0 for k in sigma}, s0).info_gain(Xc, lab[ca_idx])
        sR = [v for (k, v) in sigma.items() if (lab[ca_idx] == k).any()]
        for s, v in sc.items():
            rows.append(dict(name=G["name"], regime=R, frac=f, draw=d, partition=pname, score=s, n_cand=int(cand.sum()),
                             t_hier=t, sigma_R=float(np.mean(sR)), sigma_pooled=s0, **C.metrics(v, err)))
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, tab = C.load(a.name, a.cap)
    G.update(D=D, tab=tab, multi=len(np.unique(D["element"])) > 1, name=a.name, cluster=a.cluster)
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in a.fracs for d in a.draws]
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame([r for rs in out for r in rs])
    df.to_csv(C.RES / f"infogain_{a.name}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["regime", "frac"], columns=["partition", "score"], values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
