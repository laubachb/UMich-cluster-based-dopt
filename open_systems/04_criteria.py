#!/usr/bin/env python3
"""Step 4: score criteria C1-C6 of CRITERIA.md from results/shift_<name>.csv and results/al_sio2*.csv.
Delta = metric of a partition grade - metric of gamma_bulk on the same candidate atoms (AUROC; Spearman and recall
are printed alongside). Units for the bootstrap = (regime, draw) pairs; 10,000 resamples, seed 0.
Writes results/criteria.csv and results/shift_summary.csv.
usage: 04_criteria.py [si_gap18 sio2]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import core as C

METRICS = ["auroc", "spearman", "recall10"]


def deltas(df, part, base="bulk"):
    a = df[df.element == "all"].set_index(["regime", "frac", "draw", "partition"])[METRICS + ["rep_R"]]
    p, b = a.xs(part, level="partition"), a.xs(base, level="partition")
    out = (p[METRICS] - b[METRICS]).add_prefix("d_")
    out["rep_R"] = b.rep_R
    return out.reset_index()


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(names):
    crit, summ = [], []
    for name in names:
        f = C.RES / f"shift_{name}.csv"
        if not f.exists():
            print(f"{name}: no {f.name} yet"); continue
        df = pd.read_csv(f)
        rep = df[(df.partition == "bulk") & (df.element == "all")].groupby(["regime", "frac"]).rep_R.mean().unstack()
        print(f"\n===== {name}: bulk-basis representation of R (mean over draws)\n{rep.round(2).to_string()}")
        parts = [p for p in ("regime", "cluster", "element") if p in df.partition.unique()]
        D = {p: deltas(df, p) for p in parts}
        for p in parts:
            s = D[p].groupby(["frac", D[p].rep_R < 0.5]).agg(n=("d_auroc", "size"), **{f"d_{m}": (f"d_{m}", "mean") for m in METRICS})
            s = s.reset_index().rename(columns={"rep_R": "under_represented"}); s.insert(0, "partition", p); s.insert(0, "name", name)
            summ.append(s)
            print(f"\n-- {name}, Delta({p} - bulk) by fraction and representation < 0.5\n{s.round(3).to_string(index=False)}")
        c1 = (rep[0.25] < 0.5).sum() if 0.25 in rep else np.nan
        crit.append(dict(name=name, criterion="C1", value=c1, detail="regimes with representation < 0.5 at f = 0.25",
                         passed=bool(c1 >= 1)))
        d = D["regime"]; u = d[d.rep_R < 0.5]
        lo, hi = u[u.frac == 0.02].d_auroc.mean(), u[u.frac == 0.25].d_auroc.mean()
        crit.append(dict(name=name, criterion="C2", value=lo - hi, detail=f"mean Delta(regime) f=0.02 {lo:.3f} vs f=0.25 {hi:.3f}",
                         passed=bool(lo > hi)))
        for cname, p in (("C3", "regime"), ("C4", "cluster")):
            d = D[p]; u = d[(d.rep_R < 0.5) & (d.frac <= 0.05)]
            m, l, h = boot(u.d_auroc)
            crit.append(dict(name=name, criterion=cname, value=m, detail=f"Delta({p}) at f<=0.05, rep<0.5: {m:.3f} [{l:.3f}, {h:.3f}], n={len(u)}",
                             passed=bool(l > 0)))
        fa = C.RES / f"agreement_{name}.csv"
        if fa.exists():                                   # C7 (addendum of 2026-10-09): training-only agreement A_R
            ag = pd.read_csv(fa)[["regime", "frac", "draw", "A_R"]]
            m = D["cluster"].merge(ag, on=["regime", "frac", "draw"])
            rho = spearmanr(m.A_R, m.d_auroc)[0]
            units = m.groupby(["regime", "draw"]).indices; keys = list(units)
            rng = np.random.default_rng(0); bs = []
            for _ in range(10_000):
                idx = np.concatenate([units[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
                bs.append(spearmanr(m.A_R.iloc[idx], m.d_auroc.iloc[idx])[0])
            lo, hi = np.nanquantile(bs, [0.025, 0.975])
            low, high = m[m.A_R < 0.5].d_auroc, m[m.A_R >= 0.8].d_auroc
            crit.append(dict(name=name, criterion="C7", value=rho,
                             detail=f"Spearman(A_R, Delta cluster) {rho:.3f} [{lo:.3f}, {hi:.3f}], n={len(m)}; mean Delta A_R<0.5: "
                                    f"{low.mean():.3f} (n={len(low)}), A_R>=0.8: {high.mean():.3f} (n={len(high)}); A_R range "
                                    f"{m.A_R.min():.2f}-{m.A_R.max():.2f}",
                             passed=bool(hi < 0 and len(low) > 0 and len(high) > 0 and low.mean() > high.mean())))
            print(f"\n-- {name}: A_R and Delta(cluster) per regime (mean over f, draws)\n" +
                  m.groupby("regime")[["A_R", "d_auroc", "rep_R"]].mean().round(3).to_string())
        d = D["cluster"]; w = d[d.rep_R >= 1].groupby("frac").d_auroc.mean()
        crit.append(dict(name=name, criterion="C5", value=float(w.min()) if len(w) else np.nan,
                         detail="min over f of mean Delta(cluster) where rep>=1: " + ", ".join(f"f={k}: {v:.3f}" for k, v in w.items()),
                         passed=bool(len(w) == 0 or w.min() > -0.05)))
    fa, fr = C.RES / "al_sio2.csv", C.RES / "al_sio2_rep.csv"
    if fa.exists():
        al, rep = pd.read_csv(fa), pd.read_csv(fr)
        r_all = rep[rep.element == "all"].set_index("regime").representation
        a = al[(al.kind == "parent") & (al.element == "all")].pivot_table(index="group", columns="partition", values="auroc")
        a["rep_parent"] = r_all.reindex(a.index)
        for p in ("cluster", "regime", "element"):
            if p in a:
                a[f"d_{p}"] = a[p] - a["bulk"]
        print(f"\n===== sio2 arm B (AUROC by parent regime)\n{a.round(3).to_string()}")
        u = a[a.rep_parent < 0.5]
        ok = len(u) > 0 and (u.d_cluster > 0).all() and (u.d_regime.dropna() > 0).all()
        crit.append(dict(name="sio2", criterion="C6", value=float(u.d_cluster.mean()) if len(u) else np.nan,
                         detail="; ".join(f"{g}: rep {r.rep_parent:.2f}, dCluster {r.d_cluster:.3f}, dRegime {r.d_regime:.3f}, "
                                          f"dElement {r.get('d_element', np.nan):.3f}" for g, r in u.iterrows()) or "no parent regime with rep < 0.5",
                         passed=bool(ok)))
    cr = pd.DataFrame(crit)
    if len(cr):
        cr.to_csv(C.RES / "criteria.csv", index=False)
        print("\n===== criteria\n" + cr.to_string(index=False))
    if summ:
        pd.concat(summ).to_csv(C.RES / "shift_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["si_gap18", "sio2", "water"])
