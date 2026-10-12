#!/usr/bin/env python3
"""Step 7: score R2-1 to R2-3 of CRITERIA.md (addendum of 2026-10-12) on the open data.
Joins results/leverage_<name>*.csv (step 6) with the gamma results of arm A (results/shift_<name>.csv) on
(regime, frac, draw). Scenarios and bootstrap as C3 in 04_criteria.py: f <= 0.05, bulk-basis representation < 0.5,
10,000 resamples of scenarios, seed 0. Writes results/robustness_criteria.csv and results/robustness_by_regime.csv.
usage: 07_robustness.py [si_gap18 sio2 water]
"""
import sys

import numpy as np
import pandas as pd

import core as C


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(names):
    crit, per = [], []
    for name in names:
        g = pd.read_csv(C.RES / f"shift_{name}.csv")
        g = g[g.element == "all"].pivot_table(index=["regime", "frac", "draw"], columns="partition", values=["auroc", "rep_R"])
        rep = g["rep_R"]["bulk"].rename("rep_R")
        g = g["auroc"].add_prefix("gamma_")
        h = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"leverage_{name}*.csv") if "_al_" not in f.name])
        h = h[h.element == "all"].drop_duplicates(["regime", "frac", "draw", "partition"])
        h = h.pivot_table(index=["regime", "frac", "draw"], columns="partition", values="auroc").add_prefix("h_")
        m = g.join(h, how="inner").join(rep).reset_index()
        assert len(m) == len(g), f"{name}: {len(m)} scenarios with leverage, {len(g)} with gamma"
        u = m[(m.frac <= 0.05) & (m.rep_R < 0.5)]
        d1 = u.h_bulk - u.gamma_bulk
        d_h, d_g = u.h_regime - u.h_bulk, u.gamma_regime - u.gamma_bulk
        mh, lh, hh = boot(d_h)
        crit += [
            dict(name=name, criterion="R2-1", value=d1.mean(), passed=bool(abs(d1.mean()) < 0.05),
                 detail=f"mean AUROC h_bulk {u.h_bulk.mean():.3f} vs gamma_bulk {u.gamma_bulk.mean():.3f}, n={len(u)}"),
            dict(name=name, criterion="R2-2", value=mh, passed=bool(lh > 0),
                 detail=f"Delta(h_regime - h_bulk) {mh:.3f} [{lh:.3f}, {hh:.3f}]"),
            dict(name=name, criterion="R2-3", value=d_g.mean() - d_h.mean(), passed=bool(abs(d_g.mean() - d_h.mean()) < 0.05),
                 detail=f"Delta gamma_regime {d_g.mean():.3f} vs Delta h_regime {d_h.mean():.3f}"),
        ]
        extra = [c for c in ("element", "cluster") if f"h_{c}" in u]
        for c in extra:
            dh, dg = u[f"h_{c}"] - u.h_bulk, u[f"gamma_{c}"] - u.gamma_bulk
            mm, ll, hi = boot(dh)
            crit.append(dict(name=name, criterion=f"(descriptive) h_{c}", value=mm, passed=np.nan,
                             detail=f"Delta(h_{c} - h_bulk) {mm:.3f} [{ll:.3f}, {hi:.3f}]; Delta gamma_{c} {dg.mean():.3f}"))
        r = m[m.frac == 0.05].groupby("regime")[[c for c in m.columns if c.startswith(("gamma_", "h_"))] + ["rep_R"]].mean()
        r.insert(0, "name", name); per.append(r.reset_index())
        pd.set_option("display.width", 250)
        print(f"\n===== {name}: AUROC at f = 0.05 (mean over draws)\n" + r.round(3).to_string())
    fa = C.RES / "leverage_al_sio2.csv"
    if fa.exists():
        al = pd.read_csv(fa); al = al[(al.kind == "parent") & (al.element == "all")].pivot_table(index="group", columns="partition", values="auroc")
        ga = pd.read_csv(C.RES / "al_sio2.csv"); ga = ga[(ga.kind == "parent") & (ga.element == "all")].pivot_table(index="group", columns="partition", values="auroc")
        print("\n===== sio2 arm B, AUROC by parent regime\n" + ga.add_prefix("gamma_").join(al.add_prefix("h_")).round(3).to_string())
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "robustness_criteria.csv", index=False)
    pd.concat(per).to_csv(C.RES / "robustness_by_regime.csv", index=False)
    print("\n===== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["si_gap18", "sio2", "water"])
