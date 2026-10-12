#!/usr/bin/env python3
"""Step 11: score H3 of CRITERIA.md (addendum H) from results/infogain_<name>*.csv (step 10), with h_bulk from
results/leverage_<name>*.csv (step 6) and gamma_bulk / gamma_regime / representation from results/shift_<name>.csv.
C3 scenarios (f <= 0.05, representation < 0.5), 10,000 bootstrap resamples of scenarios, seed 0.
Writes results/info_gain_criteria.csv.
usage: 11_info_criteria.py [si_gap18 sio2 water]
"""
import sys

import numpy as np
import pandas as pd

import core as C

KEY = ["regime", "frac", "draw"]


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main(names):
    crit = []
    for name in names:
        files = list(C.RES.glob(f"infogain_{name}*.csv"))
        if not files:
            print(f"{name}: no results"); continue
        ig = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(KEY + ["partition", "score"])
        a = ig.pivot_table(index=KEY, columns=["partition", "score"], values="auroc")
        a.columns = [f"{s}_{p}" for p, s in a.columns]
        h = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"leverage_{name}*.csv") if "_al_" not in f.name])
        h = h[h.element == "all"].drop_duplicates(KEY + ["partition"]).pivot_table(index=KEY, columns="partition", values="auroc").add_prefix("h_")
        g = pd.read_csv(C.RES / f"shift_{name}.csv"); g = g[g.element == "all"]
        rep = g[g.partition == "bulk"].set_index(KEY).rep_R
        g = g.pivot_table(index=KEY, columns="partition", values="auroc").add_prefix("gamma_")
        m = a.join(h).join(g).join(rep).reset_index()
        u = m[(m.frac <= 0.05) & (m.rep_R < 0.5)]
        mm, lo, hi = boot(u.IG_hier_regime - u.h_bulk)
        crit.append(dict(name=name, criterion="H3", value=mm, passed=bool(lo > 0),
                         detail=f"AUROC IG_hier(regime) - h_bulk {mm:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(u)}"))
        for c in [c for c in m.columns if c.startswith(("IG_", "h_", "gamma_"))]:
            v, l, hh = boot(u[c] - u.h_bulk)
            crit.append(dict(name=name, criterion=f"(descriptive) {c} - h_bulk", value=v, passed=np.nan,
                             detail=f"mean AUROC {u[c].mean():.3f}; difference {v:+.3f} [{l:+.3f}, {hh:+.3f}]"))
        print(f"\n===== {name}: chosen t of IG_hier (scenario counts)\n" + ig[ig.score == "IG_hier"].groupby("partition").t_hier.value_counts().to_string())
        pd.set_option("display.width", 250)
        cols = [c for c in m.columns if c.startswith(("IG_", "h_", "gamma_"))]
        print(f"\n===== {name}: AUROC at f = 0.05 (mean over draws)\n" + m[m.frac == 0.05].groupby("regime")[cols].mean().round(3).to_string())
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "info_gain_criteria.csv", index=False)
    print("\n===== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["si_gap18", "sio2", "water"])
