#!/usr/bin/env python3
"""Step 9: score M3 of CRITERIA.md (addendum M) from results/minrule_<name>*.csv (step 8) and the bulk grade and
representation of arm A (results/shift_<name>.csv). Scenarios and bootstrap as C3 (f <= 0.05, representation < 0.5,
10,000 resamples of scenarios, seed 0). Writes results/min_rule_criteria.csv.
usage: 09_min_criteria.py [si_gap18 sio2 water]
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
    crit = []
    for name in names:
        files = list(C.RES.glob(f"minrule_{name}*.csv"))
        if not files:
            print(f"{name}: no results"); continue
        mr = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["regime", "frac", "draw", "partition", "rule"])
        a = mr.pivot_table(index=["regime", "frac", "draw"], columns=["partition", "rule"], values="auroc")
        a.columns = [f"{p}_{r}" for p, r in a.columns]
        sh = pd.read_csv(C.RES / f"shift_{name}.csv"); sh = sh[(sh.element == "all") & (sh.partition == "bulk")]
        b = sh.set_index(["regime", "frac", "draw"])[["auroc", "rep_R"]].rename(columns={"auroc": "bulk"})
        m = a.join(b).reset_index()
        u = m[(m.frac <= 0.05) & (m.rep_R < 0.5)]
        mm, lo, hi = boot(u.regime_assigned - u.regime_min)
        crit.append(dict(name=name, criterion="M3", value=mm, passed=bool(lo > 0),
                         detail=f"AUROC regime assigned - regime min {mm:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(u)} "
                                f"(scenarios {len(m)} of {len(b)})"))
        for p in ("regime", "cluster"):
            for r in ("assigned", "min"):
                v, l, h = boot(u[f"{p}_{r}"] - u.bulk)
                crit.append(dict(name=name, criterion=f"(descriptive) {p}_{r} - bulk", value=v, passed=np.nan,
                                 detail=f"{v:+.3f} [{l:+.3f}, {h:+.3f}]"))
        pd.set_option("display.width", 250)
        print(f"\n===== {name}: AUROC at f = 0.05 (mean over draws)\n" +
              m[m.frac == 0.05].groupby("regime")[[c for c in m.columns if c.endswith(("assigned", "min"))] + ["bulk", "rep_R"]].mean().round(3).to_string())
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "min_rule_criteria.csv", index=False)
    print("\n===== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["si_gap18", "sio2", "water"])
