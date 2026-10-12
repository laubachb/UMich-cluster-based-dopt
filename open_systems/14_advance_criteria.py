#!/usr/bin/env python3
"""Step 14: score N1, N2, N3b and pick the N4 candidate from results/advance_<name>*.csv (step 13), with the
representation of arm A (results/shift_<name>.csv) for the C3 scenarios (f <= 0.05, rep < 0.5). Bootstrap over
scenarios, 10,000 resamples, seed 0. With --mlearn METHOD: scores N4 from results/advance_mlearn_*.csv.
Writes results/advance_criteria_open.csv (or results/advance_criteria_mlearn.csv).
usage: 14_advance_criteria.py [si_gap18 sio2 water] | 14_advance_criteria.py --mlearn gamma_soft
"""
import sys

import numpy as np
import pandas as pd

import core as C

KEY = ["regime", "frac", "draw"]
ELEMENTS = ["Cu", "Ge", "Li", "Mo", "Ni", "Si"]


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def table(name):
    df = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"advance_{name}*.csv")]).drop_duplicates(KEY + ["score"])
    a = df.pivot_table(index=KEY, columns="score", values="auroc")
    cov = df[df.score == "gamma_cal"].set_index(KEY)[["coverage", "coverage_pooled"]] if "coverage" in df else None
    return a, cov


def main(argv):
    if argv[:1] == ["--mlearn"]:
        method = argv[1]; rows = []
        for e in ELEMENTS:
            a, _ = table(f"mlearn_{e}"); u = a[a.index.get_level_values("frac") <= 0.05]
            rows.append(dict(element=e, d_cluster=(u[method] - u.gamma_cluster).mean(), d_min=(u[method] - u.gamma_min).mean(),
                             **{s: u[s].mean() for s in u.columns}))
        per = pd.DataFrame(rows).set_index("element")
        allu = pd.concat([table(f"mlearn_{e}")[0] for e in ELEMENTS]); allu = allu[allu.index.get_level_values("frac") <= 0.05]
        m1, l1, h1 = boot(allu[method] - allu.gamma_cluster); m2, l2, h2 = boot(allu[method] - allu.gamma_min)
        ok = l1 > 0 and l2 > 0 and (per.d_cluster > 0).sum() >= 4 and (per.d_min > 0).sum() >= 4
        cr = pd.DataFrame([dict(criterion="N4", method=method, passed=bool(ok),
                                detail=f"vs gamma_cluster {m1:+.3f} [{l1:+.3f}, {h1:+.3f}], positive in {(per.d_cluster > 0).sum()}/6; "
                                       f"vs gamma_min {m2:+.3f} [{l2:+.3f}, {h2:+.3f}], positive in {(per.d_min > 0).sum()}/6; n={len(allu)}")])
        cr.to_csv(C.RES / "advance_criteria_mlearn.csv", index=False)
        pd.set_option("display.width", 250); print(per.round(3).to_string()); print(cr.to_string(index=False)); return
    names = argv or ["si_gap18", "sio2", "water"]
    U, crit = {}, []
    for name in names:
        a, cov = table(name)
        sh = pd.read_csv(C.RES / f"shift_{name}.csv"); rep = sh[(sh.element == "all") & (sh.partition == "bulk")].set_index(KEY).rep_R
        m = a.join(rep); u = m[(m.index.get_level_values("frac") <= 0.05) & (m.rep_R < 0.5)]
        U[name] = u
        if cov is not None:
            c = cov.join(rep); c = c[(c.index.get_level_values("frac") <= 0.05) & (c.rep_R < 0.5)]
            crit.append(dict(name=name, criterion="N3b coverage", value=c.coverage.mean(), passed=bool(0.85 <= c.coverage.mean() <= 0.95),
                             detail=f"mean coverage {c.coverage.mean():.3f} (pooled quantile {c.coverage_pooled.mean():.3f}), n={len(c)}"))
        pd.set_option("display.width", 250)
        print(f"\n===== {name}: mean AUROC, C3 scenarios\n" + u.drop(columns="rep_R").mean().round(3).to_string())
    pooled = pd.concat(U.values())
    for s_, other, cname in (("gamma_soft", "gamma_min", "N1 soft vs min"), ("gamma_soft", "gamma_cluster", "N1 soft vs cluster")):
        m, lo, hi = boot(pooled[s_] - pooled[other])
        crit.append(dict(name="pooled", criterion=cname, value=m, passed=bool(lo > 0), detail=f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(pooled)}"))
    ens_parts = [U[n] for n in U if "gamma_ens" in U[n] and U[n].gamma_ens.notna().any()]
    ens = pd.concat(ens_parts) if ens_parts else pd.DataFrame()
    if len(ens):
        m, lo, hi = boot(ens.gamma_ens - ens.gamma_cluster)
        crit.append(dict(name="Si+water", criterion="N2 ens vs cluster", value=m, passed=bool(lo > 0), detail=f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(ens)}"))
    gains = {s_: (pooled[s_] - pooled.gamma_cluster).mean() for s_ in ("gamma_soft", "gamma_ens", "gamma_cal") if s_ in pooled and pooled[s_].notna().any()}
    crit.append(dict(name="pooled", criterion="N4 candidate (largest gain over gamma_cluster, within-regime ranking)", value=max(gains.values()), passed=np.nan,
                     detail=", ".join(f"{k} {v:+.3f}" for k, v in gains.items()) + f" -> {max(gains, key=gains.get)}"))
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "advance_criteria_open.csv", index=False)
    print("\n===== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
