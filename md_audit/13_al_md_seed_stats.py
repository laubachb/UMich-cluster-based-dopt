#!/usr/bin/env python3
"""Step 13: collapse-event statistics at 5000 K over all MD seeds -> results/al_md_seed_stats*.csv.

Pools the 5000 K runs of results/al_md_stability.csv (every refit, MD seed 1), al_md_stability_balanced_seeds.csv
(gamma rules: seeds 2-5 for MaxVol draws 0-4 and seeds 1-5 for draws 5-9, from 08_al_step.py --n-maxvol 10;
random draws 0-9: seeds 2-5; random_5000K draws 0-9: seeds 1-5) and al_md_stability_base_seeds.csv (base model,
40 seeds at 5000 K). Every rule then has 10 models x 5 seeds = 50 runs per K. Event = econserve excursion > 10 meV/atom or an N-N pair inside
the 0.86 A inner cutoff (12_al_md_analysis.py). Only models that have all five seeds (1-5) enter the rule comparisons.

Outputs
  al_md_seed_stats_by_k.csv   rule x K: events / runs, Wilson 95% interval
  al_md_seed_stats_rules.csv  rule (pooled over K, and K <= 10): rate, Wilson interval, and the difference to
                              random_5000K and to random with a 95% interval from a bootstrap over models (a model's
                              seeds are resampled together, since runs of one model are not independent)
usage: 13_al_md_seed_stats.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD

SP = "5000K_2.0gcc"
SEEDS = [1, 2, 3, 4, 5]
RULES = ["gamma_bulk", "random", "random_5000K", "gamma_cluster", "cluster_roundrobin"]
N_BOOT = 10000


def wilson(n, N, z=1.96):
    if N == 0:
        return np.nan, np.nan
    p = n / N; d = 1 + z * z / N; c = (p + z * z / (2 * N)) / d
    h = z * np.sqrt(p * (1 - p) / N + z * z / (4 * N * N)) / d
    return c - h, c + h


def boot_diff(a, b, rng):
    """95% interval of mean event rate(a) - rate(b); a, b = per-model event fractions (models resampled)."""
    ia = rng.integers(0, len(a), (N_BOOT, len(a))); ib = rng.integers(0, len(b), (N_BOOT, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return np.percentile(d, [2.5, 97.5])


def main():
    R = MD / "results"
    runs = pd.concat([pd.read_csv(R / f) for f in
                      ("al_md_stability.csv", "al_md_stability_balanced_seeds.csv", "al_md_stability_base_seeds.csv")])
    runs = runs[runs.statepoint == SP].drop_duplicates("run")
    runs = runs[(runs.strategy != "base") | runs.run.str.startswith("base__s")]   # base: the 20-seed set only
    runs["model"] = runs.strategy + "__k" + runs.k.astype(str) + "__d" + runs.draw.astype(str)
    assert runs.completed.all(), "incomplete runs"
    base = runs[runs.strategy == "base"]
    have = runs[runs.seed.isin(SEEDS)].groupby("model").seed.nunique()
    full = runs[runs.model.isin(have[have == len(SEEDS)].index) & runs.seed.isin(SEEDS) & (runs.strategy != "base")]

    rows = [dict(strategy="base", k=0, events=int(base.event.sum()), runs=len(base), models=1)]
    for (s, k), g in full.groupby(["strategy", "k"]):
        rows.append(dict(strategy=s, k=k, events=int(g.event.sum()), runs=len(g), models=g.model.nunique()))
    byk = pd.DataFrame(rows)
    byk["rate"] = byk.events / byk.runs
    byk[["lo95", "hi95"]] = [wilson(n, N) for n, N in zip(byk.events, byk.runs)]
    byk.to_csv(R / "al_md_seed_stats_by_k.csv", index=False)

    rng = np.random.default_rng(0)
    per_model = full.groupby(["strategy", "k", "model"]).event.mean().reset_index()
    out = []
    for scope, kmax in (("all K", 40), ("K <= 10", 10)):
        pm = per_model[per_model.k <= kmax]
        for s in RULES:
            a = pm[pm.strategy == s].event.to_numpy()
            g = full[(full.strategy == s) & (full.k <= kmax)]
            lo, hi = wilson(int(g.event.sum()), len(g))
            r = dict(scope=scope, strategy=s, models=len(a), runs=len(g), events=int(g.event.sum()),
                     rate=g.event.mean(), lo95=lo, hi95=hi)
            for ref in ("random_5000K", "random"):
                b = pm[pm.strategy == ref].event.to_numpy()
                if s != ref and len(a) and len(b):
                    r[f"diff_vs_{ref}"] = a.mean() - b.mean()
                    r[f"diff_vs_{ref}_lo95"], r[f"diff_vs_{ref}_hi95"] = boot_diff(a, b, rng)
            out.append(r)
    out.append(dict(scope="base, 20 seeds", strategy="base", models=1, runs=len(base), events=int(base.event.sum()),
                    rate=base.event.mean(), lo95=wilson(int(base.event.sum()), len(base))[0],
                    hi95=wilson(int(base.event.sum()), len(base))[1]))
    rules = pd.DataFrame(out)
    rules.to_csv(R / "al_md_seed_stats_rules.csv", index=False)
    pd.set_option("display.width", 250)
    print(byk.round(3).to_string(index=False)); print(); print(rules.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
