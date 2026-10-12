#!/usr/bin/env python3
"""Step 6: variance baseline of test R2 (CRITERIA.md, addendum of 2026-10-12).

Scores every candidate atom of arm A (and of arm B with --arm al) by its relative ridge leverage (core.Leverage)
against the training atoms of its group, for the partitions bulk, element (multi-element data), regime and, with
--cluster, the manuscript's clustering. Splits, candidates and errors are rebuilt exactly as in 02_shift.py and
03_al_retro.py (the candidate counts are checked against results/shift_NAME*.csv). Leverage is deterministic, so
there is one value per atom and no MaxVol solutions.
Writes results/leverage_NAME[TAG].csv (arm A) or results/leverage_al_NAME.csv (arm B), with the columns of the
gamma tables (partition, element, n_cand, spearman, auroc, recall10).
usage: 06_leverage.py NAME [--arm shift|al] [--cluster] [--jobs N] [--only REGIME ...] [--tag TAG]
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


def score(D, train, cand, groups):
    """rows of metrics for every partition and element; `groups` maps a group label to a candidate mask."""
    err = C.error_model(D, train, cand)
    parts = C.partitions(D, train, G["multi"], cluster=G["cluster"])
    els = ["all"] + (sorted(np.unique(D["element"])) if G["multi"] else [])
    rows = []
    for pname, lab in parts.items():
        H = C.leverage_all(D["X"], train, cand, lab)
        for gname, gm in groups.items():
            for e in els:
                m = gm & (np.ones(cand.sum(), bool) if e == "all" else D["element"][cand] == e)
                ok = m & np.isfinite(H)
                if ok.sum() < 30:
                    continue
                rows.append(dict(group=gname, partition=pname, element=e, n_cand=int(m.sum()), n_graded=int(ok.sum()),
                                 **C.metrics(H[ok], err[ok])))
    return rows


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time()
    train, cand = split(R, f, d)
    rows = score(G["D"], train, cand, {R: np.ones(cand.sum(), bool)})
    for r in rows:
        r.update(name=G["name"], regime=R, frac=f, draw=d); del r["group"]
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def arm_b():
    """03_al_retro.py: reference = base frames, candidates = *_AL frames grouped by config type and parent regime."""
    D, tab = G["D"], G["tab"]
    config = tab.set_index("frame").config_type.loc[D["frame"]].to_numpy()
    train, cand = ~D["al"], D["al"].copy()
    cc, pc = config[cand], D["regime"][cand]
    groups = {f"config_type:{g}": cc == g for g in np.unique(cc)} | {f"parent:{g}": pc == g for g in np.unique(pc)}
    rows = score(D, train, cand, groups)
    for r in rows:
        r["kind"], r["group"] = r["group"].split(":", 1); r["name"] = G["name"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--arm", default="shift"); ap.add_argument("--cluster", action="store_true")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D, tab = C.load(a.name, a.cap)
    G.update(D=D, tab=tab, multi=len(np.unique(D["element"])) > 1, name=a.name, cluster=a.cluster)
    if a.arm == "al":
        df = pd.DataFrame(arm_b())
        df.to_csv(C.RES / f"leverage_al_{a.name}.csv", index=False)
        print(df[(df.kind == "parent") & (df.element == "all")].pivot_table(index="group", columns="partition", values="auroc").round(3))
        return
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in a.fracs for d in a.draws]
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame([r for rs in out for r in rs])
    shift = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"shift_{a.name}*.csv")])   # the candidate sets must be those of arm A
    s = shift[(shift.partition == "bulk") & (shift.element == "all")].drop_duplicates(["regime", "frac", "draw"]).set_index(["regime", "frac", "draw"]).n_cand
    chk = df[(df.partition == "bulk") & (df.element == "all")].set_index(["regime", "frac", "draw"]).n_cand
    assert (chk == s.reindex(chk.index)).all(), "candidate sets differ from arm A"
    print(f"candidate counts match arm A for {len(chk)} scenarios")
    df.to_csv(C.RES / f"leverage_{a.name}{a.tag}.csv", index=False)
    print(df[df.element == "all"].pivot_table(index=["regime", "frac"], columns="partition", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
