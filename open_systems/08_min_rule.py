#!/usr/bin/env python3
"""Step 8: test M3 of CRITERIA.md (addendum M): minimum over groups (Shuang et al., arXiv 2508.03405) vs assignment.

For every arm-A scenario (split, error model and partitions exactly as 02_shift.py) and the partitions regime and
cluster, every candidate atom is graded against the basis of every training group of its own element; gamma_min = the
smallest grade (a group whose rank-reduced row space does not contain the atom, out-of-span fraction > 1e-6, gives
+inf), gamma_assigned = the grade against the atom's own group. The bases of the groups that hold candidates are drawn
from the same random stream and in the same order as core.grade_all, so gamma_assigned is arm A's grade (its AUROC is
checked against results/shift_NAME*.csv); the other groups' bases use a second stream (seed 10,000 + SEED0 + t).
Writes results/minrule_NAME[TAG].csv (one row per scenario and partition: AUROC, Spearman, recall of both rules).
usage: 08_min_rule.py NAME [--jobs N] [--only REGIME ...] [--tag TAG] [--nsol 5]
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

MIN_FRAMES, OUT_TOL = 60, 1e-6
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


def grades(D, train, cand, part, nsol):
    """(nsol, n_cand) arrays: assigned grade and minimum over same-element groups."""
    X, el = D["X"], D["element"]
    tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    groups = np.unique(part[tr_idx])
    g_el = {int(g): el[tr_idx[part[tr_idx] == g][0]] for g in groups}
    Ca = X[C.rows_of(ca_idx)]
    Ga = np.full((nsol, len(ca_idx)), np.nan); Gm = np.full((nsol, len(ca_idx)), np.inf)
    for t in range(nsol):
        rng, rng2 = np.random.default_rng(C.SEED0 + t), np.random.default_rng(10_000 + C.SEED0 + t)
        for g in groups:                                     # grade_all's order: groups holding candidates use rng
            has = (part[ca_idx] == g).any()
            if not has and len(groups) == 1:
                has = True
            gt = tr_idx[part[tr_idx] == g]
            b = C.Basis(X[C.rows_of(gt)], rng if has else rng2)
            same = el[ca_idx] == g_el[int(g)]
            if not same.any():
                continue
            raw, out = b.grade(Ca[C.rows_of(np.where(same)[0])])
            Gm[t, same] = np.minimum(Gm[t, same], np.where(out > OUT_TOL, np.inf, raw))
            own = part[ca_idx][same] == g
            Ga[t, np.where(same)[0][own]] = raw[own]             # as grade_all (no out-of-span rule)
    return Ga, Gm


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time()
    D = G["D"]
    train, cand = split(R, f, d)
    err = C.error_model(D, train, cand)
    parts = C.partitions(D, train, G["multi"])
    rows = []
    for pname in ("regime", "cluster"):
        Ga, Gm = grades(D, train, cand, parts[pname], G["nsol"])
        for rule, Gr in (("assigned", Ga), ("min", Gm)):
            ok = np.isfinite(Gr).all(0)
            ms = [C.metrics(Gr[t, ok], err[ok]) for t in range(G["nsol"])]
            rows.append(dict(name=G["name"], regime=R, frac=f, draw=d, partition=pname, rule=rule, n_cand=int(cand.sum()),
                             n_graded=int(ok.sum()), **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--cap", type=int, default=32); ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.25]); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default=""); ap.add_argument("--nsol", type=int, default=C.NSOL)
    a = ap.parse_args()
    D, tab = C.load(a.name, a.cap)
    G.update(D=D, tab=tab, multi=len(np.unique(D["element"])) > 1, name=a.name, nsol=a.nsol)
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in a.fracs for d in a.draws]
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame([r for rs in out for r in rs])
    df.to_csv(C.RES / f"minrule_{a.name}{a.tag}.csv", index=False)
    shift = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"shift_{a.name}*.csv")])
    s = shift[shift.element == "all"].drop_duplicates(["regime", "frac", "draw", "partition"]).set_index(["regime", "frac", "draw", "partition"]).auroc
    chk = df[df.rule == "assigned"].set_index(["regime", "frac", "draw", "partition"]).auroc
    dev = (chk - s.reindex(chk.index)).abs().max()
    print(f"assigned AUROC vs arm A: max |diff| = {dev:.2e} over {len(chk)} scenario-partitions")
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["regime", "frac"], columns=["partition", "rule"], values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
