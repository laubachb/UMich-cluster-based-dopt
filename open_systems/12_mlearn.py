#!/usr/bin/env python3
"""Step 12: test V of CRITERIA.md (addendum of 2026-10-12, evening): regime-weighted variance (RWV) on the untouched
mlearn data (Cu, Ge, Li, Mo, Ni, Si; work/mlearn_<el>/ links the matrices prepared in cluster-gamma-threshold).

Arm A as 02_shift.py (same split seeds and error model), with every regime of >= 20 frames. Scores per scenario:
gamma_bulk (core.grade_all), gamma_regime (assigned) and gamma_min_regime (minimum over regime bases; 08_min_rule.grades),
h_bulk (core.leverage_all), IG_hom and RWV (core.BayesGrade with regime groups; RWV = IG_het of addendum H).
MaxVol grades: 5 solutions, seeds 300 + t. Writes results/mlearn_<el>.csv; --score writes results/mlearn_criteria.csv.
usage: 12_mlearn.py Cu Ge ... [--jobs N]      |      12_mlearn.py --score
"""
import argparse, importlib.util, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

MIN_FRAMES, ELEMENTS = 20, ["Cu", "Ge", "Li", "Mo", "Ni", "Si"]
spec = importlib.util.spec_from_file_location("m8", C.HERE / "08_min_rule.py")
m8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m8)
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
    parts = C.partitions(D, train, False, cluster=False)
    tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    Xr, yr, Xc = X[C.rows_of(tr_idx)], F[train].ravel(), X[C.rows_of(ca_idx)]
    reg = parts["regime"]; gr = np.repeat(reg[tr_idx], 3)
    nsol = G["nsol"]
    S = {"gamma_bulk": C.grade_all(D, train, cand, parts["bulk"], nsol)[0]}
    S["gamma_regime"], S["gamma_min_regime"] = m8.grades(D, train, cand, reg, nsol)
    S["h_bulk"] = C.leverage_all(X, train, cand, parts["bulk"])[None]
    coef = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(X.shape[1]), Xr.T @ yr)
    sigma, s0 = C.group_sigma(Xr, yr, gr, coef)
    S["IG_hom"] = C.BayesGrade(Xr, yr, gr, {k: s0 for k in sigma}, s0).info_gain(Xc, reg[ca_idx])[None]
    S["RWV"] = C.BayesGrade(Xr, yr, gr, sigma, s0).info_gain(Xc, reg[ca_idx])[None]
    rows = []
    for s, Gm in S.items():
        ok = np.isfinite(Gm).all(0)
        ms = [C.metrics(g[ok], err[ok]) for g in Gm]
        rows.append(dict(element=G["el"], regime=R, frac=f, draw=d, score=s, n_cand=int(cand.sum()), n_graded=int(ok.sum()),
                         sigma_R=sigma[int(reg[ca_idx][0])], sigma_pooled=s0, **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
    print(f"{G['el']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def score():
    df = pd.concat([pd.read_csv(C.RES / f"mlearn_{e}.csv") for e in ELEMENTS if (C.RES / f"mlearn_{e}.csv").exists()])
    a = df.pivot_table(index=["element", "regime", "frac", "draw"], columns="score", values="auroc").reset_index()
    u = a[a.frac <= 0.05]
    out = []
    for cname, other in (("V1", "h_bulk"), ("V2", "gamma_regime"), ("V3", "gamma_min_regime"), ("V4", "gamma_bulk")):
        d = u.RWV - u[other]; m, lo, hi = boot(d)
        per = (u.assign(d=d).groupby("element").d.mean())
        ok = lo > 0 and (cname != "V1" or (per > 0).sum() >= 4)
        out.append(dict(criterion=cname, comparator=other, value=m, passed=bool(ok),
                        detail=f"RWV - {other} {m:+.3f} [{lo:+.3f}, {hi:+.3f}], n={len(u)}; per element " +
                               ", ".join(f"{e} {v:+.3f}" for e, v in per.items())))
    m, lo, hi = boot(u.IG_hom - u.h_bulk)
    out.append(dict(criterion="(descriptive) IG_hom - h_bulk", comparator="h_bulk", value=m, passed=np.nan, detail=f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"))
    cr = pd.DataFrame(out); cr.to_csv(C.RES / "mlearn_criteria.csv", index=False)
    pd.set_option("display.width", 250)
    sc = ["gamma_bulk", "gamma_regime", "gamma_min_regime", "h_bulk", "IG_hom", "RWV"]
    print("== mean AUROC, f <= 0.05, by element\n" + u.groupby("element")[sc].mean().round(3).to_string())
    print("\n== mean AUROC by f (all elements)\n" + a.groupby("frac")[sc].mean().round(3).to_string())
    print("\n== criteria\n" + cr.to_string(index=False))
    print("\nDecision (fixed in advance): method claim only if V1 and V2 pass ->",
          bool(cr.set_index("criterion").loc["V1", "passed"] and cr.set_index("criterion").loc["V2", "passed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elements", nargs="*"); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--score", action="store_true")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--nsol", type=int, default=C.NSOL)
    a = ap.parse_args()
    if a.score:
        return score()
    for el in a.elements:
        D, tab = C.load(f"mlearn_{el}", a.cap)
        G.update(D=D, tab=tab, el=el, nsol=a.nsol)
        base = tab[~tab.al]; cnt = base.regime.value_counts()
        regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES]
        jobs = [(R, f, d) for R in regimes for f in (0.02, 0.05, 0.10, 0.25) for d in (0, 1, 2)]
        print(f"{el}: {len(D['element'])} atoms, regimes {regimes}, {len(jobs)} scenarios", flush=True)
        if a.jobs > 1:
            with Pool(a.jobs) as p:
                out = p.map(scenario, jobs, chunksize=1)
        else:
            out = [scenario(j) for j in jobs]
        pd.DataFrame([r for rs in out for r in rs]).to_csv(C.RES / f"mlearn_{el}.csv", index=False)


if __name__ == "__main__":
    main()
