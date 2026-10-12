#!/usr/bin/env python3
"""Step 18: test X of CRITERIA.md: the ceiling of descriptor-based within-regime ranking. Arm A splits as 02_shift.py.
Per scenario, k-nearest-neighbour regressors (k = 20, distance-weighted) of log error in the standardized 126-feature
force-row descriptor (scaler fit on the training atoms):
  oracle_shift  trained on the calibration atoms (other regimes, not in training; errors from the same ridge model)
  oracle_in     leave-frame-out within the candidates of R (trained on R's other candidate frames)
Writes results/ceiling_NAME[TAG].csv. --score: per data set, mean AUROC of both oracles against the best existing
grade per scenario (max over gamma_regime, h_regime, RWV / IG_het_regime, gamma_cal, gamma_cluster where available) in
the C3 scenarios, headroom and the fixed decision -> results/ceiling_criteria.csv.
usage: 18_ceiling.py NAME [--jobs N] [--only REGIME ...] [--tag TAG]  |  18_ceiling.py --score
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import core as C

MIN_FRAMES, K = 60, 20
G = {}


def split(R, f, d):
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
    train = np.isin(D["frame"], tr); inbase = np.isin(D["frame"], base.frame)
    return train, (D["regime"] == R) & ~train & inbase, (D["regime"] != R) & ~train & inbase


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time(); D = G["D"]; X, F = D["X"], D["F"]
    train, cand, cal = split(R, f, d)
    tr_idx = np.where(train)[0]; Xr, yr = X[C.rows_of(tr_idx)], F[train].ravel()
    coef = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(X.shape[1]), Xr.T @ yr)
    Xatom = X.reshape(len(D["element"]), -1)
    err = lambda m: np.abs((X[C.rows_of(np.where(m)[0])] @ coef).reshape(-1, 3) - F[m]).mean(1)
    ec, ek = err(cand), err(cal)
    sc = StandardScaler().fit(Xatom[train]); Zc, Zk = sc.transform(Xatom[cand]), sc.transform(Xatom[cal])
    ly = lambda e: np.log(np.maximum(e, 1e-12))
    rows = []
    pred = KNeighborsRegressor(K, weights="distance").fit(Zk, ly(ek)).predict(Zc)
    rows.append(dict(score="oracle_shift", **C.metrics(pred, ec)))
    fr = D["frame"][cand]; pred = np.full(len(ec), np.nan)
    for fid in np.unique(fr):
        m = fr == fid
        if (~m).sum() >= K:
            pred[m] = KNeighborsRegressor(K, weights="distance").fit(Zc[~m], ly(ec[~m])).predict(Zc[m])
    ok = np.isfinite(pred)
    rows.append(dict(score="oracle_in", **C.metrics(pred[ok], ec[ok])))
    for r in rows:
        r.update(name=G["name"], regime=R, frac=f, draw=d, n_cand=int(cand.sum()), n_cal=int(cal.sum()))
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def best_grade(name):
    """per scenario, the best existing within-regime AUROC among the grades of earlier tests."""
    KEY = ["regime", "frac", "draw"]; cols = []
    sh = pd.read_csv(C.RES / f"shift_{name}.csv"); sh = sh[sh.element == "all"]
    g = sh.pivot_table(index=KEY, columns="partition", values="auroc").add_prefix("gamma_"); cols.append(g[[c for c in g if c in ("gamma_regime", "gamma_cluster")]])
    rep = sh[sh.partition == "bulk"].set_index(KEY).rep_R
    h = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"leverage_{name}*.csv") if "_al_" not in f.name]); h = h[h.element == "all"].drop_duplicates(KEY + ["partition"])
    cols.append(h.pivot_table(index=KEY, columns="partition", values="auroc").add_prefix("h_")[["h_regime", "h_bulk"]])
    ig = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"infogain_{name}*.csv")]); ig = ig[(ig.partition == "regime") & (ig.score == "IG_het")].drop_duplicates(KEY)
    cols.append(ig.set_index(KEY)[["auroc"]].rename(columns={"auroc": "RWV"}))
    adv = list(C.RES.glob(f"advance_{name}*.csv"))
    if adv:
        a = pd.concat([pd.read_csv(f) for f in adv]).drop_duplicates(KEY + ["score"]); a = a[a.score == "gamma_cal"]
        if len(a):
            cols.append(a.set_index(KEY)[["auroc"]].rename(columns={"auroc": "gamma_cal"}))
    m = pd.concat(cols, axis=1).join(rep)
    m["best"] = m.drop(columns="rep_R").max(axis=1); m["best_name"] = m.drop(columns=["rep_R", "best"]).idxmax(axis=1)
    return m


def score():
    crit = []
    for name in ("si_gap18", "sio2", "water"):
        files = list(C.RES.glob(f"ceiling_{name}*.csv"))
        if not files:
            continue
        o = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["regime", "frac", "draw", "score"])
        o = o.pivot_table(index=["regime", "frac", "draw"], columns="score", values="auroc")
        m = o.join(best_grade(name)); u = m[(m.index.get_level_values("frac") <= 0.05) & (m.rep_R < 0.5)]
        head = u.oracle_in.mean() - u.best.mean()
        decision = "at ceiling (< 0.05)" if head < 0.05 else ("headroom >= 0.10: supervised error model next" if head >= 0.10 else "between 0.05 and 0.10")
        crit.append(dict(name=name, oracle_in=u.oracle_in.mean(), oracle_shift=u.oracle_shift.mean(), best_grade=u.best.mean(),
                         best_grade_names=u.best_name.value_counts().to_dict(), headroom=head, shift_minus_best=u.oracle_shift.mean() - u.best.mean(), decision=decision, n=len(u)))
        pd.set_option("display.width", 250)
        print(f"\n===== {name}: mean AUROC at f = 0.05 by regime\n" + m[m.index.get_level_values('frac') == 0.05].groupby("regime")[["oracle_in", "oracle_shift", "best", "gamma_cluster", "gamma_regime", "h_bulk", "RWV"]].mean().round(3).to_string())
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "ceiling_criteria.csv", index=False); print("\n===== ceiling\n" + cr.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?"); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--only", nargs="*")
    ap.add_argument("--tag", default=""); ap.add_argument("--score", action="store_true"); ap.add_argument("--cap", type=int, default=32)
    a = ap.parse_args()
    if a.score:
        return score()
    D, tab = C.load(a.name, a.cap); G.update(D=D, tab=tab, name=a.name)
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in (0.02, 0.05, 0.10, 0.25) for d in (0, 1, 2)]
    print(f"{a.name}: regimes {regimes}, {len(jobs)} scenarios", flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame([r for rs in out for r in rs]); df.to_csv(C.RES / f"ceiling_{a.name}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["regime", "frac"], columns="score", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
