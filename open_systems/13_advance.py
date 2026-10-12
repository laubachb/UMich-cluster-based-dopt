#!/usr/bin/env python3
"""Step 13: test N of CRITERIA.md (addendum N) on arm A of the open data, and (--mlearn, one method) the N4 confirmation.

Per scenario (split, error model and clustering exactly as 02_shift.py; the assigned grade is checked against
results/shift_NAME*.csv):
  gamma_cluster   assigned (baseline, ours)            gamma_min   minimum over cluster bases (baseline, Shuang et al.)
  gamma_soft      N-A, exp(sum_k p_k log gamma_k) with the 15-NN vote's membership probabilities (gamma_soft_arith: sum p g)
  gamma_ens       N-B, exp(mean over 15 clusterings (k = 4-8 x seeds 42, 0, 1) of log gamma_cluster)   [--methods ens]
  gamma_cal       N-C, q_k * gamma_cluster with per-cluster conformal quantiles (alpha = 0.1) of |error| / grade on the
                  calibration set = frames of the other regimes not drawn into training; gamma_cal_min likewise for
                  gamma_min; coverage = share of candidate errors below the bound (also with one pooled quantile)
All grades use mean-over-solutions scores for the calibration step; MaxVol: 5 solutions, seeds 300 + t for the bases
of clusters holding candidates (arm A's stream), separate streams otherwise.
Writes results/advance_NAME[TAG].csv (one row per scenario and score) or results/advance_mlearn_EL.csv.
usage: 13_advance.py NAME [NAME ...] [--methods soft,ens,cal] [--jobs N] [--only REGIME ...] [--tag TAG] [--mlearn]
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

OUT_TOL, ALPHA = 1e-6, 0.1
ENS = [(k, s) for k in (4, 5, 6, 7, 8) for s in (42, 0, 1)]
G = {}


def split(R, f, d):
    """02_shift.scenario's split, plus the calibration frames (other regimes, not drawn into training)."""
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
    inbase = np.isin(D["frame"], base.frame)
    cand = (D["regime"] == R) & ~train & inbase
    cal = (D["regime"] != R) & ~train & inbase
    return train, cand, cal


def partition_members(D, train):
    """manuscript clustering per element with membership probabilities; group code = element * K + cluster."""
    el = D["element"]; code = pd.Categorical(el).codes; Xatom = D["X"].reshape(len(el), -1)
    lab = np.full(len(el), -1); proba = np.zeros((len(el), C.K_CLUSTERS))
    for e in np.unique(code):
        me = code == e
        l, p = C.spectral_members(Xatom[train & me], Xatom[me])
        lab[me] = e * C.K_CLUSTERS + l; proba[me] = p
    return lab, proba, code


def group_grades(D, train, cand, atoms, lab, nsol, seed2=10_000):
    """Praw / P (nsol, n_groups, len(atoms)): grades of `atoms` against every training group of lab; P has inf outside
    the group's element or row space. Groups holding arm-A candidates draw their basis from arm A's stream."""
    X, el = D["X"], D["element"]; tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    groups = np.unique(lab[tr_idx]); g_el = {int(g): el[tr_idx[lab[tr_idx] == g][0]] for g in groups}
    Xa = X[C.rows_of(atoms)]
    Praw = np.full((nsol, len(groups), len(atoms)), np.inf); P = Praw.copy()
    for t in range(nsol):
        rng, rng2 = np.random.default_rng(C.SEED0 + t), np.random.default_rng(seed2 + C.SEED0 + t)
        for j, g in enumerate(groups):
            has = (lab[ca_idx] == g).any() or len(groups) == 1
            b = C.Basis(X[C.rows_of(tr_idx[lab[tr_idx] == g])], rng if has else rng2)
            same = np.where(el[atoms] == g_el[int(g)])[0]
            if same.size == 0:
                continue
            raw, out = b.grade(Xa[C.rows_of(same)])
            Praw[t, j, same] = raw; P[t, j, same] = np.where(out > OUT_TOL, np.inf, raw)
    return Praw, P, groups


def ensemble_grade(D, train, cand, code, nsol):
    X, el = D["X"], D["element"]; Xatom = X.reshape(len(el), -1)
    tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    logs = np.zeros((nsol, len(ca_idx)))
    for k, s in ENS:
        lab = np.full(len(el), -1)
        for e in np.unique(code):
            me = code == e
            lab[me] = e * k + C.spectral_members(Xatom[train & me], Xatom[me], k=k, rng_seed=s)[0]
        for t in range(nsol):
            rng = np.random.default_rng(30_000 + 1000 * k + 10 * s + t)
            for g in np.unique(lab[ca_idx]):
                gc = np.where(lab[ca_idx] == g)[0]; gt = tr_idx[lab[tr_idx] == g]
                if gt.size == 0:
                    continue
                logs[t, gc] += np.log(np.maximum(C.Basis(X[C.rows_of(gt)], rng).grade(X[C.rows_of(ca_idx[gc])])[0], 1e-300))
    return np.exp(logs / len(ENS))


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time(); D = G["D"]; X, F = D["X"], D["F"]; nsol = G["nsol"]; methods = G["methods"]
    train, cand, cal = split(R, f, d)
    tr_idx = np.where(train)[0]
    Xr, yr = X[C.rows_of(tr_idx)], F[train].ravel()
    coef = np.linalg.solve(Xr.T @ Xr + 1e-8 * np.eye(X.shape[1]), Xr.T @ yr)        # = the error model of arm A
    atoms = np.where(cand | cal)[0]; is_c = cand[atoms]
    err = np.abs((X[C.rows_of(atoms)] @ coef).reshape(-1, 3) - F[atoms]).mean(1)
    lab, proba, code = partition_members(D, train)
    Praw, P, groups = group_grades(D, train, cand, atoms, lab, nsol)
    gpos = {int(g): j for j, g in enumerate(groups)}
    own = np.array([gpos[int(g)] for g in lab[atoms]]); ar = np.arange(len(atoms))
    W = np.zeros((len(atoms), len(groups)))
    for j, g in enumerate(groups):
        e, l = divmod(int(g), C.K_CLUSTERS); W[:, j] = proba[atoms, l] * (code[atoms] == e)
    S = {"gamma_cluster": Praw[:, own, ar], "gamma_min": P.min(1)}
    if "soft" in methods:
        S["gamma_soft"] = np.vstack([C.soft_grade(P[t], W) for t in range(nsol)])
        S["gamma_soft_arith"] = np.vstack([C.soft_grade(P[t], W, geometric=False) for t in range(nsol)])
    if "ens" in methods:
        S["gamma_ens"] = ensemble_grade(D, train, cand, code, nsol)
    rows = []
    cov = {}
    if "cal" in methods and (~is_c).sum() >= 30:
        for base in ("gamma_cluster", "gamma_min"):
            sc = S[base].mean(0); ok = np.isfinite(sc)
            q, qp = C.mondrian_q(sc[ok & ~is_c], err[ok & ~is_c], lab[atoms][ok & ~is_c], ALPHA)
            qk = np.array([q.get(int(g), qp) for g in lab[atoms]])
            name = "gamma_cal" if base == "gamma_cluster" else "gamma_cal_min"
            S[name] = (qk * sc)[None]
            m = is_c & ok
            cov[name] = dict(coverage=float(np.mean(err[m] <= qk[m] * sc[m])), coverage_pooled=float(np.mean(err[m] <= qp * sc[m])),
                             bound_mean=float(np.mean(qk[m] * sc[m])), n_cal=int((ok & ~is_c).sum()),
                             n_cal_clusters_pooled=int(sum(1 for g in np.unique(lab[atoms][~is_c]) if (lab[atoms][~is_c] == g).sum() < 20)))
    ec = err[is_c]
    for s, Gm in S.items():
        Gc = Gm[:, is_c] if Gm.shape[1] == len(atoms) else Gm
        ok = np.isfinite(Gc).all(0)
        ms = [C.metrics(g[ok], ec[ok]) for g in Gc]
        rows.append(dict(name=G["name"], regime=R, frac=f, draw=d, score=s, n_cand=int(is_c.sum()), n_graded=int(ok.sum()),
                         **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}, **cov.get(s, {})))
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+"); ap.add_argument("--methods", default="soft,ens,cal"); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", nargs="*"); ap.add_argument("--tag", default=""); ap.add_argument("--mlearn", action="store_true")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--nsol", type=int, default=C.NSOL)
    a = ap.parse_args()
    for name in a.names:
        D, tab = C.load(name, a.cap)
        G.update(D=D, tab=tab, name=name, nsol=a.nsol, methods=a.methods.split(","))
        base = tab[~tab.al]; cnt = base.regime.value_counts()
        regimes = [r for r in sorted(cnt.index) if cnt[r] >= (20 if a.mlearn else 60) and (not a.only or r in a.only)]
        jobs = [(R, f, d) for R in regimes for f in (0.02, 0.05, 0.10, 0.25) for d in (0, 1, 2)]
        print(f"{name}: {len(D['element'])} atoms, regimes {regimes}, {len(jobs)} scenarios, methods {G['methods']}", flush=True)
        if a.jobs > 1:
            with Pool(a.jobs) as p:
                out = p.map(scenario, jobs, chunksize=1)
        else:
            out = [scenario(j) for j in jobs]
        df = pd.DataFrame([r for rs in out for r in rs])
        if not a.mlearn:
            shift = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"shift_{name}*.csv")])
            s = shift[(shift.element == "all") & (shift.partition == "cluster")].drop_duplicates(["regime", "frac", "draw"]).set_index(["regime", "frac", "draw"]).auroc
            chk = df[df.score == "gamma_cluster"].set_index(["regime", "frac", "draw"]).auroc
            print(f"assigned AUROC vs arm A: max |diff| = {(chk - s.reindex(chk.index)).abs().max():.2e} over {len(chk)} scenarios")
        out_name = f"advance_{name}{a.tag}.csv" if not a.mlearn else f"advance_{name}{a.tag}.csv"
        df.to_csv(C.RES / out_name, index=False)
        pd.set_option("display.width", 250)
        print(df.pivot_table(index=["regime", "frac"], columns="score", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
