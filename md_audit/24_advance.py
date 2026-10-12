#!/usr/bin/env python3
"""Step 24: test N of open_systems/CRITERIA.md (addendum N) on the MD of the DFT-only model.

Baselines gamma_cluster (assigned; reproduces work/atom_errors_<model>.csv) and gamma_min, bases as step 3 (reference =
the 90 training frames, bulk basis drawn first, 10 solutions). New grades: gamma_soft (membership probabilities of the
step-3 vote; argmax reproduces the stored clusters), gamma_ens (15 clusterings of all DFT-MD atoms as step 8, bases
from the training atoms, 5 solutions), gamma_cal = q_k * gamma_cluster with per-cluster conformal quantiles
(alpha = 0.1) calibrated on the 30 held-out DFT-MD frames (grades against the same bases, errors of the DFT-only model),
gamma_cal_min likewise. Mean-over-solution grades for calibration.
Reports: pooled Spearman over all 57,600 MD atoms (bootstrap over frames, 2,000, seed 0; paired to gamma_cluster),
per-state-point frame-level Spearman and coverage, and one active-learning iteration with top-K over the whole pool by
frame-max gamma_cal (10 MaxVol draws, streams 11 + draw as step 8) scored by the H4 decision rule.
Writes results/advance_md_{pooled,frame,coverage}.csv, results/al_step_advance.csv, results/advance_md_criteria.csv.
usage: 24_advance.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces
from open_systems.core import soft_grade, mondrian_q

MODEL = "shared_t0_dft_train"
N_TRIALS, N_ENS_TRIALS, N_BOOT, ALPHA = 10, 5, 2000, 0.1
ENS = [(k, s) for k in (4, 5, 6, 7, 8) for s in (42, 0, 1)]
KS = [2, 5, 10, 20, 40]
HA_BOHR_TO_KCAL_A = 627.509474 / 0.529177211
EV_TO_KCAL, KCAL2EV = 23.0605478, 0.0433641


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def all_groups(X, bases):
    return np.vstack([E.gamma_for_rows(X, inv) for inv in bases])


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    hold_keys = [k for k in by_frame if k[0].startswith("DFT.") and k not in set(train_keys)]
    kind = np.array([r["kind"] for r in man]); dft_ids = np.where(kind == "dft")[0]
    lab = np.full(len(kind), -1); lab[dft_ids] = np.load(RW / "labels_dft.npy")
    ref = np.array(sorted(a for k in train_keys for a in by_frame[k]))
    hold = np.array(sorted(a for k in hold_keys for a in by_frame[k]))
    X = A.reshape(len(kind), -1)
    sc = StandardScaler().fit(X[dft_ids]); Zd = sc.transform(X[dft_ids])
    knn = KNeighborsClassifier(15, weights="distance").fit(Zd, lab[dft_ids])

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    gidx = start[df.frame.to_numpy()] + df.atom.to_numpy()
    Xmd = np.asarray(Am[E.atom_rows_from_indices(gidx)]); Zmd = sc.transform(Xmd.reshape(len(df), -1))
    proba = knn.predict_proba(Zmd); assert (knn.classes_ == np.arange(6)).all() and (proba.argmax(1) == cl_md[gidx]).all()
    print(f"membership: {np.mean(proba.max(1) > 0.99):.3f} of MD atoms have p_max > 0.99", flush=True)

    Xh = A[E.atom_rows_from_indices(hold)]
    P = np.zeros((N_TRIALS, 6, len(df))); Ph = np.zeros((N_TRIALS, 6, len(hold)))
    for t in range(N_TRIALS):
        rng = np.random.default_rng(E.SEED0 + t)
        E.random_restart_maxvol(A[E.atom_rows_from_indices(ref)], rng)
        bases = [E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[lab[ref] == c])], rng) for c in range(6)]
        P[t] = all_groups(Xmd, bases); Ph[t] = all_groups(Xh, bases)
    Ga = P[:, cl_md[gidx], np.arange(len(df))]
    assert np.abs(Ga.mean(0) - df.gamma_cluster.to_numpy()).max() < 1e-4
    df["gamma_min"] = P.min(1).mean(0)
    df["gamma_soft"] = np.vstack([soft_grade(P[t], proba) for t in range(N_TRIALS)]).mean(0)
    df["gamma_soft_arith"] = np.vstack([soft_grade(P[t], proba, geometric=False) for t in range(N_TRIALS)]).mean(0)
    print("soft done", flush=True)

    logs = np.zeros((N_ENS_TRIALS, len(df)))
    for k, s in ENS:
        labk = SpectralClustering(n_clusters=k, affinity="nearest_neighbors", n_neighbors=20, assign_labels="kmeans",
                                  random_state=s, n_jobs=1).fit_predict(Zd)
        full = np.full(len(kind), -1); full[dft_ids] = labk
        lm = KNeighborsClassifier(15, weights="distance").fit(Zd, labk).predict(Zmd)
        for t in range(N_ENS_TRIALS):
            rng = np.random.default_rng(30_000 + 1000 * k + 10 * s + t)
            for c in np.unique(lm):
                m = np.where(lm == c)[0]; rc = ref[full[ref] == c]
                if rc.size == 0:
                    continue
                logs[t, m] += np.log(np.maximum(E.gamma_for_rows(Xmd[E.atom_rows_from_indices(m)], E.random_restart_maxvol(A[E.atom_rows_from_indices(rc)], rng)), 1e-300))
        print(f"ensemble k={k} seed={s} done", flush=True)
    df["gamma_ens"] = np.exp(logs / len(ENS)).mean(0)

    # calibration on the held-out frames: errors of the DFT-only model (vector norm, eV/A, as df.err)
    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); c0 = fit(A0, b0)
    Ah, bh = block(hold_keys)
    hold_ids = np.concatenate([by_frame[k] for k in hold_keys]); order = np.argsort(hold_ids)      # block order -> sorted hold
    err_h = np.linalg.norm((Ah @ c0 - bh).reshape(-1, 3), axis=1)[order] * KCAL2EV
    lab_h = lab[hold]
    gh = {"gamma_cluster": Ph[:, lab_h, np.arange(len(hold))].mean(0), "gamma_min": Ph.min(1).mean(0)}
    cov = []
    for base in ("gamma_cluster", "gamma_min"):
        q, qp = mondrian_q(gh[base], err_h, lab_h, ALPHA)
        name = "gamma_cal" if base == "gamma_cluster" else "gamma_cal_min"
        qk = np.array([q[int(c)] for c in cl_md[gidx]])
        df[name] = qk * df[base]
        for sp, d in df.groupby("statepoint"):
            cov.append(dict(score=name, statepoint=sp, coverage=float(np.mean(d.err <= d[name])), coverage_pooled_q=float(np.mean(d.err <= qp * d[base])),
                            bound_median=float(d[name].median()), q=";".join(f"{c}:{q[c]:.4g}" for c in sorted(q))))
    df.to_csv(MD / "work" / f"advance_{MODEL}.csv", index=False)

    scores = ["gamma_bulk", "gamma_cluster", "gamma_min", "gamma_soft", "gamma_soft_arith", "gamma_ens", "gamma_cal", "gamma_cal_min"]
    fr_all = df.groupby("frame").agg(statepoint=("statepoint", "first"))
    B = np.random.default_rng(0).integers(0, len(fr_all), (N_BOOT, len(fr_all)))
    idx_of = df.groupby("frame").indices; frames = fr_all.index.to_numpy(); y = df.err.to_numpy()
    def pooled_sp(s, b):
        ii = np.concatenate([idx_of[frames[j]] for j in b]); return spearmanr(df[s].to_numpy()[ii], y[ii])[0]
    boot = {s: np.array([pooled_sp(s, b) for b in B[:500]]) for s in scores}      # 500 resamples: 57,600 atoms each
    pooled = []
    for s in scores:
        d = boot[s] - boot["gamma_cluster"]
        pooled.append(dict(score=s, spearman_pooled=spearmanr(df[s], df.err)[0], lo=np.quantile(boot[s], .025), hi=np.quantile(boot[s], .975),
                           minus_cluster=d.mean(), mc_lo=np.quantile(d, .025), mc_hi=np.quantile(d, .975)))
    frame = []
    for sp, d in df.groupby("statepoint"):
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in scores})
        for s in scores:
            frame.append(dict(statepoint=sp, score=s, spearman=spearmanr(fr[s], fr.rmse)[0]))
    pl, frm, cv = pd.DataFrame(pooled), pd.DataFrame(frame), pd.DataFrame(cov)
    pl.to_csv(MD / "results" / "advance_md_pooled.csv", index=False); frm.to_csv(MD / "results" / "advance_md_frame.csv", index=False)
    cv.to_csv(MD / "results" / "advance_md_coverage.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== pooled Spearman over all MD atoms\n" + pl.round(3).to_string(index=False))
    print("\n== frame-level Spearman per state point\n" + frm.pivot(index="score", columns="statepoint", values="spearman").round(3).to_string())
    print("\n== coverage of 90 % bounds\n" + cv.drop(columns="q").round(3).to_string(index=False))

    # ---- one active-learning iteration by frame-max gamma_cal (and gamma_soft) over the whole pool
    FD = load_dft_forces(MD / "data" / "dft_forces.npz")
    md = fm[(fm.model == MODEL) & (fm.step > 0)]; md = md[md.frame.isin(list(FD))]
    Xf = {f: np.asarray(Am[3 * start[f]:3 * start[f + 1]]) for f in md.frame}
    Yf = {f: FD[f].ravel() * EV_TO_KCAL for f in md.frame}
    role = pd.read_csv(MD / "results" / "balanced_split.csv").set_index("frame").role
    pool = md[md.frame.map(role) == "pool"]; test = md[md.frame.map(role) == "test"]
    sp_of = pool.set_index("frame").statepoint
    ref_ids = np.concatenate([by_frame[k] for k in train_keys])
    q_cl, _ = mondrian_q(gh["gamma_cluster"], err_h, lab_h, ALPHA)

    def evaluate(c):
        out = {}
        for sp, t_ in test.groupby("statepoint"):
            e = np.concatenate([Xf[f] @ c - Yf[f] for f in t_.frame])
            out[f"rmse_md_{sp}"] = np.sqrt(np.mean(e ** 2)) / EV_TO_KCAL
        out["rmse_dft_holdout"] = np.sqrt(np.mean((Ah @ c - bh) ** 2)) / EV_TO_KCAL
        return out

    res = [dict(strategy="base", k=0, draw=0, **evaluate(c0))]
    for draw in range(N_TRIALS):
        rng = np.random.default_rng(11 + draw)
        E.random_restart_maxvol(A0, rng)
        bases = [E.random_restart_maxvol(A0[E.atom_rows_from_indices(np.where(lab[ref_ids] == c)[0])], rng) for c in range(6)]
        s_cal, s_soft = {}, {}
        for f in pool.frame:
            Pf = all_groups(Xf[f], bases); cl = cl_md[start[f]:start[f + 1]]; pr = knn.predict_proba(sc.transform(Xf[f].reshape(len(cl), -1)))
            s_cal[f] = (np.array([q_cl[int(c)] for c in cl]) * Pf[cl, np.arange(len(cl))]).max()
            s_soft[f] = soft_grade(Pf, pr).max()
        for strat, s_ in (("gamma_cal", s_cal), ("gamma_soft", s_soft)):
            order = sorted(s_, key=s_.get, reverse=True)
            for k in KS:
                ch = order[:k]
                c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
                res.append(dict(strategy=strat, k=k, draw=draw, sp_mix=sp_of.loc[ch].value_counts().to_dict(), **evaluate(c)))
        print("AL draw", draw, "done", flush=True)
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_advance.csv", index=False)
    num = [c for c in out.columns if c.startswith("rmse")]
    m = out.groupby(["strategy", "k"])[num].mean(); print(m.round(4).to_string())
    print(out[out.draw == 0][["strategy", "k", "sp_mix"]].to_string(index=False))

    ref8 = pd.read_csv(MD / "results" / "al_step_balanced_mv10.csv").groupby(["strategy", "k"]).mean(numeric_only=True)
    e0 = res[0]["rmse_md_5000K_2.0gcc"]; red = lambda tab, s, k: e0 - tab.loc[(s, k), "rmse_md_5000K_2.0gcc"]
    gc = {k: red(ref8, "gamma_cluster", k) for k in (10, 20)}; rnd300 = ref8.loc[("random", 40), "rmse_md_300K_1.0gcc"]
    pc = pl.set_index("score"); cvm = cv[cv.score == "gamma_cal"].set_index("statepoint").coverage
    rows = [dict(criterion="N3a-MD (pooled ranking)", passed=bool(pc.loc["gamma_cal", "mc_lo"] > 0),
                 detail=f"pooled Spearman gamma_cal {pc.loc['gamma_cal', 'spearman_pooled']:.3f} vs gamma_cluster {pc.loc['gamma_cluster', 'spearman_pooled']:.3f}; "
                        f"difference {pc.loc['gamma_cal', 'minus_cluster']:+.3f} [{pc.loc['gamma_cal', 'mc_lo']:+.3f}, {pc.loc['gamma_cal', 'mc_hi']:+.3f}]"),
            dict(criterion="N3b-MD (coverage 300 K, 2000 K in [0.85, 0.95])", passed=bool(all(0.85 <= cvm[sp] <= 0.95 for sp in ("300K_1.0gcc", "2000K_1.0gcc"))),
                 detail="; ".join(f"{sp}: {cvm[sp]:.3f}" for sp in cvm.index))]
    for s_ in ("gamma_cal", "gamma_soft"):
        ok5 = all(red(m, s_, k) >= 0.8 * gc[k] for k in (10, 20)); e300 = m.loc[(s_, 40), "rmse_md_300K_1.0gcc"]
        rows.append(dict(criterion=f"N3c-MD selection ({s_})", passed=bool(ok5 and abs(e300 - rnd300) <= 0.02),
                         detail="; ".join(f"K={k}: 5000 K reduction {red(m, s_, k):.3f} vs 0.8 x gamma_cluster {0.8 * gc[k]:.3f}" for k in (10, 20)) + f"; 300 K at K=40 {e300:.3f} vs random {rnd300:.3f}"))
    cr = pd.DataFrame(rows); cr.to_csv(MD / "results" / "advance_md_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
