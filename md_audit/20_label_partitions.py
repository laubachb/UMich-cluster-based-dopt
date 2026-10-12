#!/usr/bin/env python3
"""Step 20: tests R1 and R2 of open_systems/CRITERIA.md (addendum of 2026-10-12) on the MD of the DFT-only model.

Reference = the 90 training frames of shared_t0_dft_train, as in step 3. Partitions of reference and MD atoms by the
known simulation condition of their frame:
  statepoint  6 groups (an MD frame's state point is that of its run)
  density     5 groups (300 K and 2000 K merged at 1.0 g/cm3)
  phase       2 groups (1.0 g/cm3 vs the rest)
gamma_<partition>: MaxVol per group (common/dopt.py), 10 solutions, seeds 300 + t, mean over solutions per atom (as
step 6 does for gamma_bulk and gamma_cluster, which are taken from work/atom_errors_<model>.csv).
h_bulk, h_cluster, h_statepoint: relative ridge leverage (open_systems/core.Leverage) against the same reference.
Per state point: atom-level Spearman, AUROC (top 10 % error), recall at 10 %; frame-level Spearman of frame-max score
vs frame force RMSE with a bootstrap over frames (2,000 resamples, seed 0) and the paired difference to gamma_cluster.
Then one active-learning iteration (balanced split, as step 8): top-K pool frames by frame-max gamma_statepoint or
gamma_phase (10 MaxVol solutions, random streams 11 + draw), refit, test RMSE.
Writes results/label_partitions_md_{atom,frame}.csv, results/al_step_label_partitions.csv,
results/label_partitions_md_criteria.csv.
usage: 20_label_partitions.py [--no-al]
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces
from open_systems.core import Leverage

MODEL = "shared_t0_dft_train"
N_TRIALS, N_BOOT = 10, 2000
KS = [2, 5, 10, 20, 40]
HA_BOHR_TO_KCAL_A = 627.509474 / 0.529177211
EV_TO_KCAL = 23.0605478
PARTS = {"statepoint": lambda sp: sp, "density": lambda sp: sp.split("_")[1], "phase": lambda sp: sp.split("_")[1] == "1.0gcc"}


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def codes(values):
    return pd.Categorical(values).codes


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    ref = np.concatenate([by_frame[k] for k in train_keys])
    ref_sp = np.concatenate([[k[0][4:-5]] * len(by_frame[k]) for k in train_keys])     # DFT.300K_1.0gcc.xyzf -> 300K_1.0gcc
    kind = np.array([r["kind"] for r in man])
    lab = np.full(len(kind), -1); lab[kind == "dft"] = np.load(RW / "labels_dft.npy")

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    gidx = start[df.frame.to_numpy()] + df.atom.to_numpy()
    Xmd = np.asarray(Am[E.atom_rows_from_indices(gidx)])
    md_sp = df.statepoint.to_numpy()
    print(f"reference atoms {ref.size}; MD atoms {len(df)}", flush=True)

    # ---- grades with known-condition partitions
    for pname, fn in PARTS.items():
        rl = np.array([fn(s) for s in ref_sp]); ml = np.array([fn(s) for s in md_sp])
        G = np.full((N_TRIALS, len(df)), np.nan)
        for t in range(N_TRIALS):
            rng = np.random.default_rng(E.SEED0 + t)
            for g in sorted(set(ml)):
                inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[rl == g])], rng)
                loc = np.where(ml == g)[0]
                G[t, loc] = E.gamma_for_rows(Xmd[E.atom_rows_from_indices(loc)], inv)
        df[f"gamma_{pname}"] = G.mean(0)
        print(f"gamma_{pname} done", flush=True)

    def lev(rl, ml):
        h = np.full(len(df), np.nan)
        for g in sorted(set(ml)):
            loc = np.where(ml == g)[0]
            h[loc] = Leverage(A[E.atom_rows_from_indices(ref[rl == g])]).grade(Xmd[E.atom_rows_from_indices(loc)])
        return h
    df["h_bulk"] = lev(np.zeros(ref.size, int), np.zeros(len(df), int))
    df["h_cluster"] = lev(lab[ref], cl_md[gidx])
    df["h_statepoint"] = lev(ref_sp, md_sp)
    df.to_csv(MD / "work" / f"label_partitions_{MODEL}.csv", index=False)

    scores = ["gamma_bulk", "gamma_cluster", "gamma_statepoint", "gamma_density", "gamma_phase", "h_bulk", "h_cluster", "h_statepoint"]
    atom, frame = [], []
    for sp, d in df.groupby("statepoint"):
        y = d.err.to_numpy(); top = y > np.quantile(y, 0.9); k = max(1, int(round(0.1 * len(y))))
        for s in scores:
            v = d[s].to_numpy()
            atom.append(dict(statepoint=sp, score=s, spearman=spearmanr(v, y)[0], auroc_q90=roc_auc_score(top, v),
                             recall10=top[np.argsort(-v)[:k]].sum() / top.sum()))
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in scores})
        rng = np.random.default_rng(0); B = rng.integers(0, len(fr), (N_BOOT, len(fr)))
        boot = {s: np.array([spearmanr(fr[s].to_numpy()[b], fr.rmse.to_numpy()[b])[0] for b in B]) for s in scores}
        for s in scores:
            diff = boot[s] - boot["gamma_cluster"]
            frame.append(dict(statepoint=sp, score=s, n_frames=len(fr), spearman=spearmanr(fr[s], fr.rmse)[0],
                              lo=np.quantile(boot[s], 0.025), hi=np.quantile(boot[s], 0.975),
                              diff_vs_cluster=np.mean(diff), diff_lo=np.quantile(diff, 0.025), diff_hi=np.quantile(diff, 0.975)))
    at, frm = pd.DataFrame(atom), pd.DataFrame(frame)
    at.to_csv(MD / "results" / "label_partitions_md_atom.csv", index=False)
    frm.to_csv(MD / "results" / "label_partitions_md_frame.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n== atom level\n" + at.pivot_table(index="score", columns="statepoint", values=["spearman", "auroc_q90"]).round(3).to_string())
    print("\n== frame level\n" + frm.round(3).to_string(index=False))
    r = frm[(frm.statepoint == "5000K_2.0gcc") & (frm.score == "gamma_statepoint")].iloc[0]
    cr = pd.DataFrame([dict(criterion="R1-3", passed=bool(r.diff_hi < 0),
                            detail=f"5000 K frame Spearman gamma_statepoint {r.spearman:.3f} [{r.lo:.3f}, {r.hi:.3f}]; "
                                   f"minus gamma_cluster {r.diff_vs_cluster:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]")])
    cr.to_csv(MD / "results" / "label_partitions_md_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))
    if "--no-al" in sys.argv:
        return

    # ---- one active-learning iteration (as step 8, balanced split)
    hold_keys = [k for k in by_frame if k[0].startswith("DFT.") and k not in set(train_keys)]

    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); Ah, bh = block(hold_keys)
    FD = load_dft_forces(MD / "data" / "dft_forces.npz")
    md = fm[(fm.model == MODEL) & (fm.step > 0)]; md = md[md.frame.isin(list(FD))]
    Xf = {f: np.asarray(Am[3 * start[f]:3 * start[f + 1]]) for f in md.frame}
    Yf = {f: FD[f].ravel() * EV_TO_KCAL for f in md.frame}
    role = pd.read_csv(MD / "results" / "balanced_split.csv").set_index("frame").role
    pool = md[md.frame.map(role) == "pool"]; test = md[md.frame.map(role) == "test"]
    sp_of = pool.set_index("frame").statepoint

    def evaluate(c):
        out = {}
        for sp, t in test.groupby("statepoint"):
            e = np.concatenate([Xf[f] @ c - Yf[f] for f in t.frame])
            out[f"rmse_md_{sp}"] = np.sqrt(np.mean(e ** 2)) / EV_TO_KCAL
        out["rmse_dft_holdout"] = np.sqrt(np.mean((Ah @ c - bh) ** 2)) / EV_TO_KCAL
        return out

    res = [dict(strategy="base", k=0, draw=0, **evaluate(fit(A0, b0)))]
    for draw in range(N_TRIALS):
        rng = np.random.default_rng(11 + draw)
        for pname in ("statepoint", "phase"):
            fn = PARTS[pname]; rl = np.array([fn(s) for s in ref_sp])
            groups = sorted(set(fn(s) for s in sp_of))
            inv = {g: E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[rl == g])], rng) for g in groups}
            score = {f: E.gamma_for_rows(Xf[f], inv[fn(sp_of[f])]).max() for f in pool.frame}
            order = sorted(score, key=score.get, reverse=True)
            for k in KS:
                ch = order[:k]
                c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
                res.append(dict(strategy=f"gamma_{pname}", k=k, draw=draw, sp_mix=sp_of.loc[ch].value_counts().to_dict(), **evaluate(c)))
        print("AL draw", draw, "done", flush=True)
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_label_partitions.csv", index=False)
    num = [c for c in out.columns if c.startswith("rmse")]
    print(out.groupby(["strategy", "k"])[num].mean().round(4).to_string())
    print(out[out.draw == 0][["strategy", "k", "sp_mix"]].to_string(index=False))


if __name__ == "__main__":
    main()
