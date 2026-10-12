#!/usr/bin/env python3
"""Step 22: test M2 of open_systems/CRITERIA.md on the MD of the DFT-only model: minimum over groups (Shuang et al.,
arXiv 2508.03405) vs assignment to one group.

Grades: every MD atom against the basis of every group; gamma_min = smallest, gamma_assigned = own group.
  cluster    bases as step 3 (reference = sorted training atoms, solution t uses seed 300 + t, bulk basis drawn first,
             then clusters 0-5), so gamma_assigned reproduces the manuscript's gamma_cluster (checked)
  statepoint one basis per DFT-MD state point of the training frames (all 6, so the 6000-8000 K bases enter the minimum)
Frame level per state point: Spearman of frame-max (mean over 10 solutions) vs frame RMSE, bootstrap over frames
(2,000, seed 0), paired difference to gamma_cluster. Then one active-learning iteration (as step 8, balanced split):
top-K by frame-max gamma_min,cluster with the bases of step 8 (random streams 11 + draw), and the assigned rule as a
check that step 8's gamma_cluster selection is reproduced.
Writes results/min_rule_md_frame.csv, results/al_step_min_rule.csv, results/min_rule_md_criteria.csv.
usage: 22_min_rule.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces

MODEL = "shared_t0_dft_train"
N_TRIALS, N_BOOT = 10, 2000
KS = [2, 5, 10, 20, 40]
HA_BOHR_TO_KCAL_A = 627.509474 / 0.529177211
EV_TO_KCAL = 23.0605478


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def all_groups(X, bases):
    """(n_groups, n_atoms) grades of the atoms with rows X against every basis."""
    return np.vstack([E.gamma_for_rows(X, inv) for inv in bases])


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    kind = np.array([r["kind"] for r in man])
    lab = np.full(len(kind), -1); lab[kind == "dft"] = np.load(RW / "labels_dft.npy")
    sp_atom = np.array([r["source_file"][4:-5] if r["source_file"].startswith("DFT.") else "" for r in man])
    ref = np.array(sorted(a for k in train_keys for a in by_frame[k]))          # as step 3

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    gidx = start[df.frame.to_numpy()] + df.atom.to_numpy()
    Xmd = np.asarray(Am[E.atom_rows_from_indices(gidx)])

    sps = sorted(set(sp_atom[ref]))
    Ga, Gm, Gs = (np.zeros((N_TRIALS, len(df))) for _ in range(3))
    for t in range(N_TRIALS):
        rng = np.random.default_rng(E.SEED0 + t)
        E.random_restart_maxvol(A[E.atom_rows_from_indices(ref)], rng)                # bulk basis first, as step 3
        P = all_groups(Xmd, [E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[lab[ref] == c])], rng) for c in range(6)])
        Ga[t] = P[cl_md[gidx], np.arange(len(df))]; Gm[t] = P.min(0)
        rng = np.random.default_rng(E.SEED0 + t)
        Gs[t] = all_groups(Xmd, [E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[sp_atom[ref] == s])], rng) for s in sps]).min(0)
        print(f"solution {t} done", flush=True)
    dev = np.abs(Ga.mean(0) - df.gamma_cluster.to_numpy()).max()
    print(f"assigned gamma_cluster vs manuscript (atom_errors csv, float32): max |diff| = {dev:.2e}", flush=True)
    assert dev < 1e-4
    df["gamma_min_cluster"], df["gamma_min_statepoint"] = Gm.mean(0), Gs.mean(0)
    print(f"share of 5000 K atoms whose cluster minimum is their own cluster: "
          f"{np.mean((Gm[-1] == Ga[-1])[df.statepoint.to_numpy() == '5000K_2.0gcc']):.3f}", flush=True)

    scores = ["gamma_bulk", "gamma_cluster", "gamma_min_cluster", "gamma_min_statepoint"]
    frame = []
    for sp, d in df.groupby("statepoint"):
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in scores})
        B = np.random.default_rng(0).integers(0, len(fr), (N_BOOT, len(fr)))
        boot = {s: np.array([spearmanr(fr[s].to_numpy()[b], fr.rmse.to_numpy()[b])[0] for b in B]) for s in scores}
        for s in scores:
            diff = boot["gamma_cluster"] - boot[s]
            frame.append(dict(statepoint=sp, score=s, spearman=spearmanr(fr[s], fr.rmse)[0], lo=np.quantile(boot[s], .025),
                              hi=np.quantile(boot[s], .975), cluster_minus=diff.mean(), cm_lo=np.quantile(diff, .025),
                              cm_hi=np.quantile(diff, .975)))
    frm = pd.DataFrame(frame); frm.to_csv(MD / "results" / "min_rule_md_frame.csv", index=False)
    pd.set_option("display.width", 250); print("\n== frame level\n" + frm.round(3).to_string(index=False))
    r = frm[(frm.statepoint == "5000K_2.0gcc") & (frm.score == "gamma_min_cluster")].iloc[0]
    cr = pd.DataFrame([dict(criterion="M2", passed=bool(r.cm_lo > 0),
                            detail=f"5000 K frame Spearman gamma_min_cluster {r.spearman:.3f}; gamma_cluster minus it "
                                   f"{r.cluster_minus:+.3f} [{r.cm_lo:+.3f}, {r.cm_hi:+.3f}]")])
    cr.to_csv(MD / "results" / "min_rule_md_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))

    # ---- one active-learning iteration (step 8 bases: reference in train_keys order, streams 11 + draw)
    hold_keys = [k for k in by_frame if k[0].startswith("DFT.") and k not in set(train_keys)]
    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); Ah, bh = block(hold_keys)
    ref_ids = np.concatenate([by_frame[k] for k in train_keys])
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
        E.random_restart_maxvol(A0, rng)                                              # step 8 draws inv_b first
        bases = [E.random_restart_maxvol(A0[E.atom_rows_from_indices(np.where(lab[ref_ids] == c)[0])], rng) for c in range(6)]
        sa, sm = {}, {}
        for f in pool.frame:
            P = all_groups(Xf[f], bases); cl = cl_md[start[f]:start[f + 1]]
            sa[f] = P[cl, np.arange(len(cl))].max(); sm[f] = P.min(0).max()
        for strat, s in (("gamma_cluster_check", sa), ("gamma_min_cluster", sm)):
            order = sorted(s, key=s.get, reverse=True)
            for k in KS:
                ch = order[:k]
                c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
                res.append(dict(strategy=strat, k=k, draw=draw, sp_mix=sp_of.loc[ch].value_counts().to_dict(), **evaluate(c)))
        print("AL draw", draw, "done", flush=True)
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_min_rule.csv", index=False)
    num = [c for c in out.columns if c.startswith("rmse")]
    m = out.groupby(["strategy", "k"])[num].mean()
    ref8 = pd.read_csv(MD / "results" / "al_step_balanced_mv10.csv").groupby(["strategy", "k"])[num].mean()
    print(m.round(4).to_string())
    print("check: max |gamma_cluster_check - step 8 gamma_cluster| =",
          float(np.abs(m.loc["gamma_cluster_check"].to_numpy() - ref8.loc["gamma_cluster"].to_numpy()).max()))
    print(out[out.draw == 0][["strategy", "k", "sp_mix"]].to_string(index=False))


if __name__ == "__main__":
    main()
