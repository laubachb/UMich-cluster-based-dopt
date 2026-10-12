#!/usr/bin/env python3
"""Step 26: refit coefficients of the gamma_soft top-K selection of step 24 (MaxVol draw 0 only), for the MD
stability runs of the soft-membership rule (open_systems/CRITERIA.md, exploratory addendum after test S).

Reproduces exactly the gamma_soft active-learning iteration of 24_advance.py for draw 0: streams
np.random.default_rng(11 + draw) with the bulk basis drawn first, then the 6 cluster bases (reference = the 90
training frames of the DFT-only model, A0 in train_meta block order); membership probabilities from the step-3
15-NN distance-weighted vote; frame score = largest soft_grade of its atoms; top-K over the whole balanced pool,
K = 2, 5, 10, 20, 40, same fit() and test sets. Checks the test RMSEs against the draw-0 gamma_soft rows of
results/al_step_advance.csv and writes work/al_models/soft/coefs.npz (keys base, gamma_soft__k<K>__d0, the format
of step 21), for 11_al_md_setup.py soft --seeds ... --select gamma_soft:1.
usage: 26_soft_coefs.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces
from open_systems.core import soft_grade

MODEL = "shared_t0_dft_train"
DRAW = 0
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
    X = A.reshape(len(kind), -1)
    sc = StandardScaler().fit(X[dft_ids]); Zd = sc.transform(X[dft_ids])
    knn = KNeighborsClassifier(15, weights="distance").fit(Zd, lab[dft_ids])
    assert (knn.classes_ == np.arange(6)).all()

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]

    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); c0 = fit(A0, b0)
    Ah, bh = block(hold_keys)
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
        for sp, t_ in test.groupby("statepoint"):
            e = np.concatenate([Xf[f] @ c - Yf[f] for f in t_.frame])
            out[f"rmse_md_{sp}"] = np.sqrt(np.mean(e ** 2)) / EV_TO_KCAL
        out["rmse_dft_holdout"] = np.sqrt(np.mean((Ah @ c - bh) ** 2)) / EV_TO_KCAL
        return out

    # ---- draw 0 of the step-24 iteration: bulk basis first (same stream), then the 6 cluster bases
    rng = np.random.default_rng(11 + DRAW)
    E.random_restart_maxvol(A0, rng)
    bases = [E.random_restart_maxvol(A0[E.atom_rows_from_indices(np.where(lab[ref_ids] == c)[0])], rng) for c in range(6)]
    s_soft = {}
    for f in pool.frame:
        Pf = all_groups(Xf[f], bases); cl = cl_md[start[f]:start[f + 1]]
        pr = knn.predict_proba(sc.transform(Xf[f].reshape(len(cl), -1)))
        s_soft[f] = soft_grade(Pf, pr).max()
    order = sorted(s_soft, key=s_soft.get, reverse=True)
    coefs = {"base": c0}; res = [dict(strategy="base", k=0, draw=DRAW, **evaluate(c0))]
    for k in KS:
        ch = order[:k]
        c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
        coefs[f"gamma_soft__k{k}__d0"] = c
        res.append(dict(strategy="gamma_soft", k=k, draw=DRAW, sp_mix=sp_of.loc[ch].value_counts().to_dict(),
                        frames=list(ch), **evaluate(c)))
    out = pd.DataFrame(res)
    pd.set_option("display.width", 250)
    print(out.drop(columns="frames").round(4).to_string(index=False))

    # ---- check against the draw-0 rows of step 24
    ref = pd.read_csv(MD / "results" / "al_step_advance.csv")
    ref = ref[(ref.strategy == "gamma_soft") & (ref.draw == DRAW)].set_index("k")
    num = [c for c in out.columns if c.startswith("rmse")]
    chk = out[out.strategy == "gamma_soft"].set_index("k")[num]
    diff = (chk - ref[num].reindex(chk.index)).abs()
    print(f"max |RMSE - al_step_advance.csv (draw {DRAW})| over K = {KS}: {diff.to_numpy().max():.2e}")
    assert diff.to_numpy().max() < 1e-6, "refits do not reproduce step 24"
    d = MD / "work" / "al_models" / "soft"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "coefs.npz", **coefs)
    out.to_csv(MD / "results" / "al_step_soft_d0.csv", index=False)
    print(f"saved {d / 'coefs.npz'} with keys {list(coefs)}")


if __name__ == "__main__":
    main()
