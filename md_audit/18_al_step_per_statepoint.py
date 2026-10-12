#!/usr/bin/env python3
"""Step 18: one active-learning iteration with the labeling budget divided equally among state points.

Step 8 ranks the pool once and takes the top K frames, so gamma_cluster spends its first frames on 5000 K only and the
300 K / 2000 K test error rises until molecular frames are selected. Here the same frame scores are used (same MaxVol
random streams 11 + draw as step 8, balanced split), but frames are taken in turn from the three state points
(5000 K, 2000 K, 300 K, 5000 K, ...), each time the highest-scored remaining frame of that state point. The state
point of an MD frame is known from the simulation it came from, so the rule needs no extra information.
Arms: gamma_cluster_per_sp, gamma_bulk_per_sp, random_per_sp (20 draws). K = 3, 6, 12, 21, 39 (multiples of 3) and
K = 2, 5, 10, 20, 40 for comparison with step 8. Writes results/al_step_per_statepoint.csv.
usage: 18_al_step_per_statepoint.py [n_maxvol=10]
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces

HA_BOHR_TO_KCAL_A = 627.509474 / 0.529177211
EV_TO_KCAL = 23.0605478
MODEL = "shared_t0_dft_train"
KS = [2, 3, 5, 6, 10, 12, 20, 21, 39, 40]
ORDER = ["5000K_2.0gcc", "2000K_1.0gcc", "300K_1.0gcc"]
N_RANDOM = 20


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def main():
    n_maxvol = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    hold_keys = [k for k in by_frame if k[0].startswith("DFT.") and k not in set(train_keys)]

    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()

    A0, b0 = block(train_keys); Ah, bh = block(hold_keys)
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    FD = load_dft_forces(MD / "data" / "dft_forces.npz")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    md = fm[(fm.model == MODEL) & (fm.step > 0)].copy(); md = md[md.frame.isin(list(FD))]
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

    lab = np.full(A.shape[0] // 3, -1)
    kind = np.array([r["kind"] for r in man])
    lab[kind == "dft"] = np.load(RW / "labels_dft.npy")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]
    ref_ids = np.concatenate([by_frame[k] for k in train_keys])

    def frame_scores(rng):                                  # as step 8 (base set only)
        inv_b = E.random_restart_maxvol(A0, rng)
        inv_c = {c: E.random_restart_maxvol(A0[E.atom_rows_from_indices(np.where(lab[ref_ids] == c)[0])], rng) for c in range(6)}
        sb, sc = {}, {}
        for f in pool.frame:
            sb[f] = E.gamma_for_rows(Xf[f], inv_b).max()
            cl = cl_md[start[f]:start[f + 1]]; gc = np.zeros(len(cl))
            for c in np.unique(cl):
                m = np.where(cl == c)[0]
                gc[m] = E.gamma_for_rows(Xf[f][E.atom_rows_from_indices(m)], inv_c[c])
            sc[f] = gc.max()
        return sb, sc

    res = [dict(strategy="base", k=0, draw=0, **evaluate(fit(A0, b0)))]

    def run(strat, draw, queues):
        chosen = []
        for k in range(1, max(KS) + 1):
            chosen.append(queues[ORDER[(k - 1) % 3]].pop(0))
            if k in KS:
                c = fit(np.vstack([A0] + [Xf[f] for f in chosen]), np.concatenate([b0] + [Yf[f] for f in chosen]))
                res.append(dict(strategy=strat, k=k, draw=draw, **evaluate(c)))

    for draw in range(n_maxvol):
        sb, sc = frame_scores(np.random.default_rng(11 + draw))
        for strat, s in (("gamma_bulk_per_sp", sb), ("gamma_cluster_per_sp", sc)):
            run(strat, draw, {sp: sorted(sp_of.index[sp_of == sp], key=s.get, reverse=True) for sp in ORDER})
        print("maxvol draw", draw, "done", flush=True)
    for d in range(N_RANDOM):
        rng = np.random.default_rng(300 + d)
        run("random_per_sp", d, {sp: list(rng.permutation(sp_of.index[sp_of == sp].to_numpy())) for sp in ORDER})
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_per_statepoint.csv", index=False)
    pd.set_option("display.width", 250)
    num = [c for c in out.columns if c.startswith("rmse")]
    print(out.groupby(["strategy", "k"])[num].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
