#!/usr/bin/env python3
"""Step 8: one active-learning iteration on the DFT-only model's own MD, DFT as oracle (Fig. 6B).

base model : shared_t0_dft_train refit here as force-only unweighted least squares on its 90 DFT frames
             (same recipe as results/mode1_jobs: chimes_lsq.py --algorithm svd --eps 1e-8), checked against its params.txt
pool       : DFT-labelled Langevin MD frames of that model, seeds in POOL_SEEDS
test       : DFT-labelled MD frames of the held-out seed + the 30 DFT-MD frames not in the base training set
selection  : K frames by frame-max gamma_bulk, frame-max gamma_cluster, gamma_cluster round-robin over clusters
             (N_MAXVOL MaxVol solutions each), or uniformly at random (N_RANDOM draws). Grades are ranked once against
             the base set; --greedy recomputes them after every added frame (not used in the manuscript).
Writes results/al_step.csv (component force RMSE in eV/A per test set).
--split balanced : pool/test from results/balanced_split.csv (10_balanced_split.py; gamma-matched 1 ps blocks) instead
                   of the seed split; writes results/al_step_balanced.csv.
--save-coefs     : also write every refit model's 42 coefficients (kcal/mol/A units, params.txt order) to
                   work/al_models/<split>/coefs.npz, keys "base" and "<strategy>__k<K>__d<draw>" (used by 11_al_md_setup.py).
--random-sp SP   : add a control arm "random_<T>K": uniform draws (N_RANDOM) from the pool frames of state point SP only
                   (repeatable); used for the MD stability control with SP = 5000K_2.0gcc.
--n-maxvol N     : MaxVol solutions per gamma rule (default N_MAXVOL = 5). Draw d always uses the random streams 11 + d
                   and 500 + d, so draws 0-4 are identical for any N >= 5.
--out-tag TAG    : append TAG to the results file and the model directory (e.g. _mv10 -> results/al_step_balanced_mv10.csv,
                   work/al_models/balanced_mv10/), so a larger run does not overwrite the manuscript files.
usage: 08_al_step.py [--greedy] [--split seed|balanced] [--save-coefs] [--random-sp SP ...] [--n-maxvol N] [--out-tag TAG]
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
POOL_SEEDS, TEST_SEED = (1, 2), 3
KS = [2, 5, 10, 20, 40]
N_RANDOM = 20
N_MAXVOL = 5
RANDOM_SP = []        # state points for the single-state-point random control arm (--random-sp)


def dft_forces(src, k):
    L = (DATA / src).read_text().splitlines(); i = 0
    for _ in range(k):
        i += int(L[i]) + 2
    n = int(L[i])
    return np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float) * HA_BOHR_TO_KCAL_A


def fit(A, b):
    return np.linalg.lstsq(A, b, rcond=1e-8)[0]


def main():
    greedy = "--greedy" in sys.argv
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "seed"
    save = "--save-coefs" in sys.argv
    global RANDOM_SP
    RANDOM_SP = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--random-sp"]
    n_maxvol = int(sys.argv[sys.argv.index("--n-maxvol") + 1]) if "--n-maxvol" in sys.argv else N_MAXVOL
    out_tag = sys.argv[sys.argv.index("--out-tag") + 1] if "--out-tag" in sys.argv else ""
    coefs = {}
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

    A0, b0 = block(train_keys)
    c0 = fit(A0, b0)
    Ah, bh = block(hold_keys)

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    FD = load_dft_forces(MD / "data" / "dft_forces.npz")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    md = fm[(fm.model == MODEL) & (fm.step > 0)].copy()
    md = md[md.frame.isin(list(FD))]
    Xf = {f: np.asarray(Am[3 * start[f]:3 * start[f + 1]]) for f in md.frame}
    Yf = {f: FD[f].ravel() * EV_TO_KCAL for f in md.frame}
    if split == "seed":
        pool = md[md.rep.isin(POOL_SEEDS)]; test = md[md.rep == TEST_SEED]
    else:
        role = pd.read_csv(MD / "results" / f"{split}_split.csv").set_index("frame").role
        assert set(md.frame) == set(role.index), "split file does not cover the labelled frames"
        pool = md[md.frame.map(role) == "pool"]; test = md[md.frame.map(role) == "test"]
    print(f"base train {len(train_keys)} frames; DFT holdout {len(hold_keys)}; pool {len(pool)}; test {len(test)}", flush=True)

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
    lab[kind == "chimes"] = np.load(RW / "labels_candidates.npy")
    g = np.load(MD / "work" / "gamma_md.npz"); cl_md = g["cluster"]
    ref_ids = np.concatenate([by_frame[k] for k in train_keys])

    def frame_scores(extra, rng):
        """frame-max gamma_bulk / gamma_cluster of every pool frame against base + extra frames (one MaxVol solution)."""
        Aref = np.vstack([A0] + [Xf[f] for f in extra])
        lref = np.concatenate([lab[ref_ids]] + [cl_md[start[f]:start[f + 1]] for f in extra])
        inv_b = E.random_restart_maxvol(Aref, rng)
        inv_c = {c: E.random_restart_maxvol(Aref[E.atom_rows_from_indices(np.where(lref == c)[0])], rng) for c in range(6)}
        sb, sc = {}, {}
        for f in pool.frame:
            sb[f] = E.gamma_for_rows(Xf[f], inv_b).max()
            cl = cl_md[start[f]:start[f + 1]]; gc = np.zeros(len(cl))
            for c in np.unique(cl):
                m = np.where(cl == c)[0]
                gc[m] = E.gamma_for_rows(Xf[f][E.atom_rows_from_indices(m)], inv_c[c])
            sc[f] = gc.max()
        return sb, sc

    res = [dict(strategy="base", k=0, draw=0, **evaluate(c0))]
    coefs["base"] = c0

    def frame_key_cluster(f, inv_c):
        """cluster of the atom with the largest gamma_cluster in frame f (for the cluster round-robin arm)."""
        cl = cl_md[start[f]:start[f + 1]]; gc = np.zeros(len(cl))
        for c in np.unique(cl):
            m = np.where(cl == c)[0]
            gc[m] = E.gamma_for_rows(Xf[f][E.atom_rows_from_indices(m)], inv_c[c])
        return int(cl[np.argmax(gc)])

    def run_arm(strat, draw, order_fn):
        chosen = []
        for k in range(1, max(KS) + 1):
            chosen.append(order_fn(chosen))
            if k in KS:
                c = fit(np.vstack([A0] + [Xf[f] for f in chosen]), np.concatenate([b0] + [Yf[f] for f in chosen]))
                coefs[f"{strat}__k{k}__d{draw}"] = c
                res.append(dict(strategy=strat, k=k, draw=draw,
                                sp_mix=pool.set_index("frame").loc[chosen, "statepoint"].value_counts().to_dict(), **evaluate(c)))

    for draw in range(n_maxvol):
        rng = np.random.default_rng(11 + draw)
        sb, sc = frame_scores([], rng)
        for strat, score in [("gamma_bulk", sb), ("gamma_cluster", sc)]:
            def pick(chosen, score=score, strat=strat):
                if greedy and chosen:
                    s = frame_scores(chosen, np.random.default_rng(1000 * draw + len(chosen)))[0 if strat == "gamma_bulk" else 1]
                else:
                    s = score
                return max((f for f in s if f not in chosen), key=s.get)
            run_arm(strat, draw, pick)
        # cluster round-robin: each frame keyed to the cluster of its max-gamma_cluster atom; take the top frame of each
        # cluster in turn (the per-cluster MaxVol selection rule), so no single regime absorbs the budget
        Aref = A0; lref = lab[ref_ids]
        inv_c = {c: E.random_restart_maxvol(Aref[E.atom_rows_from_indices(np.where(lref == c)[0])], np.random.default_rng(500 + draw))
                 for c in range(6)}
        key = {f: frame_key_cluster(f, inv_c) for f in pool.frame}
        queues = {c: sorted([f for f in key if key[f] == c], key=sc.get, reverse=True) for c in sorted(set(key.values()))}
        rr = [q.pop(0) for _ in range(max(len(q) for q in queues.values())) for q in queues.values() if q]
        run_arm("cluster_roundrobin", draw, lambda chosen, rr=rr: rr[len(chosen)])
        print("maxvol draw", draw, "done", flush=True)
    for d in range(N_RANDOM):
        perm = np.random.default_rng(100 + d).permutation(pool.frame.to_numpy())
        for k in KS:
            ch = perm[:k]
            c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
            coefs[f"random__k{k}__d{d}"] = c
            res.append(dict(strategy="random", k=k, draw=d, sp_mix=pool.set_index("frame").loc[ch, "statepoint"].value_counts().to_dict(),
                            **evaluate(c)))
    # control arm: uniform draws from the pool frames of one state point only (separates "which frames gamma picks" from
    # "which state point it picks"); own random streams, so the arms above are unchanged
    for sp in RANDOM_SP:
        name = f"random_{sp.split('_')[0]}"
        for d in range(N_RANDOM):
            perm = np.random.default_rng(200 + d).permutation(pool[pool.statepoint == sp].frame.to_numpy())
            for k in KS:
                ch = perm[:k]
                c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
                coefs[f"{name}__k{k}__d{d}"] = c
                res.append(dict(strategy=name, k=k, draw=d, sp_mix={sp: k}, **evaluate(c)))
    tag = ("_greedy" if greedy else "") + ("" if split == "seed" else f"_{split}") + out_tag
    out = pd.DataFrame(res); out.to_csv(MD / "results" / f"al_step{tag}.csv", index=False)
    if save:
        d = MD / "work" / "al_models" / (split + ("_greedy" if greedy else "") + out_tag); d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "coefs.npz", **coefs)
        print(f"saved {len(coefs)} coefficient sets -> {d / 'coefs.npz'}")
    pd.set_option("display.width", 250)
    num = [c for c in out.columns if c.startswith("rmse")]
    print(out.groupby(["strategy", "k"])[num].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
