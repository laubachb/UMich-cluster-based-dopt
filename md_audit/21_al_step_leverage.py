#!/usr/bin/env python3
"""Step 21: test R3 of open_systems/CRITERIA.md: one active-learning iteration with frames selected by leverage.

Same base model, balanced pool/test split, K values, refits and test sets as step 8. Scores: the relative ridge
leverage (open_systems/core.Leverage) of every pool atom against the 90 base training frames, either one reference
(h_bulk), the training atoms of the atom's spectral cluster (h_cluster) or of its state point (h_statepoint); a frame's
score is its largest atom score. Rules: h_bulk and h_cluster top-K; h_bulk_per_sp and h_statepoint_per_sp with the
budget split equally among state points (5000 K, 2000 K, 300 K in turn, as step 18). Leverage is deterministic, so
each rule is one run. Writes results/al_step_leverage.csv, results/al_step_leverage_criteria.csv and the refit
coefficients to work/al_models/leverage/coefs.npz (keys <strategy>__k<K>__d0, as step 8 --save-coefs).
usage: 21_al_step_leverage.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces
from open_systems.core import Leverage

MODEL = "shared_t0_dft_train"
KS = [2, 3, 5, 6, 10, 12, 20, 21, 39, 40]
TOPK = [2, 5, 10, 20, 40]
ORDER = ["5000K_2.0gcc", "2000K_1.0gcc", "300K_1.0gcc"]
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


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    train_keys = [(f["source"], i) for f in meta["files"] for i in f["indices"]]
    hold_keys = [k for k in by_frame if k[0].startswith("DFT.") and k not in set(train_keys)]
    ref = np.concatenate([by_frame[k] for k in train_keys])
    ref_sp = np.concatenate([[k[0][4:-5]] * len(by_frame[k]) for k in train_keys])
    kind = np.array([r["kind"] for r in man])
    lab = np.full(len(kind), -1); lab[kind == "dft"] = np.load(RW / "labels_dft.npy")

    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); Ah, bh = block(hold_keys)
    assert np.allclose(A0, A[E.atom_rows_from_indices(ref)])

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]
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

    lev_b = Leverage(A0)
    lev_c = {c: Leverage(A[E.atom_rows_from_indices(ref[lab[ref] == c])]) for c in range(6)}
    lev_s = {s: Leverage(A[E.atom_rows_from_indices(ref[ref_sp == s])]) for s in ORDER}
    score = {"h_bulk": {}, "h_cluster": {}, "h_statepoint": {}}
    for f in pool.frame:
        X = Xf[f]; cl = cl_md[start[f]:start[f + 1]]
        score["h_bulk"][f] = lev_b.grade(X).max()
        score["h_cluster"][f] = max(lev_c[c].grade(X[E.atom_rows_from_indices(np.where(cl == c)[0])]).max() for c in np.unique(cl))
        score["h_statepoint"][f] = lev_s[sp_of[f]].grade(X).max()

    c0 = fit(A0, b0); coefs = {"base": c0}
    res = [dict(strategy="base", k=0, **evaluate(c0))]

    def refit(strat, chosen, ks):
        for k in ks:
            ch = chosen[:k]
            c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
            coefs[f"{strat}__k{k}__d0"] = c
            res.append(dict(strategy=strat, k=k, sp_mix=sp_of.loc[ch].value_counts().to_dict(), **evaluate(c)))

    for s in ("h_bulk", "h_cluster"):
        refit(s, sorted(score[s], key=score[s].get, reverse=True), TOPK)
    for s in ("h_bulk", "h_statepoint"):
        q = {sp: sorted(sp_of.index[sp_of == sp], key=score[s].get, reverse=True) for sp in ORDER}
        refit(f"{s}_per_sp", [q[ORDER[i % 3]][i // 3] for i in range(max(KS))], KS)
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_leverage.csv", index=False)
    d = MD / "work" / "al_models" / "leverage"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "coefs.npz", **coefs)                       # for 11_al_md_setup.py leverage --seeds ... (test S)
    pd.set_option("display.width", 250)
    print(out.drop(columns="sp_mix").round(4).to_string(index=False))
    print(out[["strategy", "k", "sp_mix"]].to_string(index=False))

    ref_al = pd.read_csv(MD / "results" / "al_step_balanced_mv10.csv").groupby(["strategy", "k"]).mean(numeric_only=True)
    e0 = res[0]["rmse_md_5000K_2.0gcc"]; o = out.set_index(["strategy", "k"])
    red = lambda tab, s, k: e0 - tab.loc[(s, k), "rmse_md_5000K_2.0gcc"]
    gc = {k: red(ref_al, "gamma_cluster", k) for k in (10, 20)}
    rnd300 = ref_al.loc[("random", 40), "rmse_md_300K_1.0gcc"]
    rows = [dict(criterion="R3-1", passed=all(red(o, "h_bulk", k) < gc[k] for k in (10, 20)),
                 detail="; ".join(f"K={k}: h_bulk {red(o, 'h_bulk', k):.3f} vs gamma_cluster {gc[k]:.3f}" for k in (10, 20))),
            dict(criterion="R3-2", passed=all(red(o, "h_cluster", k) >= 0.8 * gc[k] for k in (10, 20)),
                 detail="; ".join(f"K={k}: h_cluster {red(o, 'h_cluster', k):.3f} vs 0.8 x gamma_cluster {0.8 * gc[k]:.3f}" for k in (10, 20)))]
    for s in ("h_bulk", "h_cluster"):
        ok = all(red(o, s, k) >= 0.8 * gc[k] for k in (10, 20)) and abs(o.loc[(s, 40), "rmse_md_300K_1.0gcc"] - rnd300) <= 0.02
        rows.append(dict(criterion=f"decision ({s})", passed=ok,
                         detail=f"300 K error at K=40 {o.loc[(s, 40), 'rmse_md_300K_1.0gcc']:.3f} vs random {rnd300:.3f}"))
    cr = pd.DataFrame(rows); cr.to_csv(MD / "results" / "al_step_leverage_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
