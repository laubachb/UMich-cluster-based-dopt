#!/usr/bin/env python3
"""Step 23: tests H2, H4 and H5 of open_systems/CRITERIA.md (addendum H) on the MD of the DFT-only model.

Reference = the 90 training frames (forces in kcal/mol/A, as step 8). Groups = the manuscript's clusters (MD atoms by
the step 3 assignment). Regime noise sigma_k = RMS force-component residual of cluster k under the DFT-only model's own
least-squares fit. Scores: IG_hom, IG_het, IG_hier (open_systems/core.BayesGrade; information gain of an atom's 3 rows,
frame score = largest atom score) and gamma_w (MaxVol on rows scaled by 1/sigma_k, 10 solutions, mean per atom).
Frame level per state point: Spearman of frame-max score vs frame force RMSE, bootstrap over frames (2,000, seed 0),
paired difference to gamma_cluster. Then one active-learning iteration (as step 21): top-K over the whole pool by
frame-max IG_het and IG_hier, and per-state-point versions; decision rule H4.
Writes results/info_gain_md_frame.csv, results/al_step_info_gain.csv, results/info_gain_md_criteria.csv and the refit
coefficients to work/al_models/infogain/coefs.npz.
usage: 23_info_gain.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, DATA, RETRO_WORK as RW
from common.xyzf import load_dft_forces
from open_systems.core import BayesGrade, group_sigma, fit_hier

MODEL = "shared_t0_dft_train"
N_TRIALS, N_BOOT = 10, 2000
TOPK = [2, 5, 10, 20, 40]
KS_SP = [2, 3, 5, 6, 10, 12, 20, 21, 39, 40]
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
    kind = np.array([r["kind"] for r in man])
    lab = np.full(len(kind), -1); lab[kind == "dft"] = np.load(RW / "labels_dft.npy")
    def block(keys):
        ids = np.concatenate([by_frame[k] for k in keys])
        return A[E.atom_rows_from_indices(ids)], np.concatenate([dft_forces(*k) for k in keys]).ravel()
    A0, b0 = block(train_keys); Ah, bh = block(hold_keys)
    ref_ids = np.concatenate([by_frame[k] for k in train_keys]); gr = np.repeat(lab[ref_ids], 3)
    c0 = fit(A0, b0)
    sigma, s0 = group_sigma(A0, b0, gr, c0)
    print("regime noise sigma_k (kcal/mol/A):", {k: round(v, 3) for k, v in sigma.items()}, "pooled", round(s0, 3), flush=True)
    models = {"IG_hom": BayesGrade(A0, b0, gr, {k: s0 for k in sigma}, s0), "IG_het": BayesGrade(A0, b0, gr, sigma, s0)}
    models["IG_hier"], t, ev = fit_hier(A0, b0, gr, sigma, s0)
    print(f"IG_hier: t = {t}; log evidence by t: " + ", ".join(f"{k}: {v:.1f}" for k, v in ev.items()), flush=True)
    wr = 1 / np.array([sigma[int(k)] for k in gr])
    inv_w = [E.random_restart_maxvol(A0 * wr[:, None], np.random.default_rng(E.SEED0 + t_)) for t_ in range(N_TRIALS)]

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    start = np.r_[0, np.cumsum(fm.n_atoms.to_numpy())]
    Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    cl_md = np.load(MD / "work" / "gamma_md.npz")["cluster"]

    def scores(X, cl):
        out = {k: m.info_gain(X, cl) for k, m in models.items()}
        w = 1 / np.array([sigma[int(k)] for k in np.repeat(cl, 3)])
        out["gamma_w"] = np.mean([E.gamma_for_rows(X * w[:, None], inv) for inv in inv_w], 0)
        return out

    df = pd.read_csv(MD / "work" / f"label_partitions_{MODEL}.csv")          # step 20: errors, gamma and h columns
    gidx = start[df.frame.to_numpy()] + df.atom.to_numpy()
    for k, v in scores(np.asarray(Am[E.atom_rows_from_indices(gidx)]), cl_md[gidx]).items():
        df[k] = v
    sc = ["gamma_bulk", "gamma_cluster", "gamma_statepoint", "h_bulk", "h_cluster", "gamma_w", "IG_hom", "IG_het", "IG_hier"]
    frame = []
    for sp, d in df.groupby("statepoint"):
        fr = d.groupby("frame").agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))), **{s: (s, "max") for s in sc})
        B = np.random.default_rng(0).integers(0, len(fr), (N_BOOT, len(fr)))
        boot = {s: np.array([spearmanr(fr[s].to_numpy()[b], fr.rmse.to_numpy()[b])[0] for b in B]) for s in sc}
        for s in sc:
            dd = boot[s] - boot["gamma_cluster"]
            frame.append(dict(statepoint=sp, score=s, spearman=spearmanr(fr[s], fr.rmse)[0], lo=np.quantile(boot[s], .025),
                              hi=np.quantile(boot[s], .975), minus_cluster=dd.mean(), mc_lo=np.quantile(dd, .025), mc_hi=np.quantile(dd, .975)))
    frm = pd.DataFrame(frame); frm.to_csv(MD / "results" / "info_gain_md_frame.csv", index=False)
    pd.set_option("display.width", 250); print("\n== frame level\n" + frm.round(3).to_string(index=False))

    # ---- one active-learning iteration
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

    fsc = {k: {} for k in ("IG_het", "IG_hier")}
    for f in pool.frame:
        cl = cl_md[start[f]:start[f + 1]]
        for k in fsc:
            fsc[k][f] = models[k].info_gain(Xf[f], cl).max()
    coefs = {"base": c0}; res = [dict(strategy="base", k=0, **evaluate(c0))]
    def refit(strat, chosen, ks):
        for k in ks:
            ch = chosen[:k]
            c = fit(np.vstack([A0] + [Xf[f] for f in ch]), np.concatenate([b0] + [Yf[f] for f in ch]))
            coefs[f"{strat}__k{k}__d0"] = c
            res.append(dict(strategy=strat, k=k, sp_mix=sp_of.loc[ch].value_counts().to_dict(), **evaluate(c)))
    for s in fsc:
        refit(s, sorted(fsc[s], key=fsc[s].get, reverse=True), TOPK)
        q = {sp: sorted(sp_of.index[sp_of == sp], key=fsc[s].get, reverse=True) for sp in ORDER}
        refit(f"{s}_per_sp", [q[ORDER[i % 3]][i // 3] for i in range(max(KS_SP))], KS_SP)
    out = pd.DataFrame(res); out.to_csv(MD / "results" / "al_step_info_gain.csv", index=False)
    d_ = MD / "work" / "al_models" / "infogain"; d_.mkdir(parents=True, exist_ok=True); np.savez(d_ / "coefs.npz", **coefs)
    print(out.drop(columns="sp_mix").round(4).to_string(index=False))
    print(out[out.k.isin(TOPK)][["strategy", "k", "sp_mix"]].to_string(index=False))

    ref8 = pd.read_csv(MD / "results" / "al_step_balanced_mv10.csv").groupby(["strategy", "k"]).mean(numeric_only=True)
    e0 = res[0]["rmse_md_5000K_2.0gcc"]; o = out.set_index(["strategy", "k"])
    red = lambda tab, s, k: e0 - tab.loc[(s, k), "rmse_md_5000K_2.0gcc"]
    gc = {k: red(ref8, "gamma_cluster", k) for k in (10, 20)}; rnd300 = ref8.loc[("random", 40), "rmse_md_300K_1.0gcc"]
    r5 = frm[(frm.statepoint == "5000K_2.0gcc") & (frm.score == "IG_het")].iloc[0]
    rows = [dict(criterion="H2", passed=bool(r5.mc_lo > 0),
                 detail=f"5000 K frame Spearman IG_het {r5.spearman:.3f} [{r5.lo:.3f}, {r5.hi:.3f}]; minus gamma_cluster "
                        f"{r5.minus_cluster:+.3f} [{r5.mc_lo:+.3f}, {r5.mc_hi:+.3f}]")]
    for s in fsc:
        ok5 = all(red(o, s, k) >= 0.8 * gc[k] for k in (10, 20)); e300 = o.loc[(s, 40), "rmse_md_300K_1.0gcc"]
        rows.append(dict(criterion=f"H4 ({s})", passed=bool(ok5 and abs(e300 - rnd300) <= 0.02),
                         detail="; ".join(f"K={k}: 5000 K reduction {red(o, s, k):.3f} vs 0.8 x gamma_cluster {0.8 * gc[k]:.3f}" for k in (10, 20))
                                + f"; 300 K at K=40 {e300:.3f} vs random {rnd300:.3f}"))
    cr = pd.DataFrame(rows); cr.to_csv(MD / "results" / "info_gain_md_criteria.csv", index=False)
    print("\n== criteria\n" + cr.to_string(index=False))


if __name__ == "__main__":
    main()
