#!/usr/bin/env python3
"""Step 9: Fig. 6A/6B of the manuscript -> figures/ (gamma_bulk gray, gamma_cluster blue).

fig6A_md_flagging.png : does gamma find the MD model's true DFT force error in its own Langevin MD?
fig6B_md_al_step.png  : one active-learning step (frames added -> held-out MD force RMSE per regime)
inputs: work/atom_errors_shared_t0_dft_train.csv (step 6), results/al_step.csv (step 8)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD, FIGURES as OUT
OUT.mkdir(exist_ok=True)
C = {"gamma_bulk": "#8c8c8c", "gamma_cluster": "#1f77b4", "cluster_roundrobin": "#d95f02", "random": "black"}
LAB = {"gamma_bulk": "γ_bulk", "gamma_cluster": "γ_cluster", "cluster_roundrobin": "γ_cluster, round-robin over clusters",
       "random": "random (20 draws)"}
SP = ["300K_1.0gcc", "2000K_1.0gcc", "5000K_2.0gcc"]
SPL = {"300K_1.0gcc": "300 K\n1.0 g/cc", "2000K_1.0gcc": "2000 K\n1.0 g/cc", "5000K_2.0gcc": "5000 K\n2.0 g/cc"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6})


def boot(d, fn, n=200, seed=0):
    """frame-bootstrap 2.5/97.5 percentiles of fn(sub-dataframe)."""
    rng = np.random.default_rng(seed); fr = d.frame.unique(); g = dict(tuple(d.groupby("frame")))
    v = [fn(pd.concat([g[f] for f in rng.choice(fr, fr.size)])) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def fig_a(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(SP)); w = 0.36
    # (a) atom-level Spearman(gamma, true error)
    for i, s in enumerate(["gamma_bulk", "gamma_cluster"]):
        vals, lo, hi = [], [], []
        for sp in SP:
            d = df[df.statepoint == sp]; f = lambda t: spearmanr(t[s], t.err)[0]
            v = f(d); ci = boot(d, f); vals.append(v); lo.append(v - ci[0]); hi.append(ci[1] - v)
        ax[0].bar(x + (i - .5) * w, vals, w, color=C[s], label=LAB[s], yerr=[lo, hi], capsize=3, edgecolor="white", linewidth=1)
    ax[0].set_xticks(x, [SPL[s] for s in SP]); ax[0].set_ylabel("Spearman(γ, |ΔF| vs DFT)")
    ax[0].set_title("(a) atom-level ranking of true force error"); ax[0].legend(frameon=False, loc="upper right")
    # (b) frame-level: frame-max gamma vs frame force RMSE (the unit sent to DFT)
    fr = df.groupby(["statepoint", "frame"]).agg(rmse=("err", lambda e: np.sqrt(np.mean(e ** 2))),
                                                 gamma_bulk=("gamma_bulk", "max"), gamma_cluster=("gamma_cluster", "max")).reset_index()
    rng = np.random.default_rng(1)
    for i, s in enumerate(["gamma_bulk", "gamma_cluster"]):
        vals, lo, hi = [], [], []
        for sp in SP:
            d = fr[fr.statepoint == sp]; v = spearmanr(d[s], d.rmse)[0]
            bs = [spearmanr(*d.iloc[rng.integers(0, len(d), len(d))][[s, "rmse"]].to_numpy().T)[0] for _ in range(500)]
            vals.append(v); lo.append(v - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - v)
        ax[1].bar(x + (i - .5) * w, vals, w, color=C[s], yerr=[lo, hi], capsize=3, edgecolor="white", linewidth=1)
    ax[1].set_xticks(x, [SPL[s] for s in SP]); ax[1].set_ylabel("Spearman(frame-max γ, frame RMSE)")
    ax[1].set_title("(b) frame-level ranking (300 frames / regime)")
    # (c) error of flagged vs unflagged atoms (gamma>1 in >=80% of MaxVol trials)
    groups = [("unflagged", (df.pc_flag == 0) & (df.pb_flag == 0), "#d9d9d9"),
              ("γ_bulk > 1", df.pb_flag >= 0.8, C["gamma_bulk"]),
              ("γ_cluster > 1 only", (df.pc_flag >= 0.8) & (df.pb_flag == 0), C["gamma_cluster"])]
    w3 = 0.26
    for j, (name, m, col) in enumerate(groups):
        for i, sp in enumerate(SP):
            e = df.err[m & (df.statepoint == sp)]
            if len(e) == 0: continue
            pos = i + (j - 1) * w3
            bp = ax[2].boxplot(e, positions=[pos], widths=w3 * 0.85, patch_artist=True, showfliers=False,
                               medianprops=dict(color="black", linewidth=1.5), whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
            bp["boxes"][0].set(facecolor=col, edgecolor="white")
            ax[2].text(pos, 0.13, f"n={len(e)}", ha="center", va="bottom", fontsize=8, rotation=90, color="#333")
        ax[2].bar(0, 0, color=col, label=name)
    ax[2].set_yscale("log"); ax[2].set_xticks(range(len(SP)), [SPL[s] for s in SP]); ax[2].set_xlim(-.6, 2.6)
    ax[2].set_ylim(0.1, 40); ax[2].set_ylabel("|ΔF| vs DFT per atom (eV/Å)")
    ax[2].set_title("(c) error of flagged atoms"); ax[2].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Does γ find the MLIP's own MD errors?  DFT-only ChIMES-N model, Langevin MD, 900 DFT-labelled frames "
                 "(57,600 atoms); error bars = 95% frame bootstrap", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6A_md_flagging.png", dpi=200); plt.close(fig)


def fig_b(al):
    cols = [f"rmse_md_{s}" for s in SP] + ["rmse_dft_holdout"]
    titles = [f"(a) held-out MD, {SPL[s].replace(chr(10), ', ')}" for s in SP] + ["(d) 30 held-out DFT frames (all regimes)"]
    base = al[al.strategy == "base"].iloc[0]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    for a, c, t in zip(ax, cols, titles):
        for s in ["random", "gamma_bulk", "gamma_cluster", "cluster_roundrobin"]:
            g = al[al.strategy == s].groupby("k")[c].agg(["mean", "std"]).reset_index()
            k = np.r_[0, g.k]; m = np.r_[base[c], g["mean"]]; sd = np.r_[0, g["std"].fillna(0)]
            ls = "--" if s == "random" else "-"
            a.plot(k, m, ls, color=C[s], lw=2, marker="o" if s != "random" else None, ms=4, label=LAB[s])
            a.fill_between(k, m - sd, m + sd, color=C[s], alpha=0.12 if s == "random" else 0.18, lw=0)
        a.set_xlabel("MD frames added (DFT labels)"); a.set_title(t.replace("(a)", "(" + "abc"[cols.index(c)] + ")") if c != cols[-1] else t, fontsize=11)
        a.set_xticks([0, 5, 10, 20, 40])
    ax[0].set_ylabel("force RMSE vs DFT (eV/Å)"); ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("One active-learning step on the model's own MD: pool = 600 frames (seeds 1-2), test = seed 3; "
                 "shaded = std over 5 MaxVol solutions (γ arms) or 20 random draws", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig6B_md_al_step.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_a(pd.read_csv(MD / "work" / "atom_errors_shared_t0_dft_train.csv"))
    fig_b(pd.read_csv(MD / "results" / "al_step.csv"))
    print("wrote", *sorted(OUT.glob("fig6*.png")))
