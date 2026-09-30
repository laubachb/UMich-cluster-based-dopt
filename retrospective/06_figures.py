#!/usr/bin/env python3
"""Step 6: Figs. 1-5 of the manuscript -> figures/.

fig1: UMAP of DFT atoms colored by spectral cluster and by state point (work/umap_dft.npy, work/labels_dft.npy)
fig2-5: from results/{outliers,aggregate,by_cluster,recall}.csv written by 04_gamma_ensemble.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import csv, re, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES as PF, RETRO_RESULTS as SRC, RETRO_WORK as WORK
PF.mkdir(exist_ok=True)

# ============================== fig1: UMAP of DFT atoms ==============================
_man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
_kind = np.array([r["kind"] for r in _man]); _src = np.array([r["source_file"] for r in _man])
dft_atom_ids = np.where(_kind == "dft")[0]
sp_dft = np.array([re.match(r"^DFT\.(.+)\.xyzf$", s).group(1) for s in _src[dft_atom_ids]])
dft_labels = np.load(WORK / "labels_dft.npy")
emb = np.load(WORK / "umap_dft.npy")
# cluster x state-point atom counts (manuscript cluster-composition table)
_comp = pd.crosstab(pd.Series(dft_labels, name="cluster"), pd.Series(sp_dft, name="statepoint"))
_comp.insert(0, "n_atoms", _comp.sum(1)); _comp.insert(1, "frac", _comp.n_atoms / _comp.n_atoms.sum())
_comp.to_csv(SRC / "cluster_composition.csv")
clusters = sorted(np.unique(dft_labels).tolist())
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

ax = axes[0]
cmap = plt.colormaps["tab10"].resampled(len(clusters))
for i, cid in enumerate(clusters):
    m = dft_labels == cid
    ax.scatter(emb[m, 0], emb[m, 1], s=3, c=[cmap(i)], alpha=0.5, linewidths=0,
               rasterized=True, label=f"c{cid} ({m.sum()})")
ax.set_title("Spectral labels (raw descriptors, no EC)")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(markerscale=3, fontsize=8, frameon=False)

ax = axes[1]
dft_sps = sorted(set(sp_dft))
sp_cmap = plt.colormaps["viridis"].resampled(len(dft_sps))
for i, s in enumerate(dft_sps):
    m = sp_dft == s
    ax.scatter(emb[m, 0], emb[m, 1], s=6, c=[sp_cmap(i)], alpha=0.7, linewidths=0,
               rasterized=True, label=s)
ax.set_title("DFT statepoints")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.legend(markerscale=2, fontsize=7, frameon=False)

fig.suptitle(f"DFT descriptors → SpectralClustering, no EC (k={len(clusters)}, nn=20); n={dft_atom_ids.size}", fontsize=11)
fig.savefig(PF / "fig1_umap_cluster_structure.png", dpi=200)
plt.close(fig)
print(f"Wrote {PF / 'fig1_umap_cluster_structure.png'}")

AGG = pd.read_csv(SRC / "aggregate.csv")
BYC = pd.read_csv(SRC / "by_cluster.csv")
OUT = pd.read_csv(SRC / "outliers.csv")
REC = pd.read_csv(SRC / "recall.csv")

HIDDEN = {0, 3, 4}

# ============================== fig2: outlier counts (bar, w/ error bars) ==============================
byc = OUT[OUT.cluster != "ALL"].copy()
byc["cluster"] = byc["cluster"].astype(int)
order = sorted(byc.cluster.unique(), key=lambda c: (c not in HIDDEN, c))
g = byc.groupby("cluster").agg(
    n=("n", "first"),
    only_m=("cluster_only", "mean"), only_s=("cluster_only", "std"),
    bulk_m=("bulk_outliers", "mean"), bulk_s=("bulk_outliers", "std"),
).reindex(order).reset_index()

fig, ax = plt.subplots(figsize=(7.5, 5.1), constrained_layout=True)
x = np.arange(len(order))
width = 0.38
colors = ["#2171b5" if c in HIDDEN else "#969696" for c in order]
ax.bar(x - width / 2, g.only_m, width, yerr=g.only_s, capsize=3, color=colors, label="cluster-flagged, bulk misses (cluster_only)")
ax.bar(x + width / 2, g.bulk_m, width, yerr=g.bulk_s, capsize=3, color=colors, alpha=0.45, hatch="//", label="bulk-flagged (bulk_outliers)")
ax.set_xticks(x)
ax.set_xticklabels([f"c{c}\n(n={n})" for c, n in zip(g.cluster, g.n)], fontsize=8.5)
ax.set_ylabel("# ChIMES atoms flagged as outliers (γ > 1)")
ax.set_title(
    "Bulk γ misses internal structure in the hidden clusters\n"
    "no-EC clustering; bars = ensemble mean±std, 10 MaxVol solutions\n"
    "(blue = hidden, gray = populous)",
    fontsize=10,
)
ax.legend(fontsize=8, frameon=False)
fig.savefig(PF / "fig2_outlier_cluster_vs_bulk.png", dpi=200)
plt.close(fig)
print(f"Wrote {PF / 'fig2_outlier_cluster_vs_bulk.png'}")

# ============================== fig3: aggregate bulk vs cluster+ID, 3 protocols ==============================
protocols = ["groupkfold", "temporal_alc2_to_alc3", "temporal_alc2_to_alc4"]
protocol_labels = ["GroupKFold\n(in-dist.)", "temporal\nalc2→alc3", "temporal\nalc2→alc4"]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), constrained_layout=True)
x = np.arange(len(protocols))
width = 0.32
for metric_key, ax, title in (
    ("spearman", axes[0], "Spearman(pred, true MAE)"),
    ("auroc_q90", axes[1], "AUROC @ q90"),
):
    bulk_m = [AGG[AGG.protocol == p][f"bulk_{metric_key}"].mean() for p in protocols]
    bulk_s = [AGG[AGG.protocol == p][f"bulk_{metric_key}"].std() for p in protocols]
    cid_m = [AGG[AGG.protocol == p][f"cid_{metric_key}"].mean() for p in protocols]
    cid_s = [AGG[AGG.protocol == p][f"cid_{metric_key}"].std() for p in protocols]
    ax.bar(x - width / 2, bulk_m, width, yerr=bulk_s, capsize=3, color="#969696", label="γ_bulk")
    ax.bar(x + width / 2, cid_m, width, yerr=cid_s, capsize=3, color="#2171b5", label="γ_cluster + cluster-ID")
    ax.set_xticks(x)
    ax.set_xticklabels(protocol_labels, fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("score")
axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
fig.suptitle(
    "Uncertainty prediction: bulk-only γ vs. cluster-local γ+ID (no EC)\n"
    "bars = ensemble mean±std over 10 replicated MaxVol solutions (HGB)",
    fontsize=10.5,
)
fig.savefig(PF / "fig3_gamma_bulk_vs_cluster_id_uncertainty.png", dpi=200)
plt.close(fig)
print(f"Wrote {PF / 'fig3_gamma_bulk_vs_cluster_id_uncertainty.png'}")

# ============================== fig4: per-cluster margin ==============================
cluster_order = sorted(BYC.cluster.unique(), key=lambda c: (c not in HIDDEN, c))
xpos = np.arange(len(cluster_order))
width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
for metric_key, ax, title in (
    ("spearman", axes[0], "Δ Spearman (cluster+ID − bulk)"),
    ("auroc_q90", axes[1], "Δ AUROC@q90 (cluster+ID − bulk)"),
):
    means, stds, labels, colors = [], [], [], []
    for c in cluster_order:
        sub = BYC[BYC.cluster == c]
        delta = sub[f"cid_{metric_key}"] - sub[f"bulk_{metric_key}"]
        means.append(delta.mean())
        stds.append(delta.std())
        n = int(sub.n.iloc[0])
        labels.append(f"c{c} (n={n})")
        colors.append("#2171b5" if c in HIDDEN else "#969696")
    ax.bar(xpos, means, width, yerr=stds, capsize=3, color=colors)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.set_title(title, fontsize=10)
from matplotlib.patches import Patch
axes[1].legend(handles=[Patch(color="#2171b5", label="hidden"), Patch(color="#969696", label="populous")],
               fontsize=8, frameon=False, loc="upper right")
fig.suptitle(
    "Where does cluster-local γ+ID beat bulk γ, by cluster? (no EC)\n"
    "HGB, GroupKFold OOF; bars = ensemble mean±std, 10 replicated MaxVol solutions",
    fontsize=10.5,
)
fig.savefig(PF / "fig4_uncertainty_margin_by_cluster.png", dpi=200)
plt.close(fig)
print(f"Wrote {PF / 'fig4_uncertainty_margin_by_cluster.png'}")

# ============================== fig5: budget efficiency (trained model score) ==============================
budgets_grid = np.concatenate([np.linspace(0.01, 0.2, 20), np.linspace(0.22, 1.0, 20)])
piv_bulk = REC.groupby("budget")["recall_bulk"].agg(["mean", "std"]).reindex(budgets_grid, method="nearest")
piv_cid = REC.groupby("budget")["recall_cid"].agg(["mean", "std"]).reindex(budgets_grid, method="nearest")

fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
ax.plot(budgets_grid, piv_cid["mean"].values, marker="o", ms=2.5, color="#2171b5", lw=1.8, label="γ_cluster + cluster-ID (trained)")
ax.fill_between(budgets_grid, piv_cid["mean"].values - piv_cid["std"].values, piv_cid["mean"].values + piv_cid["std"].values,
                 color="#2171b5", alpha=0.18, lw=0)
ax.plot(budgets_grid, piv_bulk["mean"].values, marker="o", ms=2.5, color="#969696", lw=1.8, label="γ_bulk (trained)")
ax.fill_between(budgets_grid, piv_bulk["mean"].values - piv_bulk["std"].values, piv_bulk["mean"].values + piv_bulk["std"].values,
                 color="#969696", alpha=0.18, lw=0)
ax.plot([0, 1], [0, 1], ls="--", color="k", lw=0.8, label="random triage")
ax.set_xlabel("DFT-labeling budget (top-K% of hidden-regime candidates)")
ax.set_ylabel("recall of true high-error (top-10%) structures")
r10 = float(np.interp(0.10, budgets_grid, piv_cid["mean"].values))
b10 = float(np.interp(0.10, budgets_grid, piv_bulk["mean"].values))
ax.set_title(
    f"Active-learning payoff in the rare regime (no EC)\n"
    f"at 10% budget: cluster+ID recall={r10:.2f}  vs  bulk={b10:.2f}\n"
    f"shaded = ensemble std, 10 replicated MaxVol solutions",
    fontsize=10,
)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.savefig(PF / "fig5_al_budget_efficiency.png", dpi=200)
plt.close(fig)
print(f"Wrote {PF / 'fig5_al_budget_efficiency.png'}")
