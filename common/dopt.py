"""D-optimality (MaxVol) extrapolation grades and ranking metrics shared by both studies.

gamma(a) = max_j |(a A_hat^-1)_j| for a candidate descriptor row a, where A_hat is a square (rank x rank)
submatrix of the reference rows selected by rectangular MaxVol (Podryabinkin & Shapeev 2017). The per-atom grade
is the maximum over the atom's three force-component rows. MaxVol solutions are not unique, so callers draw several
solutions from random initial pivots (random_restart_maxvol) and average.

These functions are the ones that produced the manuscript numbers; do not change them without re-running both
studies. RANK = 42 is the number of ChIMES coefficients of the nitrogen basis (2B order 12 + 3B order 5).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

RANK = 42
SEED0 = 300          # MaxVol solution t uses np.random.default_rng(SEED0 + t)
THRESHOLD = 1.0      # conventional extrapolation threshold gamma > 1


def atom_rows_from_indices(idx):
    idx = np.asarray(idx, dtype=np.int64)
    return (idx[:, None] * 3 + np.arange(3)[None, :]).ravel()


def random_restart_maxvol(A_ref, rng, rank=RANK, tol=1.01, max_iters=300):
    n_rows = A_ref.shape[0]
    pivot_rows = list(rng.choice(n_rows, size=min(rank, n_rows), replace=False))
    A_hat = A_ref[pivot_rows]
    tries = 0
    while np.linalg.matrix_rank(A_hat) < len(pivot_rows) and tries < 30:
        pivot_rows = list(rng.choice(n_rows, size=min(rank, n_rows), replace=False))
        A_hat = A_ref[pivot_rows]
        tries += 1
    inv_sub = np.linalg.inv(A_hat)
    for _ in range(max_iters):
        B = A_ref @ inv_sub
        absB = np.abs(B)
        absB[pivot_rows, :] = 0.0
        i, j = np.unravel_index(np.argmax(absB), absB.shape)
        if absB[i, j] <= tol:
            break
        pivot_rows[j] = int(i)
        A_hat = A_ref[pivot_rows]
        inv_sub = np.linalg.inv(A_hat)
    return inv_sub


def gamma_for_rows(A_cand, inv_sub):
    per_row = np.max(np.abs(A_cand @ inv_sub), axis=1)
    return per_row.reshape(-1, 3).max(axis=1)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() > 2 else float("nan")


def auroc_high_error(score, y, q=0.90):
    thr = np.quantile(y, q)
    label = (y > thr).astype(np.int32)
    if label.min() == label.max():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    pos, neg = ranks[label == 1], ranks[label == 0]
    return float((pos.mean() - (len(pos) + 1) / 2) / len(neg))


def recall_curve(score, is_informative, budgets):
    order = np.argsort(-score)
    cum = np.cumsum(is_informative[order])
    n = len(score)
    n_info = is_informative.sum()
    return np.array([cum[max(1, int(round(b * n))) - 1] / n_info for b in budgets])
