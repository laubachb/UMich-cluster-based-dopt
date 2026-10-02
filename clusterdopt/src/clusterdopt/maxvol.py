"""Rectangular MaxVol and the D-optimal extrapolation grade.

gamma(a) = max_j |(a A_hat^-1)_j| for a descriptor row a, where A_hat is the square (r x r, r = number of columns)
submatrix of the reference rows selected by MaxVol (Podryabinkin & Shapeev 2017).

`maxvol` follows the random-restart MaxVol of the manuscript (common/dopt.py: random initial pivots, swap the largest
|coefficient| above `tol`, at most `max_iters` swaps) and draws from the random generator in the same order, so the
same generator state gives the same pivots. It is faster because it updates the coefficient matrix A @ A_hat^-1 by a
rank-1 correction after each swap instead of recomputing it; the returned inverse is recomputed exactly from the
final pivots.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

REFRESH_EVERY = 25       # recompute the coefficient matrix exactly every this many swaps to bound round-off drift


def _initial_pivots(A, rng, max_tries=30):
    """random full-rank initial pivots, drawn exactly as in the manuscript; pivoted QR if 30 draws are singular."""
    n, r = A.shape
    piv = rng.choice(n, size=r, replace=False)
    tries = 0
    while np.linalg.matrix_rank(A[piv]) < r and tries < max_tries:
        piv = rng.choice(n, size=r, replace=False)
        tries += 1
    if np.linalg.matrix_rank(A[piv]) < r:
        piv = scipy.linalg.qr(A.T, mode="r", pivoting=True)[1][:r]
        if np.linalg.matrix_rank(A[piv]) < r:
            raise np.linalg.LinAlgError(f"reference rows have rank < {r}: no square basis exists")
    return np.array(piv, dtype=np.int64)


def maxvol(A, rng, tol=1.01, max_iters=300):
    """Select r = A.shape[1] rows of A with locally maximal |det|.

    Returns (pivots, inverse): row indices into A and the inverse of A[pivots].
    """
    A = np.asarray(A, dtype=np.float64)
    n, r = A.shape
    if n < r:
        raise ValueError(f"MaxVol needs at least {r} reference rows, got {n}")
    piv = _initial_pivots(A, rng)
    C = A @ np.linalg.inv(A[piv])
    for it in range(max_iters):
        absC = np.abs(C)
        absC[piv, :] = 0.0
        i, j = np.unravel_index(np.argmax(absC), absC.shape)
        if absC[i, j] <= tol:
            break
        row = C[i].copy()
        row[j] -= 1.0
        C -= np.outer(C[:, j], row / C[i, j])
        piv[j] = i
        if (it + 1) % REFRESH_EVERY == 0:
            C = A @ np.linalg.inv(A[piv])
    return piv, np.linalg.inv(A[piv])


def gamma_rows(A, inverse, chunk=200_000):
    """per-row grade max_j |(a A_hat^-1)_j|."""
    A = np.asarray(A, dtype=np.float64)
    out = np.empty(A.shape[0])
    for s in range(0, A.shape[0], chunk):
        out[s:s + chunk] = np.abs(A[s:s + chunk] @ inverse).max(axis=1)
    return out


def gamma_samples(A, inverse, rows_per_sample=1):
    """per-sample grade: the maximum of gamma_rows over each sample's consecutive rows."""
    return gamma_rows(A, inverse).reshape(-1, rows_per_sample).max(axis=1)
