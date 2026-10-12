"""Shared code of the open-data extension (see CRITERIA.md).

MaxVol is the random-restart rectangular MaxVol of common/dopt.py (same initial draw, tolerance 1.01, at most 300
swaps, same pivot rule), written with the rank-one update of B = A A_hat^-1 after each swap instead of a fresh inverse,
so it scales to the 228 columns of the two-element ChIMES basis. check_maxvol() compares the two on the nitrogen data.
Each basis is selected in the row space of its own training rows (rank-reduced basis, CRITERIA.md); when the rows have
full column rank this is the manuscript's grade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
WORK, RES = HERE / "work", HERE / "results"
sys.path.insert(0, str(HERE.parent))
SEED0, NSOL, TOL, MAX_SWAPS, RANK_TOL = 300, 5, 1.01, 300, 1e-8
K_CLUSTERS, N_GRAPH, KNN_GRAPH, KNN_VOTE = 6, 15000, 20, 15


# ------------------------------------------------------------------ data
def load(name: str, cap: int, seed: int = 0):
    """Atoms of data set `name`, at most `cap` per frame (seeded). Returns a dict of per-atom arrays and the frame table."""
    w = WORK / name
    A = np.load(w / "A.npy", mmap_mode="r")
    F = np.load(w / "forces.npy", mmap_mode="r")
    sp = np.load(w / "species.npy", mmap_mode="r")
    tab = pd.read_csv(w / "frames.csv")
    if "al" not in tab:
        tab["al"] = False
    rng = np.random.default_rng(seed)
    off = np.r_[0, np.cumsum(tab.n_atoms.to_numpy())]
    pick = [off[f] + np.sort(rng.choice(n, min(n, cap), replace=False)) for f, n in enumerate(tab.n_atoms)]
    atoms = np.concatenate(pick)
    cnt = np.array([len(p) for p in pick])
    rows = (atoms[:, None] * 3 + np.arange(3)).ravel()
    X = np.asarray(A[rows], dtype=np.float64)
    keep_cols = np.abs(X).max(0) > 0
    return dict(X=X[:, keep_cols], F=np.asarray(F[atoms]), element=np.asarray(sp[atoms]).astype(str),
                frame=np.repeat(tab.frame.to_numpy(), cnt), regime=np.repeat(tab.regime.to_numpy(), cnt),
                al=np.repeat(tab.al.to_numpy(), cnt), n_cols=int(keep_cols.sum()), atoms=atoms), tab


def rows_of(idx):
    idx = np.asarray(idx, dtype=np.int64)
    return (idx[:, None] * 3 + np.arange(3)).ravel()


# ------------------------------------------------------------------ MaxVol and grades
def maxvol(A, rng):
    """rectangular MaxVol (common/dopt.py algorithm with rank-one updates). A: n x r with rank r. Returns pivots, inverse."""
    n, r = A.shape
    P = rng.choice(n, size=r, replace=False)
    tries = 0
    while np.linalg.matrix_rank(A[P]) < r and tries < 30:
        P = rng.choice(n, size=r, replace=False); tries += 1
    if np.linalg.matrix_rank(A[P]) < r:                    # fallback: pivoted QR picks r independent rows
        P = qr(A.T, pivoting=True, mode="r")[1][:r]
    B = A @ np.linalg.inv(A[P])
    isP = np.zeros(n, bool); isP[P] = True
    for _ in range(MAX_SWAPS):
        absB = np.abs(B); absB[isP] = 0.0
        i, j = np.unravel_index(np.argmax(absB), absB.shape)
        if absB[i, j] <= TOL:
            break
        bj = B[:, j] / B[i, j]; bi = B[i].copy(); bi[j] -= 1.0
        B -= np.outer(bj, bi)
        isP[P[j]] = False; isP[i] = True; P[j] = i
    return P, np.linalg.inv(A[P])


class Basis:
    """MaxVol basis of a set of training rows, in the row space of those rows."""

    def __init__(self, rows, rng):
        s, Vt = np.linalg.svd(rows, full_matrices=False)[1:]
        r = int((s > RANK_TOL * s[0]).sum())
        self.V = Vt[:r].T
        self.rank = r
        self.pivots, self.inv = maxvol(rows @ self.V, rng)

    def grade(self, rows):
        """per-atom grade (max over the atom's 3 rows) and per-atom out-of-span fraction (max over rows)."""
        p = rows @ self.V
        g = np.abs(p @ self.inv).max(1).reshape(-1, 3).max(1)
        res = np.linalg.norm(rows - p @ self.V.T, axis=1) / np.maximum(np.linalg.norm(rows, axis=1), 1e-300)
        return g, res.reshape(-1, 3).max(1)


class Leverage:
    """ridge predictive variance (leverage) of a set of training rows (CRITERIA.md, test R2):
    h(a) = sum_k (a.v_k)^2 / (s_k^2 + lam) over singular triplets with s_k > RANK_TOL * s_1, divided by the largest h of
    the training rows themselves (h <= 1 inside the set, as for gamma). Deterministic."""

    def __init__(self, rows, lam=1e-8):
        s, Vt = np.linalg.svd(rows, full_matrices=False)[1:]
        keep = s > RANK_TOL * s[0]
        self.W = Vt[keep].T / np.sqrt(s[keep] ** 2 + lam)
        self.scale = 1.0
        self.scale = self._rows(rows).max()

    def _rows(self, rows):
        return ((rows @ self.W) ** 2).sum(1) / self.scale

    def grade(self, rows):
        """per-atom score: max over the atom's 3 rows."""
        return self._rows(rows).reshape(-1, 3).max(1)


class BayesGrade:
    """Gaussian linear force model of test H (CRITERIA.md): rows X (n x p), targets y, row groups g, regime noise
    sigma[k]. tau=None: pooled coefficients (IG_het; IG_hom if all sigma equal). tau > 0: partial pooling
    c_k = c_0 + delta_k, delta_k ~ N(0, tau^2 I). Prior precision of c_0 = lam / s0^2 (s0 = pooled residual RMS)."""

    def __init__(self, X, y, g, sigma, s0, tau=None, lam=1e-8):
        self.groups = sorted(int(k) for k in np.unique(g)); self.sigma = sigma; p = X.shape[1]
        G, h = {}, {}
        for k in self.groups:
            m = g == k; w = 1.0 / sigma[k] ** 2
            G[k] = w * (X[m].T @ X[m]); h[k] = w * (X[m].T @ y[m])
        self.yWy = sum(float((y[g == k] ** 2).sum()) / sigma[k] ** 2 for k in self.groups)
        self.logdet_noise = sum(int((g == k).sum()) * np.log(sigma[k] ** 2) for k in self.groups); self.n = len(y)
        p0 = lam / s0 ** 2
        if tau is None:
            P = sum(G.values()) + p0 * np.eye(p); b = sum(h.values()); self.logdet_prior = p * np.log(p0)
        else:
            K = len(self.groups); P = np.zeros((p * (K + 1), p * (K + 1))); b = np.zeros(p * (K + 1))
            P[:p, :p] = sum(G.values()) + p0 * np.eye(p); b[:p] = sum(h.values())
            for i, k in enumerate(self.groups, 1):
                s = slice(p * i, p * (i + 1))
                P[:p, s] = P[s, :p] = G[k]; P[s, s] = G[k] + np.eye(p) / tau ** 2; b[s] = h[k]
            self.logdet_prior = p * np.log(p0) + K * p * np.log(1 / tau ** 2)
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(P, lower=True)
        S = cho_solve(cf, np.eye(len(P)))
        self.logdet_P = 2 * np.log(np.diag(cf[0])).sum()
        self.mean = S @ b; self.bSb = float(b @ self.mean)
        if tau is None:
            self.C = {k: S for k in self.groups}; self.coef = {k: self.mean for k in self.groups}
        else:
            self.C, self.coef = {}, {}
            for i, k in enumerate(self.groups, 1):
                s = slice(p * i, p * (i + 1))
                self.C[k] = S[:p, :p] + S[:p, s] + S[s, :p] + S[s, s]
                self.coef[k] = self.mean[:p] + self.mean[s]

    def log_evidence(self):
        return -0.5 * (self.yWy - self.bSb + self.logdet_P - self.logdet_prior + self.logdet_noise + self.n * np.log(2 * np.pi))

    def info_gain(self, rows, atom_group):
        """per-atom information gain of labelling its 3 force components: 0.5 log det(I3 + A3 C_k A3^T / sigma_k^2)."""
        n = len(atom_group); out = np.full(n, np.nan); R = rows.reshape(n, 3, -1)
        for k in np.unique(atom_group):
            m = atom_group == k
            M = np.einsum("aip,pq,ajq->aij", R[m], self.C[int(k)], R[m]) / self.sigma[int(k)] ** 2
            out[m] = 0.5 * np.linalg.slogdet(np.eye(3) + M)[1]
        return out


def group_sigma(X, y, g, coef):
    """RMS residual of every row group under coefficients coef (the regime noise sigma_k of test H)."""
    r = X @ coef - y
    return {int(k): float(np.sqrt(np.mean(r[g == k] ** 2))) for k in np.unique(g)}, float(np.sqrt(np.mean(r ** 2)))


def fit_hier(X, y, g, sigma, s0, ts=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """IG_hier: tau = t * RMS(pooled coefficients), t by marginal likelihood. Returns (model, t, {t: log evidence})."""
    scale = float(np.sqrt(np.mean(BayesGrade(X, y, g, sigma, s0).mean ** 2)))
    ev, best = {}, None
    for t in ts:
        m = BayesGrade(X, y, g, sigma, s0, tau=t * scale); ev[t] = m.log_evidence()
        if best is None or ev[t] > ev[best[0]]:
            best = (t, m)
    return best[1], best[0], ev


def spectral_members(Xatom_train, Xatom_all, k=K_CLUSTERS, rng_seed=42):
    """as spectral_labels, also returning the membership probabilities of the 15-NN distance-weighted vote
    (n_all, k): the vote's label is the argmax, so predict() == argmax(proba) (test N, addendum N)."""
    sc = StandardScaler().fit(Xatom_train)
    rng = np.random.default_rng(rng_seed)
    sub = rng.choice(len(Xatom_train), min(N_GRAPH, len(Xatom_train)), replace=False)
    Z = sc.transform(Xatom_train[sub])
    lab = SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=KNN_GRAPH, assign_labels="kmeans",
                             random_state=rng_seed).fit_predict(Z)
    knn = KNeighborsClassifier(KNN_VOTE, weights="distance").fit(Z, lab)
    proba = np.zeros((len(Xatom_all), k)); proba[:, knn.classes_] = knn.predict_proba(sc.transform(Xatom_all))
    return proba.argmax(1), proba


def soft_grade(P, proba, geometric=True):
    """P (n_groups, n_atoms) grades against every group (inf = not inside the group's span); proba (n_atoms, n_groups).
    Geometric: exp(sum p log g) over finite g with weights renormalised; arithmetic: sum p g."""
    W = proba.T.copy(); fin = np.isfinite(P); W[~fin] = 0.0
    s = W.sum(0); s[s == 0] = np.nan; W = W / s
    G = np.where(fin, P, 1.0)
    return np.exp((W * np.log(np.maximum(G, 1e-300))).sum(0)) if geometric else (W * G).sum(0)


def mondrian_q(score_cal, err_cal, group_cal, alpha=0.1, min_n=20):
    """per-group conformal quantile of err/score (finite-sample level ceil((n+1)(1-alpha))/n); groups with fewer than
    min_n calibration atoms use the pooled quantile. Returns (dict group -> q, pooled q)."""
    r = err_cal / np.maximum(score_cal, 1e-300)
    def q(x):
        n = len(x); lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(x, lvl))
    qp = q(r); out = {}
    for g in np.unique(group_cal):
        m = group_cal == g
        out[int(g)] = q(r[m]) if m.sum() >= min_n else qp
    return out, qp


def leverage_all(X, train, cand, part):
    """relative leverage of the candidate atoms, each against the training atoms of its own group of `part`."""
    tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    H = np.full(len(ca_idx), np.nan)
    for g in np.unique(part[ca_idx]):
        gt, gc = tr_idx[part[tr_idx] == g], np.where(part[ca_idx] == g)[0]
        if gt.size:
            H[gc] = Leverage(X[rows_of(gt)]).grade(X[rows_of(ca_idx[gc])])
    return H


# ------------------------------------------------------------------ partitions
def spectral_labels(Xatom_train, Xatom_all, rng_seed=42):
    """manuscript clustering (standardized concatenated rows, spectral k = 6, 20-NN affinity) on at most N_GRAPH training
    atoms, then 15-NN distance-weighted vote for every atom."""
    sc = StandardScaler().fit(Xatom_train)
    rng = np.random.default_rng(rng_seed)
    sub = rng.choice(len(Xatom_train), min(N_GRAPH, len(Xatom_train)), replace=False)
    Z = sc.transform(Xatom_train[sub])
    lab = SpectralClustering(K_CLUSTERS, affinity="nearest_neighbors", n_neighbors=KNN_GRAPH, assign_labels="kmeans",
                             random_state=rng_seed).fit_predict(Z)
    return KNeighborsClassifier(KNN_VOTE, weights="distance").fit(Z, lab).predict(sc.transform(Xatom_all))


def partitions(D, train, multi_element, cluster=True):
    """dict name -> integer group label for every atom (labels from training data only)."""
    el = D["element"]; out = {"bulk": np.zeros(len(el), int)}
    el_code = pd.Categorical(el).codes
    if multi_element:
        out["element"] = el_code.astype(int)
    out["regime"] = pd.Categorical(pd.Series(el).str.cat(pd.Series(D["regime"]), sep="|")).codes.astype(int)
    if not cluster:
        return out
    Xatom = D["X"].reshape(len(el), -1)
    cl = np.full(len(el), -1)
    for e in np.unique(el_code):
        me = el_code == e
        lab = spectral_labels(Xatom[train & me], Xatom[me])
        cl[me] = e * K_CLUSTERS + lab
    out["cluster"] = cl
    return out


def grade_all(D, train, cand, part, nsol=NSOL):
    """grades of the candidate atoms for one partition: arrays (nsol, n_cand) of grade and out-of-span fraction, plus
    the rank of each group basis and the bulk-basis pivots (for the representation)."""
    X = D["X"]; tr_idx, ca_idx = np.where(train)[0], np.where(cand)[0]
    G = np.full((nsol, len(ca_idx)), np.nan); R = np.zeros_like(G); ranks = {}; pivots = []
    for t in range(nsol):
        rng = np.random.default_rng(SEED0 + t)
        for g in np.unique(part[tr_idx]):
            gt, gc = tr_idx[part[tr_idx] == g], np.where(part[ca_idx] == g)[0]
            if gc.size == 0 and len(np.unique(part[tr_idx])) > 1:
                continue
            b = Basis(X[rows_of(gt)], rng)
            ranks[int(g)] = b.rank
            if len(np.unique(part[tr_idx])) == 1:
                pivots.append(gt[b.pivots // 3])
            if gc.size:
                G[t, gc], R[t, gc] = b.grade(X[rows_of(ca_idx[gc])])
    return G, R, ranks, pivots


# ------------------------------------------------------------------ error model and metrics
def error_model(D, train, cand, lam=1e-8):
    X, F = D["X"], D["F"]
    Xt = X[rows_of(np.where(train)[0])]
    c = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ F[train].ravel())
    return np.abs((X[rows_of(np.where(cand)[0])] @ c).reshape(-1, 3) - F[cand]).mean(1)


def metrics(score, err):
    top = err > np.quantile(err, 0.9)
    k = max(1, int(round(0.1 * len(err))))
    return dict(spearman=spearmanr(score, err)[0], auroc=roc_auc_score(top, score),
                recall10=top[np.argsort(-score)[:k]].sum() / top.sum())


def representation(pivot_atoms, group_of_atom, train):
    """share of bulk-basis rows from each group / share of training rows (dict group -> mean over solutions)."""
    tr_groups = group_of_atom[train]
    share = pd.Series(tr_groups).value_counts(normalize=True)
    reps = [pd.Series(group_of_atom[p]).value_counts(normalize=True).reindex(share.index, fill_value=0) / share
            for p in pivot_atoms]
    return pd.concat(reps, axis=1).mean(1).to_dict()


def check_maxvol(n_sol=3):
    """the rank-one MaxVol gives the grades of common/dopt.py on the nitrogen bulk basis (retrospective study)."""
    import csv
    from common import dopt as E
    from common.paths import RETRO_WORK as RW
    A = np.load(RW / "A_atomic.npy")
    kind = np.array([r["kind"] for r in csv.DictReader(open(RW / "atom_manifest.csv"))])
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0][:5000]
    Ad, Ac = A[E.atom_rows_from_indices(dft)], A[E.atom_rows_from_indices(ch)]
    for t in range(n_sol):
        g_ref = E.gamma_for_rows(Ac, E.random_restart_maxvol(Ad, np.random.default_rng(SEED0 + t)))
        b = Basis(Ad, np.random.default_rng(SEED0 + t))
        g_new = b.grade(Ac)[0]
        print(f"solution {t}: rank {b.rank}, max |gamma_new - gamma_ref| = {np.abs(g_new - g_ref).max():.2e}, "
              f"Spearman {spearmanr(g_new, g_ref)[0]:.6f}")
