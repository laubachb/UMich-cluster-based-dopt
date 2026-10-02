import numpy as np
import pytest

from clusterdopt import ClusterDOpt, gamma_rows, maxvol
from clusterdopt.cli import main as cli


def reference_maxvol(A_ref, rng, tol=1.01, max_iters=300):
    """random_restart_maxvol of the manuscript (common/dopt.py), with rank = number of columns."""
    rank = A_ref.shape[1]
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
        absB = np.abs(A_ref @ inv_sub)
        absB[pivot_rows, :] = 0.0
        i, j = np.unravel_index(np.argmax(absB), absB.shape)
        if absB[i, j] <= tol:
            break
        pivot_rows[j] = int(i)
        inv_sub = np.linalg.inv(A_ref[pivot_rows])
    return pivot_rows, inv_sub


def two_regimes(n_big=600, n_small=120, d=6, seed=0):
    """a large, varied regime and a small, tight regime far from it (the situation of Fig. 1)."""
    rng = np.random.default_rng(seed)
    big = rng.normal(0, 3.0, (n_big, d)) + 10.0
    small = rng.normal(0, 0.3, (n_small, d)) - 10.0
    return np.vstack([big, small]), n_big


@pytest.mark.parametrize("seed", range(5))
def test_maxvol_matches_reference(seed):
    A = np.random.default_rng(100 + seed).standard_t(3, (2000, 12))
    piv_ref, inv_ref = reference_maxvol(A, np.random.default_rng(seed))
    piv, inv = maxvol(A, np.random.default_rng(seed))
    assert list(piv) == list(piv_ref)
    np.testing.assert_allclose(inv, inv_ref, rtol=1e-12, atol=0)


def test_maxvol_basis_properties():
    A = np.random.default_rng(1).normal(size=(500, 8))
    piv, inv = maxvol(A, np.random.default_rng(0))
    g = gamma_rows(A, inv)
    assert g.max() <= 1.01 + 1e-9                         # converged: every reference row inside the basis volume
    np.testing.assert_allclose(g[piv], 1.0)


def test_maxvol_too_few_rows():
    with pytest.raises(ValueError):
        maxvol(np.ones((3, 5)), np.random.default_rng(0))


def test_training_rows_are_interpolation():
    A, _ = two_regimes()
    m = ClusterDOpt(n_clusters=2, n_solutions=3).fit(A)
    assert m.n_clusters_ == 2
    assert m.gamma(A).max() <= 1.01 + 1e-9


def test_minority_regime_extrapolation_found_by_cluster_basis():
    A, n_big = two_regimes()
    m = ClusterDOpt(n_clusters=2, n_solutions=3).fit(A)
    small = A[n_big:]
    new = small.mean(0).copy()
    new[0] += 6 * small.std(0)[0]                         # off the small regime's span, but small next to the bulk spread
    bulk_inv = maxvol(A, np.random.default_rng(0))[1]
    assert gamma_rows(new[None], bulk_inv)[0] < 1
    g, lab = m.gamma(new[None], return_clusters=True)
    assert lab[0] == m.labels_[n_big] and g[0] > 1


def test_rows_per_sample_groups_rows():
    A, _ = two_regimes(d=4)                               # 720 rows -> 240 samples of 3 rows
    m = ClusterDOpt(n_clusters=2, rows_per_sample=3, n_solutions=2).fit(A)
    assert m.labels_.size == 240
    new = A[:30] * 1.5
    g, G, lab = m.gamma(new, return_solutions=True, return_clusters=True)
    assert g.shape == (10,) and G.shape == (2, 10)
    rows = np.abs(new @ m.inverses_[0, lab[0]]).max(1)[:3]
    assert G[0, 0] == pytest.approx(rows.max())
    with pytest.raises(ValueError):
        m.gamma(A[:31])


def test_small_cluster_merged():
    A, _ = two_regimes(n_small=4)                         # 4 rows cannot support a 6 x 6 basis
    with pytest.warns(UserWarning, match="merged"):
        m = ClusterDOpt(n_clusters=2, n_solutions=1, n_neighbors_affinity=5).fit(A)
    assert m.n_clusters_ == 1


def test_save_load_roundtrip(tmp_path):
    A, _ = two_regimes()
    m = ClusterDOpt(n_clusters=2, n_solutions=2).fit(A)
    m.save(tmp_path / "m.npz")
    m2 = ClusterDOpt.load(tmp_path / "m.npz")
    new = np.random.default_rng(5).normal(0, 8, (50, A.shape[1]))
    np.testing.assert_array_equal(m.gamma(new), m2.gamma(new))
    np.testing.assert_array_equal(m.predict_cluster(new), m2.predict_cluster(new))


def test_cli(tmp_path):
    A, _ = two_regimes()
    np.save(tmp_path / "train.npy", A)
    np.save(tmp_path / "new.npy", A[:20] * 2)
    cli(["fit", str(tmp_path / "train.npy"), str(tmp_path / "m.npz"), "--n-clusters", "2", "--n-solutions", "2"])
    cli(["gamma", str(tmp_path / "m.npz"), str(tmp_path / "new.npy"), str(tmp_path / "g.npy"),
         "--clusters", str(tmp_path / "c.npy")])
    g = np.load(tmp_path / "g.npy")
    expected = ClusterDOpt.load(tmp_path / "m.npz").gamma(A[:20] * 2)
    np.testing.assert_array_equal(g, expected)
    assert np.load(tmp_path / "c.npy").shape == (20,)


def test_rank_deficient_training_matrix():
    A, _ = two_regimes()
    A = np.hstack([A, np.zeros((A.shape[0], 1))])
    with pytest.raises(ValueError, match="linearly dependent"):
        ClusterDOpt(n_clusters=2, n_solutions=1).fit(A)
