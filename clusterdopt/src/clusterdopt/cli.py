"""Command line: fit a model on a training matrix and grade new rows, both as .npy files.

  clusterdopt fit train.npy model.npz [--n-clusters 6] [--rows-per-sample 1] [--n-solutions 10]
  clusterdopt gamma model.npz new.npy gamma.npy [--clusters clusters.npy]
"""
import argparse

import numpy as np

from .model import ClusterDOpt


def main(argv=None):
    p = argparse.ArgumentParser(prog="clusterdopt", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="cluster the training rows and select one MaxVol basis per cluster")
    f.add_argument("train", help="training matrix, .npy (rows x columns)")
    f.add_argument("model", help="output model, .npz")
    f.add_argument("--n-clusters", type=int, default=6)
    f.add_argument("--rows-per-sample", type=int, default=1, help="3 for per-atom force-component rows")
    f.add_argument("--n-solutions", type=int, default=10)
    f.add_argument("--n-jobs", type=int, default=None)
    g = sub.add_parser("gamma", help="gamma_cluster of new rows")
    g.add_argument("model", help="model written by 'clusterdopt fit'")
    g.add_argument("new", help="new rows, .npy (rows x columns)")
    g.add_argument("out", help="output gamma_cluster per sample, .npy")
    g.add_argument("--clusters", help="also write the assigned cluster of every sample to this .npy")
    a = p.parse_args(argv)

    if a.cmd == "fit":
        m = ClusterDOpt(n_clusters=a.n_clusters, rows_per_sample=a.rows_per_sample, n_solutions=a.n_solutions,
                        n_jobs=a.n_jobs).fit(np.load(a.train))
        m.save(a.model)
        print(f"{m.n_clusters_} clusters, samples per cluster {np.bincount(m.labels_).tolist()} -> {a.model}")
    else:
        m = ClusterDOpt.load(a.model)
        gam, lab = m.gamma(np.load(a.new), return_clusters=True)
        np.save(a.out, gam)
        if a.clusters:
            np.save(a.clusters, lab)
        print(f"{gam.size} samples, {(gam > 1).sum()} with gamma_cluster > 1 -> {a.out}")


if __name__ == "__main__":
    main()
