#!/usr/bin/env python3
"""Rank candidate frames for DFT labeling by gamma_cluster, from ChIMES (chimes_lsq) design matrices.

Inputs
  --train A.txt        chimes_lsq design matrix of the training set
  --candidates A.txt   chimes_lsq design matrix of the candidate frames (same fm_setup.in hyperparameters)
  --xyzf F [F ...]     the candidate .xyzf file(s), in the order they were given to chimes_lsq (used only to count
                       the atoms of every frame)

Both matrices must be force-only (FITENER, FITSTRS false in fm_setup.in): 3 rows (fx, fy, fz) per atom, atoms in
frame order. Each atom is one sample (rows_per_sample=3); a frame's score is the largest gamma_cluster of its atoms,
because whole frames are what is sent to DFT.

Output: a CSV of all candidate frames sorted by decreasing frame-max gamma_cluster, and optionally the fitted model.

usage:
  python3 chimes_rank_frames.py --train train/A.txt --candidates cand/A.txt --xyzf cand.xyzf \
      --out ranking.csv [--budget 20] [--n-clusters 6] [--model model.npz]
"""
import argparse
import csv
import os

import numpy as np

from clusterdopt import ClusterDOpt


def atoms_per_frame(path):
    """atom count of every frame of an .xyzf file (first line of each frame = number of atoms)."""
    lines = open(path).read().splitlines()
    i, counts = 0, []
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        counts.append(n)
        i += n + 2
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--xyzf", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=None, help="also print the top N frames")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--model", help="write the fitted model here (.npz)")
    a = ap.parse_args()

    A_train, A_cand = np.loadtxt(a.train), np.loadtxt(a.candidates)
    frames = [(f, k, n) for f in a.xyzf for k, n in enumerate(atoms_per_frame(f))]
    n_atoms = sum(n for _, _, n in frames)
    if A_cand.shape[0] != 3 * n_atoms:
        raise SystemExit(f"{a.candidates} has {A_cand.shape[0]} rows, but the .xyzf files hold {n_atoms} atoms "
                         f"({3 * n_atoms} force rows): is the matrix force-only and are the files in chimes_lsq order?")

    model = ClusterDOpt(n_clusters=a.n_clusters, rows_per_sample=3).fit(A_train)
    print(f"training atoms per cluster: {np.bincount(model.labels_).tolist()}")
    if a.model:
        model.save(a.model)
    gamma, cluster = model.gamma(A_cand, return_clusters=True)        # one value per candidate atom

    rows, start = [], 0
    for f, k, n in frames:
        g, c = gamma[start:start + n], cluster[start:start + n]
        rows.append(dict(file=f, frame=k, n_atoms=n, gamma_max=float(g.max()), n_atoms_gamma_gt1=int((g > 1).sum()),
                         cluster_of_max=int(c[g.argmax()])))
        start += n
    rows.sort(key=lambda r: -r["gamma_max"])
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank"] + list(rows[0]))
        w.writeheader()
        w.writerows({"rank": i + 1, **r} for i, r in enumerate(rows))
    n_flag = sum(r["gamma_max"] > 1 for r in rows)
    print(f"{len(rows)} candidate frames, {n_flag} with an atom above gamma_cluster = 1 -> {a.out}")
    for r in rows[:a.budget or 0]:
        print(f"  {os.path.basename(r['file'])} frame {r['frame']}: gamma_max {r['gamma_max']:.2f} (cluster {r['cluster_of_max']}), "
              f"{r['n_atoms_gamma_gt1']} atoms > 1")


if __name__ == "__main__":
    main()
