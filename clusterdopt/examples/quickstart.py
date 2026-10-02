#!/usr/bin/env python3
"""Quickstart on synthetic data: a large, varied regime and a small, tight one (no external data needed).

A new point that lies off the small regime is inside the volume of the bulk basis (gamma_bulk < 1) but outside the
basis of its own cluster (gamma_cluster > 1). This is the situation of Fig. 1 of the manuscript.

usage: python3 quickstart.py
"""
import numpy as np

from clusterdopt import ClusterDOpt, gamma_rows, maxvol

rng = np.random.default_rng(0)
n_params = 6
big = rng.normal(10.0, 3.0, (600, n_params))           # majority regime: 600 rows, broad
small = rng.normal(-10.0, 0.3, (120, n_params))        # minority regime: 120 rows, narrow
A_train = np.vstack([big, small])

# 1. fit: cluster the training rows and select one MaxVol basis per cluster
model = ClusterDOpt(n_clusters=2, n_solutions=5).fit(A_train)
print("training samples per cluster:", np.bincount(model.labels_))

# 2. grade new rows
new = np.vstack([
    big[0] * 1.02,                                     # close to the majority regime
    small[0] * 1.02,                                   # close to the minority regime
    small.mean(0) + np.r_[2.0, np.zeros(n_params - 1)],  # off the minority regime in one direction
])
gamma, clusters = model.gamma(new, return_clusters=True)

# for comparison: the conventional grade against one basis from all training rows
_, inv_bulk = maxvol(A_train, np.random.default_rng(0))
gamma_bulk = gamma_rows(new, inv_bulk)

for name, gc, gb, c in zip(["near majority", "near minority", "off minority"], gamma, gamma_bulk, clusters):
    print(f"{name:14s} cluster {c}  gamma_cluster {gc:7.2f}  gamma_bulk {gb:5.2f}")

# 3. save and reload
model.save("quickstart_model.npz")
assert np.allclose(ClusterDOpt.load("quickstart_model.npz").gamma(new), gamma)
print("saved to quickstart_model.npz")
