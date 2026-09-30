#!/usr/bin/env python3
"""Step 2: spectral clustering of DFT atoms and assignment of candidate atoms to clusters.

Per-atom feature = the atom's three descriptor rows concatenated (126 values), standardized on the DFT atoms.
DFT atoms: SpectralClustering(k=6, 20-NN affinity, k-means labels, random_state=42).
Candidate (active-learning) atoms: distance-weighted vote of their 15 nearest DFT atoms.
Also computes the 2-D UMAP embedding of the DFT atoms used only for Fig. 1.
Writes work/labels_dft.npy, work/labels_candidates.npy (candidate atoms in atom-manifest order), work/umap_dft.npy.
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RETRO_WORK as WORK


def main():
    kind = np.array([r["kind"] for r in csv.DictReader(open(WORK / "atom_manifest.csv"))])
    A = np.load(WORK / "A_atomic.npy")
    X = A.reshape(len(kind), -1)
    dft, ch = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    sc = StandardScaler().fit(X[dft]); Xd, Xc = sc.transform(X[dft]), sc.transform(X[ch])
    lab = SpectralClustering(n_clusters=6, affinity="nearest_neighbors", n_neighbors=20, assign_labels="kmeans",
                             random_state=42, n_jobs=1).fit_predict(Xd)
    lch = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(Xd, lab).predict(Xc)
    np.save(WORK / "labels_dft.npy", lab); np.save(WORK / "labels_candidates.npy", lch)
    print("DFT cluster sizes", np.bincount(lab), "candidate cluster sizes", np.bincount(lch), flush=True)
    import umap
    emb = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, n_jobs=1, low_memory=True,
                    random_state=42).fit_transform(StandardScaler().fit_transform(X[dft]))
    np.save(WORK / "umap_dft.npy", emb)


if __name__ == "__main__":
    main()
