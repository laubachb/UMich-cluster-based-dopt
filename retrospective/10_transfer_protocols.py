#!/usr/bin/env python3
"""Step 10: within-cluster ranking of force error by the raw grades under three train/test protocols (SI analog of Fig. 4).

Each protocol has a reference set (the training data: the error model is fit to it and gamma_bulk / gamma_cluster
use its atoms as the MaxVol reference) and a test set whose atoms are ranked.
  in_dist   reference = the 90 DFT-MD training frames of step 3; test = the 30 held-out DFT-MD frames
  alc2_alc3 reference = 90 DFT-MD training frames + the ALC-2 candidate frames (labeled, as after ALC-2 of the
            campaign); test = the ALC-3 candidates
  alc2_alc4 same reference; test = the ALC-4 candidates
Error model: ridge least squares on the reference force rows (lambda = 1e-8, as step 3); per-atom error = mean |dF|
over the three components. Cluster labels: the spectral labels for DFT atoms, the 15-NN labels for candidates (step 2).
gamma_cluster of an atom of cluster c uses the reference atoms of cluster c. Grades from 5 MaxVol solutions (seeds
300 + t, as 05_controls.py). Metrics within each cluster: Spearman(gamma, error) and AUROC for the cluster's own
top-10 % error atoms.
Writes results/transfer_by_cluster.csv (protocol, trial, cluster, n, score, spearman, auroc_q90).
usage: 10_transfer_protocols.py [n_trials]
"""
import csv, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import POOLS, RETRO_WORK as WORK, RETRO_RESULTS as RES

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
RIDGE = 1e-8
_spec = importlib.util.spec_from_file_location("errors", Path(__file__).parent / "03_errors.py")
errors = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(errors)


def main():
    frames = list(csv.DictReader(open(WORK / "frame_manifest.csv")))
    man = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man]); src = np.array([r["source_file"] for r in man])
    gf = np.array([int(r["global_frame_idx"]) for r in man])
    lab = np.empty(len(man), int)
    lab[kind == "dft"] = np.load(WORK / "labels_dft.npy"); lab[kind == "chimes"] = np.load(WORK / "labels_candidates.npy")
    split = json.loads((RES / "dft_holdout.json").read_text())
    A = np.load(WORK / "A_atomic.npy"); F = errors.forces(frames)
    dtr = np.where((kind == "dft") & np.isin(gf, split["train_frames"]))[0]
    dho = np.where((kind == "dft") & np.isin(gf, split["holdout_frames"]))[0]
    pool = {k: np.where(src == v)[0] for k, v in POOLS.items()}
    ref_alc = np.concatenate([dtr, pool["alc2"]])
    protocols = {"in_dist": (dtr, dho), "alc2_alc3": (ref_alc, pool["alc3"]), "alc2_alc4": (ref_alc, pool["alc4"])}

    rows = []
    for name, (ref, test) in protocols.items():
        r = E.atom_rows_from_indices(ref)
        x = np.linalg.solve(A[r].T @ A[r] + RIDGE * np.eye(A.shape[1]), A[r].T @ F.reshape(-1)[r])
        y = np.abs((A[E.atom_rows_from_indices(test)] @ x).reshape(-1, 3) - F[test]).mean(1)
        lt, lr = lab[test], lab[ref]
        A_test = A[E.atom_rows_from_indices(test)]
        print(f"{name}: reference {len(ref)} atoms, test {len(test)} atoms, test MAE {y.mean() * 51.422:.3f} eV/A",
              flush=True)
        for t in range(N_TRIALS):
            rng = np.random.default_rng(E.SEED0 + t)
            gb = E.gamma_for_rows(A_test, E.random_restart_maxvol(A[r], rng))
            gc = np.full(len(test), np.nan)
            for c in np.unique(lr):
                m = np.where(lt == c)[0]
                inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[lr == c])], rng)
                gc[m] = E.gamma_for_rows(A_test[E.atom_rows_from_indices(m)], inv)
            for c in np.unique(lt):
                m = lt == c
                for s, v in (("raw_bulk", gb), ("raw_cluster", gc)):
                    rows.append(dict(protocol=name, trial=t, cluster=c, n=int(m.sum()), score=s,
                                     spearman=E.spearman(v[m], y[m]), auroc_q90=E.auroc_high_error(v[m], y[m])))
    df = pd.DataFrame(rows); df.to_csv(RES / "transfer_by_cluster.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["protocol", "cluster", "score"])[["spearman", "auroc_q90"]].mean().unstack("score").round(3))


if __name__ == "__main__":
    main()
