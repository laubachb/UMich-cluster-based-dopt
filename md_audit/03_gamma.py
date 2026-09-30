#!/usr/bin/env python3
"""Step 3: bulk and cluster-local gamma of every MD atom against the training set of the model that ran that MD
(models/<model>/train_meta.json), using the retrospective study's clusters (retrospective/work/labels_*.npy).

  assignment : StandardScaler fit on the DFT atoms' concatenated descriptors, 15-NN distance-weighted vote
  gamma      : random-restart MaxVol (common/dopt.py), 10 solutions, all kept
Every MD atom is also graded against all DFT atoms, for comparison.
Writes work/gamma_md.npz and results/flag_summary.csv.
Robust flag: gamma > 1 in >= 80% of solutions; "cluster-only" also requires gamma_bulk <= 1 in every solution.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, RETRO_WORK as RW

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WORK = MD / "work"


def main():
    A = np.load(RW / "A_atomic.npy")
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man])
    dft_ids, ch_ids = np.where(kind == "dft")[0], np.where(kind == "chimes")[0]
    lab = np.full(len(kind), -1)
    lab[dft_ids] = np.load(RW / "labels_dft.npy")
    lab[ch_ids] = np.load(RW / "labels_candidates.npy")
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    X = A.reshape(len(kind), -1)
    sc = StandardScaler().fit(X[dft_ids])
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(sc.transform(X[dft_ids]), lab[dft_ids])

    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    Am = np.load(WORK / "A.npy")
    nat = Am.shape[0] // 3
    assert nat == fm.n_atoms.sum(), (nat, fm.n_atoms.sum())
    frame_of = np.repeat(fm.frame.to_numpy(), fm.n_atoms.to_numpy())
    cl = knn.predict(sc.transform(Am.reshape(nat, -1)))
    print("MD cluster counts:", np.bincount(cl, minlength=6), flush=True)

    refs = {"all_dft": dft_ids}
    for m in fm.model.unique():
        meta = json.load(open(MD / "models" / m / "train_meta.json"))
        refs[m] = np.array(sorted(a for f in meta["files"] for i in f["indices"] for a in by_frame[f["source"], i]))
    out = {"cluster": cl.astype(np.int8), "frame": frame_of}
    for name, ref in refs.items():
        md = np.arange(nat) if name == "all_dft" else np.where(np.isin(frame_of, fm.frame[fm.model == name]))[0]
        gb = np.full((N_TRIALS, md.size), np.nan, np.float32); gc = gb.copy()
        for t in range(N_TRIALS):
            rng = np.random.default_rng(E.SEED0 + t)
            gb[t] = E.gamma_for_rows(Am[E.atom_rows_from_indices(md)], E.random_restart_maxvol(A[E.atom_rows_from_indices(ref)], rng))
            for c in range(6):
                rc, mc = ref[lab[ref] == c], np.where(cl[md] == c)[0]
                if mc.size == 0: continue
                inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(rc)], rng)
                gc[t, mc] = E.gamma_for_rows(Am[E.atom_rows_from_indices(md[mc])], inv)
        out[f"{name}__atoms"], out[f"{name}__gb"], out[f"{name}__gc"] = md, gb, gc
        print(f"{name}: ref atoms {ref.size} (per cluster {np.bincount(lab[ref], minlength=6)}), MD atoms {md.size}", flush=True)
    np.savez_compressed(WORK / "gamma_md.npz", **out)

    res = []
    for name in refs:
        if name == "all_dft": continue
        md, gb, gc = out[f"{name}__atoms"], out[f"{name}__gb"], out[f"{name}__gc"]
        robust_c = ((gc > 1).mean(0) >= 0.8) & ((gb > 1).mean(0) == 0)
        robust_b = ((gb > 1).mean(0) >= 0.8) & ((gc > 1).mean(0) == 0)
        for sp in sorted(fm.statepoint.unique()):
            m = fm.set_index("frame").loc[frame_of[md], "statepoint"].to_numpy() == sp
            if not m.any(): continue
            fr = frame_of[md][m]
            res.append(dict(model=name, statepoint=sp, atoms=int(m.sum()),
                            gb_gt1=float((gb[:, m] > 1).mean()), gc_gt1=float((gc[:, m] > 1).mean()),
                            robust_cluster_only=int(robust_c[m].sum()), robust_bulk_only=int(robust_b[m].sum()),
                            frames_with_robust=int(np.unique(fr[robust_c[m]]).size), frames=int(np.unique(fr).size),
                            gb_q99=float(np.quantile(gb[:, m].mean(0), .99)), gc_q99=float(np.quantile(gc[:, m].mean(0), .99))))
    df = pd.DataFrame(res); (MD / "results").mkdir(exist_ok=True)
    df.to_csv(MD / "results" / "flag_summary.csv", index=False)
    pd.set_option("display.width", 200); print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
