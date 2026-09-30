#!/usr/bin/env python3
"""Step 3: reference force error of every candidate atom.

A preliminary model is fit to 75% of the DFT frames (15 of 20 per state point; split drawn with seed 42) by
ridge least squares on the force rows (lambda = 1e-8, forces in Ha/bohr as stored in the xyzf files).
Per-atom error = mean |F_pred - F_DFT| over the three components (Ha/bohr; x 51.422 for eV/A).
Writes results/dft_holdout.json (tracked) and work/atom_errors.csv.
"""
import csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA, RETRO_WORK as WORK, RETRO_RESULTS as RES
from common.dopt import atom_rows_from_indices

HOLDOUT_FRAC, SEED, RIDGE = 0.25, 42, 1e-8


def split(frames):
    by_sp = defaultdict(list)
    for r in frames:
        if r["kind"] == "dft":
            by_sp[re.match(r"^DFT\.(.+)\.xyzf$", r["source_file"]).group(1)].append(int(r["global_frame_idx"]))
    rng = np.random.default_rng(SEED); train, hold, per = [], [], {}
    for sp, g in sorted(by_sp.items()):
        g = np.array(sorted(g), dtype=np.int64); rng.shuffle(g)
        nh = min(max(1, int(round(HOLDOUT_FRAC * len(g)))), len(g) - 1)
        hold += g[:nh].tolist(); train += g[nh:].tolist(); per[sp] = dict(n_total=len(g), n_train=len(g) - nh, n_holdout=nh)
    return sorted(train), sorted(hold), per


def forces(frames):
    cache, out = {}, []
    for fr in frames:
        src = fr["source_file"]
        if src not in cache:
            L = (DATA / src).read_text().splitlines(); i = 0; blocks = []
            while i < len(L):
                n = int(L[i]); blocks.append(np.array([l.split()[4:7] for l in L[i + 2:i + 2 + n]], float)); i += n + 2
            cache[src] = blocks
        out.append(cache[src][int(fr["frame_idx_in_file"])])
    return np.vstack(out)


def main():
    frames = list(csv.DictReader(open(WORK / "frame_manifest.csv")))
    atoms = list(csv.DictReader(open(WORK / "atom_manifest.csv")))
    gf = np.array([int(a["global_frame_idx"]) for a in atoms]); kind = np.array([a["kind"] for a in atoms])
    train, hold, per = split(frames)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "dft_holdout.json").write_text(json.dumps(dict(frac=HOLDOUT_FRAC, seed=SEED, train_frames=train,
                                                          holdout_frames=hold, per_statepoint=per), indent=2) + "\n")
    A = np.load(WORK / "A_atomic.npy"); F = forces(frames); b = F.reshape(-1)
    tr = np.where(np.isin(gf, train) & (kind == "dft"))[0]
    rows = atom_rows_from_indices(tr)
    x = np.linalg.solve(A[rows].T @ A[rows] + RIDGE * np.eye(A.shape[1]), A[rows].T @ b[rows])
    d = (A @ x).reshape(-1, 3) - F
    mae, rmse = np.abs(d).mean(1), np.sqrt((d ** 2).mean(1))
    ch = np.where(kind == "chimes")[0]
    with open(WORK / "atom_errors.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["global_atom_idx", "global_frame_idx", "source_file", "atom_idx", "mae", "rmse"])
        for i in ch:
            w.writerow([i, gf[i], atoms[i]["source_file"], atoms[i]["atom_idx"], f"{mae[i]:.8g}", f"{rmse[i]:.8g}"])
    hm = np.isin(gf, hold)
    print(f"train frames {len(train)}, holdout {len(hold)}; holdout MAE {mae[hm].mean():.6g} Ha/bohr; "
          f"candidate MAE median {np.median(mae[ch]):.6g}")


if __name__ == "__main__":
    main()
