#!/usr/bin/env python3
"""Step 1: ChIMES force descriptors (design-matrix rows) for every atom of the nitrogen dataset.

Writes to retrospective/work/:
  traj_list.dat, fm_setup.in    chimes_lsq inputs (10 xyzf files, 646 frames)
  A.txt                         chimes_lsq force-only design matrix, 3 rows (fx, fy, fz) per atom, 42 columns
  A_atomic.npy                  A.txt as float64 (172,824 x 42)
  frame_manifest.csv            one row per frame (source file, kind = dft | chimes, frame index, atom count)
  atom_manifest.csv             one row per atom; atom i owns rows 3i..3i+2 of A_atomic

Needs CHIMES_LSQ (and MPIRUN) unless work/A.txt already exists (--skip-lsq).
"""
import argparse, csv, re, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA, DFT_FILES, POOLS, RETRO, RETRO_WORK as WORK, tool, mpirun


def n_atoms_per_frame(path):
    L = path.read_text().splitlines(); i = 0; out = []
    while i < len(L):
        n = int(L[i]); out.append(n); i += n + 2
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-lsq", action="store_true"); args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    files = DFT_FILES + list(POOLS.values())
    frames, entries = [], []
    for src in files:
        nat = n_atoms_per_frame(DATA / src)
        entries.append(f"{len(nat)} {DATA / src}")
        for k, n in enumerate(nat):
            frames.append(dict(global_frame_idx=len(frames), source_file=src, kind="dft" if src.startswith("DFT.") else "chimes",
                               frame_idx_in_file=k, n_atoms=n))
    (WORK / "traj_list.dat").write_text(f"{len(files)}\n" + "\n".join(entries) + "\n")
    fm = (RETRO / "fm_setup.in").read_text()
    fm = re.sub(r"(# NFRAMES #\s*\n)\s*\d+", rf"\g<1>        {len(frames)}", fm)
    (WORK / "fm_setup.in").write_text(fm)
    if not args.skip_lsq:
        subprocess.run(mpirun() + [tool("CHIMES_LSQ"), "fm_setup.in"], cwd=WORK, check=True,
                       stdout=open(WORK / "fm_setup.log", "w"), stderr=subprocess.STDOUT)
    A = np.vstack([np.fromstring(l, sep=" ", dtype=np.float64) for l in open(WORK / "A.txt") if l.strip()])
    n_at = sum(f["n_atoms"] for f in frames)
    assert A.shape[0] == 3 * n_at, (A.shape, n_at)
    np.save(WORK / "A_atomic.npy", A)
    with open(WORK / "frame_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frames[0])); w.writeheader(); w.writerows(frames)
    with open(WORK / "atom_manifest.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["atom_row", "global_frame_idx", "source_file", "kind", "frame_idx_in_file", "atom_idx"])
        i = 0
        for fr in frames:
            for a in range(fr["n_atoms"]):
                w.writerow([i, fr["global_frame_idx"], fr["source_file"], fr["kind"], fr["frame_idx_in_file"], a]); i += 1
    print(f"A_atomic {A.shape}; {len(frames)} frames, {n_at} atoms")


if __name__ == "__main__":
    main()
