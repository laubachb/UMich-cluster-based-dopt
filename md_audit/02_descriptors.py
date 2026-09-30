#!/usr/bin/env python3
"""Step 2: ChIMES force descriptors of every MD atom (same fm_setup.in as the retrospective study).
Writes work/traj.xyzf, work/fm_setup.in, work/A.txt, work/A.npy. Needs CHIMES_LSQ (and MPIRUN)."""
import gzip, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD, RETRO, tool, mpirun


def main():
    W = MD / "work"; W.mkdir(exist_ok=True)
    with gzip.open(MD / "data" / "traj.xyzf.gz", "rb") as s, open(W / "traj.xyzf", "wb") as d:
        shutil.copyfileobj(s, d)
    n = len(pd.read_csv(MD / "data" / "frame_manifest.csv"))
    fm = (RETRO / "fm_setup.in").read_text()
    fm = re.sub(r"(# TRJFILE #\s*\n)\s*MULTI traj_list.dat", r"\1        traj.xyzf", fm)
    fm = re.sub(r"(# NFRAMES #\s*\n)\s*\d+", rf"\g<1>        {n}", fm)
    (W / "fm_setup.in").write_text(fm)
    subprocess.run(mpirun() + [tool("CHIMES_LSQ"), "fm_setup.in"], cwd=W, check=True,
                   stdout=open(W / "fm_setup.log", "w"), stderr=subprocess.STDOUT)
    np.save(W / "A.npy", pd.read_csv(W / "A.txt", sep=r"\s+", header=None, dtype=np.float64).to_numpy())
    print("A.npy", np.load(W / "A.npy", mmap_mode="r").shape)


if __name__ == "__main__":
    main()
