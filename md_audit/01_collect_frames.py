#!/usr/bin/env python3
"""Step 1: every saved MD frame -> data/traj.xyzf.gz + data/frame_manifest.csv (both tracked).

Frames are ordered by run directory name (sorted) and then by time; positions are wrapped into the cell and
written with zero forces (the ChIMES design matrix depends only on geometry). Frame indices in the manifest are
the keys used by every later step (DFT directories, DFT forces, gamma, errors).
"""
import gzip, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD


def frames(path):
    with open(path) as f:
        while f.readline():
            step = int(f.readline()); f.readline(); n = int(f.readline()); f.readline()
            L = np.array([[float(x) for x in f.readline().split()[:2]] for _ in range(3)])
            f.readline()
            yield step, L, np.array([f.readline().split()[3:6] for _ in range(n)], dtype=float)


def main():
    rows = []; (MD / "data").mkdir(exist_ok=True)
    with gzip.open(MD / "data" / "traj.xyzf.gz", "wt") as w:
        for d in sorted(p.parent for p in (MD / "work" / "md").glob("*/traj.lammpstrj")):
            model, sp, s = d.name.split("__")
            for step, L, xyz in frames(d / "traj.lammpstrj"):
                box = L[:, 1] - L[:, 0]
                xyz = (xyz - L[:, 0]) % box
                w.write(f"{len(xyz)}\nNON_ORTHO {box[0]:.8f} 0 0 0 {box[1]:.8f} 0 0 0 {box[2]:.8f} 0 0 0 0 0 0 0\n")
                np.savetxt(w, np.c_[xyz, np.zeros_like(xyz)], fmt="N %.6f %.6f %.6f %.1f %.1f %.1f")
                rows.append((len(rows), d.name, model, sp, int(s[1:]), step, len(xyz)))
    with open(MD / "data" / "frame_manifest.csv", "w") as f:
        f.write("frame,job,model,statepoint,rep,step,n_atoms\n")
        f.writelines(",".join(map(str, r)) + "\n" for r in rows)
    print(f"{len(rows)} frames, {sum(r[6] for r in rows)} atoms")


if __name__ == "__main__":
    main()
