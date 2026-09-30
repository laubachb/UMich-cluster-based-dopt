#!/usr/bin/env python3
"""Step 5: forces of every finished VASP single point (work/dft/f<frame>/OUTCAR) -> data/dft_forces.npz (tracked).
Arrays: frame (frame index in data/frame_manifest.csv), n_atoms, forces (stacked, eV/A, last TOTAL-FORCE block)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD
from common.xyzf import outcar_forces, outcar_done


def main():
    frames, nat, F = [], [], []
    for d in sorted((MD / "work" / "dft").glob("f*/"), key=lambda p: int(p.name[1:])):
        if not outcar_done(d / "OUTCAR"):
            print("not finished:", d.name); continue
        f = outcar_forces(d / "OUTCAR"); frames.append(int(d.name[1:])); nat.append(len(f)); F.append(f)
    np.savez_compressed(MD / "data" / "dft_forces.npz", frame=np.array(frames), n_atoms=np.array(nat), forces=np.vstack(F))
    print(f"{len(frames)} frames, {sum(nat)} atoms -> data/dft_forces.npz")


if __name__ == "__main__":
    main()
