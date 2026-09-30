#!/usr/bin/env python3
"""Step 0: LAMMPS inputs for the Langevin MD runs -> work/md/<model>__<statepoint>__s<seed>/.

33 runs: 5 models x {300 K, 2000 K} x seeds {1,2,3}, plus the DFT-only model x 5000 K x seeds {1,2,3}.
Each run: 64 N atoms starting from frame 0 of the DFT-MD file of its state point, pair_style chimesFF with the
model's params.txt, velocity-Verlet (fix nve) + Langevin thermostat (100 fs damping, zero net force),
dt = 0.2 fs, 50,000 steps (10 ps), coordinates dumped every 500 steps (100 fs).
Run them with slurm/run_md.cmd (needs LMP_CHIMES).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA, MD
from common.xyzf import read_lines, frame

MODELS = ["shared_t0_dft_train", "bulk_t3_k30", "cluster_t3_k30", "random_t3_k30", "stratified_t3_k30"]
STATEPOINTS = ["300K_1.0gcc", "2000K_1.0gcc", "5000K_2.0gcc"]
SEEDS = [1, 2, 3]

IN = """# Langevin NVT rerun (md_audit): model={m} statepoint={sp} seed={s}
units real
newton on
atom_style atomic
atom_modify sort 0 0.0
atom_modify map array
neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes
read_data data.in
velocity all create {T}.0 {s}7{s}1 dist gaussian mom yes rot yes loop all
pair_style chimesFF
pair_coeff * * params.txt
fix 1 all nve
fix 2 all langevin {T}.0 {T}.0 100.0 {s}3{s}9 zero yes
dump 1 all custom 500 traj.lammpstrj id type element x y z
dump_modify 1 element N sort id format float %.8f
thermo_style custom step time temp pe ke etotal press
thermo 500
timestep 0.2
run 50000
"""


def data_in(sp):
    src = f"DFT.{sp}.xyzf"; cell, x = frame(read_lines(DATA / src), 0)
    lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
    lines = [f"# from {src} frame 0", "", f"{len(x)} atoms", "1 atom types", "", f"0.0 {lx:.8f} xlo xhi",
             f"0.0 {ly:.8f} ylo yhi", f"0.0 {lz:.8f} zlo zhi", "", "Masses", "", "1 14.0067", "", "Atoms", ""]
    lines += [f"{i} 1 {a:.8f} {b:.8f} {c:.8f}" for i, (a, b, c) in enumerate(x, start=1)]
    return "\n".join(lines) + "\n"


def main():
    out = MD / "work" / "md"; out.mkdir(parents=True, exist_ok=True); jobs = []
    for m in MODELS:
        for sp in STATEPOINTS:
            if sp == "5000K_2.0gcc" and m != "shared_t0_dft_train":
                continue
            for s in SEEDS:
                d = out / f"{m}__{sp}__s{s}"; d.mkdir(exist_ok=True)
                (d / "data.in").write_text(data_in(sp))
                (d / "params.txt").write_text((MD / "models" / m / "params.txt").read_text())
                (d / "in.lammps").write_text(IN.format(m=m, sp=sp, s=s, T=sp.split("K_")[0]))
                jobs.append(d.name)
    (out / "jobs.txt").write_text("\n".join(jobs) + "\n")
    print(f"{len(jobs)} runs in {out}")


if __name__ == "__main__":
    main()
