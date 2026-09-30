#!/usr/bin/env python3
"""Step 11: MD stability of the active-learning refits (Fig. 6B models) -> work/al_md/<split>/<model>__<statepoint>/.

Every model saved by `08_al_step.py --save-coefs` (base + strategy x K x draw) is run at the state points of the Fig. 6B
panels, with the protocol of 00_setup_md.py (64 atoms from frame 0 of the DFT-MD file, Langevin 100 fs damping with zero
net force, dt = 0.2 fs, 10 ps, seed 1), plus:
  - fix langevin ... tally yes, so thermo reports ecouple (energy exchanged with the bath) and
    econserve = etotal + ecouple, the conserved quantity of Langevin dynamics;
  - thermo every 50 steps (10 fs) and coordinates every 100 steps (20 fs), for drift and closest-approach analysis.
params.txt = the base model's file with the refit coefficients written in (common/chimes_params.py; checked to reproduce
A.c forces to 1e-10 kcal/mol/A through chimes_calculator).
usage: 11_al_md_setup.py [split] [seed]      (defaults: balanced 1)
Seed mode (extra seeds for a subset of models, one directory per model x seed x state point):
  11_al_md_setup.py balanced --seeds 2,3,4,5 --sp 5000K_2.0gcc --select gamma_bulk:5,random:10 --out balanced_seeds [--append]
  --select strategy:n keeps draws 0..n-1 of that strategy, strategy:lo-hi keeps draws lo..hi-1 (the base model is
  "base:1"). Runs that already have a log.lammps are never rewritten; the runs added by a call are also written to
  jobs_added.txt (run with JOBS=jobs_added.txt).
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD
from common.chimes_params import write_params

STATEPOINTS = ["5000K_2.0gcc", "300K_1.0gcc"]        # Fig. 6B panels (a) and (b)
TEMPLATE = MD / "models" / "shared_t0_dft_train" / "params.txt"
spec = importlib.util.spec_from_file_location("setup_md", MD / "00_setup_md.py")
setup_md = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup_md)

IN = """# AL-refit MD stability (md_audit step 11): model={m} split={split} statepoint={sp} seed={s}
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
fix 2 all langevin {T}.0 {T}.0 100.0 {s}3{s}9 zero yes tally yes
dump 1 all custom 100 traj.lammpstrj id x y z
dump_modify 1 sort id format float %.6f
thermo_style custom step time temp pe ke etotal ecouple econserve press
thermo_modify flush yes
thermo 50
timestep 0.2
run 50000
"""


def main():
    args = sys.argv[1:]
    opt = lambda k, default: args[args.index(k) + 1] if k in args else default
    pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    split = pos[0] if pos else "balanced"
    if "--seeds" not in args:                       # original mode: every model, both state points, one seed
        seed = int(pos[1]) if len(pos) > 1 else 1
        coefs = np.load(MD / "work" / "al_models" / split / "coefs.npz")
        out = MD / "work" / "al_md" / split; out.mkdir(parents=True, exist_ok=True)
        data = {sp: setup_md.data_in(sp) for sp in STATEPOINTS}
        jobs = []
        for m in coefs.files:
            for sp in STATEPOINTS:
                d = out / f"{m}__{sp}"; d.mkdir(exist_ok=True)
                (d / "data.in").write_text(data[sp])
                write_params(TEMPLATE, coefs[m], d / "params.txt", comment=f"AL refit {m} ({split} split), md_audit step 8")
                (d / "in.lammps").write_text(IN.format(m=m, split=split, sp=sp, s=seed, T=sp.split("K_")[0]))
                jobs.append(d.name)
        (out / "jobs.txt").write_text("\n".join(jobs) + "\n")
        print(f"{len(jobs)} runs ({len(coefs.files)} models x {len(STATEPOINTS)} state points) in {out}")
        return
    # seed mode: extra seeds for selected models -> work/al_md/<out>/<model>__s<seed>__<statepoint>/
    seeds = [int(s) for s in opt("--seeds", "1").split(",")]
    sps = opt("--sp", ",".join(STATEPOINTS)).split(",")
    def draw_range(v):                               # "n" -> draws 0..n-1; "lo-hi" -> draws lo..hi-1
        lo, hi = (int(x) for x in v.split("-")) if "-" in v else (0, int(v))
        return range(lo, hi)
    select = dict((kv.split(":")[0], draw_range(kv.split(":")[1])) for kv in opt("--select", "").split(",") if kv)
    coefs = np.load(MD / "work" / "al_models" / split / "coefs.npz")
    out = MD / "work" / "al_md" / opt("--out", f"{split}_seeds"); out.mkdir(parents=True, exist_ok=True)
    jf = out / "jobs.txt"
    jobs = jf.read_text().split() if ("--append" in args and jf.exists()) else []
    data = {sp: setup_md.data_in(sp) for sp in sps}
    added = []
    for m in coefs.files:
        strat, draw = m.split("__")[0], int(m.rsplit("__d", 1)[1]) if "__d" in m else 0
        if strat not in select or draw not in select[strat]:
            continue
        for s in seeds:
            for sp in sps:
                d = out / f"{m}__s{s}__{sp}"
                if (d / "log.lammps").exists():          # never overwrite a run that has already been done
                    continue
                d.mkdir(exist_ok=True)
                (d / "data.in").write_text(data[sp])
                write_params(TEMPLATE, coefs[m], d / "params.txt", comment=f"AL refit {m} ({split} split), md_audit step 8")
                (d / "in.lammps").write_text(IN.format(m=m, split=split, sp=sp, s=s, T=sp.split("K_")[0]))
                if d.name not in jobs:
                    jobs.append(d.name); added.append(d.name)
    jf.write_text("\n".join(jobs) + "\n")
    # runs added by this call only: run them with JOBS=jobs_added.txt (slurm/run_al_md.cmd)
    (out / "jobs_added.txt").write_text("\n".join(added) + "\n")
    print(f"{len(added)} runs added ({len(jobs)} total) in {out}; new ones listed in jobs_added.txt")


if __name__ == "__main__":
    main()
