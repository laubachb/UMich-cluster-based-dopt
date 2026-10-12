#!/usr/bin/env python3
"""Step 1: prepare the open data sets: chimes_lsq inputs, frame tables, DFT forces, species.

  sio2      Erhard, Rohrer, Albe, Deringer et al., "Modelling atomic and nanoscale structure in the silicon-oxygen system
            through active machine-learning" (2024); Zenodo 10.5281/zenodo.10419194, database.zip,
            training.general_purpose.SiOx.xyz (CC-BY-4.0; SCAN, PAW GW potentials, 900 eV). Only the SiO2-composition
            config types (BASE + AL below) are used. Frames of a config type ending in _int_AL / _max_AL were added
            during the original active learning; their parent regime is the config type without that suffix.
  water     Ibrahim, Lysogorskiy, Drautz, Piaggi (2026); Zenodo 10.5281/zenodo.18291736,
            dataset_water_ACE_rev_PBE_D3.gzip (CC-BY-4.0; revPBE-D3), a pandas pickle of ASE atoms (needs ase; run with a
            python that has it). No phase labels: regime from the mean O-O coordination within 3.3 A (CRITERIA.md,
            addendum): tetrahedral <= 4.25 < intermediate <= 7.5 < dense. Periodic, H:O = 2, >= 8 molecules; at most
            WATER_CAP_TETRA tetrahedral frames (seeded), all others.
  si_gap18  Bartok et al., PRX 2018; prepared by cluster-gamma-threshold/experiments/open_prep.py (frames.csv,
            forces.npy, A.npy with the nitrogen 42-term recipe). Linked, not recomputed.

Frame filters (sio2): periodic, DFT forces present, >= 8 atoms, smallest cell width >= rc_2B / NLAYERS. At most CAP
frames per config type (seeded draw), to keep the design matrix near 1.4 M rows.
ChIMES basis (sio2): the nitrogen recipe (2-body order 12, 3-body order 5, Morse transform, Tersoff 0.75 smoothing) for
two elements. Per pair: r_min = 0.98 x the smallest distance of that pair in the kept frames, Morse lambda = the median
nearest-neighbour distance of that pair; common outer cutoffs rc_2B = 5.0 A, rc_3B = 3.5 A (covers the Si-O-Si
triplet, Si-Si ~ 3.1 A).
Writes work/<name>/: traj.xyzf, fm_setup.in, frames.csv, forces.npy (eV/A), species.npy, params.json.
usage: 01_prepare.py sio2 [--source PATH] | si_gap18
"""
import argparse, itertools, json, os, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
MASS = dict(Si=28.0855, O=15.9994)
BASE = ["bulk_cryst", "bulk_cryst_hp", "bulk_amo", "liquid", "quench", "vacancy", "surfaces"]
AL = ["quench_int_AL", "quench_max_AL", "vacancy_int_AL", "highpressure_int_AL", "highpressure_max_AL",
      "surfaces_int_AL", "surfaces_max_AL"]
CAP, NLAYERS, RC2, RC3 = 250, 2, 5.0, 3.5
WATER_CAP_TETRA = 600
GAP18 = Path("/work2/09982/blaubach/stampede3/cluster-gamma-threshold/work/open/si_gap18")

TEMPLATE = """####### CONTROL VARIABLES #######

# TRJFILE #
        traj.xyzf
# WRAPTRJ #
        true
# NFRAMES #
        {nframes}
# NLAYERS #
        {nlayers}
# FITCOUL #
        false
# CNSCOUL #
        false
# FITSTRS #
        false
# FITENER #
        false
# FITPOVR #
        false
# PAIRTYP #
        CHEBYSHEV 12 5 0 -1 1
# CHBTYPE #
        MORSE
# SPLITFI #
        false
# HIERARC #
        false

####### TOPOLOGY VARIABLES #######

# NATMTYP #
        2

# TYPEIDX #     # ATM_TYP #     # ATMCHRG #     # ATMMASS #
1               Si              0.0             {mSi}
2               O               0.0             {mO}

# PAIRIDX #     # ATM_TY1 #     # ATM_TY1 #     # S_MINIM #     # S_MAXIM #     # S_DELTA #     # MORSE_LAMBDA #      # USEOVRP #     # NIJBINS #     # NIKBINS #     # NJKBINS #
{pairs}

SPECIAL 3B S_MAXIM: ALL {rc3:.2f}

# FCUTTYP #
        TERSOFF 0.75

# ENDFILE #
"""


def read_extxyz(path, keep_types):
    out = []
    with open(path) as f:
        while True:
            l = f.readline()
            if not l:
                break
            n, h = int(l.split()[0]), f.readline()
            rows = [f.readline().split() for _ in range(n)]
            ct = re.search(r"config_type=(\S+)", h)
            ct = ct.group(1) if ct else None
            pbc = re.search(r'pbc="([^"]+)"', h)
            if ct not in keep_types or (pbc and pbc.group(1).split() != ["T", "T", "T"]):
                continue
            props = re.search(r"Properties=(\S+)", h).group(1).split(":")
            col, off = {}, 0
            for name, _, cnt in zip(props[0::3], props[1::3], props[2::3]):
                col[name] = off; off += int(cnt)
            if "dft_forces" not in col:
                continue
            out.append(dict(cell=np.array(re.search(r'Lattice="([^"]+)"', h).group(1).split(), float).reshape(3, 3),
                            sp=np.array([r[col["species"]] for r in rows]),
                            xyz=np.array([r[col["pos"]:col["pos"] + 3] for r in rows], float),
                            F=np.array([r[col["dft_forces"]:col["dft_forces"] + 3] for r in rows], float),
                            config_type=ct))
    return out


def geometry(fr, species=("Si", "O")):
    """per-atom nearest distance to each species, and the smallest cell width."""
    cell = fr["cell"]
    xyz = (np.linalg.solve(cell.T, fr["xyz"].T).T % 1.0) @ cell
    vol = abs(np.linalg.det(cell))
    width = min(vol / np.linalg.norm(np.cross(cell[i], cell[j])) for i, j in ((0, 1), (1, 2), (0, 2)))
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3))) @ cell
    img = (xyz[None] + shifts[:, None]).reshape(-1, 3)
    sp_img = np.tile(fr["sp"], len(shifts))
    near = {}
    for b in species:
        tree = cKDTree(img[sp_img == b])
        d = tree.query(xyz, k=2)[0]
        near[b] = np.where((fr["sp"] == b), d[:, 1], d[:, 0])      # skip the atom itself for same-species pairs
    return near, width


def prepare_sio2(source):
    frames = read_extxyz(source, set(BASE + AL))
    frames = [f for f in frames if len(f["xyz"]) >= 8]
    rng = np.random.default_rng(0)
    by_type = {}
    for f in frames:
        by_type.setdefault(f["config_type"], []).append(f)
    keep = []
    for ct in BASE + AL:
        fs = by_type.get(ct, [])
        idx = np.sort(rng.choice(len(fs), min(CAP, len(fs)), replace=False)) if fs else []
        keep += [fs[i] for i in idx]
    geo = [geometry(f) for f in keep]
    keep = [f for f, g in zip(keep, geo) if g[1] >= RC2 / NLAYERS]
    geo = [g for g in geo if g[1] >= RC2 / NLAYERS]
    P = dict(rc2=RC2, rc3=RC3, nlayers=NLAYERS, cap=CAP, pairs={})
    for a, b in (("Si", "Si"), ("Si", "O"), ("O", "O")):
        d = np.concatenate([g[0][b][f["sp"] == a] for f, g in zip(keep, geo)])
        P["pairs"][f"{a}-{b}"] = dict(rmin=round(0.98 * float(d.min()), 3), lam=round(float(np.median(d)), 3))
    out = WORK / "sio2"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "traj.xyzf", "w") as fh:
        for f in keep:
            fh.write(f"{len(f['xyz'])}\nNON_ORTHO " + " ".join(f"{x:.10f}" for x in f["cell"].ravel()) + " 0 0 0 0 0 0 0\n")
            for s, x, F in zip(f["sp"], f["xyz"], f["F"]):
                fh.write(f"{s} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f} {F[0]:.8f} {F[1]:.8f} {F[2]:.8f}\n")
    pairs = "\n".join(f"{i + 1}               {a}              {b}              {P['pairs'][f'{a}-{b}']['rmin']:.3f}"
                      f"           {RC2:.2f}            0.01            {P['pairs'][f'{a}-{b}']['lam']:.3f}"
                      "                   false           0               0               0"
                      for i, (a, b) in enumerate((("Si", "Si"), ("Si", "O"), ("O", "O"))))
    (out / "fm_setup.in").write_text(TEMPLATE.format(nframes=len(keep), nlayers=NLAYERS, mSi=MASS["Si"], mO=MASS["O"],
                                                     pairs=pairs, rc3=RC3))
    tab = pd.DataFrame(dict(frame=np.arange(len(keep)), n_atoms=[len(f["xyz"]) for f in keep],
                            config_type=[f["config_type"] for f in keep],
                            n_Si=[int((f["sp"] == "Si").sum()) for f in keep], n_O=[int((f["sp"] == "O").sum()) for f in keep],
                            f_rms=[float(np.sqrt((f["F"] ** 2).mean())) for f in keep]))
    tab["al"] = tab.config_type.str.endswith("_AL")
    tab["regime"] = tab.config_type.str.replace(r"_(int|max)_AL$", "", regex=True)
    tab.to_csv(out / "frames.csv", index=False)
    np.save(out / "forces.npy", np.vstack([f["F"] for f in keep]))
    np.save(out / "species.npy", np.concatenate([f["sp"] for f in keep]))
    (out / "params.json").write_text(json.dumps(P, indent=1) + "\n")
    print(f"sio2: {len(keep)} frames, {tab.n_atoms.sum()} atoms ({tab.n_Si.sum()} Si, {tab.n_O.sum()} O); params {P['pairs']}")
    print(tab.groupby(["al", "config_type"]).agg(frames=("frame", "size"), atoms=("n_atoms", "sum"),
                                                 O_per_Si=("n_O", "sum"), f_rms=("f_rms", "mean")).assign(
        O_per_Si=lambda d: d.O_per_Si / tab.groupby(["al", "config_type"]).n_Si.sum()).round(3).to_string())


def prepare_water(source):
    df = pd.read_pickle(source, compression="gzip")
    frames = []
    for a, F in zip(df.ase_atoms, df.forces):
        sp = np.array(a.get_chemical_symbols())
        if not a.pbc.all() or (sp == "O").sum() < 8 or (sp == "H").sum() != 2 * (sp == "O").sum():
            continue
        cell = a.cell.array
        xyz = a.get_positions(wrap=True)
        O = xyz[sp == "O"]
        shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3))) @ cell
        tree = cKDTree((O[None] + shifts[:, None]).reshape(-1, 3))
        c = float(np.mean([len(x) - 1 for x in tree.query_ball_point(O, 3.3)]))
        regime = "tetrahedral" if c <= 4.25 else ("intermediate" if c <= 7.5 else "dense")
        frames.append(dict(cell=cell, sp=sp, xyz=xyz, F=np.asarray(F, float), config_type=regime, coord=c))
    rng = np.random.default_rng(0)
    tet = [f for f in frames if f["config_type"] == "tetrahedral"]
    keep_t = set(rng.choice(len(tet), min(WATER_CAP_TETRA, len(tet)), replace=False).tolist())
    keep = [f for f in frames if f["config_type"] != "tetrahedral"] + [f for k, f in enumerate(tet) if k in keep_t]
    geo = [geometry(f, ("O", "H")) for f in keep]
    ok = [g[1] >= RC2 / NLAYERS for g in geo]
    keep, geo = [f for f, o in zip(keep, ok) if o], [g for g, o in zip(geo, ok) if o]
    P = dict(rc2=RC2, rc3=RC3, nlayers=NLAYERS, cap_tetrahedral=WATER_CAP_TETRA, pairs={})
    order = (("O", "O"), ("O", "H"), ("H", "H"))
    for a_, b_ in order:
        d = np.concatenate([g[0][b_][f["sp"] == a_] for f, g in zip(keep, geo)])
        P["pairs"][f"{a_}-{b_}"] = dict(rmin=round(0.98 * float(d.min()), 3), lam=round(float(np.median(d)), 3))
    out = WORK / "water"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "traj.xyzf", "w") as fh:
        for f in keep:
            fh.write(f"{len(f['xyz'])}\nNON_ORTHO " + " ".join(f"{x:.10f}" for x in f["cell"].ravel()) + " 0 0 0 0 0 0 0\n")
            for s_, x, F in zip(f["sp"], f["xyz"], f["F"]):
                fh.write(f"{s_} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f} {F[0]:.8f} {F[1]:.8f} {F[2]:.8f}\n")
    pairs = "\n".join(f"{i + 1}               {a_}              {b_}              {P['pairs'][f'{a_}-{b_}']['rmin']:.3f}"
                      f"           {RC2:.2f}            0.01            {P['pairs'][f'{a_}-{b_}']['lam']:.3f}"
                      "                   false           0               0               0" for i, (a_, b_) in enumerate(order))
    fm = TEMPLATE.format(nframes=len(keep), nlayers=NLAYERS, mSi=0, mO=0, pairs=pairs, rc3=RC3)
    fm = fm.replace("1               Si              0.0             0\n2               O               0.0             0",
                    "1               O               0.0             15.9994\n2               H               0.0             1.0079")
    (out / "fm_setup.in").write_text(fm)
    tab = pd.DataFrame(dict(frame=np.arange(len(keep)), n_atoms=[len(f["xyz"]) for f in keep],
                            config_type=[f["config_type"] for f in keep], coord_OO=[round(f["coord"], 3) for f in keep],
                            n_O=[int((f["sp"] == "O").sum()) for f in keep], n_H=[int((f["sp"] == "H").sum()) for f in keep],
                            f_rms=[float(np.sqrt((f["F"] ** 2).mean())) for f in keep]))
    tab["al"] = False
    tab["regime"] = tab.config_type
    tab.to_csv(out / "frames.csv", index=False)
    np.save(out / "forces.npy", np.vstack([f["F"] for f in keep]))
    np.save(out / "species.npy", np.concatenate([f["sp"] for f in keep]))
    (out / "params.json").write_text(json.dumps(P, indent=1) + "\n")
    print(f"water: {len(keep)} frames, {tab.n_atoms.sum()} atoms; params {P['pairs']}")
    print(tab.groupby("regime").agg(frames=("frame", "size"), atoms=("n_atoms", "sum"), coord=("coord_OO", "median"),
                                    f_rms=("f_rms", "mean")).round(3).to_string())


def link_si_gap18():
    out = WORK / "si_gap18"; out.mkdir(parents=True, exist_ok=True)
    for f in ("A.npy", "forces.npy", "frames.csv", "params.json", "fm_setup.in"):
        if not (out / f).exists():
            os.symlink(GAP18 / f, out / f)
    tab = pd.read_csv(out / "frames.csv")
    np.save(out / "species.npy", np.full(tab.n_atoms.sum(), "Si"))
    print(f"si_gap18: linked from {GAP18}; {len(tab)} frames, {tab.n_atoms.sum()} atoms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name", choices=["sio2", "si_gap18", "water"])
    ap.add_argument("--source", default=os.path.join(os.environ.get("SCRATCH", "."), "cdopt_open_raw", "database",
                                                     "training.general_purpose.SiOx.xyz"))
    a = ap.parse_args()
    if a.name == "water":
        prepare_water(a.source if a.source.endswith(".gzip") else os.path.join(os.environ.get("SCRATCH", "."), "cdopt_open_raw",
                                                                             "water_ace", "dataset_water_ACE_rev_PBE_D3.gzip"))
    else:
        prepare_sio2(a.source) if a.name == "sio2" else link_si_gap18()
