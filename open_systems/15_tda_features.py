#!/usr/bin/env python3
"""Step 15: per-atom topological (`tda`) and distance/angle control (`ctrl`) descriptors of test T (CRITERIA.md).

Reads work/NAME/traj.xyzf (every atom, file order = the order of A.npy rows / 3 and forces.npy). For each atom: the
neighbours within rc3 (params.json; at least the 12 nearest), periodic images from the full cell (enough image layers
for rc3), centred on the atom. tda (33): the 12 smallest H0 deaths (padded with rc3), H1 count / total / longest
persistence / mean birth / mean death, a 4 x 4 H1 persistence image. ctrl (21): 12 smallest neighbour distances
(padded), an 8-bin angle histogram over neighbour pairs, the neighbour count.
Writes work/NAME/tda.npy and work/NAME/ctrl.npy (n_atoms x d, float32). Nitrogen: --nitrogen writes
retrospective/work/{tda,ctrl}.npy in atom-manifest order from the data xyzf files.
usage: 15_tda_features.py NAME [--jobs N]  |  15_tda_features.py --nitrogen [--jobs N]
"""
import argparse, csv, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.spatial import cKDTree

import core as C
sys.path.insert(0, str(C.HERE.parent))
from common.xyzf import read_lines, frame_offsets, frame as read_frame

MIN_NEIGH, N_H0, IMG, SIG = 12, 12, 4, 0.2
G = {}


def neighbours(cell, x, rc):
    """for every atom: relative positions of all periodic-image neighbours within rc (at least MIN_NEIGH nearest)."""
    inv = np.linalg.inv(cell); widths = 1.0 / np.linalg.norm(inv, axis=0)      # perpendicular widths of the cell
    n_img = int(np.ceil(rc / widths.min())) + 1
    shifts = np.array([[i, j, k] for i in range(-n_img, n_img + 1) for j in range(-n_img, n_img + 1) for k in range(-n_img, n_img + 1)])
    frac = x @ inv; frac -= np.floor(frac); xw = frac @ cell
    sup = (xw[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    tree = cKDTree(sup); out = []
    for i, xi in enumerate(xw):
        idx = tree.query_ball_point(xi, rc)
        d = sup[idx] - xi; r = np.linalg.norm(d, axis=1); keep = r > 1e-6; d, r = d[keep], r[keep]
        if len(r) < MIN_NEIGH:
            dd, ii = tree.query(xi, MIN_NEIGH + 1); dd, ii = dd[dd > 1e-6], ii[dd > 1e-6]
            d = sup[ii] - xi; r = dd
        out.append(d[np.argsort(r)])
    return out


def features(d, rc):
    r = np.linalg.norm(d, axis=1); n = len(r)
    pts = np.vstack([np.zeros(3), d])
    dg = ripser(pts, maxdim=1, thresh=rc)["dgms"]
    h0 = np.sort(dg[0][np.isfinite(dg[0][:, 1]), 1]); h0 = np.r_[h0[:N_H0], np.full(max(0, N_H0 - len(h0)), rc)]
    h1 = dg[1]; h1 = h1[np.isfinite(h1[:, 1])] if len(h1) else h1
    if len(h1):
        pers = h1[:, 1] - h1[:, 0]
        stats = [len(h1), pers.sum(), pers.max(), h1[:, 0].mean(), h1[:, 1].mean()]
        bx, py = np.meshgrid(np.linspace(0, rc, IMG, endpoint=False) + rc / (2 * IMG), np.linspace(0, rc / 2, IMG, endpoint=False) + rc / (4 * IMG), indexing="ij")
        img = sum(w * np.exp(-((bx - b) ** 2 + (py - p) ** 2) / (2 * SIG ** 2)) for b, p, w in zip(h1[:, 0], pers, pers)).ravel()
    else:
        stats = [0, 0, 0, 0, 0]; img = np.zeros(IMG * IMG)
    tda = np.r_[h0, stats, img]
    rs = np.r_[r[:N_H0], np.full(max(0, N_H0 - n), rc)]
    u = d[:min(n, 24)] / r[:min(n, 24), None]; cos = (u @ u.T)[np.triu_indices(len(u), 1)]
    hist = np.histogram(np.degrees(np.arccos(np.clip(cos, -1, 1))), bins=8, range=(0, 180))[0]
    ctrl = np.r_[rs, hist / max(1, hist.sum()), n]
    return tda.astype(np.float32), ctrl.astype(np.float32)


def do_frame(job):
    cell, x, rc = job
    T, Cc = zip(*[features(d, rc) for d in neighbours(cell, x, rc)])
    return np.vstack(T), np.vstack(Cc)


def frames_open(name):
    L = read_lines(C.WORK / name / "traj.xyzf"); off = frame_offsets(L)
    return [read_frame(L, s) for s in off[:-1]]


def frames_nitrogen():
    from common.paths import DATA, RETRO_WORK
    man = list(csv.DictReader(open(RETRO_WORK / "frame_manifest.csv")))
    cache, out = {}, []
    for r in man:
        src = r["source_file"]
        if src not in cache:
            L = read_lines(DATA / src); cache[src] = (L, frame_offsets(L))
        L, off = cache[src]; out.append(read_frame(L, off[int(r["frame_idx_in_file"])]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name", nargs="?"); ap.add_argument("--nitrogen", action="store_true"); ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.nitrogen:
        from common.paths import RETRO_WORK
        fr, rc, out = frames_nitrogen(), 5.0, RETRO_WORK
    else:
        import json
        fr, rc, out = frames_open(a.name), float(json.load(open(C.WORK / a.name / "params.json"))["rc3"]), C.WORK / a.name
    jobs = [(c, x, rc) for c, x in fr]
    print(f"{len(jobs)} frames, {sum(len(x) for _, x in fr)} atoms, rc3 = {rc}", flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            res = p.map(do_frame, jobs, chunksize=4)
    else:
        res = [do_frame(j) for j in jobs]
    T = np.vstack([r[0] for r in res]); Cc = np.vstack([r[1] for r in res])
    np.save(out / "tda.npy", T); np.save(out / "ctrl.npy", Cc)
    print(f"tda {T.shape}, ctrl {Cc.shape}; H1 loops per atom: mean {T[:, N_H0].mean():.2f}, atoms with none {np.mean(T[:, N_H0] == 0):.3f}")


if __name__ == "__main__":
    main()
