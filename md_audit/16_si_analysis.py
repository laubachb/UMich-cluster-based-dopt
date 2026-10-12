#!/usr/bin/env python3
"""Step 16: supporting-information analyses of the MD audit -> results/si_*.csv.

thermostat    N2 bond-length standard deviation vs time for the DFT-only model at 300 K and 2000 K, Nose-Hoover
              (T_damp = 20 fs, archived first runs) vs Langevin (100 fs, data/traj.xyzf.gz); bonds = mutual nearest-
              neighbour pairs closer than 1.3 A; mean over the three seeds; DFT-MD reference from the dataset files
              -> si_thermostat.csv (needs ../archive/md_audit_work_nosehoover_runs; skipped if absent)
flag_time     fraction of atoms with gamma > 1 (mean over the 10 MaxVol solutions) vs time, per state point and grade,
              DFT-only model, mean over seeds -> si_flag_vs_time.csv
atom_rank     atom-level Spearman(gamma, error) per state point with a 95% bootstrap over frames (200 resamples)
              -> si_md_atom_rank.csv
split         frame-level gamma features with the pool/test role of each frame under the seed and the balanced split
              -> si_split_frames.csv
usage: 16_si_analysis.py [all|thermostat|flag_time|atom_rank|split]
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD, DATA, ROOT
from common.xyzf import read_lines, frame_offsets, frame

MODEL = "shared_t0_dft_train"
R = MD / "results"
NH = ROOT / "archive" / "md_audit_work_nosehoover_runs"


def bond_sd(cell, x):
    ln = np.diag(cell); d = x[:, None] - x[None]; d -= ln * np.round(d / ln)
    r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf)
    nn = r.argmin(1)
    b = [r[i, j] for i, j in enumerate(nn) if i < j and nn[j] == i and r[i, j] < 1.3]
    return np.std(b) if len(b) > 2 else np.nan


def traj_bonds(traj, manifest, label):
    fm = pd.read_csv(manifest); L = read_lines(traj); off = frame_offsets(L)
    sel = fm[(fm.model == MODEL) & fm.statepoint.isin(["300K_1.0gcc", "2000K_1.0gcc"])]
    rows = []
    for f, sp, job, step in zip(sel.frame, sel.statepoint, sel.job, sel.step):
        cell, x = frame(L, off[f])
        rows.append(dict(thermostat=label, statepoint=sp, job=job, t_ps=step * 0.2 / 1000, bond_sd=bond_sd(cell, x)))
    return pd.DataFrame(rows)


def thermostat():
    if not (NH / "traj.xyzf").exists():
        print("Nose-Hoover archive not found; skipping"); return
    df = pd.concat([traj_bonds(NH / "traj.xyzf", NH / "frame_manifest.csv", "nose_hoover"),
                    traj_bonds(MD / "data" / "traj.xyzf.gz", MD / "data" / "frame_manifest.csv", "langevin")])
    out = df.groupby(["thermostat", "statepoint", "t_ps"]).bond_sd.agg(["mean", "std", "count"]).reset_index()
    ref = []
    for sp in ["300K_1.0gcc", "2000K_1.0gcc"]:
        L = read_lines(DATA / f"DFT.{sp}.xyzf"); off = frame_offsets(L)
        ref.append(dict(thermostat="dft_md", statepoint=sp, t_ps=np.nan,
                        mean=np.nanmean([bond_sd(*frame(L, o)) for o in off[:-1]]), std=np.nan, count=len(off) - 1))
    out = pd.concat([out, pd.DataFrame(ref)])
    out.to_csv(R / "si_thermostat.csv", index=False)
    print(out.groupby(["thermostat", "statepoint"])["mean"].agg(["first", "last"]).round(4))


def flag_time():
    g = np.load(MD / "work" / "gamma_md.npz")
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv").set_index("frame")
    atoms = g[f"{MODEL}__atoms"]; fr = g["frame"][atoms]
    rows = []
    for name, key in (("bulk", "gb"), ("cluster", "gc")):
        frac = (g[f"{MODEL}__{key}"] > 1).mean(0)                  # per atom, share of solutions with gamma > 1
        s = pd.DataFrame({"frame": fr, "flag": frac}).groupby("frame").flag.mean()
        d = fm.loc[s.index, ["statepoint", "rep", "step"]].assign(flag=s.to_numpy(), grade=name)
        rows.append(d)
    df = pd.concat(rows); df["t_ps"] = df.step * 0.2 / 1000
    out = df.groupby(["grade", "statepoint", "t_ps"]).flag.agg(["mean", "std"]).reset_index()
    out.to_csv(R / "si_flag_vs_time.csv", index=False)
    print(out.groupby(["grade", "statepoint"])["mean"].agg(["first", "max", "last"]).round(3))


def atom_rank():
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    rng = np.random.default_rng(0); rows = []
    for sp, d in df.groupby("statepoint"):
        groups = {f: x for f, x in d.groupby("frame")}; frames = np.array(list(groups))
        for s in ["gamma_bulk", "gamma_cluster"]:
            v = spearmanr(d[s], d.err)[0]
            bs = []
            for _ in range(200):
                b = pd.concat([groups[f] for f in rng.choice(frames, len(frames))])
                bs.append(spearmanr(b[s], b.err)[0])
            rows.append(dict(statepoint=sp, grade=s, spearman=v, lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
    out = pd.DataFrame(rows); out.to_csv(R / "si_md_atom_rank.csv", index=False); print(out.round(3))


def split():
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    fr = df.groupby(["statepoint", "seed", "frame"]).agg(gb_max=("gamma_bulk", "max"), gc_max=("gamma_cluster", "max"),
                                                         gc_frac=("gamma_cluster", lambda g: (g > 1).mean())).reset_index()
    bal = pd.read_csv(R / "balanced_split.csv").set_index("frame").role
    fr["seed_role"] = np.where(fr.seed == 3, "test", "pool"); fr["balanced_role"] = fr.frame.map(bal)
    fr.to_csv(R / "si_split_frames.csv", index=False)
    print(fr.groupby(["statepoint", "seed_role"]).gc_max.median().round(2).to_dict())


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("thermostat", thermostat), ("flag_time", flag_time), ("atom_rank", atom_rank), ("split", split)):
        if stage in ("all", name):
            print(f"== {name}", flush=True); fn()
