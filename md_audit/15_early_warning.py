#!/usr/bin/env python3
"""Step 15: does gamma rise before a collapse event? (base model, 40 Langevin seeds at 5000 K, work/al_md/base_seeds)

Event time t_e of a run = first time the conserved quantity leaves econserve(0) by > 10 meV/atom or an N-N pair is
inside the 0.86 A inner cutoff (the event definition of 12_al_md_analysis.py); runs without an event have t_e = inf.
Frames: every STRIDE-th saved frame (20 fs apart in the dump) with t < t_e, i.e. only configurations before the event.

stages (run in order; 'all' runs every stage)
  prep         frames -> work/early_warning/traj.xyzf + frames.csv; the first two frames are MD-audit frames whose
               descriptors are already in work/A.npy (checked in 'descriptors')
  descriptors  chimes_lsq with the retrospective fm_setup.in (as 02_descriptors.py) -> A.npy; checks the two
               reference frames against work/A.npy
  gamma        per atom, 5 MaxVol solutions against the base model's training set, same draw order as 03_gamma.py:
               gamma_bulk, gamma_cluster (15-NN cluster vote as 03), and the switch hybrid of 14_hybrid_md.py;
               per frame: the maximum of each over its 64 atoms -> results/early_warning_frames.csv
  analysis     (1) hazard AUROC: does the frame score separate frames within D ps before an event (D = 0.2, 0.5, 1, 2)
               from all other pre-event frames and all frames of runs without an event? 95% interval from a bootstrap
               over runs; (2) alarm: threshold = 95th / 99th percentile of the score over runs without an event;
               share of events with an alarm in the last 1 ps before t_e, and false alarms per ps in runs without
               an event -> results/early_warning_{hazard,alarm,profile}.csv
usage: 15_early_warning.py [all|prep|descriptors|gamma|analysis]
"""
import csv, json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dopt as E
from common.paths import MD, RETRO, RETRO_WORK as RW, tool, mpirun
from common.xyzf import read_lines, frame_offsets, frame

RUNS = MD / "work" / "al_md" / "base_seeds"
W = MD / "work" / "early_warning"
SP, MODEL = "5000K_2.0gcc", "shared_t0_dft_train"
STRIDE, NAT, DT_DUMP = 2, 64, 0.02            # every 2nd dump -> 40 fs; dumps every 20 fs
KCAL_TO_MEV, EVENT_MEV, R_IN = 43.3641, 10.0, 0.86
N_SOL, N_BOOT = 5, 2000
DELTAS = [0.2, 0.5, 1.0, 2.0]
SCORES = ["gb_max", "gc_max", "sw_max"]


def dump_frames(path):
    L = Path(path).read_text().splitlines(); i = 0
    while i < len(L):
        n = int(L[i + 3]); box = np.array([[float(v) for v in L[i + 5 + k].split()[:2]] for k in range(3)])
        yield box, np.array([l.split()[1:4] for l in L[i + 9:i + 9 + n]], float)   # dump: id x y z
        i += 9 + n


def rmin(box, x):
    ln = box[:, 1] - box[:, 0]; d = x[:, None] - x[None]; d -= ln * np.round(d / ln)
    r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf); return r.min()


def econs(log):
    L = Path(log).read_text().splitlines()
    i = next(k for k, l in enumerate(L) if l.split()[:1] == ["Step"]); cols = L[i].split(); rows = []
    for l in L[i + 1:]:
        p = l.split()
        if len(p) != len(cols):
            break
        rows.append([float(v) for v in p])
    th = pd.DataFrame(rows, columns=cols)
    return th.Time.to_numpy() / 1000.0, th.Econserve.to_numpy() * KCAL_TO_MEV / NAT


def prep():
    W.mkdir(parents=True, exist_ok=True)
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv")
    refs = fm[(fm.model == MODEL) & (fm.statepoint == SP) & (fm.step > 0)].frame.to_numpy()[:2]
    L = read_lines(MD / "data" / "traj.xyzf.gz"); off = frame_offsets(L)
    rows = []
    with open(W / "traj.xyzf", "w") as w:
        for f in refs:                                              # reference frames for the descriptor check
            cell, x = frame(L, off[f])
            w.write(f"{len(x)}\nNON_ORTHO {cell[0,0]:.8f} 0 0 0 {cell[1,1]:.8f} 0 0 0 {cell[2,2]:.8f} 0 0 0 0 0 0 0\n")
            np.savetxt(w, np.c_[x, np.zeros_like(x)], fmt="N %.6f %.6f %.6f %.1f %.1f %.1f")
            rows.append(dict(run=f"ref_frame_{f}", seed=-1, t=np.nan, t_e=np.nan, rmin=np.nan, md_frame=int(f)))
        for d in sorted(p for p in RUNS.glob(f"base__s*__{SP}") if (p / "traj.lammpstrj").exists()):
            seed = int(re.search(r"__s(\d+)__", d.name).group(1))
            t, ec = econs(d / "log.lammps")
            fr = list(dump_frames(d / "traj.lammpstrj"))
            rm = np.array([rmin(b, x) for b, x in fr]); tf = np.arange(len(fr)) * DT_DUMP
            cand = [t[np.argmax(np.abs(ec - ec[0]) > EVENT_MEV)] if (np.abs(ec - ec[0]) > EVENT_MEV).any() else np.inf,
                    tf[np.argmax(rm < R_IN)] if (rm < R_IN).any() else np.inf]
            t_e = min(cand)
            for j in range(0, len(fr), STRIDE):
                if tf[j] >= t_e:
                    break
                box, x = fr[j]; ln = box[:, 1] - box[:, 0]; x = (x - box[:, 0]) % ln
                w.write(f"{len(x)}\nNON_ORTHO {ln[0]:.8f} 0 0 0 {ln[1]:.8f} 0 0 0 {ln[2]:.8f} 0 0 0 0 0 0 0\n")
                np.savetxt(w, np.c_[x, np.zeros_like(x)], fmt="N %.6f %.6f %.6f %.1f %.1f %.1f")
                rows.append(dict(run=d.name, seed=seed, t=tf[j], t_e=t_e, rmin=rm[j], md_frame=-1))
    pd.DataFrame(rows).to_csv(W / "frames.csv", index=False)
    df = pd.DataFrame(rows[2:])
    print(f"{len(rows)} frames from {df.run.nunique()} runs; {np.isfinite(df.groupby('run').t_e.first()).sum()} with an event")


def descriptors():
    n = len(pd.read_csv(W / "frames.csv"))
    fm = (RETRO / "fm_setup.in").read_text()
    fm = re.sub(r"(# TRJFILE #\s*\n)\s*MULTI traj_list.dat", r"\1        traj.xyzf", fm)
    fm = re.sub(r"(# NFRAMES #\s*\n)\s*\d+", rf"\g<1>        {n}", fm)
    (W / "fm_setup.in").write_text(fm)
    subprocess.run(mpirun() + [tool("CHIMES_LSQ"), "fm_setup.in"], cwd=W, check=True,
                   stdout=open(W / "fm_setup.log", "w"), stderr=subprocess.STDOUT)
    A = pd.read_csv(W / "A.txt", sep=r"\s+", header=None, dtype=np.float64).to_numpy()
    assert A.shape == (n * NAT * 3, E.RANK), A.shape
    np.save(W / "A.npy", A)
    # check: the reference frames reproduce the MD-audit descriptors
    fr = pd.read_csv(W / "frames.csv"); Am = np.load(MD / "work" / "A.npy", mmap_mode="r")
    start = np.r_[0, np.cumsum(pd.read_csv(MD / "data" / "frame_manifest.csv").n_atoms.to_numpy())]
    for i, f in enumerate(fr.md_frame[:2]):
        ref = np.asarray(Am[3 * start[f]:3 * start[f + 1]]); new = A[i * NAT * 3:(i + 1) * NAT * 3]
        dev = np.abs(new - ref).max() / np.abs(ref).max()
        print(f"reference frame {f}: max relative deviation {dev:.2e}")
        assert dev < 1e-5, "descriptors do not reproduce work/A.npy"


def basis_rows(A_ref, inv):
    B = np.abs(A_ref @ inv)
    return np.where((np.abs(B.max(1) - 1) < 1e-8) & (np.abs(B.sum(1) - 1) < 1e-6))[0]


def gamma():
    fr = pd.read_csv(W / "frames.csv"); A_new = np.load(W / "A.npy")
    keep = fr.md_frame.to_numpy() < 0                          # drop the two reference frames
    A_new = A_new.reshape(len(fr), NAT * 3, -1)[keep].reshape(-1, E.RANK); fr = fr[keep].reset_index(drop=True)
    man = list(csv.DictReader(open(RW / "atom_manifest.csv")))
    kind = np.array([r["kind"] for r in man]); A = np.load(RW / "A_atomic.npy")
    lab = np.full(len(kind), -1); dft = np.where(kind == "dft")[0]; lab[dft] = np.load(RW / "labels_dft.npy")
    X = A.reshape(len(kind), -1); sc = StandardScaler().fit(X[dft])
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance").fit(sc.transform(X[dft]), lab[dft])
    cl = knn.predict(sc.transform(A_new.reshape(-1, E.RANK * 3)))
    by_frame = {}
    for r in man:
        by_frame.setdefault((r["source_file"], int(r["frame_idx_in_file"])), []).append(int(r["atom_row"]))
    meta = json.load(open(MD / "models" / MODEL / "train_meta.json"))
    ref = np.array(sorted(a for f in meta["files"] for i in f["indices"] for a in by_frame[f["source"], i]))
    A_ref = A[E.atom_rows_from_indices(ref)]; row_lab = np.repeat(lab[ref], 3)
    gb = np.zeros((N_SOL, len(cl))); gc = np.zeros_like(gb); sw = np.zeros_like(gb)
    for t in range(N_SOL):
        rng = np.random.default_rng(E.SEED0 + t)               # same draw order as 03_gamma.py
        inv_b = E.random_restart_maxvol(A_ref, rng)
        gb[t] = E.gamma_for_rows(A_new, inv_b)
        brow = row_lab[basis_rows(A_ref, inv_b)]
        under = [c for c in range(6) if (brow == c).mean() / (row_lab == c).mean() < 1]
        for c in range(6):
            m = np.where(cl == c)[0]
            if m.size == 0:
                continue
            inv = E.random_restart_maxvol(A[E.atom_rows_from_indices(ref[lab[ref] == c])], rng)
            gc[t, m] = E.gamma_for_rows(A_new[E.atom_rows_from_indices(m)], inv)
        sw[t] = np.where(np.isin(cl, under), gc[t], gb[t])
        print(f"solution {t}: under-represented {under}", flush=True)
    per = lambda g: g.mean(0).reshape(len(fr), NAT).max(1)
    fr["gb_max"], fr["gc_max"], fr["sw_max"] = per(gb), per(gc), per(sw)
    fr["event"] = np.isfinite(fr.t_e)
    fr.to_csv(MD / "results" / "early_warning_frames.csv", index=False)


def auroc(pos, neg):
    x = np.r_[pos, neg]; r = np.empty(len(x)); r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (r[:len(pos)].mean() - (len(pos) + 1) / 2) / len(neg)


def analysis():
    fr = pd.read_csv(MD / "results" / "early_warning_frames.csv")
    fr["tte"] = fr.t_e - fr.t                                  # time to event (inf for runs without an event)
    runs = fr.run.unique(); rng = np.random.default_rng(0)
    groups = {r: g for r, g in fr.groupby("run")}
    hz = []
    for D in DELTAS:
        for s in SCORES:
            def score(sub):
                pos = sub[sub.tte <= D][s].to_numpy(); neg = sub[sub.tte > D][s].to_numpy()
                return auroc(pos, neg) if len(pos) and len(neg) else np.nan
            bs = [score(pd.concat([groups[r] for r in rng.choice(runs, len(runs))])) for _ in range(N_BOOT)]
            hz.append(dict(window_ps=D, score=s, auroc=score(fr), lo95=np.nanpercentile(bs, 2.5),
                           hi95=np.nanpercentile(bs, 97.5), n_pos=int((fr.tte <= D).sum()), n_events=int(fr.groupby("run").event.first().sum())))
    al = []
    stable = fr[~fr.event]; ev = fr[fr.event]
    T_stable = stable.groupby("run").t.max().sum()
    for q in (0.95, 0.99):
        for s in SCORES:
            thr = stable[s].quantile(q)
            hit = ev.groupby("run").apply(lambda g: bool((g[(g.tte <= 1.0)][s] > thr).any()), include_groups=False)
            lead = ev.groupby("run").apply(lambda g: g[g[s] > thr].tte.max() if (g[s] > thr).any() else np.nan,
                                           include_groups=False)
            fa = stable.groupby("run").apply(lambda g: int((np.diff(np.r_[0, (g[s] > thr).astype(int)]) == 1).sum()),
                                             include_groups=False).sum()
            al.append(dict(quantile=q, score=s, threshold=thr, events_alarmed_last_1ps=int(hit.sum()), n_events=len(hit),
                           median_first_alarm_before_event_ps=float(np.nanmedian(lead)),
                           false_alarms_per_ps=fa / T_stable))
    bins = np.arange(0, 3.01, 0.2)
    prof = []
    for s in SCORES:
        lo, hi = stable[s].quantile([0.5, 0.95])
        for a, b in zip(bins[:-1], bins[1:]):
            v = ev[(ev.tte > a) & (ev.tte <= b)][s]
            prof.append(dict(score=s, tte_lo=a, tte_hi=b, n=len(v), mean=v.mean(), stable_median=lo, stable_q95=hi))
    R = MD / "results"
    for name, rows in (("hazard", hz), ("alarm", al), ("profile", prof)):
        pd.DataFrame(rows).to_csv(R / f"early_warning_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(hz).round(3).to_string(index=False)); print(pd.DataFrame(al).round(3).to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in (("prep", prep), ("descriptors", descriptors), ("gamma", gamma), ("analysis", analysis)):
        if stage in ("all", name):
            print(f"== {name}", flush=True); fn()
