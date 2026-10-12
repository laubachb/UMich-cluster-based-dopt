#!/usr/bin/env python3
"""Step 16: test T of CRITERIA.md: the manuscript's partition built on the topological (tda) or control (ctrl)
representation instead of the force rows. Arm A exactly as 02_shift.py (split, error model, bases, 5 MaxVol
solutions); per scenario the assigned grade for the partitions cluster (force rows; checked against
results/shift_NAME*.csv), tda, ctrl and regime. Representations from work/NAME/{tda,ctrl}.npy (step 15), indexed by
the atoms core.load picked.
Writes results/tda_NAME[TAG].csv. --score: T1-T3 from results/tda_*.csv -> results/tda_criteria.csv.
usage: 16_tda_partition.py NAME [--jobs N] [--only REGIME ...] [--tag TAG]  |  16_tda_partition.py --score
"""
import argparse, time, warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core as C

MIN_FRAMES = 60
G = {}


def split(R, f, d):
    D, tab = G["D"], G["tab"]
    base = tab[~tab.al]
    regimes = sorted(base.regime.unique())
    rng = np.random.default_rng(10_000 + 1000 * d + 10 * regimes.index(R) + int(round(100 * f)))
    tr = []
    for reg, g in base.groupby("regime"):
        fr = g.frame.to_numpy()
        n = max(2, int(round(f * len(fr)))) if reg == R else int(round(0.75 * len(fr)))
        tr.append(rng.choice(fr, n, replace=False))
    tr = np.concatenate(tr)
    train = np.isin(D["frame"], tr)
    cand = (D["regime"] == R) & ~train & np.isin(D["frame"], base.frame)
    return train, cand


def partition_on(feat, D, train):
    """manuscript clustering (per element) on an arbitrary per-atom representation."""
    el = D["element"]; code = pd.Categorical(el).codes; lab = np.full(len(el), -1)
    for e in np.unique(code):
        me = code == e
        lab[me] = e * C.K_CLUSTERS + C.spectral_labels(feat[train & me], feat[me])
    return lab


def scenario(job):
    R, f, d = job
    warnings.filterwarnings("ignore")
    t0 = time.time(); D = G["D"]; nsol = G["nsol"]
    train, cand = split(R, f, d)
    err = C.error_model(D, train, cand)
    parts = C.partitions(D, train, G["multi"])                       # bulk, (element), regime, cluster
    parts = {"cluster": parts["cluster"], "regime": parts["regime"], "tda": partition_on(G["tda"], D, train), "ctrl": partition_on(G["ctrl"], D, train)}
    rows = []
    for pname, lab in parts.items():
        Gr = C.grade_all(D, train, cand, lab, nsol)[0]
        ok = ~np.isnan(Gr).any(0)
        ms = [C.metrics(Gr[t][ok], err[ok]) for t in range(nsol)]
        rows.append(dict(name=G["name"], regime=R, frac=f, draw=d, partition=pname, n_cand=int(cand.sum()), n_graded=int(ok.sum()),
                         n_groups_cand=int(len(np.unique(lab[cand]))), **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
    print(f"{G['name']} {R} f={f} d={d}: {time.time() - t0:.0f} s", flush=True)
    return rows


def boot(x, n=10_000, seed=0):
    x = np.asarray(x, float); m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def score():
    U = {}
    for name in ("si_gap18", "sio2", "water"):
        files = list(C.RES.glob(f"tda_{name}*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["regime", "frac", "draw", "partition"])
        a = df.pivot_table(index=["regime", "frac", "draw"], columns="partition", values="auroc")
        sh = pd.read_csv(C.RES / f"shift_{name}.csv"); rep = sh[(sh.element == "all") & (sh.partition == "bulk")].set_index(["regime", "frac", "draw"]).rep_R
        m = a.join(rep); U[name] = m[(m.index.get_level_values("frac") <= 0.05) & (m.rep_R < 0.5)]
        pd.set_option("display.width", 250)
        print(f"\n===== {name}: mean AUROC, C3 scenarios\n" + U[name].drop(columns="rep_R").mean().round(3).to_string())
    crit = []
    net = pd.concat([U[n] for n in ("si_gap18", "sio2") if n in U])
    if len(net):
        m1, l1, h1 = boot(net.tda - net.cluster); mr = (net.regime - net.cluster).mean(); m3, l3, h3 = boot(net.tda - net.ctrl)
        crit.append(dict(criterion="T1 (Si+SiO2) tda vs cluster", value=m1, passed=bool(l1 > 0 and m1 >= 0.05 and m1 >= 0.5 * mr),
                         detail=f"{m1:+.3f} [{l1:+.3f}, {h1:+.3f}]; regime gain {mr:+.3f}; n={len(net)}"))
        crit.append(dict(criterion="T3 (Si+SiO2) tda vs ctrl", value=m3, passed=bool(l3 > 0), detail=f"{m3:+.3f} [{l3:+.3f}, {h3:+.3f}]"))
    if "water" in U:
        w = U["water"]; d = (w.tda - w.cluster).mean()
        crit.append(dict(criterion="T2 (water) no loss", value=d, passed=bool(d >= -0.02), detail=f"tda - cluster {d:+.3f}, n={len(w)}"))
    cr = pd.DataFrame(crit); cr.to_csv(C.RES / "tda_criteria.csv", index=False); print("\n===== criteria\n" + cr.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?"); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--only", nargs="*")
    ap.add_argument("--tag", default=""); ap.add_argument("--score", action="store_true"); ap.add_argument("--cap", type=int, default=32)
    a = ap.parse_args()
    if a.score:
        return score()
    D, tab = C.load(a.name, a.cap)
    tda = np.load(C.WORK / a.name / "tda.npy")[D["atoms"]].astype(float); ctrl = np.load(C.WORK / a.name / "ctrl.npy")[D["atoms"]].astype(float)
    G.update(D=D, tab=tab, name=a.name, nsol=C.NSOL, multi=len(np.unique(D["element"])) > 1, tda=tda, ctrl=ctrl)
    base = tab[~tab.al]; cnt = base.regime.value_counts()
    regimes = [r for r in sorted(cnt.index) if cnt[r] >= MIN_FRAMES and (not a.only or r in a.only)]
    jobs = [(R, f, d) for R in regimes for f in (0.02, 0.05, 0.10, 0.25) for d in (0, 1, 2)]
    print(f"{a.name}: {len(D['element'])} atoms, tda {tda.shape}, regimes {regimes}, {len(jobs)} scenarios", flush=True)
    if a.jobs > 1:
        with Pool(a.jobs) as p:
            out = p.map(scenario, jobs, chunksize=1)
    else:
        out = [scenario(j) for j in jobs]
    df = pd.DataFrame([r for rs in out for r in rs])
    shift = pd.concat([pd.read_csv(f) for f in C.RES.glob(f"shift_{a.name}*.csv")])
    s = shift[(shift.element == "all") & (shift.partition == "cluster")].drop_duplicates(["regime", "frac", "draw"]).set_index(["regime", "frac", "draw"]).auroc
    chk = df[df.partition == "cluster"].set_index(["regime", "frac", "draw"]).auroc
    print(f"force-row cluster AUROC vs arm A: max |diff| = {(chk - s.reindex(chk.index)).abs().max():.2e}")
    df.to_csv(C.RES / f"tda_{a.name}{a.tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.pivot_table(index=["regime", "frac"], columns="partition", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
