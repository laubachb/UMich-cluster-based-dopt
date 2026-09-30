#!/usr/bin/env python3
"""Step 12: stability of the active-learning refits in MD (runs of 11_al_md_setup.py) -> results/al_md_stability*.csv.

Per run (log.lammps thermo every 10 fs; traj.lammpstrj every 20 fs):
  completed        run reached step 50,000 (else t_end and the LAMMPS error line)
  econs_drift      slope of a linear fit of econserve = etotal + ecouple against time, meV/atom/ps
                   (econserve is the conserved quantity of Langevin dynamics; drift = integration/model error)
  econs_maxdev     largest |econserve(t) - econserve(0)|, meV/atom
  T_mean, T_rel    mean temperature over the second half, and its ratio to the target
  pe_change        PE(last 1 ps) - PE(first 0.1 ps), meV/atom (how far the structure relaxes away from the start)
  rmin             smallest N-N distance over the trajectory (minimum image, orthorhombic box), Angstrom
  frac_inner       fraction of saved frames with any pair inside the ChIMES inner cutoff r_in = 0.86 A
  molecular        mean fraction of atoms with exactly one neighbour within 1.3 A (intact N2 units), second half
  centre_nn        mean distance from each N2 centre (mutual nearest-neighbour atom pairs) to the nearest other centre,
                   last 1 ps, Angstrom; DFT-MD at 300 K, 1.0 g/cm^3: 3.34 A (3.26-3.47 over its 20 frames)
  centre_cn        mean number of other N2 centres within 3.3 A, last 1 ps (DFT-MD 300 K: 0.5)
                   (packing metrics are meaningful for the molecular 300 K runs; at 5000 K molecules dissociate)
  event            econs_maxdev > 10 meV/atom or frac_inner > 0 (the two agree in 167 of 176 5000 K refit runs)
Summary per state point x strategy x K (mean, std, min/max over draws) -> results/al_md_stability_summary*.csv.
usage: 12_al_md_analysis.py [split]      (default balanced)
"""
import re
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD

KCAL_TO_MEV = 43.3641
N_STEPS, DT_FS = 50000, 0.2
R_IN, R_BOND = 0.86, 1.3
EVENT_MEV = 10.0      # "collapse event": |econserve - econserve(0)| > 10 meV/atom at any time, or any pair inside R_IN


def thermo(log):
    """thermo table of a LAMMPS log (first run block) and the first ERROR line, if any."""
    L = Path(log).read_text(errors="replace").splitlines()
    err = next((l.strip() for l in L if l.startswith("ERROR")), "")
    try:
        i = next(k for k, l in enumerate(L) if l.split()[:1] == ["Step"])
    except StopIteration:
        return None, err or "no thermo output"
    cols, rows = L[i].split(), []
    for l in L[i + 1:]:
        p = l.split()
        if len(p) != len(cols):
            break
        try:
            rows.append([float(x) for x in p])
        except ValueError:
            break
    return pd.DataFrame(rows, columns=cols), err


def centres(x, r, ln):
    """N2 centres (mutual nearest-neighbour atom pairs): mean nearest centre-centre distance, mean centres within 3.3 A."""
    nn = r.argmin(1); pairs = [(i, j) for i, j in enumerate(nn) if i < j and nn[j] == i]
    if len(pairs) < 2:
        return np.nan, np.nan
    c = np.array([x[i] + 0.5 * ((x[j] - x[i]) - ln * np.round((x[j] - x[i]) / ln)) for i, j in pairs])
    dc = c[:, None] - c[None]; dc -= ln * np.round(dc / ln)
    rc = np.sqrt((dc ** 2).sum(-1)); np.fill_diagonal(rc, np.inf)
    return rc.min(1).mean(), (rc < 3.3).sum(1).mean()


def dump_distances(path, last=50):
    """per saved frame: smallest pair distance and fraction of atoms with exactly one neighbour < R_BOND;
    N2-centre packing (centres()) averaged over the last `last` frames."""
    L = Path(path).read_text().splitlines()
    starts, i = [], 0
    while i < len(L):
        starts.append(i); i += 9 + int(L[i + 3])
    rmin, mol, pack = [], [], []
    for k, i in enumerate(starts):
        n = int(L[i + 3]); box = np.array([[float(v) for v in L[i + 5 + q].split()[:2]] for q in range(3)])
        x = np.array([l.split()[1:4] for l in L[i + 9:i + 9 + n]], float)
        ln = box[:, 1] - box[:, 0]
        d = x[:, None, :] - x[None, :, :]; d -= ln * np.round(d / ln)
        r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf)
        rmin.append(r.min()); mol.append(((r < R_BOND).sum(1) == 1).mean())
        if k >= len(starts) - last:
            pack.append(centres(x, r, ln))
    return np.array(rmin), np.array(mol), np.nanmean(np.array(pack), 0)


def parse_name(name):
    """<strategy>[__k<K>__d<draw>][__s<seed>]__<sp>; strategy may end in a temperature (random_5000K).
    Runs without __s<seed> used MD seed 1 (11_al_md_setup.py default)."""
    m = re.match(r"^(base|[a-z_]+?(?:_\d+K)?)(?:__k(\d+)__d(\d+))?(?:__s(\d+))?__(\d+K_[\d.]+gcc)$", name)
    strat, k, d, seed, sp = m.groups()
    return strat, int(k or 0), int(d or 0), int(seed or 1), sp


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    root = MD / "work" / "al_md" / split
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and "__" in p.name):
        strat, k, draw, seed, sp = parse_name(d.name)
        T0 = float(sp.split("K_")[0])
        r = dict(run=d.name, statepoint=sp, strategy=strat, k=k, draw=draw, seed=seed)
        if not (d / "log.lammps").exists():
            rows.append({**r, "completed": False, "error": "not run"}); continue
        th, err = thermo(d / "log.lammps")
        if th is None or th.empty:
            rows.append({**r, "completed": False, "error": err}); continue
        nat = 64
        t = th.Time.to_numpy() / 1000.0                                   # ps
        ec = th.Econserve.to_numpy() * KCAL_TO_MEV / nat                  # meV/atom
        half = t >= t[-1] / 2
        ok = th.Step.iloc[-1] >= N_STEPS and np.isfinite(ec).all()
        r.update(completed=bool(ok), error=err, t_end=t[-1],
                 econs_drift=np.polyfit(t, ec, 1)[0] if len(t) > 2 else np.nan,
                 econs_maxdev=np.abs(ec - ec[0]).max(),
                 T_mean=th.Temp[half].mean(), T_rel=th.Temp[half].mean() / T0,
                 pe_change=(th.PotEng[t >= t[-1] - 1].mean() - th.PotEng[t <= 0.1].mean()) * KCAL_TO_MEV / nat)
        if (d / "traj.lammpstrj").exists():
            rm, mol, (cnn, ccn) = dump_distances(d / "traj.lammpstrj")
            r.update(rmin=rm.min(), frac_inner=(rm < R_IN).mean(), molecular=mol[len(mol) // 2:].mean(),
                     centre_nn=cnn, centre_cn=ccn)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["statepoint", "strategy", "k", "draw"])
    df["event"] = (df.econs_maxdev > EVENT_MEV) | (df.frac_inner > 0)
    tag = "" if split == "balanced" else f"_{split}"
    df.to_csv(MD / "results" / f"al_md_stability{tag}.csv", index=False)
    num = ["econs_drift", "econs_maxdev", "T_rel", "pe_change", "rmin", "frac_inner", "molecular", "centre_nn",
           "centre_cn", "event"]
    g = df.groupby(["statepoint", "strategy", "k"])
    summ = g[num].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.insert(0, "n_completed", g.completed.sum()); summ.insert(0, "n_runs", g.size())
    summ.reset_index().to_csv(MD / "results" / f"al_md_stability_summary{tag}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(f"{df.completed.sum()} of {len(df)} runs completed")
    if (~df.completed).any():
        print(df[~df.completed][["run", "t_end", "error"]].to_string(index=False))
    print(summ[["n_runs", "n_completed"] + [f"{c}_mean" for c in num]].round(3).to_string())


if __name__ == "__main__":
    main()
