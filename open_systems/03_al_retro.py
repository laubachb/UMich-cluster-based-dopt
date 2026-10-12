#!/usr/bin/env python3
"""Step 3, arm B of CRITERIA.md (sio2 only): active-learning retrospective, the analogue of the manuscript's
retrospective study. Reference = the base (non-_AL) frames; candidates = the frames added during the original active
learning (config types *_int_AL, *_max_AL), grouped by config type and by parent regime. Error model, partitions and
grades are built from the reference; 5 MaxVol solutions. Candidates whose parent regime has no reference frames
(highpressure) cannot be graded by the regime partition and are reported with the other partitions only.
Writes results/al_<name>.csv (one row per candidate group, partition and element) and results/al_<name>_rep.csv
(bulk-basis representation of every reference regime, per element and overall).
usage: 03_al_retro.py sio2 [--cap 32] [--nsol 5]
"""
import argparse, warnings

import numpy as np
import pandas as pd

import core as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name", nargs="?", default="sio2")
    ap.add_argument("--cap", type=int, default=32); ap.add_argument("--nsol", type=int, default=C.NSOL)
    a = ap.parse_args(); warnings.filterwarnings("ignore")
    D, tab = C.load(a.name, a.cap)
    ct_of_frame = tab.set_index("frame").config_type
    config = ct_of_frame.loc[D["frame"]].to_numpy()
    train, cand = ~D["al"], D["al"].copy()
    multi = len(np.unique(D["element"])) > 1
    print(f"{a.name}: reference atoms {train.sum()}, candidate atoms {cand.sum()}, {D['n_cols']} columns", flush=True)
    err = C.error_model(D, train, cand)
    parts = C.partitions(D, train, multi)
    rows, rep_rows = [], []
    for pname, lab in parts.items():
        Gr, Out, ranks, piv = C.grade_all(D, train, cand, lab, a.nsol)
        if pname == "bulk":
            for e in ["all"] + (sorted(np.unique(D["element"])) if multi else []):
                grp = D["regime"] if e == "all" else np.where(D["element"] == e, D["regime"], "other_element")
                for reg, v in C.representation(piv, grp, train).items():
                    rep_rows.append(dict(name=a.name, element=e, regime=reg, representation=v))
        cc, pc = config[cand], D["regime"][cand]
        groups = [("config_type", g) for g in np.unique(cc)] + [("parent", g) for g in np.unique(pc)]
        for kind, g in groups:
            for e in ["all"] + (sorted(np.unique(D["element"])) if multi else []):
                m = (cc == g if kind == "config_type" else pc == g) & ((D["element"][cand] == e) if e != "all" else True)
                ok = m & ~np.isnan(Gr).any(0)
                if ok.sum() < 30:
                    continue
                ms = [C.metrics(Gr[t, ok], err[ok]) for t in range(a.nsol)]
                rows.append(dict(name=a.name, kind=kind, group=g, partition=pname, element=e, n_cand=int(m.sum()),
                                 n_graded=int(ok.sum()), outspan_median=float(np.median(Out[:, ok])),
                                 err_median=float(np.median(err[ok])),
                                 **{k: float(np.mean([x[k] for x in ms])) for k in ms[0]}))
        print(f"  {pname}: group ranks {ranks}", flush=True)
    C.RES.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(C.RES / f"al_{a.name}.csv", index=False)
    rep = pd.DataFrame(rep_rows); rep.to_csv(C.RES / f"al_{a.name}_rep.csv", index=False)
    pd.set_option("display.width", 250)
    print(rep.pivot_table(index="regime", columns="element", values="representation").round(2).to_string())
    print(df[(df.kind == "parent")].pivot_table(index=["group", "element"], columns="partition", values="auroc").round(3).to_string())


if __name__ == "__main__":
    main()
