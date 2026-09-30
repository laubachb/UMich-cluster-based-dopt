#!/usr/bin/env python3
"""Force RMSE of each calibration run (work/dft_calib/*/OUTCAR) against the dataset forces -> results/dft_calibration.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import MD
from common.xyzf import outcar_forces, outcar_done


def main():
    rows = []
    for d in sorted((MD / "work" / "dft_calib").glob("*/")):
        if not outcar_done(d / "OUTCAR"):
            continue
        F, R = outcar_forces(d / "OUTCAR"), np.load(d / "F_ref.npy")
        sp, fr, setting = d.name.split("_", 2)[0] + "_" + d.name.split("_", 2)[1], d.name.split("_")[2], "_".join(d.name.split("_")[3:])
        rows.append(dict(run=d.name, statepoint=sp, setting=setting, n_atoms=len(F), force_rmse_eV_A=np.sqrt(np.mean((F - R) ** 2)),
                         max_abs_dev=np.abs(F - R).max(), ref_force_rms=np.sqrt(np.mean(R ** 2))))
    df = pd.DataFrame(rows); df.to_csv(MD / "results" / "dft_calibration.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
