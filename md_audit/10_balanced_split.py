#!/usr/bin/env python3
"""Step 10: a pool/test split of the 900 DFT-labelled MD frames with matched gamma distributions (balanced Fig. 6B).

The seed split of 08_al_step.py (pool = seeds 1-2, test = seed 3) leaves the gamma distributions of pool and test
different at 300 K (KS ~0.5 on frame-max gamma), because seed 1 runs differently from seeds 2-3. Here each run is cut
into 1 ps blocks (10 consecutive frames), and per state point 10 of the 30 blocks are held out as test (200 pool /
100 test frames, as before). The held-out blocks minimize the largest two-sample KS statistic between pool and test
over gamma-only features; the force error is never used, so the split is not tuned to the outcome:
  frame level : max gamma_bulk, max gamma_cluster, fraction of atoms with gamma_bulk > 1 and with gamma_cluster > 1
  atom level  : gamma_bulk and gamma_cluster of every atom
(gamma = mean over the 10 MaxVol solutions of 03_gamma.py). Every seed keeps at least 2 test blocks.
Search: random starts + single-block swaps (hill climbing) on binned CDFs (100 quantile bins per feature, summed
per block, so a candidate split costs a few array sums); the reported KS values are exact.
Writes results/balanced_split.csv (frame, statepoint, seed, block, role) and results/balanced_split_diagnostics.csv.
usage: 10_balanced_split.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MD

MODEL = "shared_t0_dft_train"
BLOCK = 10            # frames per block (1 ps at 100 fs per frame)
N_TEST = 10           # test blocks per state point (of 30)
MIN_PER_SEED = 2
N_STARTS, N_SWEEPS = 200, 200
FRAME_FEATS = ["gb_max", "gc_max", "gb_frac", "gc_frac"]


def ks(a, b):
    return ks_2samp(a, b).statistic


def block_hists(fr, at, blocks, nbins=100):
    """H[feature, block, bin]: counts of each gamma feature per block on quantile bins shared by pool and test."""
    H = []
    for tab, feats in ((fr, FRAME_FEATS), (at, ["gamma_bulk", "gamma_cluster"])):
        for g in feats:
            edges = np.unique(np.quantile(tab[g], np.linspace(0, 1, nbins + 1)))
            idx = np.clip(np.searchsorted(edges, tab[g].to_numpy(), side="right") - 1, 0, len(edges) - 2)
            h = np.zeros((len(blocks), nbins))
            np.add.at(h, (tab.block.map({b: i for i, b in enumerate(blocks)}).to_numpy(), idx), 1)
            H.append(h)
    return np.array(H)


def objective(mask, H):
    """largest binned KS over features; mask[b] = block b is test."""
    t, p = H[:, mask].sum(1), H[:, ~mask].sum(1)
    d = np.abs(np.cumsum(t, 1) / t.sum(1, keepdims=True) - np.cumsum(p, 1) / p.sum(1, keepdims=True)).max(1)
    return d.max(), d


def valid(mask, seeds):
    c = np.bincount(seeds[mask], minlength=seeds.max() + 1)[np.unique(seeds)]
    return c.min() >= MIN_PER_SEED


def main():
    df = pd.read_csv(MD / "work" / f"atom_errors_{MODEL}.csv")
    fm = pd.read_csv(MD / "data" / "frame_manifest.csv").set_index("frame")
    df["block"] = df.statepoint + "__s" + df.seed.astype(str) + "__b" + ((df.step // 500 - 1) // BLOCK).astype(str)
    fr = df.groupby(["statepoint", "seed", "block", "frame"]).agg(
        gb_max=("gamma_bulk", "max"), gc_max=("gamma_cluster", "max"),
        gb_frac=("gamma_bulk", lambda g: (g > 1).mean()), gc_frac=("gamma_cluster", lambda g: (g > 1).mean()),
        rmse=("err", lambda e: np.sqrt(np.mean(e ** 2)))).reset_index()
    rng = np.random.default_rng(0)
    rows, diag = [], []
    for sp, f in fr.groupby("statepoint"):
        a = df[df.statepoint == sp][["block", "gamma_bulk", "gamma_cluster"]]
        blocks = sorted(f.block.unique())
        seeds = f.groupby("block").seed.first().reindex(blocks).to_numpy()
        assert all(f.groupby("block").size() == BLOCK) and len(blocks) == 30, sp
        H = block_hists(f, a, blocks)
        best, best_s = None, (np.inf, None)
        for _ in range(N_STARTS):
            mask = np.zeros(len(blocks), bool)
            while not valid(mask, seeds):
                mask[:] = False; mask[rng.choice(len(blocks), N_TEST, replace=False)] = True
            cur_s = objective(mask, H)
            for _ in range(N_SWEEPS):
                improved = False
                for i in rng.permutation(np.where(mask)[0]):
                    for j in rng.permutation(np.where(~mask)[0]):
                        cand = mask.copy(); cand[i], cand[j] = False, True
                        if not valid(cand, seeds): continue
                        s = objective(cand, H)
                        if s[0] < cur_s[0] - 1e-12:
                            mask, cur_s, improved = cand, s, True; break
                    if improved: break
                if not improved: break
            if cur_s[0] < best_s[0]:
                best, best_s = mask.copy(), cur_s
        test_blocks = [b for b, m in zip(blocks, best) if m]
        role = np.where(f.block.isin(test_blocks), "test", "pool")
        rows.append(pd.DataFrame(dict(frame=f.frame, statepoint=sp, seed=f.seed, block=f.block, role=role)))
        # diagnostics: balanced vs the original seed split, gamma features and (unused) error
        seed_test = (f.seed == 3).to_numpy(); bal_test = role == "test"
        for split, tm in (("seed", seed_test), ("balanced", bal_test)):
            am = a.block.isin(f.block[tm]).to_numpy()
            d = dict(statepoint=sp, split=split, n_pool=int((~tm).sum()), n_test=int(tm.sum()))
            for k in FRAME_FEATS + ["rmse"]:
                d[f"ks_{k}"] = ks(f[k].to_numpy()[~tm], f[k].to_numpy()[tm])
                d[f"pool_med_{k}"], d[f"test_med_{k}"] = f[k].to_numpy()[~tm].mean(), f[k].to_numpy()[tm].mean()
            for g in ("gamma_bulk", "gamma_cluster"):
                d[f"ks_atom_{g}"] = ks(a[g].to_numpy()[~am], a[g].to_numpy()[am])
            d["test_blocks_per_seed"] = f[tm].groupby("seed").block.nunique().to_dict()
            diag.append(d)
        print(f"{sp}: binned max KS {best_s[0]:.3f}  ({', '.join(f'{x:.2f}' for x in best_s[1])})", flush=True)
    out = pd.concat(rows).sort_values("frame")
    out.to_csv(MD / "results" / "balanced_split.csv", index=False)
    dg = pd.DataFrame(diag); dg.to_csv(MD / "results" / "balanced_split_diagnostics.csv", index=False)
    pd.set_option("display.width", 250)
    print(dg[["statepoint", "split"] + [c for c in dg.columns if c.startswith("ks_")]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
