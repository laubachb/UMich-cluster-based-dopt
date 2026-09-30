# Data: published ChIMES nitrogen dataset

`ChIMES-N_dataset/` is the training and active-learning data of the published ChIMES nitrogen model
(Lindsey et al., [doi:10.1063/5.0157238](https://doi.org/10.1063/5.0157238)), included unchanged (6 MB).
See `ChIMES-N_dataset/README` for the original provenance and `Nitrogen_data_defs.xlsx` for the cycle definitions.

## Files used by this repository

| File | Frames | Atoms/frame | Role here |
|---|---|---|---|
| `DFT.300K_1.0gcc.xyzf` | 20 | 64 | DFT-MD reference (molecular N2) |
| `DFT.2000K_1.0gcc.xyzf` | 20 | 64 | DFT-MD reference (molecular N2) |
| `DFT.5000K_2.0gcc.xyzf` | 20 | 64 | DFT-MD reference (partly dissociated) |
| `DFT.6000K_2.5gcc.xyzf` | 20 | 108 | DFT-MD reference |
| `DFT.7000K_3.1gcc.xyzf` | 20 | 108 | DFT-MD reference |
| `DFT.8000K_4.5gcc.xyzf` | 20 | 120 | DFT-MD reference |
| `ChIMES.fromALC-1.forALC-2.OUTCAR.xyzf` | 128 | mixed | active-learning candidates, pool "alc2" |
| `ChIMES.fromALC-2.forALC-3.OUTCAR.xyzf` | 160 | mixed | pool "alc3" |
| `ChIMES.fromALC-3.forALC-4.OUTCAR.xyzf` | 121 | mixed | pool "alc4" |
| `ChIMES.fromALC-4.forALC-5.OUTCAR.xyzf` | 117 | mixed | pool "alc5" |

The DFT-MD reference set has 120 frames (10,560 atoms). The candidates are 526 frames (47,048 atoms), all with DFT
forces from the original campaign. `training_data.xyzf` (the published model's full training set) is not used.

## Format

Each frame has an atom-count line, then a comment line with the 9 cell-vector components (Å), the stress tensor
(GPa: sxx syy szz sxy sxz syz) and the total energy (kcal/mol), then one line per atom: element, x y z (Å), and
Fx Fy Fz in **Hartree/bohr** (× 51.422 for eV/Å).

## DFT protocol

The dataset README says only "allPBE". Recomputing four frames (300–8000 K) showed that the stored forces are
PBE + D2 at a 1000 eV cutoff, with Fermi smearing at the MD temperature and Γ-point sampling: force RMSE
0.005–0.008 eV/Å (`md_audit/results/dft_calibration.csv`). New DFT labels in `md_audit/` use this protocol.
