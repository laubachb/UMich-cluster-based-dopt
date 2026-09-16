# Cluster-based D-optimality (Nitrogen ChIMES)

Retrospective study of **local (cluster) vs bulk MaxVol γ** for active-learning
candidate triage on the published ChIMES nitrogen dataset
([DOI 10.1063/5.0157238](https://doi.org/10.1063/5.0157238)).

## Layout

| Path | Role |
|------|------|
| `nitrogen/` | Raw ChIMES-N xyzf corpus |
| `descriptors/` | Force-only design matrix, EC spectral clustering, γ maps |
| `analysis/` | ALC 2–4 test-bed scripts (Mode II → selection → sklearn → Mode I stubs) |
| `results/` | Analysis outputs (splits, errors, selections, sklearn, refits) |
| `docs/` | Paper / design notes |

## Quick start (Mode II chain)

```bash
cd /work2/09982/blaubach/stampede3/cluster-based-dopt
chmod +x analysis/run_mode2_chain.sh
./analysis/run_mode2_chain.sh
```

Or step-by-step: see [`analysis/README.md`](analysis/README.md).

## Descriptor / γ pipeline (already run)

See [`descriptors/README.md`](descriptors/README.md). Key products:

- `descriptors/A_atomic.npy` — component force design matrix
- `descriptors/clustering_ecc_full_spectral/` — DFT spectral labels (k=6)
- `descriptors/cluster_dopt_gamma/` — γ_cluster, γ_bulk, cross-γ, figures

## External deps

- `chimes_lsq` build: `/work2/09982/blaubach/stampede3/chimes_lsq-myLLfork/build/`
- `gamma_dopt.py`: `aug2026-water-dopt-weighting/chimes_lsq-myLLfork/src/`
