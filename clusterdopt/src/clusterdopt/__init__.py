"""clusterdopt: cluster-local D-optimal extrapolation grades (gamma_cluster) for linear models."""
from .maxvol import gamma_rows, gamma_samples, maxvol
from .model import ClusterDOpt, __version__

__all__ = ["ClusterDOpt", "maxvol", "gamma_rows", "gamma_samples", "__version__"]
