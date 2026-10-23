"""Reference numerical solvers. These produce ground truth and the classical baselines."""
from nsml.solvers.burgers import BurgersFV, BurgersSpectral, coarse_grain_1d
from nsml.solvers.ns2d import NS2DSpectral, kolmogorov_forcing, resample_spectral

__all__ = [
    "BurgersFV",
    "BurgersSpectral",
    "coarse_grain_1d",
    "NS2DSpectral",
    "kolmogorov_forcing",
    "resample_spectral",
]
