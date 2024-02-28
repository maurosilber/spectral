from typing import Iterable

import numpy as np
import xarray
from dask_checkpoint import task

from . import Dim, mixing, unmixing
from .denoisers import Denoiser


@task(save=True)
def mean_root_squared_error(
    components: xarray.DataArray,
    denoisers: Iterable[Denoiser],
    /,
    *,
    seed: int,
    sigma: float,
    calibration: xarray.DataArray,
):
    spectrum = mixing.matmul(calibration, components)
    cov = unmixing.covariance_propagation(
        calibration.values.T,
        spectrum.values[..., None],
    )
    dims = [*(d for d in components.dims if d != Dim.components), Dim.components]
    var = xarray.DataArray(
        cov[..., *np.diag_indices(cov.shape[-1], 2)],
        dims=dims,
        coords={k: components.coords[k] for k in dims},
    )
    sampled_spectrum = xarray.apply_ufunc(np.random.default_rng(seed).poisson, spectrum)
    mrse = {}
    for d in denoisers:
        denoised_components = d(sampled_spectrum, sigma=sigma, calibration=calibration)
        residuals = denoised_components - components
        mrse[d.__name__] = (residuals**2 / var).mean(["x", "y"]) ** 0.5
    return xarray.Dataset(mrse)
