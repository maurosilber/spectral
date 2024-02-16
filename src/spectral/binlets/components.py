import numpy as np
import scipy.stats
import xarray
from binlets import binlets

from ..unmixing import _weighted_least_squares


def binlets_components_transform(
    data: xarray.DataArray,
    /,
    *,
    sigma: float,
    levels: int | None = None,
    dim: str,
    calibration: xarray.DataArray,
):
    dims = set(data.dims)
    try:
        dims.remove(dim)
    except KeyError:
        raise ValueError("dim is not a dimension of data")
    dims = [dim, *dims]
    data = data.transpose(*dims)

    A = calibration.transpose(dim, *set(calibration.dims).difference((dim,))).values
    pvalue = scipy.stats.chi.cdf(sigma, df=1)
    threshold = scipy.stats.chi2.ppf(pvalue, df=A.shape[1])

    def test(x: np.ndarray, y: np.ndarray):
        x = np.moveaxis(x, 0, -1)[..., None]
        y = np.moveaxis(y, 0, -1)[..., None]
        px, covx = _weighted_least_squares(A, x, x)
        py, covy = _weighted_least_squares(A, y, y)
        diff = px - py
        cov = covx + covy
        distance = (np.moveaxis(diff, -2, -1) @ np.linalg.inv(cov) @ diff)[..., 0, 0]
        mask = distance < threshold
        return mask

    denoised = binlets(data.values, levels=levels, linear=True, test=test)
    return xarray.DataArray(denoised, dims=dims, coords=data.coords)
