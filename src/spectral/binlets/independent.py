import numpy as np
import xarray
from binlets import binlets

from ..unmixing import _weighted_least_squares


def binlets_poisson(
    data: xarray.DataArray,
    /,
    *,
    sigma: float,
    levels: int | None = None,
    dim: str | None,
    independent: bool,
):
    if dim is None:
        dim = "dummy"
        if dim in data.dims:
            raise RuntimeError("data already has a dummy dimension")

        return binlets_poisson(
            data.expand_dims(dim),
            sigma=sigma,
            levels=levels,
            dim=dim,
            independent=independent,
        ).isel({dim: 0})

    def test(x: np.ndarray, y: np.ndarray):
        mask = (x - y) ** 2 < sigma**2 * (x + y)
        if not independent:
            mask = mask.all(0)
        return mask

    dims = set(data.dims)
    try:
        dims.remove(dim)
    except KeyError:
        raise ValueError("dim is not a dimension of data")
    dims = [dim, *dims]
    data = data.transpose(*dims)
    denoised = binlets(data.values, levels=levels, linear=True, test=test)
    return xarray.DataArray(denoised, dims=dims, coords=data.coords)


def binlets_independent_components(
    A: xarray.DataArray,
    b: xarray.DataArray,
    b_var: xarray.DataArray,
    /,
    *,
    sigma: float,
    levels: int | None = None,
):
    A_shape = dict(zip(A.dims, A.shape))
    b_shape = dict(zip(b.dims, b.shape))
    (common_dim,) = A_shape.keys() & b_shape.keys()
    del A_shape[common_dim]
    del b_shape[common_dim]

    A = A.transpose(common_dim, *A_shape.keys())
    b = b.transpose(*b_shape.keys(), common_dim)
    b_var = b_var.transpose(*b_shape.keys(), common_dim)
    p, cov = _weighted_least_squares(
        A.values,
        b.values[..., None],
        b_var.values[..., None],
    )

    mean_and_var_shape = A_shape | b_shape
    for k in A_shape.keys():
        mean_and_var_shape[k] *= 2
    mean_and_var = np.empty(tuple(mean_and_var_shape.values()))
    for i in range(cov.shape[-1]):
        mean_and_var[2 * i] = p[..., i, 0]
        mean_and_var[2 * i + 1] = cov[..., i, i]

    def test(x: np.ndarray, y: np.ndarray):
        d = x[0::2] - y[0::2]
        v = x[1::2] + y[1::2]
        mask = np.empty(x.shape, dtype=bool)
        mask[0::2] = mask[1::2] = d**2 < sigma**2 * v
        return mask

    mean_and_var = binlets(mean_and_var, levels=levels, linear=True, test=test)
    coords = {**A.coords, **b.coords}
    try:
        del coords[common_dim]
    except KeyError:
        pass
    return xarray.DataArray(
        mean_and_var[::2],
        coords=coords,
        dims=list(mean_and_var_shape.keys()),
    )


def _binlets_variance(
    data: np.ndarray,
    variance: np.ndarray,
    *,
    sigma: float,
    independent: bool,
    levels: int | None = None,
):
    def test(x: np.ndarray, y: np.ndarray):
        d = x[0::2] - y[0::2]
        v = x[1::2] + y[1::2]
        mask = np.empty(x.shape, dtype=bool)
        m = d**2 < sigma**2 * v
        if not independent:
            m = m.all(0)
        mask[0::2] = mask[1::2] = m
        return mask

    mean_and_var = np.empty((2 * data.shape[0], *data.shape[1:]), dtype=data.dtype)
    mean_and_var[0::2] = data
    mean_and_var[1::2] = variance
    mean_and_var = binlets(mean_and_var, levels=levels, test=test, linear=True)
    return mean_and_var[0::2], mean_and_var[1::2]


def binlets_variance(
    data: xarray.DataArray,
    variance: xarray.DataArray,
    /,
    *,
    sigma: float,
    levels: int | None = None,
    dim: str | None,
    independent: bool,
):
    if dim is None:
        dim = "dummy"
        if dim in data.dims:
            raise RuntimeError("data already has a dummy dimension")

        return binlets_variance(
            data.expand_dims(dim),
            variance.expand_dims(dim),
            sigma=sigma,
            levels=levels,
            dim=dim,
            independent=independent,
        ).isel({dim: 0})

    dims = set(data.dims)
    try:
        dims.remove(dim)
    except KeyError:
        raise ValueError("dim is not a dimension of data")
    dims = [dim, *dims]
    denoised, _ = _binlets_variance(
        data.transpose(*dims).values,
        variance.transpose(*dims).values,
        sigma=sigma,
        levels=levels,
        independent=independent,
    )
    return xarray.DataArray(denoised, dims=dims, coords=data.coords)
