import numpy as np
import xarray


def lstsq(
    A: xarray.DataArray,
    b: xarray.DataArray,
) -> xarray.DataArray:
    if A.dims[0] in b.dims:
        common_dim, output_dim = A.dims
    else:
        common_dim, output_dim = A.dims[::-1]
    remaining_dims = [d for d in b.dims if d != common_dim]

    A = A.transpose(common_dim, output_dim).sortby(common_dim)
    b = b.transpose(common_dim, *remaining_dims).sortby(common_dim)
    assert (A.coords[common_dim] == b.coords[common_dim]).all()

    x = np.linalg.lstsq(
        A.values,
        b.values.reshape(A.values.shape[0], -1),
        rcond=None,
    )[0].reshape(A.values.shape[1], *b.values.shape[1:])
    coords = {**A.coords, **b.coords}
    try:
        del coords[common_dim]
    except KeyError:
        pass
    return xarray.DataArray(
        x,
        coords=coords,
        dims=[output_dim, *remaining_dims],
    )


def weighted_least_squares(
    A: xarray.DataArray,
    b: xarray.DataArray,
    var: xarray.DataArray,
):
    if A.dims[0] in b.dims:
        common_dim, output_dim = A.dims
    else:
        common_dim, output_dim = A.dims[::-1]
    remaining_dims = [d for d in b.dims if d != common_dim]

    A = A.transpose(common_dim, output_dim).sortby(common_dim)
    b = b.transpose(*remaining_dims, common_dim).sortby(common_dim)
    var = var.transpose(*remaining_dims, common_dim).sortby(common_dim)
    assert (A.coords[common_dim] == b.coords[common_dim]).all()

    coords = {**A.coords, **b.coords}
    try:
        del coords[common_dim]
    except KeyError:
        pass

    out = _weighted_least_squares(
        A=A.values,
        b=b.values[..., None],
        cov=var.values[..., None],
    )[0][..., 0]

    return xarray.DataArray(
        out,
        coords=coords,
        dims=[*remaining_dims, output_dim],
    )


def _weighted_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    cov: np.ndarray,
):
    assert A.ndim == 2
    assert b.shape[-2] == A.shape[0]
    assert b.shape[-1] == 1

    if cov.shape[-2:] == b.shape[-2:]:
        cov = transpose(cov)
        _ATcov = transpose(A) / cov
    else:
        _ATcov = transpose(A) @ np.linalg.inv(cov)
    p_cov = np.linalg.inv(_ATcov @ A)
    p = p_cov @ (_ATcov @ b)
    return p, p_cov


def transpose(A):
    return np.swapaxes(A, -2, -1)


def covariance_propagation(
    A: np.ndarray,
    cov: np.ndarray,
):
    assert A.ndim == 2
    n = A.shape[0]
    if cov.shape[-2:] == (n, 1):
        return np.linalg.inv(np.transpose(A) @ (A / cov))
    elif cov.shape[-2:] == (n, n):
        return np.linalg.inv(np.transpose(A) @ np.linalg.inv(cov) @ A)
    else:
        raise ValueError("dimensionality mismatch:", A.shape, cov.shape)
