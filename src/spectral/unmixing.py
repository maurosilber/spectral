import numpy as np
import statsmodels.api as sm
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


def weighted_least_squares(A: xarray.DataArray, b: xarray.DataArray):
    if A.dims[0] in b.dims:
        common_dim, output_dim = A.dims
    else:
        common_dim, output_dim = A.dims[::-1]
    remaining_dims = [d for d in b.dims if d != common_dim]

    A = A.transpose(common_dim, output_dim).sortby(common_dim)
    b = b.transpose(*remaining_dims, common_dim).sortby(common_dim)
    assert (A.coords[common_dim] == b.coords[common_dim]).all()

    coords = {**A.coords, **b.coords}
    try:
        del coords[common_dim]
    except KeyError:
        pass

    A = A.values
    b = b.values

    def _wls(b: xarray.DataArray):
        return sm.WLS(b, A, weights=1 / b).fit().params

    shape = b.shape[:-1]
    out = np.empty((*shape, A.shape[1]))
    for ix in np.ndindex(shape):
        out[ix] = _wls(b[ix])

    return xarray.DataArray(
        out,
        coords=coords,
        dims=[*remaining_dims, output_dim],
    )
