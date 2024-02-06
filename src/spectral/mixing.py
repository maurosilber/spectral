import xarray


def matmul(
    A: xarray.DataArray,
    x: xarray.DataArray,
) -> xarray.DataArray:
    if A.dims[0] in x.dims:
        common_dim, output_dim = A.dims
    else:
        common_dim, output_dim = A.dims[::-1]
    remaining_dims = [d for d in x.dims if d != common_dim]

    A = A.transpose(output_dim, common_dim).sortby(common_dim)
    x = x.transpose(*remaining_dims, common_dim).sortby(common_dim)
    assert (A.coords[common_dim] == x.coords[common_dim]).all()

    b = (A.values @ x.values[..., None])[..., 0]
    coords = {**A.coords, **x.coords}
    try:
        del coords[common_dim]
    except KeyError:
        pass
    return xarray.DataArray(
        b,
        coords=coords,
        dims=[*remaining_dims, output_dim],
    )
