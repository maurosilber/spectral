import numpy as np
import pandas as pd
import xarray
import xarray.testing.strategies
from hypothesis import given, strategies

from .. import mixing


@given(
    shape=xarray.testing.strategies.dimension_sizes(min_dims=3, max_dims=5),
    seed=strategies.integers(min_value=0),
)
def test_matmul(
    shape: dict[str, int],
    seed: int,
):
    rng = np.random.default_rng(seed)

    shape_series = pd.Series(shape)
    A_shape = shape_series.iloc[-2::].iloc[::-1]
    x_shape = shape_series.iloc[:-1]
    b_shape = pd.concat((x_shape.iloc[:-1], A_shape.iloc[:1]))

    A = rng.random(tuple(A_shape))
    x = rng.random(tuple(x_shape))
    b = (A @ x[..., None])[..., 0]
    assert b.shape == tuple(b_shape.values)

    A = xarray.DataArray(A, dims=A_shape.index)
    x = xarray.DataArray(x, dims=x_shape.index)
    b = xarray.DataArray(b, dims=b_shape.index)
    assert _allclose(mixing.matmul(A, x), b)

    # With transposed dimensions
    A = A.transpose(*rng.choice(A_shape.index, size=len(A_shape.index), replace=False))
    x = x.transpose(*rng.choice(x_shape.index, size=len(x_shape.index), replace=False))
    assert _allclose(mixing.matmul(A, x), b)


def _allclose(left: xarray.DataArray, right: xarray.DataArray, /, tol=1e-15):
    return np.abs(left - right).max() < tol
