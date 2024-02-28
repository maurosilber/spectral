from typing import Sequence

import numpy as np
import skimage.color
import skimage.data
import skimage.filters
import skimage.transform
import xarray
from dask_checkpoint import task

from . import Dim, mixing

DATA_NAMES = [
    "astronaut",
    "camera",
    "cat",
    "coffee",
    "eagle",
    "grass",
    "gravel",
    "rocket",
]


@task()
def build_components(
    *,
    shape: tuple[int, int],
    dim: str,
    order: Sequence[str],
    sigma: float = 0,
):
    assert len(order) <= len(DATA_NAMES), "Not enough stock images"

    def get_image(name: str):
        image: np.ndarray = getattr(skimage.data, name)()
        if image.ndim == 3:
            image = skimage.color.rgb2gray(image)
        image = skimage.transform.resize(image, shape).astype(np.float64)
        if sigma > 0:
            image = skimage.filters.gaussian(image, sigma)
        return image

    images = np.empty((len(order), *shape), dtype=np.float64)
    for ndx, name in enumerate(DATA_NAMES[: len(order)]):
        images[ndx] = get_image(name)

    return xarray.DataArray(
        images,
        coords={dim: order},
        dims=[dim, "y", "x"],
    )


@task()
def build_components_from_composition_matrix(
    *,
    shape: tuple[int, int],
    n_composition: int,
    calibration: xarray.DataArray,
    seed: int,
):
    rng = np.random.default_rng(seed)
    components_names = calibration.coords[Dim.components]
    composition_matrix = xarray.DataArray(
        rng.random(size=(len(components_names), n_composition)),
        dims=[Dim.components, Dim.composition],
        coords={Dim.components: components_names},
    )
    composition = build_components.func(
        shape=shape,
        dim=Dim.composition,
        order=composition_matrix.coords[Dim.composition],
    )
    return mixing.matmul(composition_matrix, composition)
