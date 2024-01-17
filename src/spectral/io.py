from pathlib import Path
from typing import Iterable

import numpy as np
import xarray
from imagecodecs import jpegxr_decode, tiff_decode


def read_image(
    file: Path,
    *,
    format: str | None = None,
    out: np.ndarray | None = None,
):
    if format is None:
        format = file.suffix.removeprefix(".")

    match format.lower():
        case "tif":
            func = tiff_decode
        case "jxr":
            func = jpegxr_decode
        case _:
            raise ValueError(f"format {format} not supported")

    with file.open("rb") as f:
        return func(f.read(), out=out)


def read_stack(dir_or_files: Path | Iterable[Path], *, dim: str):
    if isinstance(dir_or_files, Path):
        if dir_or_files.is_dir():
            dir_or_files = dir_or_files.iterdir()
        else:
            dir_or_files = [dir_or_files]

    dir_or_files = sorted(dir_or_files)
    image = read_image(dir_or_files[0])
    images = np.empty((len(dir_or_files), *image.shape), dtype=image.dtype)
    images[0] = image
    for i, file in enumerate(dir_or_files[1:], start=1):
        read_image(file, out=images[i])

    return xarray.DataArray(
        images,
        coords={dim: ["_".join(f.stem.split("_")[-2:]) for f in dir_or_files]},
        dims=[dim, "y", "x"],
    )
