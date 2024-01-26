from pathlib import Path
from typing import Iterable, TypedDict

import numpy as np
import pandas as pd
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


class Crop(TypedDict):
    y: slice
    x: slice


def yield_crop_rectangles(file: Path) -> Iterable[tuple[str, Crop]]:
    with file.open() as f:
        for line in f:
            fov = line[:9]
            values = line.rstrip()[29:-1]
            d = eval(f"dict({values})")
            crop: Crop = {
                "y": slice(d["Y"], d["Y"] + d["Height"]),
                "x": slice(d["X"], d["X"] + d["Width"]),
            }
            yield fov, crop


def load_calibration(
    file: str | Path,
    *,
    input_name: str,
    output_name: str,
) -> xarray.DataArray:
    components_mapping = {
        "AutoFL": "afl",
        "DAPI": "dapi",
        "Opal520": "o520",
        "Opal540": "o540",
        "Opal570": "o570",
        "Opal620": "o620",
        "Opal650": "o650",
        "Opal690": "o690",
    }

    return (
        pd.read_excel(file)
        .drop(columns="#")
        .rename(columns={"Input": input_name, **components_mapping})
        .set_index(input_name)
        .to_xarray()
        .to_array(output_name, name="calibration")
    )
