from typing import Protocol

import xarray

from . import Dim, unmixing
from .binlets.components import binlets_components_transform
from .binlets.independent import binlets_independent_components, binlets_variance


class Denoiser(Protocol):
    def __call__(
        self,
        spectrum: xarray.DataArray,
        variance: xarray.DataArray,
        *,
        sigma: float,
        levels: int,
        calibration: xarray.DataArray,
    ) -> xarray.DataArray:
        ...


def nothing(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    return unmixing.weighted_least_squares(calibration, spectrum, variance)


def single_spectrum(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    spectrum = binlets_variance(
        spectrum,
        variance,
        sigma=sigma,
        dim=Dim.spectrum,
        independent=True,
        levels=levels,
    )
    return unmixing.weighted_least_squares(calibration, spectrum, variance)


def full_spectrum(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    spectrum = binlets_variance(
        spectrum,
        variance,
        sigma=sigma,
        dim=Dim.spectrum,
        independent=False,
        levels=levels,
    )
    return unmixing.weighted_least_squares(calibration, spectrum, variance)


def continuous_spectrum(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    spectrum = binlets_variance(
        spectrum,
        variance,
        sigma=sigma,
        dim=None,
        independent=True,  # irrelevant as len(dim=None) == 1
        levels=levels,
    )
    return unmixing.weighted_least_squares(calibration, spectrum, variance)


def single_component(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    return binlets_independent_components(
        calibration,
        spectrum,
        variance,
        sigma=sigma,
        levels=levels,
    )


def full_component(
    spectrum: xarray.DataArray,
    variance: xarray.DataArray,
    *,
    sigma: float,
    levels: int,
    calibration: xarray.DataArray,
):
    spectrum = binlets_components_transform(
        spectrum,
        sigma=sigma,
        dim=Dim.spectrum,
        calibration=calibration,
        levels=levels,
    )
    return unmixing.weighted_least_squares(calibration, spectrum, variance)
