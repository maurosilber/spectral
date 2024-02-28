from typing import Protocol

import xarray

from . import Dim, unmixing
from .binlets.components import binlets_components_transform
from .binlets.independent import binlets_independent_components, binlets_poisson


class Denoiser(Protocol):
    def __call__(
        self,
        spectrum: xarray.DataArray,
        *,
        sigma: float,
        calibration: xarray.DataArray,
    ) -> xarray.DataArray:
        ...


def nothing(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    return unmixing.weighted_least_squares(calibration, spectrum)


def single_spectrum(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    spectrum = binlets_poisson(
        spectrum,
        sigma=sigma,
        dim=Dim.spectrum,
        independent=True,
    )
    return unmixing.weighted_least_squares(calibration, spectrum)


def full_spectrum(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    spectrum = binlets_poisson(
        spectrum,
        sigma=sigma,
        dim=Dim.spectrum,
        independent=False,
    )
    return unmixing.weighted_least_squares(calibration, spectrum)


def continuous_spectrum(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    spectrum = binlets_poisson(
        spectrum,
        sigma=sigma,
        dim=None,
        independent=True,  # irrelevant as len(dim=None) == 1
    )
    return unmixing.weighted_least_squares(calibration, spectrum)


def single_component(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    return binlets_independent_components(calibration, spectrum, sigma=sigma)


def full_component(
    spectrum: xarray.DataArray,
    *,
    sigma: float,
    calibration: xarray.DataArray,
):
    spectrum = binlets_components_transform(
        spectrum,
        sigma=sigma,
        dim=Dim.spectrum,
        calibration=calibration,
    )
    return unmixing.weighted_least_squares(calibration, spectrum)
