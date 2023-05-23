#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# ----------------------------------------------------------------------- #

import abc
from typing import Dict, List, NamedTuple, NoReturn, Tuple, Union

import numpy as np
import numpy.typing as npt
import dataclasses as dt
from scipy.stats import multivariate_normal
import scipy

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import (
    ExecutionMode,
    focusing_optics_factory_method,
)
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics
from aps.ai.autoalignment.beamline28IDB.optimization import configs, movers
from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors
from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import DictionaryWrapper, Histogram, get_info
from aps.ai.autoalignment.common.util.common import AspectRatio, ColorMap, PlotMode
from aps.ai.autoalignment.common.util.wrappers import get_distribution_info as get_simulated_distribution_info
from aps.ai.autoalignment.common.util.wrappers import plot_distribution as plot_distribution_internal
from aps.ai.autoalignment.common.util.common import calculate_projections_over_noise
from aps.ai.autoalignment.common.util.shadow.common import (
    EmptyBeamException,
    HybridFailureException,
    PreProcessorFiles,
    load_shadow_beam,
)

class OptimizationCriteria:
    CENTROID                    = "centroid"
    PEAK_DISTANCE               = "peak_distance"
    FWHM                        = "fwhm"
    SIGMA                       = "sigma"
    NEGATIVE_LOG_PEAK_INTENSITY = "negative_log_peak_intensity"
    LOG_WEIGHTED_SUM_INTENSITY  = "log_weighted_sum_intensity"
    KL_DIVERGENCE_WITH_GAUSSIAN = "kl_divergence"



class SelectionAlgorithm:
    TOPSIS           = "topsis"
    NASH_EQUILIBRIUM = "nash-equilibrium"


class BeamState(NamedTuple):
    photon_beam: object
    hist: Histogram
    dw: DictionaryWrapper


class BeamParameterOutput(NamedTuple):
    parameter_value: float
    photon_beam: object
    hist: Histogram
    dw: DictionaryWrapper

@dt.dataclass
class CalculationParameters:
    execution_mode : int = ExecutionMode.SIMULATION
    implementor: int     = Implementors.SHADOW
    xrange: List[float] = None
    yrange: List[float] = None
    nbins_h: int = 256
    nbins_v: int = 256
    do_gaussian_fit: bool = False
    use_denoised : bool = True
    from_raw_image : bool = True
    random_seed : float = None
    add_noise : bool = False
    noise : float = None
    percentage_fluctuation : float = 10.0
    calculate_over_noise : bool = False
    noise_threshold : float = 1.5
    reference_h : float = 0.0
    reference_v : float = 0.0
    save_images : bool = False
    every_n_images : int = 5

@dt.dataclass
class PlotParameters:
    implementor: int = Implementors.SHADOW
    title: str = 'X,Z'
    xrange: List[float] = None
    yrange: List[float] = None
    nbins_h: int = None
    nbins_v: int = None
    plot_mode: int = PlotMode.INTERNAL
    aspect_ratio: int = AspectRatio.AUTO
    color_map: int = ColorMap.VIRIDIS


def get_distribution_info(cp: CalculationParameters, photon_beam: object = None, **kwargs):
    if cp.execution_mode == ExecutionMode.SIMULATION:
        return get_simulated_distribution_info(
            implementor=cp.implementor,
            beam=photon_beam,
            xrange=cp.xrange,
            yrange=cp.yrange,
            nbins_h=cp.nbins_h,
            nbins_v=cp.nbins_v,
            do_gaussian_fit=cp.do_gaussian_fit,
            add_noise=cp.add_noise,
            noise=cp.noise,
            percentage_fluctuation=cp.percentage_fluctuation,
            calculate_over_noise=cp.calculate_over_noise,
            noise_threshold=cp.noise_threshold,
            **kwargs
        )
    elif cp.execution_mode == ExecutionMode.HARDWARE:
        if len(photon_beam.keys()) > 4:
            return Histogram(hh=photon_beam["h_coord"],
                             vv=photon_beam["v_coord"],
                             data_2D=photon_beam["image"]), \
                   DictionaryWrapper(h_sigma=photon_beam["width"],
                                     h_fwhm=photon_beam["width"],
                                     h_centroid=photon_beam["centroid_h"],
                                     v_sigma=photon_beam["height"],
                                     v_fwhm=photon_beam["height"],
                                     v_centroid=photon_beam["centroid_v"],
                                     integral_intensity=np.sum(photon_beam["image"]),
                                     peak_intensity=np.max(photon_beam["image"]),
                                     gaussian_fit=None)
        else:
            return get_info(x_array=photon_beam["h_coord"],
                            y_array=photon_beam["v_coord"],
                            z_array=photon_beam["image_denoised"] if cp.use_denoised else photon_beam["image"],
                            do_gaussian_fit=cp.do_gaussian_fit,
                            calculate_over_noise=cp.calculate_over_noise,
                            noise_threshold=cp.noise_threshold)
    else:
        raise ValueError("Executione Mode not valid")

def plot_distribution(ppm: PlotParameters, photon_beam: object, **kwargs):
    params = dt.asdict(ppm)
    params.update(kwargs)

    plot_distribution_internal(beam=photon_beam, **params)

def get_random_init(cp : CalculationParameters,
                    focusing_system: AbstractFocusingOptics = None, 
                    motor_types: List[str] = None,
                    motor_types_and_ranges: dict = None,
                    verbose : bool = True,
                    intensity_sum_threshold: float = None,
                    **kwargs):
    error_msg = "Need to supply one of 'motor_types' and 'motor_types_and_ranges'."
    if motor_types_and_ranges is not None:
        if motor_types is not None: raise ValueError(error_msg)
        motor_types = list(motor_types_and_ranges.keys())
        init_range = list(motor_types_and_ranges.values())
    elif motor_types is not None:
        init_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) for mt in motor_types]
    else:
        raise ValueError(error_msg)

    for r in init_range:
        if len(r) != 2: raise ValueError("Need to supply min and max value for the range.")

    initial_motor_positions = movers.get_absolute_positions(focusing_system, motor_types)
    regenerate              = True

    while regenerate:
        initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in init_range]
        focusing_system = movers.move_motors(focusing_system, motor_types, initial_guess, movement="relative")
        centroid, photon_beam, hist, dw = get_centroid_distance(cp, focusing_system, None, **kwargs)

        if not (centroid < 1e4):
            if verbose: print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            focusing_system = movers.move_motors(focusing_system, motor_types, initial_motor_positions, movement="absolute")
            continue

        if intensity_sum_threshold is not None:
            if hist.data_2D.sum() <= intensity_sum_threshold:
                focusing_system = movers.move_motors(focusing_system, motor_types, initial_motor_positions, movement="absolute")
                continue
        regenerate = False

    print("Random initialization is (ABSOLUTE)", motor_types, movers.get_absolute_positions(focusing_system, motor_types))
    print("Random initialization is (RELATIVE)", motor_types, initial_guess)

    return initial_guess, focusing_system, BeamState(photon_beam, hist, dw)

def reinitialize(input_beam_path: str,
                 layout: int,
                 input_features: DictionaryWrapper,
                 bender: bool = True,
                 execution_mode: int = ExecutionMode.SIMULATION,
                 implementor: int = Implementors.SHADOW,
                 **kwargs) -> AbstractFocusingOptics:
    if execution_mode == ExecutionMode.SIMULATION:
        clean_up()
        focusing_system = focusing_optics_factory_method(execution_mode=execution_mode, implementor=implementor, bender=bender, **kwargs)
        focusing_system.initialize(input_photon_beam=load_shadow_beam(input_beam_path), rewrite_preprocessor_files=PreProcessorFiles.NO, layout=layout, input_features=input_features, **kwargs)
    else:
        focusing_system = focusing_optics_factory_method(execution_mode=execution_mode, implementor=implementor, **kwargs)
        focusing_system.initialize(**kwargs)

    return focusing_system

def get_beam(focusing_system: AbstractFocusingOptics, **kwargs):
    return focusing_system.get_photon_beam(**kwargs)

# -------------------------------------------------------------------- #

def check_input_for_beam(focusing_system: AbstractFocusingOptics, photon_beam: object, **kwargs):
    if photon_beam is None:
        if focusing_system is None:
            raise ValueError("Need to supply at least one of photon_beam or focusing_system.")
        photon_beam = get_beam(focusing_system=focusing_system, **kwargs)

    return photon_beam

def check_beam_out_of_bounds(focusing_system: AbstractFocusingOptics, photon_beam, **kwargs) -> object:
    try:
        photon_beam = check_input_for_beam(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)
    except Exception as exception:
        if (
            isinstance(exception, EmptyBeamException)
            or (isinstance(exception, HybridFailureException))
            or "Diffraction plane is set on Z, but the beam has no extention in that direction" in str(exception)
        ):
            # Assuming that the beam is outside the screen and returning the default out of bounds value.
            photon_beam = None
        elif isinstance(exception, ValueError) and "array must not contain infs or NaNs" in str(exception):
            photon_beam = None
        else:
            raise exception

    return photon_beam

# -------------------------------------------------------------------- #

def get_beam_hist_dw(cp : CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, **kwargs) -> BeamState:
    if cp.execution_mode == ExecutionMode.SIMULATION:
        kwargs["random_seed"] = cp.random_seed
        photon_beam = check_beam_out_of_bounds(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)
    else:
        kwargs["from_raw_image"] = cp.from_raw_image
        photon_beam = check_input_for_beam(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)

    if photon_beam is None: return BeamState(None, None, None)

    hist, dw = get_distribution_info(cp, photon_beam, **kwargs)

    return BeamState(photon_beam, hist, dw)

# -------------------------------------------------------------------- #

def get_fwhm(cp : CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 1e4, **kwargs) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    fwhm = _get_fwhm_from_dw(dw, cp.reference_h, cp.reference_v, cp.do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(fwhm, photon_beam, hist, dw)

def get_sigma(cp : CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 1e4, **kwargs) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    sigma = _get_sigma_from_dw(dw, cp.reference_h, cp.reference_v, cp.do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(sigma, photon_beam, hist, dw)

def get_peak_distance(cp :CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 1e4, **kwargs) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs )
    peak_distance = _get_peak_distance_from_dw(dw, cp.reference_h, cp.reference_v, cp.do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(peak_distance, photon_beam, hist, dw)

def get_centroid_distance(cp : CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 1e4,  **kwargs ) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    centroid_distance = _get_centroid_distance_from_dw(dw, cp.reference_h, cp.reference_v, cp.do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(centroid_distance, photon_beam, hist, dw)

def get_weighted_sum_intensity(cp: CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 0.0, radial_weight_power : float = 0.0, **kwargs) -> BeamParameterOutput:
    if cp.do_gaussian_fit: raise NotImplementedError

    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    sum_intensity         = _get_weighted_sum_intensity_from_hist(cp, hist, radial_weight_power, no_beam_value)

    return BeamParameterOutput(sum_intensity, photon_beam, hist, dw)

def get_peak_intensity(cp : CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam : object, no_beam_value : float = 0.0, **kwargs) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    peak = _get_peak_intensity_from_dw(dw, cp.do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(peak, photon_beam, hist, dw)

def get_kl_divergence_with_gaussian_from_hist(cp: CalculationParameters, focusing_system: AbstractFocusingOptics, photon_beam: object,
                                              no_beam_value: float = 0.0,  ref_pdf: npt.NDArray[float] = None, ref_fwhm: Tuple[float] = (1e-2, 1e-2),
                                              eps: float = 1e-8, return_ref_pdf: bool = False, **kwargs) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(cp, focusing_system, photon_beam, **kwargs)
    kl_div = _get_kl_divergence_with_gaussian_from_hist(hist, ref_pdf=ref_pdf, ref_fwhm=ref_fwhm, no_beam_value=no_beam_value,
                                                        eps=eps, calculate_over_noise=cp.calculate_over_noise,
                                                        noise_threshold=cp.noise_threshold, return_ref_pdf=return_ref_pdf)
    return BeamParameterOutput(kl_div, photon_beam, hist, dw)


# -------------------------------------------------------------------- #

def _get_fwhm_from_dw(dw: DictionaryWrapper,
                      reference_h: float = 0,
                      reference_v: float = 0,
                      do_gaussian_fit: bool = False,
                      no_beam_value: float = 1e4) -> float:
    if dw is None: return no_beam_value
    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf: return no_beam_value
        h_fwhm = dw.get_parameter("gaussian_fit")["fwhm_x"]
        v_fwhm = dw.get_parameter("gaussian_fit")["fwhm_y"]
    else:
        h_fwhm = dw.get_parameter("h_fwhm")
        v_fwhm = dw.get_parameter("v_fwhm")
    h_fwhm = 0 if h_fwhm is None else h_fwhm
    v_fwhm = 0 if v_fwhm is None else v_fwhm
    return ((h_fwhm - reference_h) ** 2 + (v_fwhm - reference_v) ** 2) ** 0.5


def _get_sigma_from_dw(dw: DictionaryWrapper,
                       reference_h: float = 0,
                       reference_v: float = 0,
                       do_gaussian_fit: bool = False,
                       no_beam_value: float = 1e4 ) -> float:
    if dw is None: return no_beam_value
    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf: return no_beam_value
        h_sigma = dw.get_parameter("gaussian_fit")["sigma_x"]
        v_sigma = dw.get_parameter("gaussian_fit")["sigma_y"]
    else:
        h_sigma = dw.get_parameter("h_sigma")
        v_sigma = dw.get_parameter("v_sigma")

    return ((h_sigma - reference_h) ** 2 + (v_sigma - reference_v) ** 2) ** 0.5

def _get_centroid_distance_from_dw(dw: DictionaryWrapper,
                                   reference_h: float = 0,
                                   reference_v: float = 0,
                                   do_gaussian_fit: bool = False,
                                   no_beam_value: float = 1e4) -> float:
    if dw is None: return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf: return no_beam_value
        h_centroid = gf["center_x"]
        v_centroid = gf["center_y"]
    else:
        h_centroid = dw.get_parameter("h_centroid")
        v_centroid = dw.get_parameter("v_centroid")

    return ((h_centroid - reference_h) ** 2 + (v_centroid - reference_v) ** 2) ** 0.5


def _get_peak_distance_from_dw(dw: DictionaryWrapper,
                               reference_h: float = 0,
                               reference_v: float = 0,
                               do_gaussian_fit: bool = False,
                               no_beam_value: float = 1e4) -> float:
    if dw is None: return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf: return no_beam_value
        h_peak = gf["center_x"]
        v_peak = gf["center_y"]
    else:
        h_peak = dw.get_parameter("h_peak")
        v_peak = dw.get_parameter("v_peak")

    return ((h_peak - reference_h) ** 2 + (v_peak - reference_v) ** 2) ** 0.5

def _get_weighted_sum_intensity_from_hist(cp: CalculationParameters, hist: Histogram, radial_weight_power: float = 0, no_beam_value: float = 0,
                                          verbose: bool = False) -> float:
    if hist is None: return no_beam_value

    data_2D = hist.data_2D
    if cp.calculate_over_noise:
        data_2D, *_ = calculate_projections_over_noise(data_2D, cp.noise_threshold)

    mesh = np.meshgrid(hist.hh, hist.vv)
    radius = (mesh[0]**2 + mesh[1]**2)**0.5
    weight = radius**radial_weight_power
    weighted_hist = data_2D*weight.T

    return weighted_hist.sum()


def _get_kl_divergence_with_gaussian_from_hist(cp: CalculationParameters, hist: Histogram,  ref_pdf: npt.NDArray[float] = None, 
                                               reference_h: float=1e-3, refernece_v: float=1e-3,
                                               eps: float = 1e-8, no_beam_value: float = 0, 
                                               return_ref_pdf: bool = False, verbose: bool = False) -> float:
    # ref_fwhm is in mm
    
    if hist is None: return no_beam_value

    data_2D = hist.data_2D
    if cp.calculate_over_noise:
        data_2D, *_ = calculate_projections_over_noise(data_2D, cp.noise_threshold)

    dsum = data_2D.sum()
    dat_pdf = data_2D / dsum + eps

    if ref_pdf is None:
        if verbose:
            print("Ref pdf is not supplied. Using reference_h and reference_v to create a Gaussian.")
        ref_fwhm = np.array([reference_h, refernece_v]) + eps
        vv, hh = np.meshgrid(hist.vv, hist.hh)
        pos = np.dstack((hh, vv))
        std_dev = ref_fwhm / (2 * (2 * np.log(2)) ** 0.5)
        cov = np.array([[std_dev[0] ** 2, 0], [0, std_dev[1] ** 2]])
        ref_pdf = multivariate_normal.pdf(pos, cov=cov)
        ref_pdf = ref_pdf / ref_pdf.sum() + 1e-8
    else:
        if verbose:
            print("Ref pdf is supplied. Ignoring reference_h and refernece_v")

    kl_div = np.sum(dat_pdf * np.log(2 * dat_pdf / (ref_pdf + dat_pdf))
                   + ref_pdf * np.log(2 * ref_pdf / (ref_pdf + dat_pdf)))
    #kl_div = np.sum(dat_pdf * np.log(dat_pdf / ref_pdf))
    if not return_ref_pdf:
        return kl_div
    else:
        return kl_div, ref_pdf

def _get_wasserstein_dist_with_gaussian_from_hist(cp: CalculationParameters, hist: Histogram,  ref_pdf: npt.NDArray[float] = None, 
                                               reference_h: float=1e-3, refernece_v: float=1e-3,
                                               eps: float = 1e-8, no_beam_value: float = 0, 
                                               return_ref_pdf: bool = False, verbose: bool = False) -> float:
    # ref_fwhm is in mm
    raise NotImplementedError
    if hist is None: return no_beam_value

    data_2D = hist.data_2D
    if cp.calculate_over_noise:
        data_2D, *_ = calculate_projections_over_noise(data_2D, cp.noise_threshold)

    dsum = data_2D.sum()
    dat_pdf = data_2D / dsum + eps

    if ref_pdf is None:
        if verbose:
            print("Ref pdf is not supplied. Using reference_h and reference_v to create a Gaussian.")
        ref_fwhm = np.array([reference_h, refernece_v]) + eps
        vv, hh = np.meshgrid(hist.vv, hist.hh)
        pos = np.dstack((hh, vv))
        std_dev = ref_fwhm / (2 * (2 * np.log(2)) ** 0.5)
        cov = np.array([[std_dev[0] ** 2, 0], [0, std_dev[1] ** 2]])
        ref_pdf = multivariate_normal.pdf(pos, cov=cov)
        ref_pdf = ref_pdf / ref_pdf.sum() + 1e-8
    else:
        if verbose:
            print("Ref pdf is supplied. Ignoring reference_h and refernece_v")

    #kl_div = np.sum(dat_pdf * np.log(2 * dat_pdf / (ref_pdf + dat_pdf))
    #                + ref_pdf * np.log(2 * ref_pdf / (ref_pdf + dat_pdf)))

    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    d = cdist(Y1, Y2)
    assignment = linear_sum_assignment(d)
    kl_div = np.sum(dat_pdf * np.log(dat_pdf / ref_pdf))
    if not return_ref_pdf:
        return kl_div
    else:
        return kl_div, ref_pdf


def _get_peak_intensity_from_dw( dw: DictionaryWrapper, do_gaussian_fit: bool = False, no_beam_value: float = 0.0) -> float:
    if dw is None: return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf: peak = no_beam_value
        else:      peak = dw.get_parameter("gaussian_fit")["amplitude"]
    else: peak = dw.get_parameter("peak_intensity")

    return peak

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #

class OptimizationCommon(abc.ABC):
    class TrialInstanceLossFunction:
        def __init__(self, opt_common: object, verbose: bool = False) -> NoReturn:
            self.opt_common = opt_common
            self.x_absolute_prev = 0
            self.current_loss = None
            self.verbose = verbose

        def loss(self, x_absolute_this: Union[List[float], "np.ndarray"], verbose: bool = None) -> float:
            if np.ndim(x_absolute_this) > 0:
                x_absolute_this = np.array(x_absolute_this)
            x_relative_this = x_absolute_this - self.x_absolute_prev
            self.x_absolute_prev = x_absolute_this
            self.current_loss = self.opt_common.loss_function(x_relative_this, verbose=False)
            verbose = verbose if verbose is not None else self.verbose
            if verbose:
                abs_trans_str = "".join([f"{x:8.2g}" for x in x_absolute_this])
                rel_trans_str = "".join([f"{x:8.2g}" for x in x_relative_this])
                print(
                    f"motors {self.opt_common.motor_types} trans abs {abs_trans_str} trans rel {rel_trans_str}"
                    f"current loss {self.current_loss:4.3g}"
                )

                absolute_set = np.array(self.opt_common.initial_motor_positions) + np.array(x_absolute_this)
                absolute_true = movers.get_absolute_positions(self.opt_common.focusing_system, self.opt_common.motor_types)
                abs_true_str = "".join([f"{x:8.2g}" for x in absolute_true])
                abs_set_str = "".join([f"{x:8.2g}" for x in absolute_set])
                print(f"motors {self.opt_common.motor_types} absolute movements are set to {abs_set_str}")
                print(f"motors {self.opt_common.motor_types} absolute movements are actually {abs_true_str}")

            return self.current_loss

    def __init__(self,
                 calculation_parameters : CalculationParameters,
                 focusing_system : AbstractFocusingOptics,
                 motor_types: List[str],
                 loss_parameters: List[str] = OptimizationCriteria.CENTROID,
                 reference_parameters_h_v: Dict[str, Tuple] = None, # repeated info, it's ok for now
                 loss_min_value: float = None,
                 no_beam_loss: float = 1e4,
                 intensity_no_beam_loss: float = 0,
                 multi_objective_optimization: bool = False,
                 **kwargs,
    ):
        self.cp                      = calculation_parameters
        self.focusing_system         = focusing_system
        self.motor_types             = motor_types if np.ndim(motor_types) > 0 else [motor_types]
        self.initial_motor_positions = movers.get_absolute_positions(self.focusing_system, self.motor_types)
        self.loss_parameters         = list(np.atleast_1d(loss_parameters))
        self.reference_parameter_h_v = configs.DEFAULT_LOSS_REFERENCE_VALUES

        if reference_parameters_h_v is not None:
            for k in reference_parameters_h_v:
                if k not in configs.DEFAULT_LOSS_REFERENCE_VALUES: raise NotImplementedError(f"Reference value not implemented for {k}")
                if np.ndim(reference_parameters_h_v[k]) != 1: raise ValueError("For now, reference parameters should be in the format (h, v). ")
                self.reference_parameter_h_v[k] = reference_parameters_h_v[k]

        self._loss_function_list = []
        temp_loss_min_value = 0
        for loss_type in self.loss_parameters:
            self._loss_function_list.append(self.get_beam_property_function_for_loss(loss_type))
            temp_loss_min_value += configs.DEFAULT_LOSS_TOLERANCES[loss_type]
            if loss_type == OptimizationCriteria.KL_DIVERGENCE_WITH_GAUSSIAN:
                self._ref_pdf = None

        self._multi_objective_optimization = multi_objective_optimization
        self._loss_min_value = temp_loss_min_value if loss_min_value is None else loss_min_value
        self._opt_trials_motor_positions = []
        self._opt_trials_losses = []
        self._opt_fn_call_counter = 0
        self._no_beam_loss = no_beam_loss  # this is a ridiculous arbitrarily high value.
        self._intensity_no_beam_loss = intensity_no_beam_loss
        self._kwargs = kwargs

        cond1 = self.cp.xrange is not None and np.size(self.cp.xrange) != 2
        cond2 = self.cp.yrange is not None and np.size(self.cp.yrange) != 2
        if cond1 or cond2: raise ValueError("Enter limits for xrange (and yrange) in the format [xrange_min, xramge_max] " + "in units of microns.")

        self._update_beam_state()

        self.guesses_all = []
        self.results_all = []

    def get_beam_property_function_for_loss(self, beam_prop: str):
        property_functions = {OptimizationCriteria.CENTROID: self.get_centroid_distance,
                              OptimizationCriteria.PEAK_DISTANCE: self.get_peak_distance,
                              OptimizationCriteria.NEGATIVE_LOG_PEAK_INTENSITY: self.get_negative_log_peak_intensity,
                              OptimizationCriteria.FWHM: self.get_fwhm,
                              OptimizationCriteria.SIGMA: self.get_sigma,
                              OptimizationCriteria.LOG_WEIGHTED_SUM_INTENSITY: self.get_log_weighted_sum_intensity,
                              OptimizationCriteria.KL_DIVERGENCE_WITH_GAUSSIAN: self.get_kl_divergence_with_gaussian_from_hist}
        if beam_prop not in property_functions:
            raise ValueError("Supplied loss option is not valid.")
        return property_functions[beam_prop]

    def _update_beam_state(self) -> bool:
        current_beam, current_hist, current_dw = get_beam_hist_dw(self.cp, self.focusing_system, None, **self._kwargs)

        if current_hist is None:  current_hist = Histogram(hh=np.zeros(self.cp.nbins_h),
                                                           vv=np.zeros(self.cp.nbins_v),
                                                           data_2D=np.zeros((self.cp.nbins_h, self.cp.nbins_v)))
        self.beam_state = BeamState(current_beam, current_hist, current_dw)

        return True

    def get_peak_intensity(self) -> float:
        return _get_peak_intensity_from_dw(self.beam_state.dw, self.cp.do_gaussian_fit, self._intensity_no_beam_loss)

    def get_negative_log_peak_intensity(self) -> float:
        peak_intensity = self.get_peak_intensity()

        if peak_intensity == 0: log_peak = self._no_beam_loss
        else:                   log_peak = -np.log(peak_intensity)

        return log_peak

    def get_sum_intensity(self) -> float:
        return _get_weighted_sum_intensity_from_hist(self.cp, self.beam_state.hist, 0, self._intensity_no_beam_loss)

    def get_weighted_sum_intensity(self) -> float:
        return _get_weighted_sum_intensity_from_hist(self.cp, self.beam_state.hist, 2, self._intensity_no_beam_loss)

    def get_log_weighted_sum_intensity(self) -> float:
        weighted_sum_intensity = self.get_weighted_sum_intensity()

        if weighted_sum_intensity == 0: log_weighted_sum_intensity = self._no_beam_loss
        else:                           log_weighted_sum_intensity = np.log(weighted_sum_intensity)

        return log_weighted_sum_intensity

    def get_peak_distance(self) -> float:
        return _get_peak_distance_from_dw(self.beam_state.dw,
                                          self.reference_parameter_h_v[OptimizationCriteria.PEAK_DISTANCE][0],
                                          self.reference_parameter_h_v[OptimizationCriteria.PEAK_DISTANCE][1],
                                          self.cp.do_gaussian_fit,
                                          self._no_beam_loss)

    def get_centroid_distance(self) -> float:
        return _get_centroid_distance_from_dw(self.beam_state.dw,
                                              self.reference_parameter_h_v[OptimizationCriteria.CENTROID][0],
                                              self.reference_parameter_h_v[OptimizationCriteria.CENTROID][1],
                                              self.cp.do_gaussian_fit,
                                              self._no_beam_loss)

    def get_fwhm(self) -> float:
        return _get_fwhm_from_dw(self.beam_state.dw,
                                 self.reference_parameter_h_v[OptimizationCriteria.FWHM][0],
                                 self.reference_parameter_h_v[OptimizationCriteria.FWHM][1],
                                 self.cp.do_gaussian_fit,
                                 self._no_beam_loss)

    def get_sigma(self) -> float:
        return _get_sigma_from_dw(self.beam_state.dw,
                                  self.reference_parameter_h_v[OptimizationCriteria.SIGMA][0],
                                  self.reference_parameter_h_v[OptimizationCriteria.SIGMA][1],
                                  self.cp.do_gaussian_fit,
                                  self._no_beam_loss)

    def get_kl_divergence_with_gaussian_from_hist(self) -> float:
        kl_div, self._ref_pdf = _get_kl_divergence_with_gaussian_from_hist(self.cp, self.beam_state.hist, self._ref_pdf,
                                                          self.reference_parameter_h_v[OptimizationCriteria.FWHM][0],
                                                          self.reference_parameter_h_v[OptimizationCriteria.FWHM][1],
                                                          no_beam_value=self._intensity_no_beam_loss,
                                                          return_ref_pdf=True)
        return kl_div



    def loss_function(self, translations: Union[List[float], "np.ndarray"], verbose: bool = True) -> float:
        """This mutates the state of the focusing system."""
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types, translations, movement="relative")
        self._update_beam_state()

        loss = np.array([lossfn() for lossfn in self._loss_function_list])
        if not self._multi_objective_optimization: loss = loss.sum()
        self._opt_trials_motor_positions.append(translations)
        self._opt_trials_losses.append(loss)
        self._opt_fn_call_counter += 1
        if verbose: print("motors", self.motor_types, "trans", translations, "current loss", loss)

        return loss

    def _check_initial_loss(self, verbose=False) -> NoReturn:
        size = np.size(self.motor_types)
        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        initial_loss = lossfn_obj_this.loss(np.atleast_1d(np.zeros(size)), verbose=False)
        if initial_loss >= self._no_beam_loss: raise EmptyBeamException("Initial beam is out of bounds.")

        print("Initial loss is", initial_loss)

    def reset(self) -> NoReturn:
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types, self.initial_motor_positions, movement="absolute")
        self._update_beam_state()

    def _get_guess_ranges(self, guess_range: List[float] = None):
        if guess_range is None:                                   guess_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) / 2 for mt in self.motor_types]
        elif np.ndim(guess_range) == 1 and len(guess_range) == 2: guess_range = [guess_range for mt in self.motor_types]
        elif np.ndim(guess_range) != 2 or len(guess_range) != len(self.motor_types): raise ValueError("Invalid range supplied for guesses.")

        return guess_range

    def get_random_init(self, guess_range: List[float] = None, verbose=True):
        guess_range = self._get_guess_ranges(guess_range)

        initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)
        if verbose: print("Random guess", initial_guess, "has loss", guess_loss)

        while guess_loss >= self._no_beam_loss or np.isnan(guess_loss):
            self.reset()
            if verbose: print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
            if verbose: print("Random guess is", initial_guess)
            guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)

        return initial_guess

    @abc.abstractmethod
    def set_optimizer_options(self) -> NoReturn: pass

    @abc.abstractmethod
    def _optimize(self) -> Tuple[object, List[float], bool]: pass

    @abc.abstractmethod
    def trials(self, *args: List, **kwargs: Dict): pass
