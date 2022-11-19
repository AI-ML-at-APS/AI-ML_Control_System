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

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import (
    ExecutionMode,
    focusing_optics_factory_method,
)
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics
from aps.ai.autoalignment.beamline28IDB.optimization import configs, movers
from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors
from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import DictionaryWrapper, Histogram, get_info
from aps.ai.autoalignment.common.util.wrappers import get_distribution_info as get_simulated_distribution_info
from aps.ai.autoalignment.common.util.shadow.common import (
    EmptyBeamException,
    HybridFailureException,
    PreProcessorFiles,
    load_shadow_beam,
)


def get_distribution_info(
    execution_mode=ExecutionMode.SIMULATION,
    implementor=Implementors.SHADOW,
    beam=None,
    xrange=None,
    yrange=None,
    nbins_h=201,
    nbins_v=201,
    do_gaussian_fit=False,
    **kwargs,
):
    if execution_mode == ExecutionMode.SIMULATION:
        return get_simulated_distribution_info(
            implementor=implementor,
            beam=beam,
            xrange=xrange,
            yrange=yrange,
            nbins_h=nbins_h,
            nbins_v=nbins_v,
            do_gaussian_fit=do_gaussian_fit,
        )
    elif execution_mode == ExecutionMode.HARDWARE:
        if len(beam.keys()) > 3:
            return Histogram(hh=beam["h_coord"], vv=beam["v_coord"], data_2D=beam["image"]), DictionaryWrapper(
                h_sigma=beam["width"],
                h_fwhm=beam["width"],
                h_centroid=beam["centroid_h"],
                v_sigma=beam["height"],
                v_fwhm=beam["height"],
                v_centroid=beam["centroid_v"],
                integral_intensity=np.sum(beam["image"]),
                peak_intensity=np.max(beam["image"]),
                gaussian_fit=None,
            )
        else:
            return get_info(
                x_array=beam["h_coord"], y_array=beam["v_coord"], z_array=beam["image"], do_gaussian_fit=do_gaussian_fit
            )
    else:
        raise ValueError("Executione Mode not valid")


class BeamState(NamedTuple):
    photon_beam: object
    hist: Histogram
    dw: DictionaryWrapper


class BeamParameterOutput(NamedTuple):
    parameter_value: float
    photon_beam: object
    hist: Histogram
    dw: DictionaryWrapper


def reinitialize(
    input_beam_path: str,
    layout: int,
    input_features: DictionaryWrapper,
    bender: bool = True,
    execution_mode: int = ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> AbstractFocusingOptics:
    if execution_mode == ExecutionMode.SIMULATION:
        clean_up()
        input_beam = load_shadow_beam(input_beam_path)

        focusing_system = focusing_optics_factory_method(
            execution_mode=execution_mode,
            implementor=implementor,
            bender=bender,
        )

        focusing_system.initialize(
            input_photon_beam=input_beam,
            rewrite_preprocessor_files=PreProcessorFiles.NO,
            layout=layout,
            input_features=input_features,
        )
    else:
        focusing_system = focusing_optics_factory_method(execution_mode=execution_mode, implementor=implementor)
        focusing_system.initialize()

    return focusing_system


def get_beam(focusing_system: AbstractFocusingOptics, **kwargs):
    return focusing_system.get_photon_beam(**kwargs)


def check_input_for_beam(focusing_system: AbstractFocusingOptics, photon_beam: object, **kwargs):
    if photon_beam is None:
        if focusing_system is None:
            raise ValueError("Need to supply at least one of photon_beam or focusing_system.")
        photon_beam = get_beam(focusing_system=focusing_system, **kwargs)

    return photon_beam


def check_beam_out_of_bounds(focusing_system: AbstractFocusingOptics, photon_beam, **kwargs) -> object:
    try:
        photon_beam = check_input_for_beam(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)
    except Exception as exc:
        if (
            isinstance(exc, EmptyBeamException)
            or (isinstance(exc, HybridFailureException))
            or "Diffraction plane is set on Z, but the beam has no extention in that direction" in str(exc)
        ):
            # Assuming that the beam is outside the screen and returning the default out of bounds value.
            photon_beam = None
        elif isinstance(exc, ValueError) and "array must not contain infs or NaNs" in str(exc):
            photon_beam = None
        else:
            raise exc

    return photon_beam


def get_beam_hist_dw(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamState:
    if execution_mode == ExecutionMode.SIMULATION:
        photon_beam = check_beam_out_of_bounds(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)
    else:
        photon_beam = check_input_for_beam(focusing_system=focusing_system, photon_beam=photon_beam, **kwargs)

    if photon_beam is None:
        return BeamState(None, None, None)

    hist, dw = get_distribution_info(
        execution_mode=execution_mode,
        implementor=implementor,
        beam=photon_beam,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
    )

    return BeamState(photon_beam, hist, dw)


def get_peak_intensity(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 0,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(
        execution_mode=execution_mode,
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
        implementor=implementor,
    )
    peak = _get_peak_intensity_from_dw(dw, do_gaussian_fit, no_beam_value)
    return BeamParameterOutput(peak, photon_beam, hist, dw)


def _get_peak_intensity_from_dw(
    dw: DictionaryWrapper, do_gaussian_fit: bool = False, no_beam_value: float = 0
) -> float:
    if dw is None:
        return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf:
            peak = no_beam_value
        else:
            peak = dw.get_parameter("gaussian_fit")["amplitude"]
    else:
        peak = dw.get_parameter("peak_intensity")
    return peak


def get_weighted_sum_intensity(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 0,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    radial_weight_power: float = 0,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    if do_gaussian_fit:
        raise NotImplementedError
    photon_beam, hist, dw = get_beam_hist_dw(
        execution_mode=execution_mode,
        implementor=implementor,
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
    )
    sum_intensity = _get_weighted_sum_intensity_from_hist(hist, radial_weight_power, no_beam_value)
    return BeamParameterOutput(sum_intensity, photon_beam, hist, dw)


def _get_weighted_sum_intensity_from_hist(hist: Histogram, radial_weight_power: float = 0, no_beam_value: float = 0) -> float:
    if hist is None: return no_beam_value

    mesh = np.meshgrid(hist.hh, hist.vv)
    radius = (mesh[0]**2 + mesh[1]**2)**0.5
    weight = radius**radial_weight_power
    weighted_hist = hist.data_2D*weight.T

    return weighted_hist.sum()


def _get_centroid_distance_from_dw(
    dw: DictionaryWrapper,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    no_beam_value: float = 1e4,
) -> float:
    if dw is None:
        return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf:
            return no_beam_value
        h_centroid = gf["center_x"]
        v_centroid = gf["center_y"]
    else:
        h_centroid = dw.get_parameter("h_centroid")
        v_centroid = dw.get_parameter("v_centroid")

    centroid_distance = ((h_centroid - reference_h) ** 2 + (v_centroid - reference_v) ** 2) ** 0.5

    return centroid_distance


def _get_peak_distance_from_dw(
    dw: DictionaryWrapper,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    no_beam_value: float = 1e4,
) -> float:
    if dw is None:
        return no_beam_value

    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf:
            return no_beam_value
        h_peak = gf["center_x"]
        v_peak = gf["center_y"]
    else:
        h_peak = dw.get_parameter("h_peak")
        v_peak = dw.get_parameter("v_peak")

    peak_distance = ((h_peak - reference_h) ** 2 + (v_peak - reference_v) ** 2) ** 0.5

    return peak_distance


def get_peak_distance(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 1e4,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(
        execution_mode=execution_mode,
        implementor=implementor,
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
    )
    peak_distance = _get_peak_distance_from_dw(dw, reference_h, reference_v, do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(peak_distance, photon_beam, hist, dw)


def get_centroid_distance(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 1e4,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(
        execution_mode=execution_mode,
        implementor=implementor,
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
    )
    centroid_distance = _get_centroid_distance_from_dw(dw, reference_h, reference_v, do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(centroid_distance, photon_beam, hist, dw)


def _get_fwhm_from_dw(
    dw: DictionaryWrapper,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    no_beam_value: float = 1e4,
) -> float:
    if dw is None:
        return no_beam_value
    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf:
            return no_beam_value
        h_fwhm = dw.get_parameter("gaussian_fit")["fwhm_x"]
        v_fwhm = dw.get_parameter("gaussian_fit")["fwhm_y"]
    else:
        h_fwhm = dw.get_parameter("h_fwhm")
        v_fwhm = dw.get_parameter("v_fwhm")
    fwhm = ((h_fwhm - reference_h) ** 2 + (v_fwhm - reference_v) ** 2) ** 0.5

    return fwhm


def _get_sigma_from_dw(
    dw: DictionaryWrapper,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    no_beam_value: float = 1e4,
) -> float:
    if dw is None:
        return no_beam_value
    if do_gaussian_fit:
        gf = dw.get_parameter("gaussian_fit")
        if not gf:
            return no_beam_value
        h_sigma = dw.get_parameter("gaussian_fit")["sigma_x"]
        v_sigma = dw.get_parameter("gaussian_fit")["sigma_y"]
    else:
        h_sigma = dw.get_parameter("h_sigma")
        v_sigma = dw.get_parameter("v_sigma")
    sigma = ((h_sigma - reference_h) ** 2 + (v_sigma - reference_v) ** 2) ** 0.5
    return sigma


def get_fwhm(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 1e4,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
        execution_mode=execution_mode,
        implementor=implementor,
    )
    fwhm = _get_fwhm_from_dw(dw, reference_h, reference_v, do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(fwhm, photon_beam, hist, dw)


def get_sigma(
    focusing_system: AbstractFocusingOptics = None,
    photon_beam: object = None,
    random_seed: float = None,
    no_beam_value: float = 1e4,
    xrange: List[float] = None,
    yrange: List[float] = None,
    nbins_h: int = 256,
    nbins_v: int = 256,
    reference_h: float = 0,
    reference_v: float = 0,
    do_gaussian_fit: bool = False,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **kwargs,
) -> BeamParameterOutput:
    photon_beam, hist, dw = get_beam_hist_dw(
        focusing_system=focusing_system,
        photon_beam=photon_beam,
        random_seed=random_seed,
        xrange=xrange,
        yrange=yrange,
        nbins_h=nbins_h,
        nbins_v=nbins_v,
        do_gaussian_fit=do_gaussian_fit,
        execution_mode=execution_mode,
        implementor=implementor,
    )
    sigma = _get_sigma_from_dw(dw, reference_h, reference_v, do_gaussian_fit, no_beam_value)

    return BeamParameterOutput(sigma, photon_beam, hist, dw)


def get_random_init(
    focusing_system,
    motor_types: List[str] = None,
    motor_types_and_ranges: dict = None,
    verbose=True,
    intensity_sum_threshold: float = None,
    random_seed: int = None,
    execution_mode=ExecutionMode.SIMULATION,
    implementor: int = Implementors.SHADOW,
    **hist_kwargs,
):
    error_msg = "Need to supply one of 'motor_types' and 'motor_types_and_ranges'."
    if motor_types_and_ranges is not None:
        if motor_types is not None:
            raise ValueError(error_msg)
        motor_types = list(motor_types_and_ranges.keys())
        init_range = list(motor_types_and_ranges.values())
    elif motor_types is not None:
        init_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) for mt in motor_types]
    else:
        raise ValueError(error_msg)

    for r in init_range:
        if len(r) != 2:
            raise ValueError("Need to supply min and max value for the range.")

    initial_motor_positions = movers.get_absolute_positions(focusing_system, motor_types)
    regenerate = True

    while regenerate:
        initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in init_range]
        focusing_system = movers.move_motors(focusing_system, motor_types, initial_guess, movement="relative")
        centroid, photon_beam, hist, dw = get_centroid_distance(
            focusing_system=focusing_system,
            random_seed=random_seed,
            execution_mode=execution_mode,
            implementor=implementor,
            **hist_kwargs,
        )
        if not (centroid < 1e4):
            if verbose:
                print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            focusing_system = movers.move_motors(
                focusing_system, motor_types, initial_motor_positions, movement="absolute"
            )
            continue

        if intensity_sum_threshold is not None:
            if hist.data_2D.sum() <= intensity_sum_threshold:
                focusing_system = movers.move_motors(
                    focusing_system, motor_types, initial_motor_positions, movement="absolute"
                )
                continue
        regenerate = False

    print(
        "Random initialization is (ABSOLUTE)", motor_types, movers.get_absolute_positions(focusing_system, motor_types)
    )
    print("Random initialization is (RELATIVE)", motor_types, initial_guess)

    return initial_guess, focusing_system, BeamState(photon_beam, hist, dw)


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
                absolute_true = movers.get_absolute_positions(
                    self.opt_common.focusing_system, self.opt_common.motor_types
                )
                abs_true_str = "".join([f"{x:8.2g}" for x in absolute_true])
                abs_set_str = "".join([f"{x:8.2g}" for x in absolute_set])
                print(f"motors {self.opt_common.motor_types} absolute movements are set to {abs_set_str}")
                print(f"motors {self.opt_common.motor_types} absolute movements are actually {abs_true_str}")

            return self.current_loss

    def __init__(
        self,
        focusing_system: AbstractFocusingOptics,
        motor_types: List[str],
        random_seed: int = None,
        loss_parameters: List[str] = "centroid",
        reference_parameters_h_v: Dict[str, Tuple] = None,
        loss_min_value: float = None,
        xrange: List[float] = None,
        yrange: List[float] = None,
        nbins_h: int = 256,
        nbins_v: int = 256,
        do_gaussian_fit: bool = False,
        no_beam_loss: float = 1e4,
        intensity_no_beam_loss: float = 0,
        multi_objective_optimization: bool = False,
        execution_mode: int = ExecutionMode.SIMULATION,
        implementor: int = Implementors.SHADOW,
        **kwargs,
    ):
        self.focusing_system = focusing_system
        self.motor_types = motor_types if np.ndim(motor_types) > 0 else [motor_types]
        self.random_seed = random_seed

        self.initial_motor_positions = movers.get_absolute_positions(focusing_system, self.motor_types)
        self.loss_parameters = list(np.atleast_1d(loss_parameters))
        self.reference_parameter_h_v = configs.DEFAULT_LOSS_REFERENCE_VALUES

        if reference_parameters_h_v is not None:
            for k in reference_parameters_h_v:
                if k not in configs.DEFAULT_LOSS_REFERENCE_VALUES:
                    raise NotImplementedError(f"Reference value not implemented for {k}")
                if np.ndim(reference_parameters_h_v[k]) != 1:
                    raise ValueError("For now, reference parameters should be in the format (h, v). ")
                self.reference_parameter_h_v[k] = reference_parameters_h_v[k]

        self._loss_function_list = []
        temp_loss_min_value = 0
        for loss_type in self.loss_parameters:
            if loss_type == "centroid":
                self._loss_function_list.append(self.get_centroid_distance)
            elif loss_type == "peak_distance":
                self._loss_function_list.append(self.get_peak_distance)
            elif loss_type == "negative_log_peak_intensity":
                print("Warning: Stopping condition for the peak intensity case is not supported.")
                self._loss_function_list.append(self.get_negative_log_peak_intensity)
            elif loss_type == "fwhm":
                self._loss_function_list.append(self.get_fwhm)
            elif loss_type == "sigma":
                self._loss_function_list.append(self.get_sigma)
            elif loss_type == "log_weighted_sum_intensity":
                self._loss_function_list.append(self.get_log_weighted_sum_intensity)
            else:
                raise ValueError("Supplied loss parameter is not valid.")
            temp_loss_min_value += configs.DEFAULT_LOSS_TOLERANCES[loss_type]

        self._multi_objective_optimization = multi_objective_optimization
        self._loss_min_value = temp_loss_min_value if loss_min_value is None else loss_min_value
        self._opt_trials_motor_positions = []
        self._opt_trials_losses = []
        self._opt_fn_call_counter = 0
        self._no_beam_loss = no_beam_loss  # this is a ridiculous arbitrarily high value.
        self._intensity_no_beam_loss = intensity_no_beam_loss

        self._xrange = xrange
        self._yrange = yrange
        self._nbins_h = nbins_h
        self._nbins_v = nbins_v
        self._do_gaussian_fit = do_gaussian_fit
        self._execution_mode = execution_mode
        self._implementor = implementor

        cond1 = xrange is not None and np.size(xrange) != 2
        cond2 = yrange is not None and np.size(yrange) != 2
        if cond1 or cond2:
            raise ValueError(
                "Enter limits for xrange (and yrange) in the format [xrange_min, xramge_max] " + "in units of microns."
            )
        self._update_beam_state()
        self.guesses_all = []
        self.results_all = []

    def _update_beam_state(self) -> bool:
        (current_beam, current_hist, current_dw) = get_beam_hist_dw(
            focusing_system=self.focusing_system,
            random_seed=self.random_seed,
            xrange=self._xrange,
            yrange=self._yrange,
            nbins_h=self._nbins_h,
            nbins_v=self._nbins_v,
            do_gaussian_fit=self._do_gaussian_fit,
            execution_mode=self._execution_mode,
            implementor=self._implementor,
        )

        if current_hist is None:
            current_hist = Histogram(
                hh=np.zeros(self._nbins_h),
                vv=np.zeros(self._nbins_v),
                data_2D=np.zeros((self._nbins_h, self._nbins_v)),
            )
        self.beam_state = BeamState(current_beam, current_hist, current_dw)
        return True

    def get_peak_intensity(self) -> float:
        return _get_peak_intensity_from_dw(self.beam_state.dw, self._do_gaussian_fit, self._intensity_no_beam_loss)

    def get_negative_log_peak_intensity(self) -> float:
        peak_intensity = self.get_peak_intensity()
        if peak_intensity == 0:
            log_peak = self._no_beam_loss
        else:
            log_peak = -np.log(peak_intensity)
        return log_peak

    def get_sum_intensity(self) -> float:
        return _get_weighted_sum_intensity_from_hist(self.beam_state.hist, 0, self._intensity_no_beam_loss)

    def get_weighted_sum_intensity(self) -> float:
        return _get_weighted_sum_intensity_from_hist(self.beam_state.hist, 2, self._intensity_no_beam_loss)

    def get_log_weighted_sum_intensity(self) -> float:
        weighted_sum_intensity = self.get_weighted_sum_intensity()
        if weighted_sum_intensity == 0:
            log_weighted_sum_intensity = self._no_beam_loss
        else:
            log_weighted_sum_intensity = np.log(weighted_sum_intensity)
        return log_weighted_sum_intensity

    def get_peak_distance(self) -> float:
        return _get_peak_distance_from_dw(
            self.beam_state.dw,
            self.reference_parameter_h_v["peak_distance"][0],
            self.reference_parameter_h_v["peak_distance"][1],
            self._do_gaussian_fit,
            self._no_beam_loss,
        )

    def get_centroid_distance(self) -> float:
        return _get_centroid_distance_from_dw(
            self.beam_state.dw,
            self.reference_parameter_h_v["centroid"][0],
            self.reference_parameter_h_v["centroid"][1],
            self._do_gaussian_fit,
            self._no_beam_loss,
        )

    def get_fwhm(self) -> float:
        return _get_fwhm_from_dw(
            self.beam_state.dw,
            self.reference_parameter_h_v["fwhm"][0],
            self.reference_parameter_h_v["fwhm"][1],
            self._do_gaussian_fit,
            self._no_beam_loss,
        )

    def get_sigma(self) -> float:
        return _get_sigma_from_dw(
            self.beam_state.dw,
            self.reference_parameter_h_v["sigma"][0],
            self.reference_parameter_h_v["sigma"][1],
            self._do_gaussian_fit,
            self._no_beam_loss,
        )

    def loss_function(self, translations: Union[List[float], "np.ndarray"], verbose: bool = True) -> float:
        """This mutates the state of the focusing system."""

        self.focusing_system = movers.move_motors(
            self.focusing_system, self.motor_types, translations, movement="relative"
        )
        self._update_beam_state()

        loss = np.array([lossfn() for lossfn in self._loss_function_list])
        if not self._multi_objective_optimization:
            loss = loss.sum()
        self._opt_trials_motor_positions.append(translations)
        self._opt_trials_losses.append(loss)
        self._opt_fn_call_counter += 1
        if verbose:
            print("motors", self.motor_types, "trans", translations, "current loss", loss)
        return loss

    def _check_initial_loss(self, verbose=False) -> NoReturn:
        size = np.size(self.motor_types)
        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        initial_loss = lossfn_obj_this.loss(np.atleast_1d(np.zeros(size)), verbose=False)
        if initial_loss >= self._no_beam_loss:
            raise EmptyBeamException("Initial beam is out of bounds.")
        print("Initial loss is", initial_loss)

    def reset(self) -> NoReturn:
        self.focusing_system = movers.move_motors(
            self.focusing_system, self.motor_types, self.initial_motor_positions, movement="absolute"
        )

        self._update_beam_state()

    def _get_guess_ranges(self, guess_range: List[float] = None):
        if guess_range is None:
            guess_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) / 2 for mt in self.motor_types]
        elif np.ndim(guess_range) == 1 and len(guess_range) == 2:
            guess_range = [guess_range for mt in self.motor_types]
        elif np.ndim(guess_range) != 2 or len(guess_range) != len(self.motor_types):
            raise ValueError("Invalid range supplied for guesses.")
        return guess_range

    def get_random_init(self, guess_range: List[float] = None, verbose=True):
        guess_range = self._get_guess_ranges(guess_range)

        initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
        lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
        guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)
        if verbose:
            print("Random guess", initial_guess, "has loss", guess_loss)

        while guess_loss >= self._no_beam_loss or np.isnan(guess_loss):
            self.reset()
            if verbose:
                print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
            if verbose:
                print("Random guess is", initial_guess)
            guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)
        return initial_guess

    @abc.abstractmethod
    def set_optimizer_options(self) -> NoReturn:
        pass

    @abc.abstractmethod
    def _optimize(self) -> Tuple[object, List[float], bool]:
        pass

    @abc.abstractmethod
    def trials(self, *args: List, **kwargs: Dict):
        pass
