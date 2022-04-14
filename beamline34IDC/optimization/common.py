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

from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.simulation.facade.focusing_optics_factory import simulated_focusing_optics_factory_method
from beamline34IDC.util.shadow.common import get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles, EmptyBeamException
from beamline34IDC.util import clean_up
import numpy as np
import abc
from beamline34IDC.optimization import movers, configs
from typing import Callable, NoReturn, Tuple, List, NamedTuple


class BeamParameterOutput(NamedTuple):
    parameter_value: float
    photon_beam: object
    hist: object
    dw: object


def reinitialize(input_beam_path: str, bender: bool = True) -> object:
    clean_up()
    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = simulated_focusing_optics_factory_method(implementor=Implementors.SHADOW, bender=bender)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    return focusing_system


def get_beam(focusing_system: object, random_seed: float = None, remove_lost_rays: bool = True) -> object:
    photon_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return photon_beam


def check_input_for_beam(focusing_system: object, photon_beam: object, random_seed: float) -> object:
    if photon_beam is None:
        if focusing_system is None:
            raise ValueError("Need to supply at least one of photon_beam or focusing_system.")
        photon_beam = get_beam(focusing_system, random_seed=random_seed)
    return photon_beam


def check_beam_out_of_bounds(focusing_system: object, photon_beam: object, random_seed: float) -> object:
    try:
        photon_beam = check_input_for_beam(focusing_system, photon_beam, random_seed)
    except Exception as exc:
        if (isinstance(exc, EmptyBeamException) or
                "Diffraction plane is set on Z, but the beam has no extention in that direction" in str(exc)):
            # Assuming that the beam is outside the screen and returning the default out of bounds value.
            photon_beam = None
        else:
            raise exc
    return photon_beam


def get_peak_intensity(focusing_system: object = None, photon_beam: object = None,
                       random_seed: float = None, out_of_bounds_value: float = 1e4) -> BeamParameterOutput:

    photon_beam = check_beam_out_of_bounds(focusing_system, photon_beam, random_seed)
    if photon_beam is None:
        return BeamParameterOutput(out_of_bounds_value, None, None, None)

    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    peak = dw.get_parameter('peak_intensity')
    return BeamParameterOutput(peak, photon_beam, hist, dw)


def get_centroid_distance(focusing_system: object = None, photon_beam: object = None,
                          random_seed: float = None, out_of_bounds_value: float = 1e4) -> BeamParameterOutput:

    photon_beam = check_beam_out_of_bounds(focusing_system, photon_beam, random_seed)
    if photon_beam is None:
        return BeamParameterOutput(out_of_bounds_value, None, None, None)

    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    h_centroid = dw.get_parameter('h_centroid')
    v_centroid = dw.get_parameter('v_centroid')
    centroid_distance = (h_centroid ** 2 + v_centroid ** 2) ** 0.5
    return BeamParameterOutput(centroid_distance, photon_beam, hist, dw)


def get_fwhm(focusing_system: object = None, photon_beam: object = None,
             random_seed: float = None, out_of_bounds_value: float = 1e4) -> BeamParameterOutput:

    photon_beam = check_beam_out_of_bounds(focusing_system, photon_beam, random_seed)
    if photon_beam is None:
        return BeamParameterOutput(out_of_bounds_value, None, None, None)

    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    h_fwhm = dw.get_parameter('h_fwhm')
    v_fwhm = dw.get_parameter('v_fwhm')
    fwhm = (h_fwhm ** 2 + v_fwhm ** 2) ** 0.5
    return BeamParameterOutput(fwhm, photon_beam, hist, dw)


class OptimizationCommon(abc.ABC):
    class TrialInstanceLossFunction:
        def __init__(self, opt_common: object, verbose: bool = False) -> NoReturn:
            self.opt_common = opt_common
            self.x_absolute_prev = 0
            self.current_loss = None
            self.verbose = verbose

        def loss(self, x_absolute_this: List[float], verbose: bool = None) -> float:
            if np.ndim(x_absolute_this) > 0:
                x_absolute_this = np.array(x_absolute_this)
            x_relative_this = x_absolute_this - self.x_absolute_prev
            self.x_absolute_prev = x_absolute_this
            self.current_loss = self.opt_common.loss_function(x_relative_this, verbose=False)
            verbose = verbose if verbose is not None else self.verbose
            if verbose:
                print("motors", self.opt_common.motor_types,
                      "trans", x_absolute_this, "current loss", self.current_loss)
            return self.current_loss

    def __init__(self, focusing_system: object,
                 motor_types: List[str],
                 initial_motor_positions: List[float] = None,
                 random_seed: int = None,
                 loss_parameters: List[str] = 'centroid',
                 loss_min_value: float = None) -> NoReturn:
        self.focusing_system = focusing_system
        self.motor_types = motor_types if np.ndim(motor_types) > 0 else [motor_types]
        self.random_seed = random_seed

        if initial_motor_positions is None:
            initial_motor_positions = np.zeros(len(self.motor_types))
        self.initial_motor_positions = initial_motor_positions

        self.loss_parameters = np.atleast_1d(loss_parameters)

        self._loss_function_list = []
        temp_loss_min_value = 0
        for loss_type in self.loss_parameters:
            if loss_type == 'centroid':
                self._loss_function_list.append(self.get_centroid_distance)
            elif loss_type == 'peak_intensity':
                print("Warning: Stopping condition for the peak intensity case is not supported.")
                self._loss_function_list.append(self.get_negative_log_peak_intensity)
            elif loss_type == 'fwhm':
                self._loss_function_list.append(self.get_fwhm)
            else:
                raise ValueError("Supplied loss parameter is not valid.")
            temp_loss_min_value += configs.DEFAULT_LOSS_TOLERANCES[loss_type]
        self._loss_function = lambda: np.sum([lossfn() for lossfn in self._loss_function_list])

        self._loss_min_value = temp_loss_min_value if loss_min_value is None else loss_min_value
        self._opt_trials_motor_positions = []
        self._opt_trials_losses = []
        self._opt_fn_call_counter = 0
        self._out_of_bounds_loss = 1e4 # this is a ridiculous arbitrarily high value.
        self.guesses_all = []
        self.results_all = []

    def get_beam(self) -> object:
        return get_beam(self.focusing_system, self.random_seed, remove_lost_rays=True)

    def get_negative_log_peak_intensity(self) -> float:
        peak, photon_beam, hist, dw = get_peak_intensity(focusing_system=self.focusing_system,
                                                         random_seed=self.random_seed,
                                                         out_of_bounds_value=self._out_of_bounds_loss)
        return -np.log(peak)

    def get_centroid_distance(self) -> float:
        centroid_distance, photon_beam, hist, dw = get_centroid_distance(focusing_system=self.focusing_system,
                                                                         random_seed=self.random_seed,
                                                                         out_of_bounds_value=self._out_of_bounds_loss)
        return centroid_distance

    def get_fwhm(self) -> float:
        fwhm, photon_beam, hist, dw = get_fwhm(focusing_system=self.focusing_system,
                                               random_seed=self.random_seed,
                                               out_of_bounds_value=self._out_of_bounds_loss)
        return fwhm

    def loss_function(self, translations: List[float], verbose: bool = True) -> float:
        """This mutates the state of the focusing system."""
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types, translations,
                                                  movement='relative')
        loss = self._loss_function()
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
        if initial_loss == self._out_of_bounds_loss:
            raise EmptyBeamException("Initial beam is out of bounds.")
        print("Initial loss is", initial_loss)

    def set_initial_motor_positions(self, motor_positions: List[float], movement: str = 'absolute') -> NoReturn:
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types,
                                                  motor_positions, movement=movement)
        self.initial_motor_positions = motor_positions

    @abc.abstractmethod
    def set_optimizer_options(self) -> NoReturn: pass

    @abc.abstractmethod
    def _optimize(self) -> Tuple[object, List[float], bool]: pass

    @abc.abstractmethod
    def trials(self, n_guesses: int = 5): pass
