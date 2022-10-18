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
from typing import List, NamedTuple, NoReturn, Tuple

import numpy as np
from aps.ai.autoalignment.common.facade.parameters import ExecutionMode
from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors

from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import DictionaryWrapper, Histogram
from aps.ai.autoalignment.common.util.shadow.common import EmptyBeamException, HybridFailureException, PreProcessorFiles, \
                                             get_shadow_beam_spatial_distribution, load_shadow_beam

from aps.ai.autoalignment.beamline34IDC.facade.focusing_optics_factory import focusing_optics_factory_method
from aps.ai.autoalignment.beamline34IDC.optimization import configs, movers
from aps.ai.autoalignment.beamline34IDC.simulation.facade.focusing_optics_interface import get_default_input_features


class BeamState(NamedTuple):
    photon_beam: object
    hist: Histogram
    dw:  DictionaryWrapper


class BeamParameterOutput(NamedTuple):
    parameter_value: float
    photon_beam: object
    hist: Histogram
    dw:  DictionaryWrapper


def reinitialize(input_beam_path: str, bender: int = 1) -> object:
    clean_up()
    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION, implementor=Implementors.SHADOW, bender=2)

    input_features = get_default_input_features()

    focusing_system.initialize(input_photon_beam=input_beam,
                                input_features=input_features,
                                power=1,
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
        if (isinstance(exc, EmptyBeamException) or (isinstance(exc, HybridFailureException)) or
                "Diffraction plane is set on Z, but the beam has no extention in that direction" in str(exc)):
            # Assuming that the beam is outside the screen and returning the default out of bounds value.
            photon_beam = None
        else:
            raise exc
    return photon_beam


def get_beam_hist_dw(focusing_system: object = None,
                     photon_beam: object = None,
                     random_seed: float = None,
                     xrange: List[float] = None,
                     yrange: List[float] = None,
                     nbins: int = 256,
                     do_gaussian_fit: bool = False):
    photon_beam = check_beam_out_of_bounds(focusing_system, photon_beam, random_seed)
    if photon_beam is None:
        return None, None, None
    hist, dw = get_shadow_beam_spatial_distribution(photon_beam, xrange=xrange, yrange=yrange, nbins=nbins,
                                                    do_gaussian_fit=do_gaussian_fit)
    return BeamState(photon_beam, hist, dw)


def get_peak_intensity(focusing_system: object = None, photon_beam: object = None,
                       random_seed: float = None, no_beam_value: float = 1e4,
                       xrange: List[float] = None, yrange: List[float] = None,
                       nbins: int = 256, use_gaussian_fit: bool = False) -> BeamParameterOutput:

    photon_beam, hist, dw = get_beam_hist_dw(focusing_system, photon_beam, random_seed, xrange, yrange, nbins,
                                             do_gaussian_fit=use_gaussian_fit)
    if dw is not None:
        if use_gaussian_fit:
            gf = dw.get_parameter('gaussian_fit')
            if not gf:
                peak = no_beam_value
            else:
                peak = dw.get_parameter('gaussian_fit')['amplitude']
        else:
            peak = dw.get_parameter('peak_intensity')
    else:
        peak = no_beam_value
    return BeamParameterOutput(peak, photon_beam, hist, dw)


def _get_centroid_distance_from_dw(dw: object, use_gaussian_fit: bool = False,
                                   no_beam_value: float = 1e4) -> float:
    if dw is None:
        return no_beam_value

    if use_gaussian_fit:
        gf = dw.get_parameter('gaussian_fit')
        if not gf:
            return no_beam_value
        h_centroid = gf['center_x']
        v_centroid = gf['center_y']
    else:
        h_centroid = dw.get_parameter('h_centroid')
        v_centroid = dw.get_parameter('v_centroid')
    centroid_distance = (h_centroid ** 2 + v_centroid ** 2) ** 0.5
    return centroid_distance


def get_centroid_distance(focusing_system: object = None, photon_beam: object = None,
                          random_seed: float = None, no_beam_value: float = 1e4,
                          xrange: List[float] = None, yrange: List[float] = None,
                          nbins: int = 256,
                          use_gaussian_fit: bool = False) -> BeamParameterOutput:

    photon_beam, hist, dw = get_beam_hist_dw(focusing_system, photon_beam, random_seed, xrange, yrange, nbins,
                                             use_gaussian_fit)
    centroid_distance = _get_centroid_distance_from_dw(dw, use_gaussian_fit, no_beam_value)
    return BeamParameterOutput(centroid_distance, photon_beam, hist, dw)


def _get_fwhm_from_dw(dw: object, use_gaussian_fit: bool = False,
                      no_beam_value: float = 1e4) -> float:
    if dw is None:
        return no_beam_value
    if use_gaussian_fit:
        gf = dw.get_parameter('gaussian_fit')
        if not gf:
            return no_beam_value
        h_fwhm = dw.get_parameter('gaussian_fit')['fwhm_x']
        v_fwhm = dw.get_parameter('gaussian_fit')['fwhm_y']
    else:
        h_fwhm = dw.get_parameter('h_fwhm')
        v_fwhm = dw.get_parameter('v_fwhm')
    fwhm = (h_fwhm ** 2 + v_fwhm ** 2) ** 0.5
    return fwhm

def _get_sigma_from_dw(dw: object, use_gaussian_fit: bool = False,
                      no_beam_value: float = 1e4) -> float:
    if dw is None:
        return no_beam_value
    if use_gaussian_fit:
        gf = dw.get_parameter('gaussian_fit')
        if not gf:
            return no_beam_value
        h_sigma = dw.get_parameter('gaussian_fit')['sigma_x']
        v_sigma = dw.get_parameter('gaussian_fit')['sigma_y']
    else:
        h_sigma = dw.get_parameter('h_sigma')
        v_sigma = dw.get_parameter('v_sigma')
    sigma = (h_sigma ** 2 + v_sigma ** 2) ** 0.5
    return sigma


def get_fwhm(focusing_system: object = None, photon_beam: object = None,
             random_seed: float = None, no_beam_value: float = 1e4,
             xrange: List[float] = None, yrange: List[float] = None,
             nbins: int = 256, use_gaussian_fit: bool = False) -> BeamParameterOutput:

    photon_beam, hist, dw = get_beam_hist_dw(focusing_system, photon_beam, random_seed, xrange, yrange, nbins,
                                             use_gaussian_fit)
    fwhm = _get_fwhm_from_dw(dw, use_gaussian_fit, no_beam_value)
    return BeamParameterOutput(fwhm, photon_beam, hist, dw)

def get_sigma(focusing_system: object = None, photon_beam: object = None,
             random_seed: float = None, no_beam_value: float = 1e4,
             xrange: List[float] = None, yrange: List[float] = None,
             nbins: int = 256, use_gaussian_fit: bool = False) -> BeamParameterOutput:

    photon_beam, hist, dw = get_beam_hist_dw(focusing_system, photon_beam, random_seed, xrange, yrange, nbins,
                                             use_gaussian_fit)
    sigma = _get_sigma_from_dw(dw, use_gaussian_fit, no_beam_value)
    return BeamParameterOutput(sigma, photon_beam, hist, dw)


def get_random_init(focusing_system, motor_types: List[str], init_range: List[float]=None, verbose=True,
                    integral_threshold: float = None, random_seed: int = None,
                    **hist_kwargs):
    
    for mt in motor_types:
        if mt not in configs.DEFAULT_MOVEMENT_RANGES: 
            raise ValueError
    
    if init_range is None:
        init_range = [np.array(configs.DEFAULT_MOVEMENT_RANGES[mt]) for mt in motor_types]
    elif np.ndim(init_range) == 1 and len(init_range) == 2:
        init_range = [init_range for mt in motor_types]
    elif np.ndim(init_range) != 2 or len(init_range) != len(motor_types):
        raise ValueError("Invalid range supplied for inits.")
    
    initial_motor_positions = movers.get_absolute_positions(focusing_system, motor_types)
    regenerate = True
    while regenerate:
        initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in init_range]
        movers.move_motors(focusing_system, motor_types, initial_guess,  movement='relative')
        centroid, photon_beam, hist, dw = get_centroid_distance(focusing_system=focusing_system, random_seed=random_seed, 
                                                                **hist_kwargs)
        
        if not (centroid < 1e4):
            if verbose:
                print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            movers.move_motors(focusing_system, motor_types, initial_motor_positions, movement='absolute')
            continue
        
        if integral_threshold is not None:
            if hist.data_2D.sum() <= integral_threshold:
                movers.move_motors(focusing_system, motor_types, initial_motor_positions, movement='absolute')
                continue
        regenerate = False
    print("Random initialization is", motor_types, initial_guess)
    return initial_guess, focusing_system, photon_beam, hist, dw


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
                abs_trans_str = ''.join([f'{x:8.2g}' for x in x_absolute_this])
                rel_trans_str = ''.join([f'{x:8.2g}' for x in x_relative_this])
                print(f"motors {self.opt_common.motor_types} trans abs {abs_trans_str} trans rel {rel_trans_str}"
                      f"current loss {self.current_loss:4.3g}")

                absolute_set = np.array(self.opt_common.initial_motor_positions) + np.array(x_absolute_this)
                absolute_true = movers.get_absolute_positions(self.opt_common.focusing_system, 
                                                              self.opt_common.motor_types)
                abs_true_str = ''.join([f'{x:8.2g}' for x in absolute_true])
                abs_set_str = ''.join([f'{x:8.2g}' for x in absolute_set])
                print(f"motors {self.opt_common.motor_types} absolute movements are set to {abs_set_str}")
                print(f"motors {self.opt_common.motor_types} absolute movements are actually {abs_true_str}")

                
            return self.current_loss

    def __init__(self, focusing_system: object,
                 motor_types: List[str], 
                 random_seed: int = None,
                 loss_parameters: List[str] = 'centroid',
                 loss_min_value: float = None,
                 camera_xrange: List[float] = None,
                 camera_yrange: List[float] = None,
                 camera_nbins: int = 256,
                 use_gaussian_fit: bool = False,
                 no_beam_loss: float = 1e4,
                 multi_objective_optimization: bool = False) -> NoReturn:
        self.focusing_system = focusing_system
        self.motor_types = motor_types if np.ndim(motor_types) > 0 else [motor_types]
        self.random_seed = random_seed

        self.initial_motor_positions = movers.get_absolute_positions(focusing_system, self.motor_types)
        
        self.multi_objective_optimization = multi_objective_optimization
        self.loss_parameters = list(np.atleast_1d(loss_parameters))

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
            elif loss_type == 'sigma':
                self._loss_function_list.append(self.get_sigma)
            else:
                raise ValueError("Supplied loss parameter is not valid.")
            temp_loss_min_value += configs.DEFAULT_LOSS_TOLERANCES[loss_type]

        self._loss_min_value = temp_loss_min_value if loss_min_value is None else loss_min_value
        self._opt_trials_motor_positions = []
        self._opt_trials_losses = []
        self._opt_fn_call_counter = 0
        self._no_beam_loss = no_beam_loss # this is a ridiculous arbitrarily high value.

        self._camera_xrange = camera_xrange
        self._camera_yrange = camera_yrange
        self._camera_nbins = camera_nbins
        self._use_gaussian_fit = use_gaussian_fit

        cond1 = camera_xrange is not None and np.size(camera_xrange) != 2
        cond2 = camera_yrange is not None and np.size(camera_yrange) != 2
        if cond1 or cond2:
            raise ValueError("Enter limits for xrange (and yrange) in the format [xrange_min, xramge_max] "
                             + "in units of microns.")
        self._update_beam_state()
        self.guesses_all = []
        self.results_all = []

    def _update_beam_state(self) -> NoReturn:
        current_beam, current_hist, current_dw = get_beam_hist_dw(
            focusing_system=self.focusing_system,
            random_seed=self.random_seed,
            xrange=self._camera_xrange,
            yrange=self._camera_yrange,
            nbins=self._camera_nbins,
            do_gaussian_fit=self._use_gaussian_fit
        )

        if current_hist is None:
            current_hist = Histogram(hh=np.zeros(self._camera_nbins),
                                           vv=np.zeros(self._camera_nbins),
                                           data_2D=np.zeros((self._camera_nbins, self._camera_nbins)))
        self.beam_state = BeamState(current_beam, current_hist, current_dw)

    def get_beam(self) -> object:
        return get_beam(self.focusing_system, self.random_seed, remove_lost_rays=True)

    def get_negative_log_peak_intensity(self) -> float:
        if self.beam_state.dw is not None:
            peak_intensity = self.beam_state.dw.get_parameter('peak_intensity')
        else:
            peak_intensity = self._no_beam_loss
        return -np.log(peak_intensity, self._use_gaussian_fit)

    def get_centroid_distance(self) -> float:
        return _get_centroid_distance_from_dw(self.beam_state.dw, self._use_gaussian_fit, self._no_beam_loss)

    def get_fwhm(self) -> float:
        return _get_fwhm_from_dw(self.beam_state.dw, self._use_gaussian_fit, self._no_beam_loss)
    
    def get_sigma(self) -> float:
        return _get_sigma_from_dw(self.beam_state.dw, self._use_gaussian_fit, self._no_beam_loss)

    def loss_function(self, translations: List[float], verbose: bool = True) -> float:
        """This mutates the state of the focusing system."""

        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types, translations,
                                                  movement='relative')
        self._update_beam_state()
        
        loss = np.array([lossfn() for lossfn in self._loss_function_list])
        if not self.multi_objective_optimization:
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
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types,
                                                  self.initial_motor_positions, movement='absolute')
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
            print('Random guess', initial_guess, 'has loss', guess_loss)

        while guess_loss >= self._no_beam_loss or np.isnan(guess_loss):
            self.reset()
            if verbose:
                print("Random guess", initial_guess, "produces beam out of bounds. Trying another guess.")
            initial_guess = [np.random.uniform(m1, m2) for (m1, m2) in guess_range]
            if verbose:
                print('Random guess is', initial_guess)
            guess_loss = lossfn_obj_this.loss(initial_guess, verbose=False)
        return initial_guess


    @abc.abstractmethod
    def set_optimizer_options(self) -> NoReturn: pass

    @abc.abstractmethod
    def _optimize(self) -> Tuple[object, List[float], bool]: pass

    @abc.abstractmethod
    def trials(self, n_guesses: int = 5): pass
