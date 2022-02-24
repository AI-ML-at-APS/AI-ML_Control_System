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
from beamline34IDC.simulation.facade.focusing_optics_factory import focusing_optics_factory_method
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

from beamline34IDC.util.shadow.common import get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles, EmptyBeamException
from beamline34IDC.util import clean_up
import numpy as np
import scipy
import beamline34IDC.optimization.movers as movers


def reinitialize(input_beam_path, random_seed=None, remove_lost_rays=True):
    clean_up()

    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    return focusing_system


def get_beam(focusing_system, random_seed=None, remove_lost_rays=True):
    photon_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return photon_beam


def check_input_for_beam(focusing_system, photon_beam, random_seed):
    if photon_beam is None:
        if focusing_system is None:
            raise ValueError("Need to supply at least one of photon_beam or focusing_system.")
        photon_beam = get_beam(focusing_system, random_seed=random_seed)
    return photon_beam


def get_peak_intensity(focusing_system=None, photon_beam=None, random_seed=None):
    try:
        photon_beam = check_input_for_beam(focusing_system, photon_beam, random_seed)
    except EmptyBeamException:
        # Assuming that the beam is outside the screen and returning 0 as a default value.
        return 0, None, None, None
    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    peak = dw.get_parameter('peak_intensity')
    return peak, photon_beam, hist, dw


def get_centroid_distance(focusing_system=None, photon_beam=None, random_seed=None):
    try:
        photon_beam = check_input_for_beam(focusing_system, photon_beam, random_seed)
    except EmptyBeamException:
        # Assuming that the beam is outside the screen and returning 0 as a default value.
        return np.inf, None, None, None
    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    h_centroid = dw.get_parameter('h_centroid')
    v_centroid = dw.get_parameter('v_centroid')
    centroid_distance = (h_centroid ** 2 + v_centroid ** 2) ** 0.5
    return centroid_distance, photon_beam, hist, dw


def get_fwhm(focusing_system=None, photon_beam=None, random_seed=None):
    try:
        photon_beam = check_input_for_beam(focusing_system, photon_beam, random_seed)
    except EmptyBeamException:
        # Assuming that the beam is outside the screen and returning 0 as a default value.
        return np.inf, None, None, None
    hist, dw = get_shadow_beam_spatial_distribution(photon_beam)
    h_fwhm = dw.get_parameter('h_fwhm')
    v_fwhm = dw.get_parameter('v_fwhm')
    fwhm = (h_fwhm ** 2 + v_fwhm ** 2) ** 0.5
    return fwhm, photon_beam, hist, dw


class OptimizationCommon:
    class TrialInstanceLossFunction:
        def __init__(self, opt_common, verbose=False):
            self.opt_common = opt_common
            self.x_absolute_prev = 0
            self.current_loss = None
            self.verbose = verbose

        def loss(self, x_absolute_this):
            if np.ndim(x_absolute_this) > 0:
                x_absolute_this = np.array(x_absolute_this)
            x_relative_this = x_absolute_this - self.x_absolute_prev
            self.x_absolute_prev = x_absolute_this
            self.current_loss = self.opt_common.loss_function(x_relative_this, verbose=False)
            if self.verbose:
                print("motors", self.opt_common.motor_types,
                      "trans", x_absolute_this, "current loss", self.current_loss)
            return self.current_loss

    def __init__(self, focusing_system, motor_types,
                 initial_motor_positions=None, random_seed=None,
                 loss_parameter='centroid', loss_min_value=None):
        self.focusing_system = focusing_system
        self.motor_types = motor_types if np.ndim(motor_types) >0 else [motor_types]
        self.random_seed = random_seed

        if initial_motor_positions is None:
            self.initial_motor_positions = np.zeros(len(self.motor_types))
        self.initial_motor_positions = initial_motor_positions

        self._default_optimization_fn = self.scipy_optimize

        if loss_parameter == 'centroid':
            self._loss_function = self.get_centroid_distance
            self._loss_min_value = loss_min_value if loss_min_value is not None else 5e-4
        elif loss_parameter == 'peak_intensity':
            self._loss_function = self.get_negative_peak_intensity
            self._loss_min_value = loss_min_value if loss_min_value is not None else -40
        elif loss_parameter == 'fwhm':
            self._loss_function = self.get_fwhm
            self._loss_min_value = loss_min_value if loss_min_value is not None else 5e-3
        else:
            raise ValueError("Supplied loss parameter is not valid.")

    def get_beam(self):
        return get_beam(self.focusing_system, self.random_seed, remove_lost_rays=True)

    def get_negative_peak_intensity(self):
        peak, photon_beam, hist, dw = get_peak_intensity(focusing_system=self.focusing_system,
                                                      random_seed=self.random_seed)
        return -peak

    def get_centroid_distance(self):
        centroid_distance, photon_beam, hist, dw = get_centroid_distance(focusing_system=self.focusing_system,
                                                                      random_seed=self.random_seed)
        return centroid_distance

    def get_fwhm(self):
        fwhm, photon_beam, hist, dw = get_fwhm(focusing_system=self.focusing_system,
                                            random_seed=self.random_seed)
        return fwhm

    def loss_function(self, translations, verbose=True):
        """This mutates the state of the focusing system."""
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types, translations,
                                                  movement='relative')
        loss = self._loss_function()
        if verbose:
            print("motors", self.motor_types, "trans", translations, "current loss", loss)
        return loss

    def scipy_optimize(self, lossfn, initial_guess):

        opt_result = scipy.optimize.minimize(lossfn, initial_guess, method='Nelder-Mead',
                                             options={'adaptive': True}) #'maxiter': 50,
        loss = opt_result.fun
        sol = opt_result.x
        status = opt_result.status
        if loss < self._loss_min_value:
            return opt_result, sol, True
        return opt_result, sol, False

    def set_gaussian_process_optimizer(self, bounds, **default_opt_params):
        """Changes the default optimization from scipy-based to Nelder-Mead method to
        using Gaussian Processes from skopt. Need to explicitly set bounds on the variables if we want
        to use this optimizer."""
        if np.ndim(bounds) == 1:
            bounds = np.array([bounds])
        if len(bounds) != len(self.motor_types):
            raise ValueError
        self._optimization_bounds = bounds
        self._default_opt_params = default_opt_params
        self._default_optimization_fn = self.skopt_gp_optimize

    def skopt_gp_optimize(self, lossfn, initial_guess):
        import skopt
        # print(initial_guess)
        opt_result = skopt.gp_minimize(lossfn, self._optimization_bounds, **self._default_opt_params)
        loss = opt_result.fun
        sol = opt_result.x
        if loss < 5e-4:
            return opt_result, sol, True
        return opt_result, sol, False

    def set_initial_motor_positions(self, motor_positions, movement='absolute'):
        self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types,
                                                  motor_positions, movement=movement)
        self.initial_motor_positions = motor_positions

    def trials(self, n_guesses=5, verbose=False, guess_min=-0.05, guess_max=0.05):
        guesses_all = []
        results_all = []
        for n_trial in range(n_guesses):
            guess_this = np.random.uniform(guess_min, guess_max, size=np.size(self.motor_types))
            guess_this = np.atleast_1d(guess_this)

            lossfn_obj_this = self.TrialInstanceLossFunction(self, verbose=verbose)
            print("Initial loss is", lossfn_obj_this.loss(np.zeros_like(guess_this)))

            result, solution, success_status = self._default_optimization_fn(lossfn_obj_this.loss, guess_this)
            guesses_all.append(guess_this)
            results_all.append(result)
            if success_status:
                return results_all, guesses_all, solution, True

            if n_trial < n_guesses:
                self.focusing_system = movers.move_motors(self.focusing_system, self.motor_types,
                                                          self.initial_motor_positions, movement='absolute')
            # centroid, photon_beam, *_ = getCentroidDistance()

        return results_all, guesses_all, solution, False