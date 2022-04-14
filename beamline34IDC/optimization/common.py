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
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

from beamline34IDC.util.shadow.common import get_shadow_beam_spatial_distribution,\
    load_shadow_beam, PreProcessorFiles, EmptyBeamException
from beamline34IDC.util import clean_up
import numpy as np
import scipy

MOTOR_TYPES = ['vkb_4', 'hkb_4']

def getBeam(focusing_system, random_seed=None, remove_lost_rays=True):
    out_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return out_beam


def getPeakIntensity(focusing_system, random_seed=None):
    try:
        out_beam = getBeam(focusing_system, random_seed=random_seed)
    except EmptyBeamException:
        # Assuming that the beam is outside the screen and returning 0 as a default value.
        return 0
    hist, dw = get_shadow_beam_spatial_distribution(out_beam)
    peak = dw.get_parameter('peak_intensity')
    return peak, out_beam, hist, dw


def getCentroidDistance(focusing_system, random_seed=None):
    try:
        out_beam = getBeam(focusing_system, random_seed=random_seed)
    except EmptyBeamException:
        # Assuming that the centroid is outside the screen and returning 0.5 microns as a default value.
        return 0.5,
    hist, dw = get_shadow_beam_spatial_distribution(out_beam)
    h_centroid = dw.get_parameter('h_centroid')
    v_centroid = dw.get_parameter('v_centroid')
    centroid_distance = (h_centroid ** 2 + v_centroid ** 2) ** 0.5
    return centroid_distance, out_beam, hist, dw


def moveHkb4(focusing_system, trans, movement=Movement.RELATIVE):
    focusing_system.move_hkb_motor_4_translation(trans, movement=movement)
    return focusing_system

def moveVkb4(focusing_system, trans, movement=Movement.RELATIVE):
    focusing_system.move_vkb_motor_4_translation(trans, movement=movement)
    return focusing_system

def moveMotors(focusing_system,  motor_types, translations, movement='relative'):
    if movement == 'relative' or movement == Movement.RELATIVE:
        movement = Movement.RELATIVE
    elif movement == 'absolute' or movement == Movement.ABSOLUTE:
        movement = Movement.ABSOLUTE
    else:
        raise ValueError
    if np.ndim(motor_types) == 0:
        motor_types = [motor_types]
    if np.ndim(translations) == 0:
        translations = [translations]
    for tdx, motor_type in zip(translations, motor_types):
        if motor_type == 'vkb_4':
            focusing_system = moveVkb4(focusing_system, tdx, movement=movement)
        elif motor_type == 'hkb_4':
            focusing_system = moveHkb4(focusing_system, tdx, movement=movement)
        else:
            raise ValueError
    return focusing_system

def lossFunction(focusing_system, motor_types, translations, random_seed=None, movement='relative', verbose=True):
    focusing_system = moveMotors(focusing_system,  motor_types, translations, movement)
    centroid_distance, *__ = getCentroidDistance(focusing_system,
                                                 random_seed=random_seed)
    if verbose:
        print("motors", motor_types, "trans", translations, "current loss", centroid_distance)
    return focusing_system, centroid_distance


def reinitialize(input_beam_path, random_seed=None, remove_lost_rays=True):
    clean_up()

    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = simulated_focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)
    output_beam = focusing_system.get_photon_beam(random_seed=random_seed, remove_lost_rays=remove_lost_rays)
    return focusing_system, output_beam


class OptimizationLossFunction:
    def __init__(self, focusing_system, motor_types, random_seed):
        self.x_absolute_prev = 0
        self.random_seed = random_seed
        self.focusing_system = focusing_system
        self.motor_types = motor_types
        self.current_loss = None

    def loss(self, x_absolute_this, verbose=False):
        x_relative_this = x_absolute_this - self.x_absolute_prev
        self.x_absolute_prev = x_absolute_this
        self.focusing_system, self.current_loss = lossFunction(self.focusing_system, self.motor_types, x_relative_this,
                                                       random_seed=self.random_seed, verbose=verbose)
        return self.current_loss


def optimizationTrials(focusing_system, motor_types, initial_positions, random_seed,
                       n_guesses=5, verbose=False):
    guesses_all = []
    focusing_system_this = focusing_system
    for n_trial in range(n_guesses):
        guess_this = np.random.uniform(-0.05, 0.05, size=np.size(initial_positions))
        guess_this = np.atleast_1d(guess_this)
        lossfn_obj = OptimizationLossFunction(focusing_system_this, motor_types, random_seed)
        lossfn = lambda x: lossfn_obj.loss(x, verbose=verbose)

        print("Initial loss is", lossfn(np.zeros_like(guess_this)))
        print(guess_this)
        opt = scipy.optimize.minimize(lossfn, guess_this,
                                      method='Nelder-Mead',
                                      options={'maxiter': 50, 'adaptive': True})
        guesses_all.append(guess_this)
        focusing_system_this = lossfn_obj.focusing_system
        if opt.fun < 5e-4:
            return focusing_system_this, guesses_all, True
        focusing_system_this = moveMotors(focusing_system_this, motor_types, initial_positions)
        centroid, out_beam, *_ = getCentroidDistance()

    return focusing_system_this, guesses_all, False