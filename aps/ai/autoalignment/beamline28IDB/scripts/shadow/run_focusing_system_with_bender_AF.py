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
import os
import sys

import numpy

from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import focusing_optics_factory_method, ExecutionMode
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import get_default_input_features, Layout
from aps.ai.autoalignment.beamline28IDB.hardware.epics.focusing_optics import DISTANCE_V_MOTORS

from aps.ai.autoalignment.common.facade.parameters import Movement, AngularUnits, DistanceUnits

from aps.ai.autoalignment.common.util.common import PlotMode, AspectRatio, ColorMap
from aps.ai.autoalignment.common.util.wrappers import plot_distribution, load_beam
from aps.ai.autoalignment.common.util.shadow.common import PreProcessorFiles
from aps.ai.autoalignment.common.util import clean_up

from aps.ai.autoalignment.common.util.wrappers import EXPERIMENTAL_NOISE_TO_SIGNAL_RATIO

def get_v_bimorph_mirror_motor_pitch(di, do, u):
    return numpy.degrees(numpy.arcsin((di - get_v_bimorph_mirror_motor_translation(do, u)) /
                                      (0.5 * DISTANCE_V_MOTORS)))

def get_v_bimorph_mirror_motor_translation(do, u):
    return 0.5 * (u + do)

if __name__ == "__main__":
    verbose = False

    plot_mode = PlotMode.INTERNAL
    aspect_ratio = AspectRatio.CARTESIAN
    color_map = ColorMap.RAINBOW

    nbins_h = 2160
    nbins_v = 2560

    detector_x = nbins_h * 0.65 * 1e-3
    detector_y = nbins_v * 0.65 * 1e-3

    x_range = [-detector_x/2, detector_x/2]
    y_range = [-detector_y/2, detector_y/2]

    x_range = y_range = [-0.1, 0.1]
    nbins_h = int(nbins_h*0.2/detector_x)
    nbins_v = int(nbins_v*0.2/detector_y)

    add_noise = False #True
    noise     = EXPERIMENTAL_NOISE_TO_SIGNAL_RATIO * 70

    try: os.chdir("../../../../../../work_directory/28-ID")
    except: os.chdir("../../../../../work_directory/28-ID")

    clean_up()

    input_beam = load_beam(Implementors.SHADOW, "primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION, implementor=Implementors.SHADOW, bender=True)
    focusing_system.initialize(input_photon_beam=input_beam,
                               input_features=get_default_input_features(layout=Layout.AUTO_FOCUSING),
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               layout=Layout.AUTO_FOCUSING)

    print("Nr Shadow Rays: " + str(len(input_beam._beam.rays[numpy.where(input_beam._beam.rays[:, 9]==1)])))

    # ----------------------------------------------------------------
    # perturbation of the incident beam to make adjustements necessary

    random_seed = 2120 # for repeatability

    focusing_system.perturbate_input_photon_beam(shift_h=0.0, shift_v=0.0)

    '''
    hb 1    (V)  :-170.0
    hb 2    (V)  :-157.0
    hb pitch (deg):0.17164
    hb trans (deg):0.0
    vb       (V)  :384.0
    vb (mm)       : 0.54126, 0.54126, -0.54126
    
    hb up    (V)  :-184.0
    hb down  (V)  :-155.0
    hb pitch (deg):0.17089
    hb trans (deg):-0.013599999999999998
    vb       (V)  :372.0
    vb pitch (deg):0.17232866863693438
    vb trans (deg):0.015600000000000003
    '''

    '''
    focusing_system.move_h_bendable_mirror_motor_1_bender(-174,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_2_bender(-162,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_pitch(0.17202,   movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_h_bendable_mirror_motor_translation(0.0136, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)

    focusing_system.move_h_bendable_mirror_motor_1_bender(-174,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_2_bender(-161,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_pitch(0.17203,   movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_h_bendable_mirror_motor_translation(0.018, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)

    focusing_system.move_h_bendable_mirror_motor_1_bender(-173,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_2_bender(-162,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_pitch(0.17202,   movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_h_bendable_mirror_motor_translation(0.01, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
    
    Figure 7:
    focusing_system.move_h_bendable_mirror_motor_1_bender(-174,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_2_bender(-162,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_pitch(0.17202,   movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_h_bendable_mirror_motor_translation(0.0136, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
    '''

    focusing_system.move_h_bendable_mirror_motor_1_bender(-174,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_2_bender(-162,   movement=Movement.ABSOLUTE)
    focusing_system.move_h_bendable_mirror_motor_pitch(0.17202,   movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_h_bendable_mirror_motor_translation(0.0136, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)

    focusing_system.move_v_bimorph_mirror_motor_bender(420, movement=Movement.ABSOLUTE) # vertical focus
    #focusing_system.move_v_bimorph_mirror_motor_pitch(get_v_bimorph_mirror_motor_pitch(0.54126, 0.54126, -0.54126),    movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    #focusing_system.move_v_bimorph_mirror_motor_translation(get_v_bimorph_mirror_motor_translation(0.54126, -0.54126), movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
    focusing_system.move_v_bimorph_mirror_motor_pitch(0.1721,    movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
    focusing_system.move_v_bimorph_mirror_motor_translation(0.0, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)

    print(focusing_system.get_h_bendable_mirror_q_distance(), focusing_system.get_v_bimorph_mirror_q_distance())

    output_beam = focusing_system.get_photon_beam(near_field_calculation=True, verbose=verbose, debug_mode=False, random_seed=random_seed)

    plot_distribution(Implementors.SHADOW, output_beam,
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Initial Beam",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map, add_noise=add_noise, noise=noise)

    sys.exit(0)
    #--------------------------------------------------
    # interaction with the beamline

    focusing_system.move_h_bendable_mirror_motor_1_bender(-50, movement=Movement.ABSOLUTE)

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change H-KB Shape",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    focusing_system.move_h_bendable_mirror_motor_pitch(0.0005, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change H-KB Pitch",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    print(focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.MILLIRADIANS))

    focusing_system.move_h_bendable_mirror_motor_translation(10.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change H-KB Translation",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    print(focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MICRON))

    #--------------------------------------------------

    focusing_system.move_v_bimorph_mirror_motor_bender(450, movement=Movement.ABSOLUTE) # vertical focus

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change V-KB Shape",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)


    focusing_system.move_v_bimorph_mirror_motor_pitch(-0.0005, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change V-KB Pitch",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    print(focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.MILLIRADIANS))

    focusing_system.move_v_bimorph_mirror_motor_translation(-10.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    plot_distribution(Implementors.SHADOW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False, random_seed=random_seed),
                      xrange=x_range, yrange=y_range, nbins_h=nbins_h, nbins_v=nbins_v,
                      title="Change V-KB Translation",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    print(focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MICRON))

    # ----------------------------------------------------------------

    clean_up()
