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

from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.facade.focusing_optics_factory import focusing_optics_factory_method, ExecutionMode
from beamline34IDC.facade.focusing_optics_interface import Movement, AngularUnits, DistanceUnits

from beamline34IDC.util.shadow.common import plot_shadow_beam_spatial_distribution, load_shadow_beam, PreProcessorFiles
from beamline34IDC.util import clean_up
from beamline34IDC.util.wrappers import PlotMode

if __name__ == "__main__":
    verbose = False

    os.chdir("../../work_directory")

    clean_up()

    input_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION, implementor=Implementors.SHADOW, bender=2)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)

    print("Initial V-KB bender positions and q (up, down) ",
          focusing_system.get_vkb_motor_1_bender(units=DistanceUnits.MICRON),
          focusing_system.get_vkb_motor_2_bender(units=DistanceUnits.MICRON),
          focusing_system.get_vkb_q_distance())
    print("Initial H-KB bender positions and q (up, down)",
          focusing_system.get_hkb_motor_1_bender(units=DistanceUnits.MICRON),
          focusing_system.get_hkb_motor_2_bender(units=DistanceUnits.MICRON),
          focusing_system.get_hkb_q_distance())

    # ----------------------------------------------------------------
    # perturbation of the incident beam to make adjustements necessary

    random_seed = 2120 # for repeatability

    focusing_system.perturbate_input_photon_beam(shift_h=0.0, shift_v=0.0)

    output_beam = focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed)

    plot_shadow_beam_spatial_distribution(output_beam, xrange=[-0.01, 0.01], yrange=[-0.01, 0.01], plot_mode=PlotMode.NATIVE)

    #--------------------------------------------------
    # interaction with the beamline

    focusing_system.move_vkb_motor_2_bender(pos_downstream=-12.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    print("VKB Q", focusing_system.get_vkb_q_distance())

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=[-0.01, 0.01], yrange=[-0.01, 0.01], plot_mode=PlotMode.NATIVE)

    focusing_system.move_vkb_motor_1_bender(pos_upstream=-10.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    print("VKB Q", focusing_system.get_vkb_q_distance())

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=[-0.01, 0.01], yrange=[-0.01, 0.01])

    focusing_system.move_vkb_motor_3_pitch(0.1, movement=Movement.RELATIVE, units=AngularUnits.MILLIRADIANS)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    focusing_system.move_vkb_motor_4_translation(5, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    #--------------------------------------------------

    focusing_system.move_hkb_motor_1_bender(pos_upstream=5.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    print("HKB Q", focusing_system.get_hkb_q_distance())

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    focusing_system.move_hkb_motor_3_pitch(0.2, movement=Movement.RELATIVE, units=AngularUnits.MILLIRADIANS)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    focusing_system.move_hkb_motor_4_translation(-1, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    # ----------------------------------------------------------------

    clean_up()
