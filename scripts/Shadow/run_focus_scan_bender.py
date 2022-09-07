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
import os, numpy
import sys

from beamline34IDC.simulation.facade import Implementors
from beamline34IDC.facade.focusing_optics_factory import focusing_optics_factory_method, ExecutionMode
from beamline34IDC.facade.focusing_optics_interface import Movement, DistanceUnits
from beamline34IDC.simulation.facade.focusing_optics_interface import get_default_input_features

from beamline34IDC.util.shadow.common import get_shadow_beam_spatial_distribution, load_shadow_beam, PreProcessorFiles
from beamline34IDC.util import clean_up

from plot_focus_scan_bender import plot_3D

if __name__ == "__main__":
    verbose = False

    os.chdir("../../work_directory")

    clean_up()

    input_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION, implementor=Implementors.SHADOW, bender=2)

    input_features = get_default_input_features()
    input_features.set_parameter("coh_slits_h_aperture", 0.15)
    input_features.set_parameter("coh_slits_v_aperture", 0.15)
    input_features.set_parameter("vkb_motor_1_bender_position", 138.0)
    input_features.set_parameter("vkb_motor_2_bender_position", 243.5)
    input_features.set_parameter("hkb_motor_1_bender_position", 215.5)
    input_features.set_parameter("hkb_motor_2_bender_position", 110.5)

    focusing_system.initialize(input_photon_beam=input_beam,
                               input_features=input_features,
                               power=1,
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

    random_seed = 2120 # for repeatability

    output_beam = focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=True, debug_mode=False, random_seed=random_seed)

    v_pos_up   = focusing_system.get_vkb_motor_1_bender(units=DistanceUnits.MICRON)
    v_pos_down = focusing_system.get_vkb_motor_2_bender(units=DistanceUnits.MICRON)
    h_pos_up   = focusing_system.get_hkb_motor_1_bender(units=DistanceUnits.MICRON)
    h_pos_down = focusing_system.get_hkb_motor_2_bender(units=DistanceUnits.MICRON)

    n_points = [25, 25]
    rel_pos = [-6.0, 6.0]
    xrange = [-0.005, 0.005]
    yrange = [-0.005, 0.005]

    v_abs_pos_up   = numpy.linspace(rel_pos[0], rel_pos[1], n_points[0]) + v_pos_up
    v_abs_pos_down = numpy.linspace(rel_pos[0], rel_pos[1], n_points[1]) + v_pos_down
    h_abs_pos_up   = numpy.linspace(rel_pos[0], rel_pos[1], n_points[0]) + h_pos_up
    h_abs_pos_down = numpy.linspace(rel_pos[0], rel_pos[1], n_points[1]) + h_pos_down

    sigma_v = numpy.zeros((len(v_abs_pos_up), len(v_abs_pos_down)))
    sigma_h = numpy.zeros((len(h_abs_pos_up), len(h_abs_pos_down)))

    positions_up = numpy.zeros((2, n_points[0]))
    positions_up[0, :] = v_abs_pos_up
    positions_up[1, :] = h_abs_pos_up
    positions_down = numpy.zeros((2, n_points[1]))
    positions_down[0, :] = v_abs_pos_down
    positions_down[1, :] = h_abs_pos_down
    with open("positions_up.npy", 'wb') as f: numpy.save(f, positions_up, allow_pickle=False)
    with open("positions_down.npy", 'wb') as f: numpy.save(f, positions_down, allow_pickle=False)

    min_v = +numpy.inf
    pos_min_v = None
    min_h = +numpy.inf
    pos_min_h = None

    for i in range(n_points[0]):
        focusing_system.move_vkb_motor_1_bender(pos_upstream=v_abs_pos_up[i],
                                                movement=Movement.ABSOLUTE,
                                                units=DistanceUnits.MICRON)
        focusing_system.move_hkb_motor_1_bender(pos_upstream=h_abs_pos_up[i],
                                                movement=Movement.ABSOLUTE,
                                                units=DistanceUnits.MICRON)

        for j in range(n_points[1]):
            focusing_system.move_vkb_motor_2_bender(pos_downstream=v_abs_pos_down[j],
                                                    movement=Movement.ABSOLUTE,
                                                    units=DistanceUnits.MICRON)
            focusing_system.move_hkb_motor_2_bender(pos_downstream=h_abs_pos_down[j],
                                                    movement=Movement.ABSOLUTE,
                                                    units=DistanceUnits.MICRON)

            try:
                _, dict = get_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose,
                                                                                               near_field_calculation=True,
                                                                                               debug_mode=False,
                                                                                               random_seed=random_seed),
                                                               nbins=201, xrange=xrange, yrange=yrange)

                s_v = dict.get_parameter("v_sigma")
                s_h = dict.get_parameter("h_sigma")

                if s_v < min_v:
                    min_v = s_v
                    pos_min_v = [v_abs_pos_up[i], v_abs_pos_down[j]]

                if s_h < min_h:
                    min_h = s_h
                    pos_min_h = [h_abs_pos_up[i], h_abs_pos_down[j]]

                sigma_v[i, j] = s_v
                sigma_h[i, j] = s_h
            except Exception as e:
                pass
                raise e

        print("Percentage completed: " + str(round(100*(1+i)*n_points[0] / (n_points[0]*n_points[1]), 2)))

    with open("sigma_v.npy", 'wb') as f: numpy.save(f, sigma_v, allow_pickle=False)
    with open("sigma_h.npy", 'wb') as f: numpy.save(f, sigma_h, allow_pickle=False)

    print("V-KB: sigma min " + str(min_v) + " found at (U,D): " + str(pos_min_v))
    print("H-KB: sigma min " + str(min_h) + " found at (U,D): " + str(pos_min_h))

    plot_3D(v_abs_pos_up, v_abs_pos_down, sigma_v*1e6, "Sigma (V)")
    plot_3D(h_abs_pos_up, h_abs_pos_down, sigma_h*1e6, "Sigma (H)")

    clean_up()
