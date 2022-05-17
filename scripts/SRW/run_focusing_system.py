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

from beamline34IDC.util.wrappers import get_distribution_info, plot_distribution, load_beam, save_beam
from beamline34IDC.util.common import PlotMode, AspectRatio, ColorMap

if __name__ == "__main__":
    verbose = False

    plot_mode = PlotMode.NATIVE
    aspect_ratio = AspectRatio.AUTO
    color_map = ColorMap.GRAY

    os.chdir("../../work_directory")

    input_beam = load_beam(Implementors.SRW, "primary_optics_system_srw_wavefront.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION, implementor=Implementors.SRW)

    focusing_system.initialize(input_photon_beam=input_beam, rewrite_height_error_profile_files=False)

    # ----------------------------------------------------------------
    # perturbation of the incident beam to make adjustements necessary

    focusing_system.perturbate_input_photon_beam(shift_h=0.0, shift_v=0.0)

    output_beam = focusing_system.get_photon_beam(verbose=verbose, debug_mode=False)

    plot_distribution(Implementors.SRW, output_beam,
                      xrange=[-0.005, 0.005], yrange=[-0.005, 0.005], title="Initial Beam",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    #--------------------------------------------------
    # interaction with the beamline

    focusing_system.change_vkb_shape(10, movement=Movement.RELATIVE)

    print("V-KB Q", focusing_system.get_vkb_q_distance(), "mm")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=[-0.005, 0.005], yrange=[-0.005, 0.005], title="Change V Shape",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    focusing_system.move_vkb_motor_3_pitch(0.1, movement=Movement.RELATIVE, units=AngularUnits.MILLIRADIANS)

    print("V-KB Pitch", focusing_system.get_vkb_motor_3_pitch(units=AngularUnits.MILLIRADIANS), "mrad")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=[-0.005, 0.005], yrange=None, title="Move V Pitch",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    focusing_system.move_vkb_motor_4_translation(10.0, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    print("V-KB Z", focusing_system.get_vkb_motor_4_translation(units=DistanceUnits.MILLIMETERS), "mm")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=[-0.005, 0.005], yrange=None, title="Move V Translation",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    #--------------------------------------------------

    focusing_system.change_hkb_shape(50, movement=Movement.RELATIVE)

    print("H-KB Q", focusing_system.get_hkb_q_distance(), "mm")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=[-0.005, 0.005], yrange=[-0.005, 0.005], title="Change H Shape",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    focusing_system.move_hkb_motor_3_pitch(-0.2, movement=Movement.RELATIVE, units=AngularUnits.MILLIRADIANS)

    print("H-KB Pitch", focusing_system.get_hkb_motor_3_pitch(units=AngularUnits.MILLIRADIANS), "mrad")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=None, yrange=None, title="Move H Pitch",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)

    focusing_system.move_hkb_motor_4_translation(20, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

    print("H-KB Z", focusing_system.get_hkb_motor_4_translation(units=DistanceUnits.MILLIMETERS), "mm")

    plot_distribution(Implementors.SRW, focusing_system.get_photon_beam(verbose=verbose, debug_mode=False),
                      xrange=None, yrange=None, title="Move H Translation",
                      plot_mode=plot_mode, aspect_ratio=aspect_ratio, color_map=color_map)


