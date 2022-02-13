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
from beamline34IDC.simulation.facade.focusing_optics_factory import focusing_optics_factory_method
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement

from beamline34IDC.util.shadow.common import plot_shadow_beam_spatial_distribution, load_shadow_beam, PreProcessorFiles
from beamline34IDC.util import clean_up

if __name__ == "__main__":
    verbose = False

    os.chdir("../work_directory")

    clean_up()

    input_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = focusing_optics_factory_method(implementor=Implementors.SHADOW)

    focusing_system.initialize(input_photon_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)

    # ----------------------------------------------------------------
    # perturbation of the incident beam to make adjustements necessary

    random_seed = 2120 # for repeatability

    focusing_system.perturbate_input_photon_beam(shift_h=0.0, shift_v=0.0)

    output_beam = focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed)

    plot_shadow_beam_spatial_distribution(output_beam, xrange=[-0.01, 0.01], yrange=[-0.01, 0.01])

    #--------------------------------------------------
    # interaction with the beamline

    focusing_system.change_vkb_shape(10, movement=Movement.RELATIVE)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    focusing_system.move_vkb_motor_3_pitch(1e-4, movement=Movement.RELATIVE)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          xrange=None, yrange=None)

    focusing_system.move_vkb_motor_4_translation(0.005, movement=Movement.RELATIVE)

    plot_shadow_beam_spatial_distribution(focusing_system.get_photon_beam(verbose=verbose, near_field_calculation=False, debug_mode=False, random_seed=random_seed),
                                          range=None, yrange=None)

    # ----------------------------------------------------------------

    clean_up()
