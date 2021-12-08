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

from orangecontrib.ml.util.data_structures import DictionaryWrapper
from beamline34IDC.util import clean_up
from beamline34IDC.raytracing.source_simulation import get_source_beam
from beamline34IDC.raytracing.beamline_simulation import run_invariant_shadow_simulation, run_ML_shadow_simulation, extract_output_parameters


import sys

if __name__ == "__main__":
    arguments = sys.argv[1:]

    try:
        input_beam = run_invariant_shadow_simulation(get_source_beam(arguments))

        #TODO: add displacement of the beam as input parameter: it simulates the real beam and let the ML system take action to compensate

        output_beam = run_ML_shadow_simulation(input_beam,
                                               #TODO: remove useless parameters and convert real movement into shadow movement
                                               input_features=DictionaryWrapper(coh_slits_h_aperture=0.03,
                                                                                coh_slits_h_center=0.0,
                                                                                coh_slits_v_aperture=0.07,
                                                                                coh_slits_v_center=0.0,
                                                                                vkb_p_distance=221,
                                                                                vkb_offset_x=0.0,
                                                                                vkb_offset_y=0.0,
                                                                                vkb_offset_z=0.0,
                                                                                vkb_rotation_x=0.0,
                                                                                vkb_rotation_y=0.0,
                                                                                vkb_rotation_z=0.0,
                                                                                hkb_p_distance=120,
                                                                                hkb_offset_x=0.0,
                                                                                hkb_offset_y=0.0,
                                                                                hkb_offset_z=0.0,
                                                                                hkb_rotation_x=0.0,
                                                                                hkb_rotation_y=0.0,
                                                                                hkb_rotation_z=0.0),
                                               use_benders=False,
                                               verbose=True)

        extract_output_parameters(output_beam=output_beam, plot=True)

        # TODO: extract as output the horizontal and vertical profile + convolution with particle as option (input size)

    except Exception as e:
        print(e)

    clean_up()
