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
import numpy

from orangecontrib.ml.util.data_structures import ListOfParameters, DictionaryWrapper
from beamline34IDC.raytracing.beamline_simulation import run_ML_shadow_simulation, extract_output_parameters

def build_training_database(input_beam):

    # create a list of possible values to map the change in curvature of both the kb mirrors
    # FROM OASYS p value change in range +-20 mm with step 1 (visual scanning loop)

    # vkb_p_distances = numpy.arange(start=201.0, stop=242.0, step=1.0)
    # hkb_p_distances = numpy.arange(start=100.0, stop=141.0, step=1.0)
    vkb_p_distances = numpy.arange(start=219.0, stop=224.0, step=1.0)
    hkb_p_distances = numpy.arange(start=118.0, stop=123.0, step=1.0)

    input_features_list = ListOfParameters()

    for vkb_p_distance in vkb_p_distances:
        for hkb_p_distance in hkb_p_distances:
            input_features_list.add_parameters(DictionaryWrapper(coh_slits_h_aperture=0.03,
                                                                 coh_slits_h_center=0.0,
                                                                 coh_slits_v_aperture=0.07,
                                                                 coh_slits_v_center=0.0,
                                                                 vkb_p_distance=vkb_p_distance,
                                                                 vkb_offset_x=0.0,
                                                                 vkb_offset_y=0.0,
                                                                 vkb_offset_z=0.0,
                                                                 vkb_rotation_x=0.0,
                                                                 vkb_rotation_y=0.0,
                                                                 vkb_rotation_z=0.0,
                                                                 hkb_p_distance=hkb_p_distance,
                                                                 hkb_offset_x=0.0,
                                                                 hkb_offset_y=0.0,
                                                                 hkb_offset_z=0.0,
                                                                 hkb_rotation_x=0.0,
                                                                 hkb_rotation_y=0.0,
                                                                 hkb_rotation_z=0.0))

    # store the input file
    input_features_list.to_npy_file("input")

    # load the input file
    input_features_list = ListOfParameters()
    input_features_list.from_npy_file("input.npy")

    output_parameters_list = ListOfParameters()


    try:
        for index in range(input_features_list.get_number_of_parameters()):
            output_beam = run_ML_shadow_simulation(input_beam, input_features=input_features_list.get_parameters(index))
            output_parameters_list.add_parameters(extract_output_parameters(output_beam))
            print("Run simulation # ", index + 1)
    except Exception as e:
        print(e)

    output_parameters_list.to_npy_file("output.npy")

    #####################
    # SAVE/LOAD AS TRAINING DATA

    result = numpy.full((1, 2), None)
    result[0, 0] = input_features_list.to_numpy_matrix()
    result[0, 1] = output_parameters_list.to_numpy_matrix()

    numpy.save("database.npy", result, allow_pickle=True)
