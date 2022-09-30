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
import numpy
import pickle
from srxraylib.metrology import dabam
from oasys.util.error_profile_util import DabamInputParameters, calculate_dabam_profile
from orangecontrib.srw.util.srw_util import write_error_profile_file
from oasys_srw.uti_plot import uti_plot2d1d, uti_plot_init, uti_plot_show
from oasys_srw.srwlib import array, srwl, deepcopy

from aps_ai.common.util.common import get_info, plot_2D, Flip, PlotMode, AspectRatio, ColorMap

uti_plot_init(backend="Qt5Agg")

def __get_arrays(srw_wavefront):
    _, x_array, y_array, i = srw_wavefront.get_intensity(multi_electron=False)

    x_array *= 1000  # mm
    y_array *= 1000  # mm
    z_array = i[0]

    return x_array, y_array, z_array

def get_srw_wavefront_distribution_info(srw_wavefront, xrange=None, yrange=None, do_gaussian_fit=False):
    x_array, y_array, z_array = __get_arrays(srw_wavefront)

    return get_info(x_array, y_array, z_array, xrange, yrange, do_gaussian_fit)

def plot_srw_wavefront_spatial_distribution(srw_wavefront, title="X,Z", xrange=None, yrange=None, plot_mode=PlotMode.INTERNAL, aspect_ratio=AspectRatio.AUTO, color_map=ColorMap.RAINBOW):
    if plot_mode in [PlotMode.INTERNAL, PlotMode.BOTH]:
        x_array, y_array, z_array = __get_arrays(srw_wavefront)

        plot_2D(x_array, y_array, z_array, title, xrange, yrange, flip=Flip.VERTICAL, aspect_ratio=aspect_ratio, color_map=color_map)

    if plot_mode in [PlotMode.NATIVE, PlotMode.BOTH]:
        mesh = deepcopy(srw_wavefront.mesh)
        arI = array('f', [0] * mesh.nx * mesh.ny)  # "flat" 2D array to take intensity data
        srwl.CalcIntFromElecField(arI, srw_wavefront, 6, 0, 3, mesh.eStart, 0, 0)

        plotMeshx = [1000 * mesh.xStart, 1000 * mesh.xFin, mesh.nx]
        plotMeshy = [1000 * mesh.yStart, 1000 * mesh.yFin, mesh.ny]

        uti_plot2d1d(arI, plotMeshx, plotMeshy, labels=['Horizontal Position [mm]', 'Vertical Position [mm]', title])
        uti_plot_show()

def save_srw_wavefront(srw_wavefront, file_name="srw_wavefront.dat"):
    out_s = open(os.path.join(os.getcwd(),  file_name), 'wb')
    pickle.dump(srw_wavefront, out_s)
    out_s.flush()
    out_s.close()

def load_srw_wavefront(file_name="srw_wavefront.dat"):
    in_s = open(os.path.join(os.getcwd(), file_name), 'rb')
    srw_wavefront = pickle.load(in_s)
    in_s.close()

    return srw_wavefront

def write_dabam_file(figure_error_rms=None, dabam_entry_number=20, heigth_profile_file_name="KB.dat", seed=8787):
    server = dabam.dabam()
    server.set_input_silent(True)
    server.set_server(dabam.default_server)
    server.load(dabam_entry_number)

    input_parameters = DabamInputParameters(dabam_server=server)
    input_parameters.si_to_user_units = 1.0
    input_parameters.center_y = 1
    input_parameters.modify_y = 2
    input_parameters.new_length_y = 0.1
    input_parameters.filler_value_y = 0.0
    if figure_error_rms is None:
        input_parameters.renormalize_y = 0
    else:
        input_parameters.renormalize_y = 1
        input_parameters.error_type_y = 0
        input_parameters.rms_y = 3.5
    input_parameters.kind_of_profile_x = 0
    input_parameters.dimension_x = 0.05
    input_parameters.step_x = 0.001
    input_parameters.power_law_exponent_beta_x = 2.0
    input_parameters.montecarlo_seed_x = seed
    input_parameters.error_type_x = 0
    input_parameters.rms_x = 0.5

    xx, yy, zz = calculate_dabam_profile(input_parameters)

    write_error_profile_file(zz, xx, yy, heigth_profile_file_name)

    return heigth_profile_file_name
