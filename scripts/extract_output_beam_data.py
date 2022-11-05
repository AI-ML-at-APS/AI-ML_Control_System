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

import matplotlib.pyplot as plt
from matplotlib import cm

try:
    from mpl_toolkits.mplot3d import Axes3D  # necessario per caricare i plot 3D
except:
    pass

from aps.ai.autoalignment.common.util.shadow.common import get_shadow_beam_spatial_distribution, plot_shadow_beam_spatial_distribution, load_shadow_beam

def plot_3D(xx, yy, zz):
    figure = plt.figure(figsize=(10, 7))
    figure.patch.set_facecolor('white')

    axis = figure.add_subplot(111, projection='3d')

    axis.set_zlabel("Intensity")

    axis.clear()

    x_to_plot, y_to_plot = numpy.meshgrid(xx, yy)
    z_to_plot = zz

    axis.plot_surface(x_to_plot, y_to_plot, z_to_plot,
                           rstride=1, cstride=1, cmap=cm.autumn, linewidth=0.5, antialiased=True)

    axis.set_xlabel("X [mm]")
    axis.set_ylabel("Y [mm]")
    axis.set_zlabel("Intensity [A.U.]")
    axis.set_title("Spatial Distribution Plot")
    axis.mouse_init()

    plt.show()

if __name__ == "__main__":

    os.chdir("../work_directory")

    shadow_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # default plot
    plot_shadow_beam_spatial_distribution(shadow_beam, xrange=[-0.01, 0.01], yrange=[-0.01, 0.01])

    # extracting data 2D and statistical information
    shadow_histogram, statistical_data = get_shadow_beam_spatial_distribution(shadow_beam, do_gaussian_fit=True)

    plot_3D(shadow_histogram.hh, shadow_histogram.vv, shadow_histogram.data_2D)

    print(statistical_data.get_parameter("gaussian_fit"))

