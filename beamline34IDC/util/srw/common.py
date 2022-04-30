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

from matplotlib import pyplot as plt
from matplotlib import cm

import numpy

from wofrysrw.propagator.wavefront2D.srw_wavefront import SRWWavefront
from wofrysrw.util.srw_hdf5 import load_hdf5_2_wfr, save_wfr_2_hdf5

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average
from orangecontrib.ml.util.data_structures import DictionaryWrapper

from beamline34IDC.util.common import Histogram
from beamline34IDC.util.gaussian_fit import calculate_2D_gaussian_fit

def srw_wavefront_get_distribution_info(srw_wavefront, do_gaussian_fit=False):
    _, x_array, y_array, i = srw_wavefront.get_intensity(multi_electron=False)
    x_array *= 1000 # mm
    y_array *= 1000 # mm
    z_array = i[0]

    ticket = {'error': 0}
    ticket['nbins_h'] = len(x_array)
    ticket['nbins_v'] = len(y_array)

    xrange = [x_array.min(), x_array.max()]
    yrange = [y_array.min(), y_array.max()]
    pixel_area = (x_array[1] - x_array[0]) * (y_array[1] - y_array[0])

    hh = z_array
    hh_h = hh.sum(axis=1)
    hh_v = hh.sum(axis=0)
    xx = x_array
    yy = y_array

    ticket['xrange'] = xrange
    ticket['yrange'] = yrange
    ticket['bin_h'] = xx
    ticket['bin_v'] = yy
    ticket['histogram'] = hh
    ticket['histogram_h'] = hh_h
    ticket['histogram_v'] = hh_v
    ticket['total'] = numpy.sum(z_array) * pixel_area

    ticket['fwhm_h'], ticket['fwhm_quote_h'], ticket['fwhm_coordinates_h'] = get_fwhm(hh_h, xx)
    ticket['sigma_h'] = get_sigma(hh_h, xx)

    ticket['fwhm_v'], ticket['fwhm_quote_v'], ticket['fwhm_coordinates_v'] = get_fwhm(hh_v, yy)
    ticket['sigma_v'] = get_sigma(hh_v, yy)
    ticket['is_multi_energy'] = False

    ticket['centroid_h'] = get_average(ticket['histogram_h'], ticket['bin_h'])
    ticket['centroid_v'] = get_average(ticket['histogram_v'], ticket['bin_v'])

    if do_gaussian_fit:
        try:    gaussian_fit = calculate_2D_gaussian_fit(data_2D=hh, x=xx, y=yy)
        except Exception as e:
            print("Gaussian fit failed: ", e)
            gaussian_fit = {}
    else:       gaussian_fit = {}

    return Histogram(hh=xx, vv=yy, data_2D=hh), \
           DictionaryWrapper(
               h_sigma=ticket['sigma_h'],
               h_fwhm=ticket['fwhm_h'],
               h_centroid=ticket['centroid_h'],
               v_sigma=ticket['sigma_v'],
               v_fwhm=ticket['fwhm_v'],
               v_centroid=ticket['centroid_v'],
               integral_intensity=ticket['total'],
               peak_intensity=numpy.average(hh[numpy.where(hh >= numpy.max(hh) * 0.90)]),
               gaussian_fit=gaussian_fit
    )


def plot_srw_wavefront_spatial_distribution(srw_wavefront, title="X,Z", xrange=None, yrange=None):
    _, x_array, y_array, i = srw_wavefront.get_intensity(multi_electron=False)
    x_array *= 1000 # mm
    y_array *= 1000 # mm
    z_array = i[0]

    if xrange is None: xrange = [x_array[0], x_array[-1]]
    if yrange is None: yrange = [y_array[0], y_array[-1]]

    plt.imshow(X=z_array, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], cmap=cm.rainbow)
    plt.xlabel("horizontal direction [mm]")
    plt.ylabel("vertical direction [mm]")
    plt.show()

def save_srw_wavefront(srw_wavefront, file_name="srw_wavefront.h5"):
    save_wfr_2_hdf5(srw_wavefront, file_name, subgroupname="wfr", intensity=True, phase=False, overwrite=True)

def load_srw_wavefront(file_name="srw_wavefront.h5"):
    return SRWWavefront.decorateSRWWF(load_hdf5_2_wfr(file_name, "wfr"))
