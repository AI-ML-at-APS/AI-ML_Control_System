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

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average
from orangecontrib.ml.util.data_structures import DictionaryWrapper
from aps_ai.common.util.gaussian_fit import calculate_1D_gaussian_fit

def get_beam_info(scan_data_h=None, scan_data_v=None, do_gaussian_fit=False):
    beam_info = DictionaryWrapper()

    def __get_scan_info(data, suffix):
        fwhm, _, _         = get_fwhm(data[1], data[0])
        sigma              = get_sigma(data[1], data[0])
        centroid           = get_average(data[1], data[0])
        peak_intensity     = numpy.average(data[1][numpy.where(data[1] >= numpy.max(data[1]) * 0.90)])
        integral_intensity = numpy.sum(data[1])

        beam_info.set_parameter("fwhm_" + suffix, fwhm)
        beam_info.set_parameter("sigma_" + suffix, sigma)
        beam_info.set_parameter("centroid_" + suffix, centroid)
        beam_info.set_parameter("peak_intensity_" + suffix, peak_intensity)
        beam_info.set_parameter("integral_intensity_" + suffix, integral_intensity)

        if do_gaussian_fit: beam_info.set_parameter("gaussian_fit_" + suffix, calculate_1D_gaussian_fit(data_1D=data[1], x=data[0]))

    if not scan_data_h is None: __get_scan_info(scan_data_h, "h")
    if not scan_data_v is None: __get_scan_info(scan_data_v, "v")

    return beam_info
