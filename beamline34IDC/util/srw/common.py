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

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average
from orangecontrib.ml.util.data_structures import DictionaryWrapper
from srxraylib.metrology import dabam
from oasys.util.error_profile_util import DabamInputParameters, calculate_dabam_profile, ErrorProfileInputParameters, calculate_heigth_profile
from orangecontrib.srw.util.srw_util import write_error_profile_file

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
    hh_h = hh.sum(axis=0)
    hh_v = hh.sum(axis=1)
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

    cursor_x = numpy.where(numpy.logical_and(x_array >= xrange[0], x_array <= xrange[1]))
    cursor_y = numpy.where(numpy.logical_and(y_array >= yrange[0], y_array <= yrange[1]))

    xx = x_array[cursor_x]
    yy = y_array[cursor_y]
    hh = z_array[tuple(numpy.meshgrid(cursor_x, cursor_y))]

    hh_h = hh.sum(axis=0)
    hh_v = hh.sum(axis=1)

    fwhm_h, _, _ = get_fwhm(hh_h, xx)
    sigma_h = get_sigma(hh_h, xx)
    fwhm_v, _, _ = get_fwhm(hh_v, yy)
    sigma_v = get_sigma(hh_v, yy)
    centroid_h = get_average(hh_h, xx)
    centroid_v = get_average(hh_v, yy)

    integral_intensity = numpy.sum(hh) * (xx[1] - xx[0]) * (yy[1] - yy[0])
    peak_intensity = numpy.average(hh[numpy.where(hh >= numpy.max(hh) * 0.90)]),

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2.5, 1]})
    fig.set_size_inches(9, 5)

    ax1.imshow(X=hh, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], aspect='auto', cmap=cm.rainbow)
    ax1.set_title(title)
    ax1.set_xlabel("horizontal direction [mm]")
    ax1.set_ylabel("vertical direction [mm]")

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    def as_si(x, ndp):
        s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
        m, e = s.split('e')
        return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

    textstr = '\n'.join((
        r'$fwhm(H)=%.3f$ $\mu$m' % (fwhm_h*1e3,),
        r'$fwhm(V)=%.3f$ $\mu$m' % (fwhm_v*1e3,),
        r'',
        r'$\sigma(H)=%.3f$ $\mu$m' % (sigma_h*1e3,),
        r'$\sigma(V)=%.3f$ $\mu$m' % (sigma_v*1e3,),
        r'',
        r'$centroid(H)=%.3f$ $\mu$m' % (centroid_h * 1e3,),
        r'$centroid(V)=%.3f$ $\mu$m' % (centroid_v * 1e3,),
        r'',
        r'$Integral={0:s}$ $ph/s/0.1\%BW$'.format(as_si(integral_intensity[0] if isinstance(integral_intensity, tuple) else integral_intensity,2)),
        r'$Peak={0:s}$ $ph/s/mm^2/0.1\%BW$'.format(as_si(peak_intensity[0] if isinstance(peak_intensity, tuple) else peak_intensity, 2))
    ))

    # place a text box in upper left in axes coords
    ax2.text(0.05, 0.98, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axhline(y=0.87, xmin=0.05, xmax=1.0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.72, xmin=0.05, xmax=1.0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.58, xmin=0.05, xmax=1.0, color='black', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()

import pickle

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
