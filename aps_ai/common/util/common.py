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
from matplotlib import pyplot as plt
from matplotlib import cm
from oasys.util.oasys_util import get_sigma, get_fwhm, get_average
from orangecontrib.ml.util.data_structures import DictionaryWrapper

from aps_ai.common.util.gaussian_fit import calculate_2D_gaussian_fit

class Histogram():
    def __init__(self, hh, vv, data_2D):
        self.hh = hh
        self.vv = vv
        self.data_2D = data_2D

def get_info(x_array, y_array, z_array, xrange=None, yrange=None, do_gaussian_fit=False):
    ticket = {'error': 0}
    ticket['nbins_h'] = len(x_array)
    ticket['nbins_v'] = len(y_array)

    if xrange is None: xrange = [x_array.min(), x_array.max()]
    if yrange is None: yrange = [y_array.min(), y_array.max()]
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
    ticket['peak']  = numpy.average(hh[numpy.where(hh >= numpy.max(hh) * 0.95)])

    ticket['fwhm_h'], ticket['fwhm_quote_h'], ticket['fwhm_coordinates_h'] = get_fwhm(hh_h, xx)
    ticket['sigma_h'] = get_sigma(hh_h, xx)

    ticket['fwhm_v'], ticket['fwhm_quote_v'], ticket['fwhm_coordinates_v'] = get_fwhm(hh_v, yy)
    ticket['sigma_v'] = get_sigma(hh_v, yy)

    ticket['centroid_h'] = get_average(ticket['histogram_h'], -ticket['bin_h'])
    ticket['centroid_v'] = get_average(ticket['histogram_v'], -ticket['bin_v'])

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
               peak_intensity=ticket['peak'],
               gaussian_fit=gaussian_fit
    )

class Flip:
    NO = 0
    HORIZONTAL = 1
    VERTICAL = 2
    BOTH = 3

class PlotMode:
    INTERNAL = 0
    NATIVE = 1
    BOTH = 2

class AspectRatio:
    AUTO = 0
    CARTESIAN = 1

class ColorMap:
    RAINBOW = cm.rainbow
    GRAY    = cm.gray
    VIRIDIS = cm.viridis

def plot_2D(x_array, y_array, z_array, title="X,Z", xrange=None, yrange=None,
            int_um="$ph/s/0.1\%BW$", peak_um="$ph/s/mm^2/0.1\%BW$",
            flip=Flip.VERTICAL, aspect_ratio=AspectRatio.AUTO, color_map=ColorMap.RAINBOW):
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

    integral_intensity = numpy.sum(hh) * (1 if int_um=="" else (xx[1] - xx[0]) * (yy[1] - yy[0]))
    peak_intensity     = numpy.average(hh[numpy.where(hh >= numpy.max(hh) * 0.90)]),

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2.5, 1]})
    fig.set_size_inches(9, 5)

    data_2D = hh
    if flip==Flip.VERTICAL:     data_2D = numpy.flip(data_2D, axis=0)
    elif flip==Flip.HORIZONTAL: data_2D = numpy.flip(data_2D, axis=1)
    elif flip==Flip.BOTH:       data_2D = numpy.flip(numpy.flip(data_2D), axis=1)

    if   aspect_ratio == AspectRatio.AUTO:      aspect = 'auto'
    elif aspect_ratio == AspectRatio.CARTESIAN: aspect = 'equal'

    ax1.imshow(X=data_2D, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], aspect=aspect, cmap=color_map)
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
        r'$Integral={0:s}$'.format(as_si(integral_intensity[0] if isinstance(integral_intensity, tuple) else integral_intensity,2)) + ' ' + int_um,
        r'$Peak={0:s}$'.format(as_si(peak_intensity[0] if isinstance(peak_intensity, tuple) else peak_intensity, 2)) + ' ' + peak_um
    ))

    # place a text box in upper left in axes coords
    ax2.text(0.05, 0.98, textstr, transform=ax2.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.axhline(y=0.87, xmin=0.05, xmax=0.5, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.72, xmin=0.05, xmax=0.5, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.58, xmin=0.05, xmax=0.5, color='black', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()
