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
import Shadow
from orangecontrib.shadow.util.shadow_util import ShadowPhysics

import scipy.constants as codata
m2ev = codata.c * codata.h / codata.e

from Shadow.ShadowPreprocessorsXraylib import prerefl, bragg
from oasys.widgets import congruence

def rotate(origin, point, angle):
    """
    Rotate a point counter-clockwise by a given angle around a given origin.
    """
    # Convert negative angles to positive
    angle = normalise_angle(angle)

    # Convert to radians
    angle = math.radians(angle)

    # Convert to radians
    ox, oy = origin
    px, py = point

    # Move point 'p' to origin (0,0)
    _px = px - ox
    _py = py - oy

    # Rotate the point 'p'
    qx = (math.cos(angle) * _px) - (math.sin(angle) * _py)
    qy = (math.sin(angle) * _px) + (math.cos(angle) * _py)

    # Move point 'p' back to origin (ox, oy)
    qx = ox + qx
    qy = oy + qy

    return [qx, qy]

def normalise_angle(angle):
    """ If angle is negative then convert it to positive. """
    return (360 + angle) if ((angle != 0) & (abs(angle) == (angle * -1))) else angle

# WEIRD MEMORY INITIALIZATION BY FORTRAN. JUST A FIX.
def fix_Intensity(beam_out, polarization=0):
    if polarization == 0:
        beam_out._beam.rays[:, 15] = 0
        beam_out._beam.rays[:, 16] = 0
        beam_out._beam.rays[:, 17] = 0

    return beam_out

def get_shadow_beam_spatial_distribution(shadow_beam, nbins=201, nolost=1, xrange=[-2.0, +2.0], yrange=[-2.0, +2.0]):
    return shadow_beam._beam.histo2(1, 3, nbins=nbins, nolost=nolost, title=title, xrange=xrange, yrange=yrange)

def get_shadow_beam_divergence_distribution(shadow_beam, nbins=201, nolost=1, xrange=[-1e-3, +1e-3], yrange=[-1e-3, +1e-3]):
    return shadow_beam._beam.histo2(4, 6, nbins=nbins, nolost=nolost, title=title, xrange=xrange, yrange=yrange)

def plot_shadow_beam_spatial_distribution(shadow_beam, nbins=201, nolost=1, title="X,Z", xrange=[-2.0, +2.0], yrange=[-2.0, +2.0]):
    return Shadow.ShadowTools.plotxy(shadow_beam._beam, 1, 3, nbins=nbins, nolost=nolost, title=title, xrange=xrange, yrange=yrange)

def plot_shadow_beam_divergence_distribution(shadow_beam, nbins=201, nolost=1, title="X',Z'", xrange=[-1e-3, +1e-3], yrange=[-1e-3, +1e-3]):
    return Shadow.ShadowTools.plotxy(shadow_beam._beam, 4, 6, nbins=nbins, nolost=nolost, title=title, xrange=xrange, yrange=yrange)

def write_reflectivity_file(symbol="Pt", shadow_file_name="Pt.dat", energy_range=[5000, 15000], energy_step=1.0):
    symbol = symbol.strip()
    density = ShadowPhysics.getMaterialDensity(symbol)

    prerefl(interactive=False,
            SYMBOL=symbol,
            DENSITY=density,
            E_MIN=energy_range[0],
            E_MAX=energy_range[1],
            E_STEP=energy_step,
            FILE=congruence.checkFileName(shadow_file_name))

    return shadow_file_name

def write_bragg_file(crystal="Si", miller_indexes=[1, 1, 1], shadow_file_name="Si111.dat", energy_range=[5000, 15000], energy_step=1.0):
    bragg(interactive=False,
          DESCRIPTOR=crystal.strip(),
          H_MILLER_INDEX=miller_indexes[0],
          K_MILLER_INDEX=miller_indexes[1],
          L_MILLER_INDEX=miller_indexes[2],
          TEMPERATURE_FACTOR=1.0,
          E_MIN=energy_range[0],
          E_MAX=energy_range[1],
          E_STEP=energy_step,
          SHADOW_FILE=congruence.checkFileName(shadow_file_name))

    return shadow_file_name
