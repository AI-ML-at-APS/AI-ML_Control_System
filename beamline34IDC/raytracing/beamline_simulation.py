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
import numpy

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average

from orangecontrib.ml.util.mocks import MockWidget
from orangecontrib.ml.util.data_structures import DictionaryWrapper

# OASYS + HYBRID library, to add correction for diffraction and error profiles interference effects.
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement
from orangecontrib.shadow.util.hybrid         import hybrid_control

def run_invariant_shadow_simulation(source_beam):
    #####################################################
    # SHADOW 3 INITIALIZATION

    #
    # initialize shadow3 source (oe0) and beam
    #
    oe1 = Shadow.OE()
    oe2 = Shadow.OE()
    oe3 = Shadow.OE()
    oe4 = Shadow.OE()

    #
    # Define variables. See meaning of variables in:
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
    #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
    #

    # WB SLITS
    oe1.DUMMY = 0.1
    oe1.FWRITE = 3
    oe1.F_REFRAC = 2
    oe1.F_SCREEN = 1
    oe1.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    oe1.N_SCREEN = 1
    oe1.RX_SLIT = numpy.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe1.RZ_SLIT = numpy.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe1.T_IMAGE = 0.0
    oe1.T_INCIDENCE = 0.0
    oe1.T_REFLECTION = 180.0
    oe1.T_SOURCE = 26800.0

    # PLANE MIRROR
    oe2.DUMMY = 0.1
    oe2.FHIT_C = 1
    oe2.FILE_REFL = b'Pt.dat'
    oe2.FWRITE = 1
    oe2.F_REFLEC = 1
    oe2.RLEN1 = 250.0
    oe2.RLEN2 = 250.0
    oe2.RWIDX1 = 10.0
    oe2.RWIDX2 = 10.0
    oe2.T_IMAGE = 0.0
    oe2.T_INCIDENCE = 89.7135211024
    oe2.T_REFLECTION = 89.7135211024
    oe2.T_SOURCE = 2800.0

    # DCM-1
    oe3.DUMMY = 0.1
    oe3.FHIT_C = 1
    oe3.FILE_REFL = b'Si111.dat'
    oe3.FWRITE = 1
    oe3.F_CENTRAL = 1
    oe3.F_CRYSTAL = 1
    oe3.PHOT_CENT = 5000.0
    oe3.RLEN1 = 50.0
    oe3.RLEN2 = 50.0
    oe3.RWIDX1 = 50.0
    oe3.RWIDX2 = 50.0
    oe3.R_LAMBDA = 5000.0
    oe3.T_IMAGE = 0.0
    oe3.T_INCIDENCE = 66.7041090078
    oe3.T_REFLECTION = 66.7041090078
    oe3.T_SOURCE = 15400.0

    # DCM-2
    oe4.ALPHA = 180.0
    oe4.DUMMY = 0.1
    oe4.FHIT_C = 1
    oe4.FILE_REFL = b'Si111.dat'
    oe4.FWRITE = 1
    oe4.F_CENTRAL = 1
    oe4.F_CRYSTAL = 1
    oe4.PHOT_CENT = 5000.0
    oe4.RLEN1 = 50.0
    oe4.RLEN2 = 50.0
    oe4.RWIDX1 = 50.0
    oe4.RWIDX2 = 50.0
    oe4.R_LAMBDA = 5000.0
    oe4.T_IMAGE = 5494.324
    oe4.T_INCIDENCE = 66.7041090078
    oe4.T_REFLECTION = 66.7041090078
    oe4.T_SOURCE = 8.259

    output_beam = ShadowBeam.traceFromOE(source_beam.duplicate(), ShadowOpticalElement(oe1), widget_class_name="ScreenSlits")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe2), widget_class_name="PlaneMirror")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe3), widget_class_name="PlaneCrystal")
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe4), widget_class_name="PlaneCrystal")

    return output_beam

def get_hybrid_input_parameters(shadow_beam, diffraction_plane=1, calcType=1, nf=0, verbose=False):
    input_parameters = hybrid_control.HybridInputParameters()
    input_parameters.ghy_lengthunit = 2
    input_parameters.widget = MockWidget(verbose=verbose)
    input_parameters.shadow_beam = shadow_beam
    input_parameters.ghy_diff_plane = diffraction_plane
    input_parameters.ghy_calcType = calcType
    input_parameters.ghy_distance = -1
    input_parameters.ghy_focallength = -1
    input_parameters.ghy_nf = nf
    input_parameters.ghy_nbins_x = 100
    input_parameters.ghy_nbins_z = 100
    input_parameters.ghy_npeak = 10
    input_parameters.ghy_fftnpts = 10000
    input_parameters.file_to_write_out = 0
    input_parameters.ghy_automatic = 0

    return input_parameters

def run_ML_shadow_simulation(input_beam, input_features=DictionaryWrapper(), verbose=False):
    oe5 = Shadow.OE()
    oe6 = Shadow.OE()
    oe7 = Shadow.OE()
    oe8 = Shadow.OE()

    # COHERENCE SLITS
    oe5.DUMMY = 0.1
    oe5.FWRITE = 3
    oe5.F_REFRAC = 2
    oe5.F_SCREEN = 1
    oe5.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    oe5.N_SCREEN = 1
    oe5.CX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.CZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.RX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.RZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    oe5.T_IMAGE = 0.0
    oe5.T_INCIDENCE = 0.0
    oe5.T_REFLECTION = 180.0
    oe5.T_SOURCE = 0.0

    # V-KB
    oe6.ALPHA = 180.0
    oe6.DUMMY = 0.1
    oe6.FCYL = 1
    oe6.FHIT_C = 1
    oe6.FILE_REFL = b'Pt.dat'
    oe6.FILE_RIP = b'VKB.dat'
    oe6.FMIRR = 2
    oe6.FWRITE = 1
    oe6.F_DEFAULT = 0
    oe6.F_G_S = 2
    oe6.F_REFLEC = 1
    oe6.F_RIPPLE = 1

    oe6.RLEN1 = 50.0
    oe6.RLEN2 = 50.0
    oe6.RWIDX1 = 10.0
    oe6.RWIDX2 = 10.0
    oe6.SIMAG = input_features.get_parameter("vkb_p_distance")
    oe6.SSOUR = 50667.983
    oe6.THETA = 89.8281126615
    oe6.T_IMAGE = 101.0
    oe6.T_INCIDENCE = 89.8281126615
    oe6.T_REFLECTION = 89.8281126615
    oe6.T_SOURCE = 150.0

    # DISPLACEMENT
    oe6.F_MOVE = 1
    oe6.OFFX = input_features.get_parameter("vkb_offset_x")
    oe6.OFFY = input_features.get_parameter("vkb_offset_y")
    oe6.OFFZ = input_features.get_parameter("vkb_offset_z")
    oe6.X_ROT = input_features.get_parameter("vkb_rotation_x")
    oe6.Y_ROT = input_features.get_parameter("vkb_rotation_y")
    oe6.Z_ROT = input_features.get_parameter("vkb_rotation_z")

    # H-KB
    oe7.ALPHA = 90.0
    oe7.DUMMY = 0.1
    oe7.FCYL = 1
    oe7.FHIT_C = 1
    oe7.FILE_REFL = b'Pt.dat'
    oe7.FILE_RIP = b'HKB.dat'
    oe7.FMIRR = 2
    oe7.FWRITE = 1
    oe7.F_DEFAULT = 0
    oe7.F_G_S = 2
    oe7.F_MOVE = 1
    oe7.F_REFLEC = 1
    oe7.F_RIPPLE = 1
    oe7.RLEN1 = 50.0
    oe7.RLEN2 = 50.0
    oe7.RWIDX1 = 10.0
    oe7.RWIDX2 = 10.0
    oe7.SIMAG = input_features.get_parameter("hkb_p_distance")
    oe7.SSOUR = 50768.983
    oe7.THETA = 89.8281126615
    oe7.T_IMAGE = 120.0
    oe7.T_INCIDENCE = 89.8281126615
    oe7.T_REFLECTION = 89.8281126615
    oe7.T_SOURCE = 0.0

    # DISPLACEMENT
    oe7.F_MOVE = 1
    oe7.OFFX = input_features.get_parameter("hkb_offset_x")
    oe7.OFFY = input_features.get_parameter("hkb_offset_y")
    oe7.OFFZ = input_features.get_parameter("hkb_offset_z")
    oe7.X_ROT = input_features.get_parameter("hkb_rotation_x")
    oe7.Y_ROT = input_features.get_parameter("hkb_rotation_y")
    oe7.Z_ROT = input_features.get_parameter("hkb_rotation_z")

    oe8.ALPHA = 270.0
    oe8.DUMMY = 0.1
    oe8.FWRITE = 3
    oe8.F_REFRAC = 2
    oe8.T_IMAGE = 0.0
    oe8.T_INCIDENCE = 0.0
    oe8.T_REFLECTION = 180.0
    oe8.T_SOURCE = 0.0

    # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
    output_beam = ShadowBeam.traceFromOE(input_beam.duplicate(), ShadowOpticalElement(oe5), widget_class_name="ScreenSlits")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                    calcType=1,  # Diffraction by Simple Aperture
                                                                    verbose=verbose)).ff_beam
    # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe6), widget_class_name="EllypticalMirror")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=1,  # Tangential
                                                                    calcType=3,  # Diffraction by Mirror Size + Errors
                                                                    nf=1,
                                                                    verbose=verbose)).nf_beam

    # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe7), widget_class_name="EllypticalMirror")
    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                    diffraction_plane=1,  # Tangential
                                                                    calcType=3,  # Diffraction by Mirror Size + Errors
                                                                    nf=1,
                                                                    verbose=verbose)).nf_beam

    output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe8), widget_class_name="EmptyElement")

    return output_beam

def extract_output_parameters(output_beam, plot=False):
    to_micron = 1e3
    to_urad = 1e6

    output_beam._beam.rays[:, 0] *= to_micron
    output_beam._beam.rays[:, 1] *= to_micron
    output_beam._beam.rays[:, 2] *= to_micron

    output_beam._beam.rays[:, 3] *= to_urad
    output_beam._beam.rays[:, 4] *= to_urad
    output_beam._beam.rays[:, 5] *= to_urad

    if plot:
        ticket = Shadow.ShadowTools.plotxy(output_beam._beam, 1, 3, nbins=201, nolost=1, title="Focal Spot Size", xrange=[-2.0, +2.0], yrange=[-2.0, +2.0])
    else:
        ticket = output_beam._beam.histo2(1, 3, nbins=201, nolost=1, xrange=[-2.0, +2.0], yrange=[-2.0, +2.0])

    ticket['fwhm_h'], ticket['fwhm_quote_h'], ticket['fwhm_coordinates_h'] = get_fwhm(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['fwhm_v'], ticket['fwhm_quote_v'], ticket['fwhm_coordinates_v'] = get_fwhm(ticket['histogram_v'], ticket['bin_v_center'])
    ticket['sigma_h'] = get_sigma(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['sigma_v'] = get_sigma(ticket['histogram_v'], ticket['bin_v_center'])
    ticket['centroid_h'] = get_average(ticket['histogram_h'], ticket['bin_h_center'])
    ticket['centroid_v'] = get_average(ticket['histogram_v'], ticket['bin_v_center'])

    histogram = ticket["histogram"]

    peak_intensity = numpy.average(histogram[numpy.where(histogram >= numpy.max(histogram) * 0.90)])
    integral_intensity = numpy.sum(histogram)

    output_parameters = DictionaryWrapper(
        h_sigma=ticket['sigma_h'],
        h_fwhm=ticket['fwhm_h'],
        h_centroid=ticket['centroid_h'],
        v_sigma=ticket['sigma_v'],
        v_fwhm=ticket['fwhm_v'],
        v_centroid=ticket['centroid_v'],
        integral_intensity=integral_intensity,
        peak_intensity=peak_intensity
    )

    if plot:
        ticket = Shadow.ShadowTools.plotxy(output_beam._beam, 4, 6, nbins=201, nolost=1, title="Focal Spot Divergence", xrange=[-150.0, +150.0], yrange=[-200.0, +200.0])
    else:
        ticket = output_beam._beam.histo2(4, 6, nbins=201, nolost=1, xrange=[-150.0, +150.0], yrange=[-200.0, +200.0])

    output_parameters.set_parameter("h_divergence", get_average(ticket['histogram_h'], ticket['bin_h_center']))
    output_parameters.set_parameter("v_divergence", get_average(ticket['histogram_v'], ticket['bin_v_center']))

    return output_parameters
