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
import os, numpy

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average

from orangecontrib.ml.util.mocks import MockWidget
from orangecontrib.ml.util.data_structures import DictionaryWrapper

# OASYS + HYBRID library, to add correction for diffraction and error profiles interference effects.
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowOpticalElement
from orangecontrib.shadow.util.hybrid         import hybrid_control
from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl import bendable_ellipsoid_mirror_bl as bem

class MockBendableEllipsoidMirror(MockWidget):
    def __init__(self, oe=Shadow.OE(), verbose=False,  workspace_units=2):
        MockWidget.__init__(self, verbose, workspace_units)

        self.add_acceptance_slits = 0

        self.dim_x_minus = oe.RWIDX1
        self.dim_x_plus = oe.RWIDX2
        self.dim_y_minus = oe.RLEN1
        self.dim_y_plus = oe.RLEN2

        self.modified_surface = oe.F_G_S

        if self.modified_surface > 0:
            self.ms_defect_file_name = oe.FILE_RIP.decode('utf-8')
            self.set_output_file_name(output_file_name="bender_" + self.ms_defect_file_name)
        else:
            self.set_output_file_name()

        self.bender_bin_x = 100
        self.bender_bin_y = 500

        self.E = 131000
        self.h = 6.2

        self.kind_of_bender = bem.DOUBLE_MOMENTUM
        self.shape = bem.TRAPEZIUM

        self.which_length = 0
        self.optimized_length = 0.0
        self.n_fit_steps = 3

        self.M1    = 0.0
        self.ratio = 0.5
        self.e     = 0.3

        self.M1_out    = 0.0
        self.ratio_out = 0.0
        self.e_out     = 0.0

        self.M1_fixed    = False
        self.ratio_fixed = False
        self.e_fixed     = False

        self.M1_min    = 0.0
        self.ratio_min = 0.0
        self.e_min     = 0.0

        self.M1_max    = 10000.0
        self.ratio_max = 1.0
        self.e_max     = 1.0

    def set_output_file_name(self, output_file_name="mirror_bender.dat"):
        self.output_file_name = output_file_name

        if os.path.isabs(self.output_file_name):
            self.output_file_name_full = self.output_file_name
        else:
            if self.output_file_name.startswith(os.path.sep):
                self.output_file_name_full = os.getcwd() + self.output_file_name
            else:
                self.output_file_name_full = os.getcwd() + os.path.sep + self.output_file_name

    def manage_acceptance_slits(self, shadow_oe):
        if self.add_acceptance_slits==1:
            shadow_oe.add_acceptance_slits(self.auto_slit_width_xaxis,
                                           self.auto_slit_height_zaxis,
                                           self.auto_slit_center_xaxis,
                                           self.auto_slit_center_zaxis)

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

def __get_hybrid_input_parameters(shadow_beam, diffraction_plane=1, calcType=1, nf=0, verbose=False):
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
    input_parameters.ghy_npeak = 20
    input_parameters.ghy_fftnpts = 50000
    input_parameters.file_to_write_out = 0
    input_parameters.ghy_automatic = 0

    return input_parameters

def __rotate_axis_system(input_beam, rotation_angle=270.0):
    oe8 = Shadow.OE()

    oe8.ALPHA = rotation_angle
    oe8.DUMMY = 0.1
    oe8.FWRITE = 3
    oe8.F_REFRAC = 2
    oe8.T_IMAGE = 0.0
    oe8.T_INCIDENCE = 0.0
    oe8.T_REFLECTION = 180.0
    oe8.T_SOURCE = 0.0

    return ShadowBeam.traceFromOE(input_beam.duplicate(), ShadowOpticalElement(oe8), widget_class_name="EmptyElement")

def __run_ML_shadow_simulation_common(input_beam, input_features=DictionaryWrapper(), verbose=False):
    oe5 = Shadow.OE()

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

    # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
    output_beam = ShadowBeam.traceFromOE(input_beam.duplicate(), ShadowOpticalElement(oe5), widget_class_name="ScreenSlits")
    output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                      diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                      calcType=1,  # Diffraction by Simple Aperture
                                                                      verbose=verbose)).ff_beam

    return output_beam

def __get_KB_OEs(input_features):
    oe6 = Shadow.OE()
    oe7 = Shadow.OE()
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

    return oe6, oe7

def __get_KB_mock_widgets(oe6, oe7):
    vkb_widget = MockBendableEllipsoidMirror(oe6)
    vkb_widget.dim_x_minus = 20.95
    vkb_widget.dim_x_plus = 20.95
    vkb_widget.dim_y_minus = 60
    vkb_widget.dim_y_plus = 60
    vkb_widget.h = 6.2
    vkb_widget.e = 0.050505050505051
    vkb_widget.e_fixed = True
    vkb_widget.M1 = 1000
    vkb_widget.ratio = 0.7
    vkb_widget.which_length = 1
    vkb_widget.optimized_length = 20.0

    vkb_widget.e = 0.050505050505051

    hkb_widget = MockBendableEllipsoidMirror(oe7)
    hkb_widget.dim_x_minus = 24.75
    hkb_widget.dim_x_plus = 24.75
    hkb_widget.dim_y_minus = 60
    hkb_widget.dim_y_plus = 60
    hkb_widget.h = 6.2
    hkb_widget.e = 0.050505050505051
    hkb_widget.e_fixed = True
    hkb_widget.M1 = 3000
    hkb_widget.ratio = 0.14
    vkb_widget.which_length = 1
    vkb_widget.optimized_length = 20.0

    return vkb_widget, hkb_widget

def run_ML_shadow_simulation(input_beam, input_features=DictionaryWrapper(), use_benders=False, verbose=False, near_field=True):
    output_beam = __run_ML_shadow_simulation_common(input_beam, input_features, verbose)

    oe6, oe7 = __get_KB_OEs(input_features)

    if not use_benders:
        # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
        output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe6), widget_class_name="EllypticalMirror")
        if near_field:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=1,
                                                                              verbose=verbose)).nf_beam
        else:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=0,
                                                                              verbose=verbose)).ff_beam

        # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
        output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), ShadowOpticalElement(oe7), widget_class_name="EllypticalMirror")
        if near_field:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=1,
                                                                              verbose=verbose)).nf_beam
        else:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=0,
                                                                              verbose=verbose)).ff_beam

    else:
        vkb_widget, hkb_widget = __get_KB_mock_widgets(oe6, oe7)

        vkb_shadow_oe, _ = bem.apply_bender_surface(vkb_widget, output_beam, ShadowOpticalElement(oe6))

        if verbose: print("V-KB: ", vkb_widget.M1_out, vkb_widget.ratio_out, vkb_widget.e_out)

        # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
        output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), vkb_shadow_oe, widget_class_name="EllypticalMirror")
        if near_field:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                          diffraction_plane=1,  # Tangential
                                                                          calcType=3,  # Diffraction by Mirror Size + Errors
                                                                          nf=1,
                                                                          verbose=verbose)).nf_beam
        else:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                          diffraction_plane=1,  # Tangential
                                                                          calcType=3,  # Diffraction by Mirror Size + Errors
                                                                          nf=0,
                                                                          verbose=verbose)).ff_beam

        hkb_shadow_oe, _ = bem.apply_bender_surface(hkb_widget, output_beam, ShadowOpticalElement(oe7))

        if verbose: print("H-KB: ", hkb_widget.M1_out, hkb_widget.ratio_out, hkb_widget.e_out)

        # HYBRID CORRECTION TO CONSIDER MIRROR SIZE AND ERROR PROFILE
        output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), hkb_shadow_oe, widget_class_name="EllypticalMirror")

        if near_field:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=1,
                                                                              verbose=verbose)).nf_beam
        else:
            output_beam = hybrid_control.hy_run(__get_hybrid_input_parameters(output_beam,
                                                                              diffraction_plane=1,  # Tangential
                                                                              calcType=3,  # Diffraction by Mirror Size + Errors
                                                                              nf=0,
                                                                              verbose=verbose)).ff_beam

    return __rotate_axis_system(output_beam, rotation_angle=270.0)

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
