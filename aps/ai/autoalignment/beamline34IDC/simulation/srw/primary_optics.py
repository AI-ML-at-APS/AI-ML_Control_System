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
from aps.ai.autoalignment.common.simulation.facade.primary_optics_interface import AbstractPrimaryOptics

from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.shape import Rectangle

from wofry.propagator.propagator import PropagationManager, PropagationParameters
from wofrysrw.propagator.propagators2D.srw_fresnel_native import SRW_APPLICATION
from wofrysrw.propagator.propagators2D.srw_propagation_mode import SRWPropagationMode

from wofrysrw.beamline.srw_beamline import SRWBeamline, Where
from wofrysrw.propagator.wavefront2D.srw_wavefront import WavefrontPropagationParameters
from wofrysrw.propagator.propagators2D.srw_fresnel_native import FresnelSRWNative

from wofrysrw.beamline.optical_elements.absorbers.srw_aperture import SRWAperture
from wofrysrw.beamline.optical_elements.mirrors.srw_mirror import Orientation
from wofrysrw.beamline.optical_elements.mirrors.srw_plane_mirror import SRWPlaneMirror

def srw_primary_optics_factory_method():
    return __PrimaryOpticsSystem()

class __PrimaryOpticsSystem(AbstractPrimaryOptics):
    def __init__(self):
        self.__beamline = None
        self.__source_wavefront = None

    def initialize(self, source_photon_beam, **kwargs):
        self.__source_wavefront = source_photon_beam
        self.__beamline = SRWBeamline()

        # Front-End Aperture
        width = 0.0001
        height = 0.01
        front_end_slit = SRWAperture(boundary_shape=Rectangle(x_left=-0.5 * width,
                                                              x_right=0.5 * width,
                                                              y_bottom=-0.5 * height,
                                                              y_top=0.5 * height))

        beamline_element = BeamlineElement(optical_element=front_end_slit, coordinates=ElementCoordinates())

        srw_oe_wavefront_propagation_parameters = WavefrontPropagationParameters(
                                                                 allow_semianalytical_treatment_of_quadratic_phase_term = 0, # Standard
                                                                 horizontal_range_modification_factor_at_resizing       = 0.2,
                                                                 horizontal_resolution_modification_factor_at_resizing  = 1.0,
                                                                 vertical_range_modification_factor_at_resizing         = 1.0,
                                                                 vertical_resolution_modification_factor_at_resizing    = 1.0
                                                             )

        self.__beamline.append_beamline_element(beamline_element)
        self.__beamline.append_wavefront_propagation_parameters(None, None, Where.DRIFT_BEFORE)
        self.__beamline.append_wavefront_propagation_parameters(srw_oe_wavefront_propagation_parameters, None, Where.OE)
        self.__beamline.append_wavefront_propagation_parameters(None, None, Where.DRIFT_AFTER)

        # Mirror 1
        mirror = SRWPlaneMirror(tangential_size=0.5,
                                sagittal_size=0.02,
                                grazing_angle=0.005,
                                orientation_of_reflection_plane=Orientation.UP,
                                invert_tangent_component = False,
                                add_acceptance_slit=True,
                                height_profile_data_file=None)

        beamline_element = BeamlineElement(optical_element=mirror,
                                           coordinates=ElementCoordinates(p=2.8,
                                                                          q=20.9,
                                                                          angle_radial=0.5*numpy.pi-0.005,
                                                                          angle_azimuthal=0.0))

        srw_drift_before_wavefront_propagation_parameters = WavefrontPropagationParameters(
            allow_semianalytical_treatment_of_quadratic_phase_term=2,  # Quadratic Term Special
            horizontal_range_modification_factor_at_resizing=1.0,
            horizontal_resolution_modification_factor_at_resizing=1.0,
            vertical_range_modification_factor_at_resizing=1.0,
            vertical_resolution_modification_factor_at_resizing=1.0
        )

        srw_oe_wavefront_propagation_parameters = WavefrontPropagationParameters(
            allow_semianalytical_treatment_of_quadratic_phase_term=0,  # Standard
            horizontal_range_modification_factor_at_resizing=1.0,
            horizontal_resolution_modification_factor_at_resizing=1.0,
            vertical_range_modification_factor_at_resizing=1.0,
            vertical_resolution_modification_factor_at_resizing=1.0
        )

        srw_drift_after_wavefront_propagation_parameters = WavefrontPropagationParameters(
            allow_semianalytical_treatment_of_quadratic_phase_term=2,  # Quadratic Term Special
            horizontal_range_modification_factor_at_resizing=1.0,
            horizontal_resolution_modification_factor_at_resizing=1.0,
            vertical_range_modification_factor_at_resizing=1.0,
            vertical_resolution_modification_factor_at_resizing=1.0
        )

        self.__beamline.append_beamline_element(beamline_element)
        self.__beamline.append_wavefront_propagation_parameters(srw_drift_before_wavefront_propagation_parameters, None, Where.DRIFT_BEFORE)
        self.__beamline.append_wavefront_propagation_parameters(srw_oe_wavefront_propagation_parameters, None, Where.OE)
        self.__beamline.append_wavefront_propagation_parameters(srw_drift_after_wavefront_propagation_parameters, None, Where.DRIFT_AFTER)


    def get_photon_beam(self, **kwargs):
        PropagationManager.Instance().set_propagation_mode(SRW_APPLICATION, SRWPropagationMode.WHOLE_BEAMLINE)

        propagation_parameters = PropagationParameters(wavefront=self.__source_wavefront.duplicate(), propagation_elements=None)
        propagation_parameters.set_additional_parameters("working_beamline", self.__beamline) # Propagation mode: WHOLE BEAMLINE

        return PropagationManager.Instance().do_propagation(propagation_parameters=propagation_parameters, handler_name=FresnelSRWNative.HANDLER_NAME)

