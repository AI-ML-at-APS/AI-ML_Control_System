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

from beamline34IDC.facade.focusing_optics_interface import Movement, DistanceUnits, AngularUnits
from beamline34IDC.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics, get_default_input_features
from beamline34IDC.util.srw.common import write_dabam_file

from syned.beamline.element_coordinates import ElementCoordinates
from syned.beamline.beamline_element import BeamlineElement
from syned.beamline.shape import Rectangle

from wofry.propagator.propagator import PropagationManager, PropagationParameters

from wofrysrw.beamline.srw_beamline import SRWBeamline, Where
from wofrysrw.propagator.wavefront2D.srw_wavefront import WavefrontPropagationParameters
from wofrysrw.propagator.propagators2D.srw_fresnel_native import FresnelSRWNative

from wofrysrw.beamline.optical_elements.absorbers.srw_aperture import SRWAperture
from wofrysrw.beamline.optical_elements.mirrors.srw_mirror import Orientation
from wofrysrw.beamline.optical_elements.mirrors.srw_elliptical_mirror import SRWEllipticalMirror

def srw_focusing_optics_factory_method(**kwargs):
    try:
        if kwargs["bender"] == True: return __BendableFocusingOptics()
        else:                        return __IdealFocusingOptics()
    except: return __IdealFocusingOptics()

class _FocusingOpticsCommon(AbstractSimulatedFocusingOptics):
    def __init__(self):
        self._input_wavefront = None
        self._beamline = None

        self._slits_beam = None
        self._vkb_beam = None
        self._hkb_beam = None
        self._coherence_slits = None
        self._vkb = None
        self._hkb = None
        self._modified_elements = None

    def initialize(self, input_photon_beam, input_features=get_default_input_features(), **kwargs):
        try:    rewrite_height_error_profile_files = kwargs["rewrite_height_error_profile_files"]
        except: rewrite_height_error_profile_files = False

        self._input_wavefront          = input_photon_beam.duplicate()
        self.__initial_input_wavefront = input_photon_beam.duplicate()

        if rewrite_height_error_profile_files == True:
            vkb_error_profile_file = write_dabam_file(dabam_entry_number=92, heigth_profile_file_name="VKB-LTP_srw.dat", seed=8787)
            hkb_error_profile_file = write_dabam_file(dabam_entry_number=93, heigth_profile_file_name="HKB-LTP_srw.dat", seed=2345345)
        else:
            vkb_error_profile_file = "VKB-LTP_srw.dat"
            hkb_error_profile_file = "HKB-LTP_srw.dat"

        self._beamline = SRWBeamline()

        # Coherence Slits
        width    = input_features.get_parameter("coh_slits_h_aperture") * 1e-3 # m
        height   = input_features.get_parameter("coh_slits_v_aperture") * 1e-3
        h_center = input_features.get_parameter("coh_slits_h_center") * 1e-3
        v_center = input_features.get_parameter("coh_slits_v_center") * 1e-3

        self._coherence_slits = SRWAperture(boundary_shape=Rectangle(x_left=-0.5 * width + h_center,
                                                                     x_right=0.5 * width + h_center,
                                                                     y_bottom=-0.5 * height + v_center,
                                                                     y_top=0.5 * height + v_center))

        beamline_element = BeamlineElement(optical_element=self._coherence_slits, coordinates=ElementCoordinates(q=0.15))

        srw_oe_wavefront_propagation_parameters = WavefrontPropagationParameters(
                                                                 allow_semianalytical_treatment_of_quadratic_phase_term = 0, # Standard
                                                                 horizontal_range_modification_factor_at_resizing       = 0.1,
                                                                 horizontal_resolution_modification_factor_at_resizing  = 5.0,
                                                                 vertical_range_modification_factor_at_resizing         = 0.1,
                                                                 vertical_resolution_modification_factor_at_resizing    = 5.0
                                                             )

        srw_after_wavefront_propagation_parameters = WavefrontPropagationParameters(
                                                                 allow_semianalytical_treatment_of_quadratic_phase_term = 2, # Standard
                                                                 horizontal_range_modification_factor_at_resizing       = 1.0,
                                                                 horizontal_resolution_modification_factor_at_resizing  = 2.0,
                                                                 vertical_range_modification_factor_at_resizing         = 1.0,
                                                                 vertical_resolution_modification_factor_at_resizing    = 2.0
                                                             )

        self._beamline.append_beamline_element(beamline_element)
        self._beamline.append_wavefront_propagation_parameters(None, None, Where.DRIFT_BEFORE)
        self._beamline.append_wavefront_propagation_parameters(srw_oe_wavefront_propagation_parameters, None, Where.OE)
        self._beamline.append_wavefront_propagation_parameters(srw_after_wavefront_propagation_parameters, None, Where.DRIFT_AFTER)

        self._initialize_kb(input_features, vkb_error_profile_file, hkb_error_profile_file)

        self._modified_elements = [self._coherence_slits, self._vkb, self._hkb]

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None): pass
    def restore_input_photon_beam(self): pass

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None, units=DistanceUnits.MICRON):

        boundaries = self._coherence_slits._boundary_shape.get_boundaries()

        if units==DistanceUnits.MILLIMETERS: factor = 1e-3
        elif units==DistanceUnits.MICRON:    factor = 1e-6
        else: ValueError("Units not recognized")

        coh_slits_h_center = abs(boundaries[1]-boundaries[0]) if coh_slits_h_center is None else factor*coh_slits_h_center
        coh_slits_v_center = abs(boundaries[3]-boundaries[2]) if coh_slits_v_center is None else factor*coh_slits_v_center
        coh_slits_h_aperture = 0.5*(boundaries[1]+boundaries[0]) if coh_slits_h_aperture is None else factor*coh_slits_h_aperture
        coh_slits_v_aperture = 0.5*(boundaries[3]+boundaries[2]) if coh_slits_v_aperture is None else factor*coh_slits_v_aperture

        self._coherence_slits_boundary_shape=Rectangle(x_left=-0.5 * coh_slits_h_aperture + coh_slits_h_center,
                                                       x_right=0.5 * coh_slits_h_aperture + coh_slits_h_center,
                                                       y_bottom=-0.5 * coh_slits_v_aperture + coh_slits_v_center,
                                                       y_top=0.5 * coh_slits_v_aperture + coh_slits_v_center)

    def get_coherence_slits_parameters(self, units=DistanceUnits.MICRON):  # center x, center z, aperture x, aperture z
        boundaries = self._coherence_slits._boundary_shape.get_boundaries()

        if units == DistanceUnits.MILLIMETERS: factor = 1e3
        elif units == DistanceUnits.MICRON:    factor = 1e6
        else: ValueError("Units not recognized")

        coh_slits_h_center = factor * abs(boundaries[1] - boundaries[0])
        coh_slits_v_center = factor * abs(boundaries[3] - boundaries[2])
        coh_slits_h_aperture = factor * 0.5 * (boundaries[1] + boundaries[0])
        coh_slits_v_aperture = factor * 0.5 * (boundaries[3] + boundaries[2])

        return coh_slits_h_center, coh_slits_v_center, coh_slits_h_aperture, coh_slits_v_aperture

    def get_photon_beam(self, **kwargs):
        propagation_parameters = PropagationParameters(wavefront=self._input_wavefront.duplicate(), propagation_elements=None)
        propagation_parameters.set_additional_parameters("working_beamline", self.__beamline) # Propagation mode: WHOLE BEAMLINE

        return PropagationManager.Instance().do_propagation(propagation_parameters=propagation_parameters, handler_name=FresnelSRWNative.HANDLER_NAME)


class __IdealFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        super.__init__()

    def _initialize_kb(self, input_features, vkb_error_profile_file, hkb_error_profile_file):
        pass

    # V-KB -----------------------

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_vkb_q_distance(self): pass

    # H-KB -----------------------

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE): pass
    def get_hkb_q_distance(self): pass

class __BendableFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        super().__init__()

    def _initialize_kb(self, input_features, vkb_error_profile_file, hkb_error_profile_file):
        pass


    # V-KB -----------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON): pass

    # H-KB -----------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON): pass

    def get_photon_beam(self, **kwargs): pass
