#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
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
import os.path

import numpy
import Shadow

from orangecontrib.shadow.util.shadow_objects import ShadowOpticalElement, ShadowBeam
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control

from beamline34IDC.util.shadow.common import HybridFailureException, rotate_axis_system, get_hybrid_input_parameters
from beamline34IDC.facade.focusing_optics_interface import Movement, MotorResolution, AngularUnits, DistanceUnits

from beamline34IDC.simulation.shadow.focusing_optics.abstract_focusing_optics import FocusingOpticsCommon
from beamline34IDC.simulation.shadow.focusing_optics.bender.calibrated_bender import CalibratedBenderManager, HKBMockWidget, VKBMockWidget
from beamline34IDC.simulation.facade.focusing_optics_interface import get_default_input_features

from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.double_rod_bendable_ellispoid_mirror_bl import apply_bender_surface

class TwoOEBendableFocusingOptics(FocusingOpticsCommon):
    def __init__(self):
        super(FocusingOpticsCommon, self).__init__()

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):

        super().initialize(input_photon_beam, input_features, **kwargs)

        self.__vkb_bender_manager = CalibratedBenderManager(kb_raytracing=None,
                                                            kb_upstream=VKBMockWidget(self._vkb[0], verbose=True, label="Upstream"),
                                                            kb_downstream=VKBMockWidget(self._vkb[1], verbose=True, label="Downstream"))
        self.__vkb_bender_manager.load_calibration("V-KB")
        self.__vkb_bender_manager.set_positions(input_features.get_parameter("vkb_motor_1_bender_position"),
                                                input_features.get_parameter("vkb_motor_2_bender_position"))
        self.__vkb_bender_manager.remove_bender_files()

        self.__hkb_bender_manager = CalibratedBenderManager(kb_raytracing=None,
                                                            kb_upstream=HKBMockWidget(self._hkb[0], verbose=True, label="Upstream"),
                                                            kb_downstream=HKBMockWidget(self._hkb[1], verbose=True, label="Downstream"))
        self.__hkb_bender_manager.load_calibration("H-KB")
        self.__hkb_bender_manager.set_positions(input_features.get_parameter("hkb_motor_1_bender_position"),
                                                input_features.get_parameter("hkb_motor_2_bender_position"))
        self.__hkb_bender_manager.remove_bender_files()

    def _initialize_kb(self, input_features, reflectivity_file, vkb_error_profile_file, hkb_error_profile_file):
        # V-KB --------------------
        vkb_motor_3_pitch_angle = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
        vkb_pitch_angle_displacement_shadow = numpy.degrees(vkb_motor_3_delta_pitch_angle)
        vkb_motor_4_translation = input_features.get_parameter("vkb_motor_4_translation")

        vkb_up = Shadow.OE()
        vkb_up.ALPHA = 180.0
        vkb_up.DUMMY = 0.1
        vkb_up.FCYL = 1
        vkb_up.FHIT_C = 1
        vkb_up.FILE_REFL = reflectivity_file.encode()
        vkb_up.FILE_RIP = vkb_error_profile_file.encode()
        vkb_up.FMIRR = 2
        vkb_up.FWRITE = 1
        vkb_up.F_DEFAULT = 0
        vkb_up.F_G_S = 2
        vkb_up.F_REFLEC = 1
        vkb_up.F_RIPPLE = 1
        vkb_up.RLEN1 = 50.0
        vkb_up.RLEN2 = 50.0
        vkb_up.RWIDX1 = 20.95
        vkb_up.RWIDX2 = 20.95
        vkb_up.SIMAG = -999
        vkb_up.SSOUR = 50667.983
        vkb_up.THETA = vkb_pitch_angle_shadow
        vkb_up.T_IMAGE = 108.0
        vkb_up.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb_up.T_REFLECTION = vkb_pitch_angle_shadow
        vkb_up.T_SOURCE = 150.0

        # DISPLACEMENTS
        vkb_up.F_MOVE = 1
        vkb_up.OFFY = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_up.OFFZ = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_up.X_ROT = vkb_pitch_angle_displacement_shadow

        vkb_down = Shadow.OE()
        vkb_down.ALPHA = 180.0
        vkb_down.DUMMY = 0.1
        vkb_down.FCYL = 1
        vkb_down.FHIT_C = 1
        vkb_down.FILE_REFL = reflectivity_file.encode()
        vkb_down.FILE_RIP = vkb_error_profile_file.encode()
        vkb_down.FMIRR = 2
        vkb_down.FWRITE = 1
        vkb_down.F_DEFAULT = 0
        vkb_down.F_G_S = 2
        vkb_down.F_REFLEC = 1
        vkb_down.F_RIPPLE = 1
        vkb_down.RLEN1 = 50.0
        vkb_down.RLEN2 = 50.0
        vkb_down.RWIDX1 = 20.95
        vkb_down.RWIDX2 = 20.95
        vkb_down.SIMAG = -999
        vkb_down.SSOUR = 50667.983
        vkb_down.THETA = vkb_pitch_angle_shadow
        vkb_down.T_IMAGE = 108.0
        vkb_down.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb_down.T_REFLECTION = vkb_pitch_angle_shadow
        vkb_down.T_SOURCE = 150.0

        # DISPLACEMENTS
        vkb_down.F_MOVE = 1
        vkb_down.OFFY = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_down.OFFZ = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_down.X_ROT = vkb_pitch_angle_displacement_shadow

        # H-KB --------------------

        hkb_motor_3_pitch_angle = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
        hkb_pitch_angle_displacement_shadow = numpy.degrees(hkb_motor_3_delta_pitch_angle)
        hkb_motor_4_translation = input_features.get_parameter("hkb_motor_4_translation")

        hkb_up = Shadow.OE()
        hkb_up.ALPHA = 90.0
        hkb_up.DUMMY = 0.1
        hkb_up.FCYL = 1
        hkb_up.FHIT_C = 1
        hkb_up.FILE_REFL = reflectivity_file.encode()
        hkb_up.FILE_RIP = hkb_error_profile_file.encode()
        hkb_up.FMIRR = 2
        hkb_up.FWRITE = 1
        hkb_up.F_DEFAULT = 0
        hkb_up.F_G_S = 2
        hkb_up.F_REFLEC = 1
        hkb_up.F_RIPPLE = 1
        hkb_up.RLEN1 = 50.0  # dim plus
        hkb_up.RLEN2 = 50.0  # dim minus
        hkb_up.RWIDX1 = 24.75
        hkb_up.RWIDX2 = 24.75
        hkb_up.SIMAG = -999
        hkb_up.SSOUR = 50775.983
        hkb_up.THETA = hkb_pitch_angle_shadow
        hkb_up.T_IMAGE = 123.0
        hkb_up.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb_up.T_REFLECTION = hkb_pitch_angle_shadow
        hkb_up.T_SOURCE = 0.0

        # DISPLACEMENT
        hkb_up.F_MOVE = 1
        hkb_up.X_ROT = hkb_pitch_angle_displacement_shadow
        hkb_up.OFFY = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb_up.OFFZ = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        # H-KB
        hkb_down = Shadow.OE()
        hkb_down.ALPHA = 90.0
        hkb_down.DUMMY = 0.1
        hkb_down.FCYL = 1
        hkb_down.FHIT_C = 1
        hkb_down.FILE_REFL = reflectivity_file.encode()
        hkb_down.FILE_RIP = hkb_error_profile_file.encode()
        hkb_down.FMIRR = 2
        hkb_down.FWRITE = 1
        hkb_down.F_DEFAULT = 0
        hkb_down.F_G_S = 2
        hkb_down.F_REFLEC = 1
        hkb_down.F_RIPPLE = 1
        hkb_down.RLEN1 = 50.0  # dim plus
        hkb_down.RLEN2 = 50.0  # dim minus
        hkb_down.RWIDX1 = 24.75
        hkb_down.RWIDX2 = 24.75
        hkb_down.SIMAG = -999
        hkb_down.SSOUR = 50775.983
        hkb_down.THETA = hkb_pitch_angle_shadow
        hkb_down.T_IMAGE = 123.0
        hkb_down.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb_down.T_REFLECTION = hkb_pitch_angle_shadow
        hkb_down.T_SOURCE = 0.0

        # DISPLACEMENT
        hkb_down.F_MOVE = 1
        hkb_down.X_ROT = hkb_pitch_angle_displacement_shadow
        hkb_down.OFFY = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb_down.OFFZ = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        self._vkb = [ShadowOpticalElement(vkb_up), ShadowOpticalElement(vkb_down)]
        self._hkb = [ShadowOpticalElement(hkb_up), ShadowOpticalElement(hkb_down)]

    # ---- H-KB ---------------------------------------------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[0]

    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[1]

    def get_vkb_q_distance(self):
        return self._get_q_distance(self._vkb[0]), self._get_q_distance(self._vkb[1])

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._vkb[0], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1], invert=True)
        self._move_motor_3_pitch(self._vkb[1], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1], invert=True)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._vkb[0], units, invert=True)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._vkb[0], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1], invert=True)
        self._move_motor_4_transation(self._vkb[1], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1], invert=True)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._vkb[0], units, invert=True)

    # ---- H-KB ---------------------------------------------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[0]

    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[1]

    def get_hkb_q_distance(self):
        return self._get_q_distance(self._hkb[0]), self._get_q_distance(self._hkb[1])

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._hkb[0], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])
        self._move_motor_3_pitch(self._hkb[1], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._hkb[0], units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._hkb[0], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])
        self._move_motor_4_transation(self._hkb[1], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._hkb[0], units)

    # IMPLEMENTATION OF PROTECTED METHODS FROM SUPERCLASS

    def _trace_vkb(self, random_seed, remove_lost_rays, verbose):
        output_beam_upstream, cursor_upstream, output_beam_downstream, cursor_downstream = \
            self.__trace_kb(bender_manager=self.__vkb_bender_manager,
                            input_beam=self._slits_beam,
                            widget_class_name="DoubleRodBenderEllypticalMirror",
                            oe_name="V-KB",
                            remove_lost_rays=remove_lost_rays)

        def run_hybrid(output_beam, increment):
            # NOTE: Near field not possible for vkb (beam is untraceable)
            try:
                return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                         diffraction_plane=2,  # Tangential
                                                                         calcType=3,  # Diffraction by Mirror Size + Errors
                                                                         verbose=verbose,
                                                                         random_seed=None if random_seed is None else (random_seed + increment))).ff_beam
            except Exception:
                raise HybridFailureException(oe="V-KB")

        output_beam_upstream = run_hybrid(output_beam_upstream, increment=200)
        output_beam_upstream._beam.rays = output_beam_upstream._beam.rays[cursor_upstream]

        output_beam_downstream = run_hybrid(output_beam_downstream, increment=201)
        output_beam_downstream._beam.rays = output_beam_downstream._beam.rays[cursor_downstream]

        return ShadowBeam.mergeBeams(output_beam_upstream, output_beam_downstream, which_flux=3, merge_history=0)

    def _trace_hkb(self, near_field_calculation, random_seed, remove_lost_rays, verbose):
        output_beam_upstream, cursor_upstream, output_beam_downstream, cursor_downstream = \
            self.__trace_kb(bender_manager=self.__hkb_bender_manager,
                            input_beam=self._vkb_beam,
                            widget_class_name="DoubleRodBenderEllypticalMirror",
                            oe_name="H-KB",
                            remove_lost_rays=remove_lost_rays)

        def run_hybrid(output_beam, increment):
            try:
                if not near_field_calculation:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment))).ff_beam
                else:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             nf=1,
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment))).nf_beam
            except Exception:
                raise HybridFailureException(oe="H-KB")

        output_beam_upstream = run_hybrid(output_beam_upstream, increment=300)
        output_beam_upstream._beam.rays = output_beam_upstream._beam.rays[cursor_upstream]

        output_beam_downstream = run_hybrid(output_beam_downstream, increment=301)
        output_beam_downstream._beam.rays = output_beam_downstream._beam.rays[cursor_downstream]

        output_beam = ShadowBeam.mergeBeams(output_beam_upstream, output_beam_downstream, which_flux=3, merge_history=0)

        return rotate_axis_system(output_beam, rotation_angle=270.0)

    # PRIVATE METHODS

    def __trace_kb(self, bender_manager, input_beam, widget_class_name, oe_name, remove_lost_rays):
        upstream_widget   = bender_manager._kb_upstream
        downstream_widget = bender_manager._kb_downstream

        upstream_oe   = upstream_widget.shadow_oe.duplicate()
        downstream_oe = downstream_widget.shadow_oe.duplicate()

        upstream_oe._oe.RLEN1   = 0.0  # no positive part
        downstream_oe._oe.RLEN2 = 0.0  # no negative part

        # trace both sides separately and get the beams:
        upstream_beam_cursor = numpy.where(self._trace_oe(input_beam=input_beam,
                                                          shadow_oe=upstream_oe,
                                                          widget_class_name=widget_class_name,
                                                          oe_name=oe_name + "_UPSTREAM",
                                                          remove_lost_rays=False,
                                                          history=False)._beam.rays[:, 9] == 1)

        downstream_beam_cursor = numpy.where(self._trace_oe(input_beam=input_beam,
                                                            shadow_oe=downstream_oe,
                                                            widget_class_name=widget_class_name,
                                                            oe_name=oe_name + "_DOWNSTREAM",
                                                            remove_lost_rays=False,
                                                            history=False)._beam.rays[:, 9] == 1)

        # this make HYBRID FAIL! we have to do it after the hybrid calculation
        # upstream_input_beam   = input_beam.duplicate()
        # downstream_input_beam = input_beam.duplicate()
        # upstream_input_beam._beam.rays   = upstream_input_beam._beam.rays[upstream_beam_cursor]
        # downstream_input_beam._beam.rays = downstream_input_beam._beam.rays[downstream_beam_cursor]

        def calculate_bender(input_beam, widget, do_calculation=True):
            widget.R0 = widget.R0_out  # use last fit result

            if do_calculation:
                widget.shadow_oe._oe.FILE_RIP = bytes(widget.ms_defect_file_name, 'utf-8')  # restore original error profile

                apply_bender_surface(widget=widget, shadow_oe=widget.shadow_oe)
            else:
                widget.shadow_oe._oe.F_RIPPLE = 1
                widget.shadow_oe._oe.F_G_S = 2
                widget.shadow_oe._oe.FILE_RIP = bytes(widget.output_file_name_full, 'utf-8')

        q_upstream, q_downstream = bender_manager.get_q_distances()

        if (q_upstream != bender_manager.q_upstream_previous) or (q_downstream != bender_manager.q_downstream_previous) or \
                (not os.path.exists(upstream_widget.output_file_name_full)) or (not os.path.exists(downstream_widget.output_file_name_full)):            # trace both the beam on the whole bender widget
            calculate_bender(input_beam, upstream_widget)
            calculate_bender(input_beam, downstream_widget)
        else:
            calculate_bender(input_beam, upstream_widget, do_calculation=False)
            calculate_bender(input_beam, downstream_widget, do_calculation=False)

        bender_manager.q_upstream_previous   = q_upstream
        bender_manager.q_downstream_previous = q_downstream

        # Redo raytracing with the bender correction as error profile
        return self._trace_oe(input_beam=input_beam,
                              shadow_oe=upstream_widget.shadow_oe,
                              widget_class_name=widget_class_name,
                              oe_name=oe_name,
                              remove_lost_rays=remove_lost_rays), \
               upstream_beam_cursor, \
               self._trace_oe(input_beam=input_beam,
                              shadow_oe=downstream_widget.shadow_oe,
                              widget_class_name=widget_class_name,
                              oe_name=oe_name,
                              remove_lost_rays=remove_lost_rays), \
               downstream_beam_cursor

    @classmethod
    def __move_motor_1_2_bender(cls, bender_manager, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=2):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        current_pos_upstream, current_pos_downstream = bender_manager.get_positions()

        def check_pos(pos, current_pos):
            if not pos is None:
                if units == DistanceUnits.MILLIMETERS:
                    return round(pos * 1e3, round_digit)
                elif units == DistanceUnits.MICRON:
                    return round(pos, round_digit)
                else:
                    raise ValueError("Distance units not recognized")
            else:
                return 0.0 if movement == Movement.RELATIVE else current_pos

        pos_upstream = check_pos(pos_upstream, current_pos_upstream)
        pos_downstream = check_pos(pos_downstream, current_pos_downstream)

        if movement == Movement.ABSOLUTE:
            bender_manager.set_positions(pos_upstream, pos_downstream)
        elif movement == Movement.RELATIVE:
            current_pos_upstream, current_pos_downstream = bender_manager.get_positions()
            bender_manager.set_positions(current_pos_upstream + pos_upstream, current_pos_downstream + pos_downstream)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_1_2_bender(cls, bender_manager, units=DistanceUnits.MICRON):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        pos_upstream, pos_downstream = bender_manager.get_positions()

        if units == DistanceUnits.MILLIMETERS:
            pos_upstream *= 1e-3
            pos_downstream *= 1e-3
        elif units == DistanceUnits.MICRON:
            pass
        else:
            raise ValueError("Distance units not recognized")

        return pos_upstream, pos_downstream
