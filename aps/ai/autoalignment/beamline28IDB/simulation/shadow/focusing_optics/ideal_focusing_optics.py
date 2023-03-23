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

import numpy
import Shadow

from orangecontrib.shadow.util.shadow_objects import ShadowOpticalElement

from aps.ai.autoalignment.common.facade.parameters import Movement, AngularUnits, DistanceUnits
from aps.ai.autoalignment.beamline28IDB.simulation.shadow.focusing_optics.focusing_optics_common import FocusingOpticsCommonAbstract, Layout

class IdealFocusingOptics(FocusingOpticsCommonAbstract):

    def __init__(self):
        super(IdealFocusingOptics, self).__init__()

    def _initialize_mirrors(self, input_features, reflectivity_file, h_bendable_mirror_error_profile_file):
        h_bendable_mirror_motor_pitch_angle              = input_features.get_parameter("h_bendable_mirror_motor_pitch_angle")
        h_bendable_mirror_motor_pitch_angle_shadow       = 90 - numpy.degrees(h_bendable_mirror_motor_pitch_angle)
        h_bendable_mirror_motor_pitch_delta_angle        = input_features.get_parameter("h_bendable_mirror_motor_pitch_delta_angle")
        h_bendable_mirror_motor_pitch_delta_angle_shadow = numpy.degrees(h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_motor_translation              = input_features.get_parameter("h_bendable_mirror_motor_translation")

        h_bendable_mirror = Shadow.OE()
        h_bendable_mirror.ALPHA = 90.0
        h_bendable_mirror.DUMMY = 0.1
        h_bendable_mirror.FCYL = 1
        h_bendable_mirror.FHIT_C = 1
        h_bendable_mirror.FILE_REFL = reflectivity_file.encode()
        h_bendable_mirror.FILE_RIP = h_bendable_mirror_error_profile_file.encode()
        h_bendable_mirror.FMIRR = 2
        h_bendable_mirror.FWRITE = 1
        h_bendable_mirror.F_DEFAULT = 0
        h_bendable_mirror.F_G_S = 2
        h_bendable_mirror.F_REFLEC = 1
        h_bendable_mirror.F_RIPPLE = 1
        h_bendable_mirror.RLEN1 = 140.0
        h_bendable_mirror.RLEN2 = 140.0
        h_bendable_mirror.RWIDX1 = 18.14
        h_bendable_mirror.RWIDX2 = 18.14
        h_bendable_mirror.SIMAG = input_features.get_parameter("h_bendable_mirror_q_distance")
        h_bendable_mirror.SSOUR = 36527.0 + self._shift_horizontal_mirror # instead of 63870.0, because of the convexity of M2
        h_bendable_mirror.THETA = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror.T_IMAGE = 0.0
        h_bendable_mirror.T_INCIDENCE = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror.T_REFLECTION = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror.T_SOURCE = 1370.0 + self._shift_horizontal_mirror

        # DISPLACEMENTS
        h_bendable_mirror.F_MOVE = 1
        h_bendable_mirror.OFFY = h_bendable_mirror_motor_translation * numpy.sin(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror.OFFZ = h_bendable_mirror_motor_translation * numpy.cos(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror.X_ROT = h_bendable_mirror_motor_pitch_delta_angle_shadow

        v_bimorph_mirror_motor_pitch_angle              = input_features.get_parameter("v_bimorph_mirror_motor_pitch_angle")
        v_bimorph_mirror_motor_pitch_angle_shadow       = 90 - numpy.degrees(v_bimorph_mirror_motor_pitch_angle)
        v_bimorph_mirror_motor_pitch_delta_angle        = input_features.get_parameter("v_bimorph_mirror_motor_pitch_delta_angle")
        v_bimorph_mirror_motor_pitch_delta_angle_shadow = numpy.degrees(v_bimorph_mirror_motor_pitch_delta_angle)
        v_bimorph_mirror_motor_translation              = input_features.get_parameter("v_bimorph_mirror_motor_translation")

        # V-KB
        v_bimorph_mirror = Shadow.OE()
        v_bimorph_mirror.ALPHA = 270.0
        v_bimorph_mirror.DUMMY = 0.1
        v_bimorph_mirror.FCYL = 1
        v_bimorph_mirror.FHIT_C = 1
        v_bimorph_mirror.FILE_REFL = reflectivity_file.encode()
        v_bimorph_mirror.FMIRR = 1 #2 switch to spherical for coma aberration
        v_bimorph_mirror.FWRITE = 1
        v_bimorph_mirror.F_DEFAULT = 0
        v_bimorph_mirror.F_REFLEC = 1
        v_bimorph_mirror.RLEN1 = 75.0
        v_bimorph_mirror.RLEN2 = 75.0
        v_bimorph_mirror.RWIDX1 = 4.0
        v_bimorph_mirror.RWIDX2 = 4.0
        v_bimorph_mirror.SIMAG = input_features.get_parameter("v_bimorph_mirror_q_distance")
        v_bimorph_mirror.SSOUR = 65000.0
        v_bimorph_mirror.THETA = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_IMAGE = 2500.0 + self._shift_detector
        v_bimorph_mirror.T_INCIDENCE = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_REFLECTION = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_SOURCE = 1130.0 - self._shift_horizontal_mirror

        # DISPLACEMENTS
        v_bimorph_mirror.F_MOVE = 1
        v_bimorph_mirror.OFFY  = v_bimorph_mirror_motor_translation * numpy.sin(v_bimorph_mirror_motor_pitch_angle + v_bimorph_mirror_motor_pitch_delta_angle)
        v_bimorph_mirror.OFFZ  = v_bimorph_mirror_motor_translation * numpy.cos(v_bimorph_mirror_motor_pitch_angle + v_bimorph_mirror_motor_pitch_delta_angle)
        v_bimorph_mirror.X_ROT = v_bimorph_mirror_motor_pitch_delta_angle_shadow

        self._h_bendable_mirror = ShadowOpticalElement(h_bendable_mirror)
        self._v_bimorph_mirror = ShadowOpticalElement(v_bimorph_mirror)

    def _trace_h_bendable_mirror(self, random_seed, remove_lost_rays, verbose):
        return self._trace_oe(input_beam=self._input_beam,
                              shadow_oe=self._h_bendable_mirror,
                              widget_class_name="EllypticalMirror",
                              oe_name="H-Bendable-Mirror",
                              remove_lost_rays=remove_lost_rays)

    def _trace_v_bimorph_mirror(self, random_seed, remove_lost_rays, verbose):
        return self._trace_oe(input_beam=self._h_bendable_mirror_beam,
                              shadow_oe=self._v_bimorph_mirror,
                              widget_class_name="EllypticalMirror",
                              oe_name="V-Bimorph-Mirror",
                              remove_lost_rays=remove_lost_rays)

    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_pitch_motor(self._h_bendable_mirror, angle, movement, units,
                               round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_pitch", units=AngularUnits.DEGREES)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements:  self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self._get_pitch_motor_value(self._h_bendable_mirror, units)

    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translation_motor(self._h_bendable_mirror, translation, movement, units,
                                     round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_translation", units=DistanceUnits.MILLIMETERS)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements: self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MICRON):
        return self._get_translation_motor_value(self._h_bendable_mirror, units)

    def change_h_bendable_mirror_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._change_shape(self._h_bendable_mirror, q_distance, movement)

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements:  self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_q_distance(self):
        return self._get_q_distance(self._h_bendable_mirror)

    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_pitch_motor(self._v_bimorph_mirror, angle, movement, units,
                                 round_digit=self._motor_resolution.get_motor_resolution("v_bimorph_mirror_motor_pitch", units=AngularUnits.DEGREES)[1])

        if not self._v_bimorph_mirror in self._modified_elements: self._modified_elements.append(self._v_bimorph_mirror)

    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self._get_pitch_motor_value(self._v_bimorph_mirror, units)

    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translation_motor(self._v_bimorph_mirror, translation, movement, units,
                                     round_digit=self._motor_resolution.get_motor_resolution("v_bimorph_mirror_motor_translation", units=DistanceUnits.MILLIMETERS)[1])

        if not self._v_bimorph_mirror in self._modified_elements: self._modified_elements.append(self._v_bimorph_mirror)

    def get_v_bimorph_mirror_motor_translation(self, units=DistanceUnits.MICRON):
        return self._get_translation_motor_value(self._v_bimorph_mirror, units)


    def change_v_bimorph_mirror_shape(self, q_distance, movement=Movement.ABSOLUTE):
        self._change_shape(self._v_bimorph_mirror, q_distance, movement)

        if not self._v_bimorph_mirror in self._modified_elements: self._modified_elements.append(self._v_bimorph_mirror)

    def get_v_bimorph_mirror_q_distance(self):
        return self._get_q_distance(self._v_bimorph_mirror)
