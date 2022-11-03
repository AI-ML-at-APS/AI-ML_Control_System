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

from aps.ai.autoalignment.beamline28IDB.simulation.shadow.focusing_optics.focusing_optics_common import FocusingOpticsCommon

from aps.ai.autoalignment.beamline28IDB.simulation.shadow.focusing_optics.calibrated_bender import TwoMotorsCalibratedBenderManager, OneMotorCalibratedBenderManager, HKBMockWidget
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import get_default_input_features
from aps.ai.autoalignment.common.facade.parameters import Movement, AngularUnits, DistanceUnits

from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.bendable_ellipsoid_mirror_bl import apply_bender_surface

class BendableFocusingOptics(FocusingOpticsCommon):
    def __init__(self):
        super(BendableFocusingOptics, self).__init__()

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):

        super().initialize(input_photon_beam, input_features, **kwargs)

        self.__hkb_bender_manager = TwoMotorsCalibratedBenderManager(kb_upstream=HKBMockWidget(self._h_bendable_mirror[0], verbose=True, label="Upstream"),
                                                                     kb_downstream=HKBMockWidget(self._h_bendable_mirror[1], verbose=True, label="Downstream"))
        self.__hkb_bender_manager.load_calibration("H-KB")
        self.__hkb_bender_manager.set_voltages(input_features.get_parameter("h_bendable_mirror_motor_1_bender_voltage"),
                                               input_features.get_parameter("h_bendable_mirror_motor_2_bender_voltage"))
        self.__hkb_bender_manager.remove_bender_files()

        self.__vkb_bender_manager = OneMotorCalibratedBenderManager(shadow_oe=self._v_bimorph_mirror)
        self.__vkb_bender_manager.load_calibration("V-KB")
        self.__vkb_bender_manager.set_voltage(input_features.get_parameter("v_bimorph_mirror_motor_bender_voltage"))

    def _initialize_mirrors(self, input_features, reflectivity_file, h_bendable_mirror_error_profile_file):
        h_bendable_mirror_motor_pitch_angle              = input_features.get_parameter("h_bendable_mirror_motor_pitch_angle")
        h_bendable_mirror_motor_pitch_angle_shadow       = 90 - numpy.degrees(h_bendable_mirror_motor_pitch_angle)
        h_bendable_mirror_motor_pitch_delta_angle        = input_features.get_parameter("h_bendable_mirror_motor_pitch_delta_angle")
        h_bendable_mirror_motor_pitch_delta_angle_shadow = numpy.degrees(h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_motor_translation              = input_features.get_parameter("h_bendable_mirror_motor_translation")

        h_bendable_mirror_up = Shadow.OE()
        h_bendable_mirror_up.ALPHA = 90.0
        h_bendable_mirror_up.DUMMY = 0.1
        h_bendable_mirror_up.FCYL = 1
        h_bendable_mirror_up.FHIT_C = 1
        h_bendable_mirror_up.FILE_REFL = reflectivity_file.encode()
        h_bendable_mirror_up.FILE_RIP = h_bendable_mirror_error_profile_file.encode()
        h_bendable_mirror_up.FMIRR = 2
        h_bendable_mirror_up.FWRITE = 1
        h_bendable_mirror_up.F_DEFAULT = 0
        h_bendable_mirror_up.F_G_S = 2
        h_bendable_mirror_up.F_REFLEC = 1
        h_bendable_mirror_up.F_RIPPLE = 1
        h_bendable_mirror_up.RLEN1 = 140.0
        h_bendable_mirror_up.RLEN2 = 140.0
        h_bendable_mirror_up.RWIDX1 = 18.14
        h_bendable_mirror_up.RWIDX2 = 18.14
        h_bendable_mirror_up.SIMAG = -999
        h_bendable_mirror_up.SSOUR = 63870.0
        h_bendable_mirror_up.THETA = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_up.T_IMAGE = 0.0
        h_bendable_mirror_up.T_INCIDENCE = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_up.T_REFLECTION = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_up.T_SOURCE = 1370.0

        # DISPLACEMENTS
        h_bendable_mirror_up.F_MOVE = 1
        h_bendable_mirror_up.OFFY = h_bendable_mirror_motor_translation * numpy.sin(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_up.OFFZ = h_bendable_mirror_motor_translation * numpy.cos(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_up.X_ROT = h_bendable_mirror_motor_pitch_delta_angle_shadow

        h_bendable_mirror_down = Shadow.OE()
        h_bendable_mirror_down.ALPHA = 90.0
        h_bendable_mirror_down.DUMMY = 0.1
        h_bendable_mirror_down.FCYL = 1
        h_bendable_mirror_down.FHIT_C = 1
        h_bendable_mirror_down.FILE_REFL = reflectivity_file.encode()
        h_bendable_mirror_down.FILE_RIP = h_bendable_mirror_error_profile_file.encode()
        h_bendable_mirror_down.FMIRR = 2
        h_bendable_mirror_down.FWRITE = 1
        h_bendable_mirror_down.F_DEFAULT = 0
        h_bendable_mirror_down.F_G_S = 2
        h_bendable_mirror_down.F_REFLEC = 1
        h_bendable_mirror_down.F_RIPPLE = 1
        h_bendable_mirror_down.RLEN1 = 140.0
        h_bendable_mirror_down.RLEN2 = 140.0
        h_bendable_mirror_down.RWIDX1 = 18.14
        h_bendable_mirror_down.RWIDX2 = 18.14
        h_bendable_mirror_down.SIMAG = -999
        h_bendable_mirror_down.SSOUR = 36527.0 # instead of 63870.0, because of the convexity of M2
        h_bendable_mirror_down.THETA = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_down.T_IMAGE = 0.0
        h_bendable_mirror_down.T_INCIDENCE = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_down.T_REFLECTION = h_bendable_mirror_motor_pitch_angle_shadow
        h_bendable_mirror_down.T_SOURCE = 1370.0

        # DISPLACEMENTS
        h_bendable_mirror_down.F_MOVE = 1
        h_bendable_mirror_down.OFFY = h_bendable_mirror_motor_translation * numpy.sin(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_down.OFFZ = h_bendable_mirror_motor_translation * numpy.cos(h_bendable_mirror_motor_pitch_angle + h_bendable_mirror_motor_pitch_delta_angle)
        h_bendable_mirror_down.X_ROT = h_bendable_mirror_motor_pitch_delta_angle_shadow

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
        v_bimorph_mirror.FMIRR = 2
        v_bimorph_mirror.FWRITE = 1
        v_bimorph_mirror.F_DEFAULT = 0
        v_bimorph_mirror.F_REFLEC = 1
        v_bimorph_mirror.RLEN1 = 75.0
        v_bimorph_mirror.RLEN2 = 75.0
        v_bimorph_mirror.RWIDX1 = 4.0
        v_bimorph_mirror.RWIDX2 = 4.0
        v_bimorph_mirror.SIMAG = -999
        v_bimorph_mirror.SSOUR = 65000.0
        v_bimorph_mirror.THETA = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_IMAGE = 3000.0
        v_bimorph_mirror.T_INCIDENCE = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_REFLECTION = v_bimorph_mirror_motor_pitch_angle_shadow
        v_bimorph_mirror.T_SOURCE = 1130.0

        # DISPLACEMENTS
        v_bimorph_mirror.F_MOVE = 1
        v_bimorph_mirror.OFFY  = v_bimorph_mirror_motor_translation * numpy.sin(v_bimorph_mirror_motor_pitch_angle + v_bimorph_mirror_motor_pitch_delta_angle)
        v_bimorph_mirror.OFFZ  = v_bimorph_mirror_motor_translation * numpy.cos(v_bimorph_mirror_motor_pitch_angle + v_bimorph_mirror_motor_pitch_delta_angle)
        v_bimorph_mirror.X_ROT = v_bimorph_mirror_motor_pitch_delta_angle_shadow

        self._h_bendable_mirror = [ShadowOpticalElement(h_bendable_mirror_up), ShadowOpticalElement(h_bendable_mirror_down)]
        self._v_bimorph_mirror = ShadowOpticalElement(v_bimorph_mirror)

    def _trace_h_bendable_mirror(self, random_seed, remove_lost_rays, verbose): 
        upstream_widget   = self.__hkb_bender_manager._kb_upstream
        downstream_widget = self.__hkb_bender_manager._kb_downstream

        def calculate_bender(input_beam, widget, do_calculation=True):
            widget.M1    = widget.M1_out  # use last fit result
            widget.ratio = widget.ratio_out

            if do_calculation:
                widget._shadow_oe._oe.FILE_RIP = bytes(widget.ms_defect_file_name, 'utf-8')  # restore original error profile

                apply_bender_surface(widget=widget, input_beam=input_beam, shadow_oe=widget._shadow_oe)
            else:
                widget._shadow_oe._oe.F_RIPPLE = 1
                widget._shadow_oe._oe.F_G_S = 2
                widget._shadow_oe._oe.FILE_RIP = bytes(widget.output_file_name_full, 'utf-8')

        q_upstream, q_downstream = self.__hkb_bender_manager.get_q_distances()

        if (q_upstream != self.__hkb_bender_manager.q_upstream_previous) or (q_downstream != self.__hkb_bender_manager.q_downstream_previous) or \
                (not os.path.exists(upstream_widget.output_file_name_full)) or (not os.path.exists(downstream_widget.output_file_name_full)):  # trace both the beam on the whole bender widget
            calculate_bender(self._input_beam, upstream_widget)
            calculate_bender(self._input_beam, downstream_widget)
        else:
            calculate_bender(self._input_beam, upstream_widget, do_calculation=False)
            calculate_bender(self._input_beam, downstream_widget, do_calculation=False)

        self.__hkb_bender_manager.q_upstream_previous   = q_upstream
        self.__hkb_bender_manager.q_downstream_previous = q_downstream

        upstream_widget._shadow_oe._oe.RLEN1   = 0.0  # no positive part
        downstream_widget._shadow_oe._oe.RLEN2 = 0.0  # no negative part

        # Redo raytracing with the bender correction as error profile
        output_beam_upstream   = self._trace_oe(input_beam=self._input_beam,
                                                shadow_oe=upstream_widget._shadow_oe,
                                                widget_class_name="BendableEllipsoidMirror",
                                                oe_name="H-KB_UPSTREAM",
                                                remove_lost_rays=remove_lost_rays)
        output_beam_downstream = self._trace_oe(input_beam=self._input_beam,
                                                shadow_oe=downstream_widget._shadow_oe,
                                                widget_class_name="BendableEllipsoidMirror",
                                                oe_name="H-KB_DOWNSTREAM",
                                                remove_lost_rays=remove_lost_rays)

        return ShadowBeam.mergeBeams(output_beam_upstream, output_beam_downstream, which_flux=3, merge_history=0)

    def _trace_v_bimorph_mirror(self,  random_seed, remove_lost_rays, verbose):
        return self._trace_oe(input_beam=self._h_bendable_mirror_beam,
                              shadow_oe=self.__vkb_bender_manager._shadow_oe,
                              widget_class_name="EllipticalMirror",
                              oe_name="V-KB",
                              remove_lost_rays=remove_lost_rays)
        

    def move_h_bendable_mirror_motor_1_bender(self, volt_upstream, movement=Movement.ABSOLUTE):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, volt_upstream, None, movement,
                                     round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_bender", units=DistanceUnits.OTHER)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements:  self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_1_bender(self):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager)[0]

    def move_h_bendable_mirror_motor_2_bender(self, volt_downstream, movement=Movement.ABSOLUTE):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, None, volt_downstream, movement,
                                     round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_bender", units=DistanceUnits.OTHER)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements:  self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_2_bender(self):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager)[1]

    def get_h_bendable_mirror_q_distance(self):
        return self._get_q_distance(self._h_bendable_mirror[0]), self._get_q_distance(self._h_bendable_mirror[1])

    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_pitch_motor(self._h_bendable_mirror[0], angle, movement, units,
                               round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_pitch", units=AngularUnits.DEGREES)[1])
        self._move_pitch_motor(self._h_bendable_mirror[1], angle, movement, units,
                               round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_pitch", units=AngularUnits.DEGREES)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements:  self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self._get_pitch_motor_value(self._h_bendable_mirror[0], units)

    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translation_motor(self._h_bendable_mirror[0], translation, movement, units,
                                     round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_translation", units=DistanceUnits.MILLIMETERS)[1])
        self._move_translation_motor(self._h_bendable_mirror[0], translation, movement, units,
                                     round_digit=self._motor_resolution.get_motor_resolution("h_bendable_mirror_motor_translation", units=DistanceUnits.MILLIMETERS)[1])

        if not self._h_bendable_mirror in self._modified_elements: self._modified_elements.append(self._h_bendable_mirror)
        if not self._v_bimorph_mirror in self._modified_elements: self._modified_elements.append(self._v_bimorph_mirror)

    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MICRON):
        return self._get_translation_motor_value(self._h_bendable_mirror[0], units)

    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE):
        if self.__vkb_bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        round_digit  = self._motor_resolution.get_motor_resolution("v_bimorph_mirror_motor_bender", units=DistanceUnits.OTHER)[1]
        current_volt = self.__vkb_bender_manager.get_voltage()

        def check_volt(volt, current_volt):
            if not volt is None: return round(volt, round_digit)
            else:                return 0.0 if movement == Movement.RELATIVE else current_volt

        volt = check_volt(actuator_value, current_volt)

        if movement == Movement.ABSOLUTE:   self.__vkb_bender_manager.set_voltage(volt)
        elif movement == Movement.RELATIVE: self.__vkb_bender_manager.set_voltage(current_volt + volt)
        else:
            raise ValueError("Movement not recognized")

    def get_v_bimorph_mirror_motor_bender(self):
        if self.__vkb_bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        return self.__vkb_bender_manager.get_voltage()


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

    def get_v_bimorph_mirror_q_distance(self):
        return self._get_q_distance(self._v_bimorph_mirror)

    @classmethod
    def __move_motor_1_2_bender(cls, bender_manager, volt_upstream, volt_downstream, movement=Movement.ABSOLUTE, round_digit=0):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        current_volt_upstream, current_volt_downstream = bender_manager.get_voltages()

        def check_volt(volt, current_volt):
            if not volt is None: return round(volt, round_digit)
            else:                return 0.0 if movement == Movement.RELATIVE else current_volt

        volt_upstream   = check_volt(volt_upstream, current_volt_upstream)
        volt_downstream = check_volt(volt_downstream, current_volt_downstream)

        if movement == Movement.ABSOLUTE:
            bender_manager.set_voltages(volt_upstream, volt_downstream)
        elif movement == Movement.RELATIVE:
            current_volt_upstream, current_volt_downstream = bender_manager.get_voltages()
            bender_manager.set_voltages(current_volt_upstream + volt_upstream, current_volt_downstream + volt_downstream)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_1_2_bender(cls, bender_manager):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        return bender_manager.get_voltages()

