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
from orangecontrib.shadow.util.shadow_util import ShadowMath, ShadowCongruence
from orangecontrib.shadow.util.shadow_objects import ShadowBeam

from aps.common.ml.data_structures import DictionaryWrapper

from aps.ai.autoalignment.common.facade.parameters import Movement, AngularUnits, DistanceUnits
from aps.ai.autoalignment.common.util.shadow.common import EmptyBeamException
from aps.ai.autoalignment.common.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics

class AbstractShadowFocusingOptics(AbstractSimulatedFocusingOptics):
    def __init__(self):
        self._input_beam = None
        self.__initial_input_beam = None
        self._modified_elements = None

    def initialize(self, **kwargs):
        input_photon_beam = kwargs["input_photon_beam"]

        self._input_beam          = input_photon_beam.duplicate()
        self.__initial_input_beam = input_photon_beam.duplicate()

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None):
        if self._input_beam is None: raise ValueError("Focusing Optical System is not initialized")

        good_only = numpy.where(self._input_beam._beam.rays[:, 9] == 1)

        if not shift_h is None: self._input_beam._beam.rays[good_only, 0] += shift_h
        if not shift_v is None: self._input_beam._beam.rays[good_only, 2] += shift_v

        v_out = [self._input_beam._beam.rays[good_only, 3],
                 self._input_beam._beam.rays[good_only, 4],
                 self._input_beam._beam.rays[good_only, 5]]

        if not rotation_h is None: v_out = ShadowMath.vector_rotate([0, 0, 1], rotation_h, v_out)
        if not rotation_v is None: v_out = ShadowMath.vector_rotate([1, 0, 0], rotation_v, v_out)

        if not (rotation_h is None and rotation_v is None):
            self._input_beam._beam.rays[good_only, 3] = v_out[0]
            self._input_beam._beam.rays[good_only, 4] = v_out[1]
            self._input_beam._beam.rays[good_only, 5] = v_out[2]

    def restore_input_photon_beam(self):
        if self._input_beam is None: raise ValueError("Focusing Optical System is not initialized")
        self._input_beam = self.__initial_input_beam.duplicate()

    # PROTECTED GENERIC MOTOR METHODS
    @classmethod
    def _move_pitch_motor(cls, element, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, round_digit=4, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == AngularUnits.MILLIRADIANS: angle = numpy.degrees(angle * 1e-3)
        elif units == AngularUnits.DEGREES:    pass
        elif units == AngularUnits.RADIANS:    angle = numpy.degrees(angle)
        else: raise ValueError("Angular units not recognized")

        sign = -1 if invert else 1

        if movement == Movement.ABSOLUTE:   element._oe.X_ROT = sign * (round(angle, round_digit) - (90 - element._oe.T_INCIDENCE))
        elif movement == Movement.RELATIVE: element._oe.X_ROT += sign * round(angle, round_digit)
        else: raise ValueError("Movement not recognized")

    @classmethod
    def _move_translation_motor(cls, element, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=3, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON:        translation *= 1e-3
        elif units == DistanceUnits.MILLIMETERS: pass
        else: raise ValueError("Distance units not recognized")

        sign = -1 if invert else 1

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + sign * element._oe.X_ROT)

        if movement == Movement.ABSOLUTE:
            element._oe.OFFY = round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ = round(translation, round_digit) * numpy.cos(total_pitch_angle)
        elif movement == Movement.RELATIVE:
            element._oe.OFFY += round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ += round(translation, round_digit) * numpy.cos(total_pitch_angle)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def _change_shape(cls, element, q_distance, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if movement == Movement.ABSOLUTE:   element._oe.SIMAG = q_distance
        elif movement == Movement.RELATIVE: element._oe.SIMAG += q_distance
        else: raise ValueError("Movement not recognized")

    @classmethod
    def _get_pitch_motor_value(cls, element, units=AngularUnits.MILLIRADIANS, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        sign = -1 if invert else 1

        angle = sign*(element._oe.X_ROT + sign * (90 - element._oe.T_INCIDENCE))

        if units == AngularUnits.MILLIRADIANS: return 1000 * numpy.radians(angle)
        elif units == AngularUnits.DEGREES:    return angle
        elif units == AngularUnits.RADIANS:    return numpy.radians(angle)
        else: raise ValueError("Angular units not recognized")

    @classmethod
    def _get_translation_motor_value(cls, element, units=DistanceUnits.MICRON, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        pitch_angle = cls._get_pitch_motor_value(element, units=AngularUnits.RADIANS, invert=invert)
        translation = numpy.average([element._oe.OFFY / numpy.sin(pitch_angle), element._oe.OFFZ / numpy.cos(pitch_angle)])

        if units == DistanceUnits.MICRON:        return translation * 1e3
        elif units == DistanceUnits.MILLIMETERS: return translation
        else: raise ValueError("Distance units not recognized")

    @classmethod
    def _get_q_distance(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")
        return element._oe.SIMAG

    @classmethod
    def _trace_oe(cls, input_beam, shadow_oe, widget_class_name, oe_name, remove_lost_rays, history=True):
        return cls._check_beam(ShadowBeam.traceFromOE(input_beam,
                                                      shadow_oe.duplicate(),
                                                      widget_class_name=widget_class_name,
                                                      history=history,
                                                      recursive_history=False),
                               oe_name, remove_lost_rays)

    @classmethod
    def _check_beam(cls, output_beam, oe, remove_lost_rays):
        if ShadowCongruence.checkEmptyBeam(output_beam):
            if ShadowCongruence.checkGoodBeam(output_beam):
                if remove_lost_rays:
                    output_beam._beam.rays = output_beam._beam.rays[numpy.where(output_beam._beam.rays[:, 9] == 1)]
                    output_beam._beam.rays[:, 11] = numpy.arange(1, output_beam._beam.rays.shape[0] + 1, 1)
                return output_beam
            else: raise EmptyBeamException(oe)
        else: raise EmptyBeamException(oe)
