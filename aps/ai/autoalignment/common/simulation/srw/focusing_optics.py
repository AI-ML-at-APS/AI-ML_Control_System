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

from aps.ai.autoalignment.common.facade.parameters import Movement, DistanceUnits, AngularUnits
from aps.ai.autoalignment.common.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics

from aps.common.ml.data_structures import DictionaryWrapper

from wofrysrw.beamline.srw_beamline import SRWBeamline
from wofrysrw.beamline.optical_elements.srw_optical_element import SRWOpticalElementDisplacement
from wofrysrw.beamline.optical_elements.mirrors.srw_mirror import Orientation

class SRWFocusingOptics(AbstractSimulatedFocusingOptics):
    def __init__(self):
        self._input_wavefront = None
        self.__initial_input_wavefront = None
        self._beamline = None
        self._modified_elements = None

    def initialize(self, **kwargs):
        input_photon_beam = kwargs["input_photon_beam"]

        self._input_wavefront          = input_photon_beam.duplicate()
        self.__initial_input_wavefront = input_photon_beam.duplicate()
        self._beamline                 = SRWBeamline()

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None):
        pass

    def restore_input_photon_beam(self):
        if self._input_wavefront is None: raise ValueError("Focusing Optical System is not initialized")
        self._input_wavefront = self.__initial_input_wavefront.duplicate()

    # PROTECTED GENERIC MOTOR METHODS
    @classmethod
    def _move_pitch_motor(cls, element, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, round_digit=4, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == AngularUnits.MILLIRADIANS: angle = angle * 1e-3
        elif units == AngularUnits.DEGREES:    angle = numpy.radians(angle)
        elif units == AngularUnits.RADIANS:     pass
        else: raise ValueError("Angular units not recognized")

        sign = -1 if invert else 1

        if movement == Movement.ABSOLUTE:
            if element.orientation_of_reflection_plane == Orientation.LEFT or \
                    element.orientation_of_reflection_plane == Orientation.RIGHT:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=sign * round(angle, round_digit),
                                                                     rotation_y=element.displacement.rotation_y)
            else:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=sign * round(angle, round_digit))
        elif movement == Movement.RELATIVE:
            if element.orientation_of_reflection_plane == Orientation.LEFT or \
                    element.orientation_of_reflection_plane == Orientation.RIGHT:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=element.displacement.rotation_x + sign * round(angle, round_digit),
                                                                     rotation_y=element.displacement.rotation_y)
            else:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=element.displacement.rotation_y + sign * round(angle, round_digit))
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def _move_translation_motor(cls, element, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=3):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON:        translation *= 1e-6
        elif units == DistanceUnits.MILLIMETERS: translation *= 1e-3
        else: raise ValueError("Distance units not recognized")

        if movement == Movement.ABSOLUTE:
            if element.orientation_of_reflection_plane == Orientation.UP or \
                    element.orientation_of_reflection_plane == Orientation.DOWN:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=round(translation, round_digit),
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=element.displacement.rotation_y)
            else:
                element.displacement = SRWOpticalElementDisplacement(shift_x=round(translation, round_digit),
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=element.displacement.rotation_y)
        elif movement == Movement.RELATIVE:
            if element.orientation_of_reflection_plane == Orientation.UP or \
                    element.orientation_of_reflection_plane == Orientation.DOWN:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x,
                                                                     shift_y=element.displacement.shift_y + round(translation, round_digit),
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=element.displacement.rotation_y)
            else:
                element.displacement = SRWOpticalElementDisplacement(shift_x=element.displacement.shift_x + round(translation, round_digit),
                                                                     shift_y=element.displacement.shift_y,
                                                                     rotation_x=element.displacement.rotation_x,
                                                                     rotation_y=element.displacement.rotation_y)
        else:
            raise ValueError("Movement not recognized")


    @classmethod
    def _change_shape(cls, element, q_distance, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        p, q = element.get_surface_shape().get_p_q(element.grazing_angle)

        if movement == Movement.ABSOLUTE:   q = q_distance * 1e-3
        elif movement == Movement.RELATIVE: q += q_distance * 1e-3
        else: raise ValueError("Movement not recognized")

        element.get_surface_shape().initialize_from_p_q(p, q, element.grazing_angle)

    @classmethod
    def _get_pitch_motor_value(cls, element, units=AngularUnits.MILLIRADIANS, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        sign = -1 if invert else 1

        if element.orientation_of_reflection_plane == Orientation.LEFT or \
                element.orientation_of_reflection_plane == Orientation.RIGHT:
            pitch_angle = element.grazing_angle + sign * element.displacement.rotation_x
        else:
            pitch_angle = element.grazing_angle + sign * element.displacement.rotation_y

        if units == AngularUnits.MILLIRADIANS: return 1000 * pitch_angle
        elif units == AngularUnits.DEGREES:    return numpy.degrees(pitch_angle)
        elif units == AngularUnits.RADIANS:    return pitch_angle
        else: raise ValueError("Angular units not recognized")

    @classmethod
    def _get_translation_motor_value(cls, element, units=DistanceUnits.MICRON):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if element.orientation_of_reflection_plane == Orientation.UP or \
                element.orientation_of_reflection_plane == Orientation.DOWN:
            translation = element.displacement.shift_y
        else:
            translation = element.displacement.shift_x

        if units == DistanceUnits.MICRON:        return translation * 1e6
        elif units == DistanceUnits.MILLIMETERS: return translation * 1e3
        else: raise ValueError("Distance units not recognized")

    @classmethod
    def _get_q_distance(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")
        _, q = element.get_surface_shape().get_p_q(element.grazing_angle)

        return q
