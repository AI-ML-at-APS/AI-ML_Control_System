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

from aps.ai.common.facade.parameters import Movement, DistanceUnits, AngularUnits

from epics import caget, caput

class AbstractEpicsOptics:

    # PRIVATE METHODS

    @classmethod
    def _move_translational_motor(cls, motor, pos, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MILLIMETERS: pos *= 1e3
        elif units == DistanceUnits.MICRON: pass
        else: raise ValueError("Distance units not recognized")

        if movement == Movement.ABSOLUTE:   caput(motor + ".VAL", pos)
        elif movement == Movement.RELATIVE: caput(motor + ".RLV", pos)
        else: raise ValueError("Movement not recognized")

    @classmethod
    def _move_rotational_motor(cls, motor, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        if units == AngularUnits.MILLIRADIANS: pass
        elif units == AngularUnits.DEGREES:    angle = 1e3 * numpy.radians(angle)
        elif units == AngularUnits.RADIANS:    angle = 1e3 * angle
        else:  raise ValueError("Angular units not recognized")

        if movement == Movement.ABSOLUTE:   caput(motor + ".VAL", angle)
        elif movement == Movement.RELATIVE: caput(motor + ".RLV", angle)
        else: raise ValueError("Movement not recognized")

    @classmethod
    def _get_translational_motor_position(cls, motor, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MICRON:        return caget(motor + ".VAL")
        elif units == DistanceUnits.MILLIMETERS: return 1e-3 * caget(motor + ".VAL")
        else: raise ValueError("Distance units not recognized")

    @classmethod
    def _get_rotational_motor_angle(cls, motor, units=AngularUnits.MILLIRADIANS):
        if units == AngularUnits.MILLIRADIANS: return caget(motor)
        elif units == AngularUnits.DEGREES:    return numpy.degrees(caget(motor) * 1e-3)
        elif units == AngularUnits.RADIANS:    return caget(motor) * 1e-3
        else: raise ValueError("Angular units not recognized")
