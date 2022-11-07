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

from epics import PV

class AbstractEpicsOptics():

    def __init__(self, translational_units=DistanceUnits.MICRON, angular_units=AngularUnits.MILLIRADIANS):
        self.__translational_units=translational_units
        self.__angular_units=angular_units

    # PRIVATE METHODS

    def _move_translational_motor(self, pv : PV, pos, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, wait=True):
        if units == DistanceUnits.MILLIMETERS: pos *= 1e3
        elif units == DistanceUnits.MICRON: pass
        else: raise ValueError("Distance units not recognized")

        if movement == Movement.ABSOLUTE:   pv.put(pos, wait=wait)
        elif movement == Movement.RELATIVE: pv.put(pv.get() + pos, wait=wait)
        else: raise ValueError("Movement not recognized")

    def _move_rotational_motor(self, pv : PV, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, wait=True):
        if units == AngularUnits.MILLIRADIANS:
            if self.__angular_units   == AngularUnits.MILLIRADIANS: pass
            elif self.__angular_units == AngularUnits.DEGREES:      angle = numpy.degrees(1e-3 * angle)
            elif self.__angular_units == AngularUnits.RADIANS:      angle = 1e-3 * angle
        elif units == AngularUnits.DEGREES:
            if self.__angular_units   == AngularUnits.MILLIRADIANS: angle = 1e3 * numpy.radians(angle)
            elif self.__angular_units == AngularUnits.DEGREES:      pass
            elif self.__angular_units == AngularUnits.RADIANS:      angle = numpy.radians(angle)
        elif units == AngularUnits.RADIANS:
            if self.__angular_units   == AngularUnits.MILLIRADIANS: angle = 1e3 * angle
            elif self.__angular_units == AngularUnits.DEGREES:      angle = numpy.degrees(angle)
            elif self.__angular_units == AngularUnits.RADIANS:      pass
        else:  raise ValueError("Angular units not recognized")

        if movement == Movement.ABSOLUTE:   pv.put(angle, wait=wait)
        elif movement == Movement.RELATIVE: pv.put(pv.get() + angle, wait=wait)
        else: raise ValueError("Movement not recognized")

    def _get_translational_motor_position(self, pv : PV, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MICRON:        return pv.get()
        elif units == DistanceUnits.MILLIMETERS: return 1e-3 * pv.get()
        else: raise ValueError("Distance units not recognized")

    def _get_rotational_motor_angle(self, pv : PV, units=AngularUnits.MILLIRADIANS):
        if units == AngularUnits.MILLIRADIANS: return pv.get()
        elif units == AngularUnits.DEGREES:    return numpy.degrees(pv.get() * 1e-3)
        elif units == AngularUnits.RADIANS:    return pv.get() * 1e-3
        else: raise ValueError("Angular units not recognized")
