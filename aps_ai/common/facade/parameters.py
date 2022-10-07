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

class ExecutionMode:
    SIMULATION = 0
    HARDWARE = 1

class Movement:
    ABSOLUTE = 0
    RELATIVE = 1

class AngularUnits:
    MILLIRADIANS = 0
    DEGREES = 1
    RADIANS = 2

class DistanceUnits:
    MILLIMETERS = 0
    MICRON      = 1
    OTHER       = 2


class MotorType:
    ROTATIONAL    = 0
    TRANSLATIONAL = 1
    OTHER         = 2

class MotorResolution:
    def __init__(self, resolution: float, motor_type : MotorType):
        self.resolution = resolution
        self.motor_type = motor_type

import decimal

class MotorResolutionSet:
    def __init__(self, motors : dict = {}):
        self.__motors = motors

    def get_motor_resolution(self, name, units) -> tuple:
        if not name in self.__motors.keys(): raise ValueError("Motor not present")

        motor_resolution  = self.__motors[name]

        if   motor_resolution.motor_type == MotorType.ROTATIONAL:    return MotorResolutionSet.__get_rotational_resolution(motor_resolution.resolution, units)
        elif motor_resolution.motor_type == MotorType.TRANSLATIONAL: return MotorResolutionSet.__get_translational_resolution(motor_resolution.resolution, units)
        elif motor_resolution.motor_type == MotorType.OTHER:         return MotorResolutionSet.__get_other_resolution(motor_resolution.resolution)

    @classmethod
    def __get_translational_resolution(cls, resolution : float, units=DistanceUnits.MICRON) -> tuple:
        significant_digits = decimal.Decimal(resolution).as_tuple().exponent

        if units==DistanceUnits.MILLIMETERS: return [resolution,     significant_digits]
        elif units==DistanceUnits.MICRON:    return [1e3*resolution, significant_digits - 3]
        else: raise ValueError("Units not recognized")

    @classmethod
    def __get_rotational_resolution(cls, resolution, units=AngularUnits.MILLIRADIANS) -> tuple:
        significant_digits = decimal.Decimal(resolution).as_tuple().exponent

        if units==AngularUnits.DEGREES:        return [resolution,                    significant_digits]
        elif units==AngularUnits.MILLIRADIANS: return [1e3*numpy.radians(resolution), significant_digits - 1]
        elif units==AngularUnits.RADIANS:      return [numpy.radians(resolution),     significant_digits + 2]
        else: raise ValueError("Units not recognized")

    @classmethod
    def __get_other_resolution(cls, resolution : float) -> tuple:
        return [resolution, decimal.Decimal(resolution).as_tuple().exponent]

'''
                                           #value #digits to round
    __coh_slits_motors_resolution        = [1e-7, 7]  # mm
    __vkb_motor_1_2_bender_resolution    = [1e-7, 7]  # mm
    __vkb_motor_3_pitch_resolution       = [1e-4, 4]  # deg
    __vkb_motor_4_translation_resolution = [1e-4, 4]  # mm
    __hkb_motor_1_2_bender_resolution    = [1e-7, 7]  # mm
    __hkb_motor_3_pitch_resolution       = [1e-4, 4]  # deg
    __hkb_motor_4_translation_resolution = [1e-4, 4]  # mm

    def get_coh_slits_motors_resolution(self, units=DistanceUnits.MICRON):        return self.__get_translational_resolution(self.__coh_slits_motors_resolution, units)
    def get_vkb_motor_1_2_bender_resolution(self, units=DistanceUnits.MICRON):    return self.__get_translational_resolution(self.__vkb_motor_1_2_bender_resolution, units)
    def get_vkb_motor_3_pitch_resolution(self, units=AngularUnits.MILLIRADIANS):  return self.__get_rotational_resolution(self.__vkb_motor_3_pitch_resolution, units)
    def get_vkb_motor_4_translation_resolution(self, units=DistanceUnits.MICRON): return self.__get_translational_resolution(self.__vkb_motor_4_translation_resolution, units)
    def get_hkb_motor_1_2_bender_resolution(self, units=DistanceUnits.MICRON):    return self.__get_translational_resolution(self.__hkb_motor_1_2_bender_resolution, units)
    def get_hkb_motor_3_pitch_resolution(self, units=AngularUnits.MILLIRADIANS):  return self.__get_rotational_resolution(self.__hkb_motor_3_pitch_resolution, units)
    def get_hkb_motor_4_translation_resolution(self, units=DistanceUnits.MICRON): return self.__get_translational_resolution(self.__hkb_motor_4_translation_resolution, units)


'''


class MotorResolutionRegistry:
    __instance = None
    __registry = {}

    @staticmethod
    def getInstance():
        if MotorResolutionRegistry.__instance == None: MotorResolutionRegistry.__instance = MotorResolutionRegistry()

        return MotorResolutionRegistry.__instance

    def __init__(self):
      if MotorResolutionRegistry.__instance != None: raise Exception("This class is a singleton!")
      else: MotorResolutionRegistry.__instance = self

    def register_motor_resolution_set(self, motor_resolution_set : MotorResolutionSet, beamline : str):
        if beamline is None or beamline.strip()=="": raise ValueError("empty beamline name")
        if motor_resolution_set is None: raise ValueError("Motor resolution set is None")
        if beamline in self.__registry.keys(): raise ValueError("beamline's motor resolution set is already registered")
        else:  self.__registry[beamline] = motor_resolution_set

    def get_motor_resolution_set(self, beamline: str):
        if not beamline in self.__registry.keys(): raise ValueError("beamline's motor resolution set is not registered")
        else: return self.__registry[beamline]
