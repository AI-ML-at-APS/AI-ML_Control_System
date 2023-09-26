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
from aps.ai.autoalignment.common.facade.focusing_optics_interface import AbstractFocusingOptics as CommonAbstractFocusingOptics
from aps.ai.autoalignment.common.facade.parameters import MotorResolutionRegistry, MotorResolutionSet, MotorType, MotorResolution, DistanceUnits, Movement, AngularUnits
from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

APPLICATION_NAME = "motors configuration"
register_ini_instance(ini_mode=IniMode.LOCAL_FILE, application_name=APPLICATION_NAME, ini_file_name="motors_configuration.ini")

ini_file = get_registered_ini_instance(APPLICATION_NAME)

res_v_bimorph_mirror_motor_bender       = ini_file.get_float_from_ini(section="Motor-Resolution",  key="VKB-Bender",      default=1.0)
res_v_bimorph_mirror_motor_translation  = ini_file.get_float_from_ini(section="Motor-Resolution",  key="VKB-Translation", default=0.0001)
res_h_bendable_mirror_motor_bender      = ini_file.get_float_from_ini(section="Motor-Resolution",  key="HKB-Bender",      default=1.0)
res_h_bendable_mirror_motor_pitch       = ini_file.get_float_from_ini(section="Motor-Resolution",  key="HKB-Pitch",       default=0.00001)
res_h_bendable_mirror_motor_translation = ini_file.get_float_from_ini(section="Motor-Resolution",  key="HKB-Translation", default=0.0001)

DISTANCE_V_MOTORS = 360 # mm

res_v_bimorph_mirror_motor_pitch = numpy.round(numpy.degrees(numpy.arcsin(res_v_bimorph_mirror_motor_translation / DISTANCE_V_MOTORS)), 5)

print("28-ID - VKB Pitch Motor Resolution: " + str(res_v_bimorph_mirror_motor_pitch) + " deg")

ini_file.set_value_at_ini(section="Motor-Resolution",  key="VKB-Bender",      value=res_v_bimorph_mirror_motor_bender)
ini_file.set_value_at_ini(section="Motor-Resolution",  key="VKB-Pitch",       value=res_v_bimorph_mirror_motor_pitch)
ini_file.set_value_at_ini(section="Motor-Resolution",  key="VKB-Translation", value=res_v_bimorph_mirror_motor_translation)
ini_file.set_value_at_ini(section="Motor-Resolution",  key="HKB-Bender",      value=res_h_bendable_mirror_motor_bender)
ini_file.set_value_at_ini(section="Motor-Resolution",  key="HKB-Pitch",       value=res_h_bendable_mirror_motor_pitch)
ini_file.set_value_at_ini(section="Motor-Resolution",  key="HKB-Translation", value=res_h_bendable_mirror_motor_translation)

ini_file.push()


motors = {}
motors["v_bimorph_mirror_motor_bender"]       = MotorResolution(res_v_bimorph_mirror_motor_bender,       MotorType.OTHER)         # Bimorph mirror: bender is an actuator, "position" is in Volt
motors["v_bimorph_mirror_motor_pitch"]        = MotorResolution(res_v_bimorph_mirror_motor_pitch,        MotorType.ROTATIONAL)    # deg
motors["v_bimorph_mirror_motor_translation"]  = MotorResolution(res_v_bimorph_mirror_motor_translation,  MotorType.TRANSLATIONAL) # mm
motors["h_bendable_mirror_motor_bender"]      = MotorResolution(res_h_bendable_mirror_motor_bender,      MotorType.OTHER)         # Deming's bender: motors are micron
motors["h_bendable_mirror_motor_pitch"]       = MotorResolution(res_h_bendable_mirror_motor_pitch,       MotorType.ROTATIONAL)    # deg
motors["h_bendable_mirror_motor_translation"] = MotorResolution(res_h_bendable_mirror_motor_translation, MotorType.TRANSLATIONAL) # mm

MotorResolutionRegistry.getInstance().register_motor_resolution_set(MotorResolutionSet(motors=motors), "28-ID-B")

class AbstractFocusingOptics(CommonAbstractFocusingOptics):

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    # V-KB -----------------------

    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_bender(self): raise NotImplementedError()
    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.DEGREES): raise NotImplementedError()
    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS): raise NotImplementedError()

    # H-KB -----------------------

    def move_h_bendable_mirror_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_h_bendable_mirror_motor_1_bender(self): raise NotImplementedError()
    def move_h_bendable_mirror_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_h_bendable_mirror_motor_2_bender(self): raise NotImplementedError()
    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES): raise NotImplementedError()
    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.DEGREES): raise NotImplementedError()
    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS): raise NotImplementedError()
    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS): raise NotImplementedError()

