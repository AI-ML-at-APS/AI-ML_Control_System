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
from aps_ai.common.facade.parameters import MotorResolutionRegistry, MotorResolutionSet, MotorType, MotorResolution, DistanceUnits, Movement, AngularUnits

motors = {}
motors["v_bimorph_mirror_motor_bender"]       = MotorResolution(1.0,  MotorType.OTHER)         # Bimorph mirror: bender is an actuator, "position" is in Volt
motors["v_bimorph_mirror_motor_pitch"]        = MotorResolution(1e-4, MotorType.ROTATIONAL)    # deg
motors["v_bimorph_mirror_motor_translation"]  = MotorResolution(1e-4, MotorType.TRANSLATIONAL) # mm
motors["h_bendable_mirror_motor_bender"]      = MotorResolution(1e-7, MotorType.TRANSLATIONAL) # mm
motors["h_bendable_mirror_motor_pitch"]       = MotorResolution(1e-4, MotorType.ROTATIONAL)    # deg
motors["h_bendable_mirror_motor_translation"] = MotorResolution(1e-4, MotorType.TRANSLATIONAL) # mm

MotorResolutionRegistry.getInstance().register_motor_resolution_set(MotorResolutionSet(motors=motors), "28-ID-B")

class AbstractFocusingOptics():

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    # V-KB -----------------------

    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_bender(self): raise NotImplementedError()
    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_v_bimorph_mirror_motor_translation(self, units=DistanceUnits.MICRON): raise NotImplementedError()

    # H-KB -----------------------

    def move_h_bendable_mirror_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_h_bendable_mirror_motor_1_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_h_bendable_mirror_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_h_bendable_mirror_motor_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MICRON): raise NotImplementedError()

