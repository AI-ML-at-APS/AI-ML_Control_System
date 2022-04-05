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

from orangecontrib.ml.util.data_structures import DictionaryWrapper

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

def get_default_input_features():
    return DictionaryWrapper(coh_slits_h_aperture=0.03,
                             coh_slits_h_center=0.0,
                             coh_slits_v_aperture=0.07,
                             coh_slits_v_center=0.0,
                             vkb_q_distance=221,
                             vkb_motor_4_translation=0.0,
                             vkb_motor_3_pitch_angle=0.003,
                             vkb_motor_3_delta_pitch_angle=0.0,
                             hkb_q_distance=120,
                             hkb_motor_4_translation=0.0,
                             hkb_motor_3_pitch_angle=0.003,
                             hkb_motor_3_delta_pitch_angle=0.0)

class MotorResolution:
    __instance = None
                                           #value #digits to round
    __vkb_motor_1_2_bender_resolution    = [1e-5, 5]  # mm
    __vkb_motor_3_pitch_resolution       = [1e-4, 4]  # deg
    __vkb_motor_4_translation_resolution = [1e-3, 3]  # mm
    __hkb_motor_1_2_bender_resolution    = [1e-5, 5]  # mm
    __hkb_motor_3_pitch_resolution       = [1e-4, 4]  # deg
    __hkb_motor_4_translation_resolution = [1e-3, 3]  # mm

    @staticmethod
    def getInstance():
      if MotorResolution.__instance == None: MotorResolution()
      return MotorResolution.__instance

    def __init__(self):
      if MotorResolution.__instance != None: raise Exception("This class is a singleton!")
      else: MotorResolution.__instance = self

    def get_vkb_motor_1_2_bender_resolution(self):    return self.__vkb_motor_1_2_bender_resolution
    def get_vkb_motor_3_pitch_resolution(self):       return self.__vkb_motor_3_pitch_resolution
    def get_vkb_motor_4_translation_resolution(self): return self.__vkb_motor_4_translation_resolution
    def get_vkb_motor_1_2_bender_resolution(self):    return self.__hkb_motor_1_2_bender_resolution
    def get_hkb_motor_3_pitch_resolution(self):       return self.__hkb_motor_3_pitch_resolution
    def get_hkb_motor_4_translation_resolution(self): return self.__hkb_motor_4_translation_resolution

class AbstractFocusingOptics():
    def initialize(self, input_photon_beam, input_features=get_default_input_features(), **kwargs): raise NotImplementedError()

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None): raise NotImplementedError()
    def restore_input_photon_beam(self): raise NotImplementedError()

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None): raise NotImplementedError()
    def get_coherence_slits_parameters(self): raise NotImplementedError() # center x, center z, aperture x, aperture z

    # V-KB -----------------------

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_vkb_motor_4_translation(self): raise NotImplementedError()
    def move_vkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_motor_1_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_q_distance(self): raise NotImplementedError()

    # H-KB -----------------------

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): raise NotImplementedError()
    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_hkb_motor_4_translation(self): raise NotImplementedError()
    def move_hkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_hkb_motor_1_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_hkb_q_distance(self): raise NotImplementedError()

    #####################################################################################
    # Run the simulation

    def get_photon_beam(self, **kwargs): raise NotImplementedError()
