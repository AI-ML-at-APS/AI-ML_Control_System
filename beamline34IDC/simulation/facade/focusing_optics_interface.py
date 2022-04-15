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

from beamline34IDC.facade.focusing_optics_interface import AbstractFocusingOptics, Movement, DistanceUnits

def get_default_input_features():
    return DictionaryWrapper(coh_slits_h_aperture=0.03,
                             coh_slits_h_center=0.0,
                             coh_slits_v_aperture=0.07,
                             coh_slits_v_center=0.0,
                             vkb_q_distance=221,
                             vkb_motor_4_translation=0.0,
                             vkb_motor_3_pitch_angle=0.003,
                             vkb_motor_3_delta_pitch_angle=0.0,
                             vkb_motor_1_bender_position=142.5,
                             vkb_motor_2_bender_position=299.5,
                             hkb_q_distance=120,
                             hkb_motor_4_translation=0.0,
                             hkb_motor_3_pitch_angle=0.003,
                             hkb_motor_3_delta_pitch_angle=0.0,
                             hkb_motor_1_bender_position=250.0515,
                             hkb_motor_2_bender_position=157.0341
                             )

class AbstractSimulatedFocusingOptics(AbstractFocusingOptics):
    def initialize(self, input_photon_beam, input_features=get_default_input_features(), **kwargs): raise NotImplementedError()

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None): raise NotImplementedError()
    def restore_input_photon_beam(self): raise NotImplementedError()

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline. FOR SIMULATION PURPOSES ONLY

    # V-KB -----------------------

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_q_distance(self): raise NotImplementedError()

    # H-KB -----------------------

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_hkb_q_distance(self): raise NotImplementedError()

