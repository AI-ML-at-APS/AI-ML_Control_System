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

from aps.ai.common.facade.parameters import Movement, DistanceUnits
from aps.ai.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics

def get_default_input_features(): # units: mm, mrad and micron for the bender
    return DictionaryWrapper(v_bimorph_mirror_q_distance=1500.0,
                             v_bimorph_mirror_motor_translation=0.0,
                             v_bimorph_mirror_motor_pitch_angle=0.003,
                             v_bimorph_mirror_motor_pitch_delta_angle=0.0,
                             v_bimorph_mirror_motor_bender_voltage=500,
                             h_bendable_mirror_q_distance=2250.0, #2630.0,
                             h_bendable_mirror_motor_translation=0.0,
                             h_bendable_mirror_motor_pitch_angle=0.003,
                             h_bendable_mirror_motor_pitch_delta_angle=0.0,
                             h_bendable_mirror_motor_1_bender_position=100,
                             h_bendable_mirror_motor_2_bender_position=100
                             )

class AbstractSimulatedFocusingOptics(AbstractFocusingOptics):

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline. FOR SIMULATION PURPOSES ONLY

    # V-KB -----------------------

    def change_h_bendable_mirror_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_h_bendable_mirror_q_distance(self): raise NotImplementedError()

    # H-KB -----------------------

    def change_v_bimorph_mirror_shape(self, q_distance, movement=Movement.ABSOLUTE): raise NotImplementedError()
    def get_v_bimorph_mirror_q_distance(self): raise NotImplementedError()

