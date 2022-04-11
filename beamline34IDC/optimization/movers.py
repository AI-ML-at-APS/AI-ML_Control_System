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

import numpy as np
from beamline34IDC.simulation.facade.focusing_optics_interface import Movement



# All distance movements are in millimeters: motors 3 and the 'q' parameter.
def get_movement(movement):
    movement_types = {'relative': Movement.RELATIVE,
                      'absolute': Movement.ABSOLUTE}
    if movement in movement_types:
        return movement_types[movement]
    if movement in movement_types.values():
        return movement
    raise ValueError

def get_motor_move_fn(focusing_system, motor):
    motor_move_fns = {'hkb_4': focusing_system.move_hkb_motor_4_translation,
                      'hkb_3': focusing_system.move_hkb_motor_3_pitch,
                      'hkb_q': focusing_system.change_hkb_shape,
                      'vkb_4': focusing_system.move_vkb_motor_4_translation,
                      'vkb_3': focusing_system.move_vkb_motor_3_pitch,
                      'vkb_q': focusing_system.change_vkb_shape,
                      'hkb_1_2': focusing_system.move_hkb_motor_1_2_bender,
                      'vkb_1_2': focusing_system.move_vkb_motor_1_2_bender}
    if motor in motor_move_fns:
        return motor_move_fns[motor]
    if motor in motor_move_fns.values():
        return motor
    raise ValueError

def move_motors(focusing_system, motors, translations, movement='relative'):
    movement = get_movement(movement)
    if np.ndim(motors) == 0:
        motors = [motors]
    if np.ndim(translations) == 0:
        translations = [translations]
    for motor, trans in zip(motors, translations):
        motor_move_fn = get_motor_move_fn(focusing_system, motor)
        if motor in ['hkb_1_2', 'vkb_1_2']:
            if np.ndim(trans) == 0 and movement == Movement.RELATIVE:
                motor_move_fn(trans, trans, movement=movement)
            elif np.ndim(trans) == 1:
                motor_move_fn(trans[0], trans[1], movement=movement)
            else:
                raise ValueError("For absolute movement for motors 1 and 2, " +
                                 "translation for both the bender motors should be supplied together")
        else:
            motor_move_fn(trans, movement=movement)
    return focusing_system


def get_motor_absolute_position_fn(focusing_system, motor):
    motor_get_pos_fns = {'hkb_4': focusing_system.get_hkb_motor_4_translation,
                         'hkb_3': focusing_system.get_hkb_motor_3_pitch,
                         'hkb_q': focusing_system.get_hkb_q_distance,
                         'vkb_4': focusing_system.get_vkb_motor_4_translation,
                         'vkb_3': focusing_system.get_vkb_motor_3_pitch,
                         'vkb_q': focusing_system.get_vkb_q_distance,
                         'hkb_1_2': focusing_system.get_hkb_motor_1_2_bender,
                         'vkb_1_2': focusing_system.get_vkb_motor_1_2_bender}
    if motor in motor_get_pos_fns:
        return motor_get_pos_fns[motor]
    if motor in motor_get_pos_fns.values():
        return motor
    raise ValueError


def get_absolute_positions(focusing_system, motors):

    if np.ndim(motors) == 0:
        motors = [motors]

    positions = []
    for motor in motors:
        position = get_motor_absolute_position_fn(focusing_system, motor)()
        positions.append(position)
    return positions