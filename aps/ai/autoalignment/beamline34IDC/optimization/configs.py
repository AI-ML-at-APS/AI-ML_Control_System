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

from aps.ai.autoalignment.beamline34IDC.facade.focusing_optics_interface import MotorResolutionRegistry

motor_resolutions = MotorResolutionRegistry.getInstance().get_motor_resolution_set("34-ID-C")

# not sure about the movement ranges for hkb4 and vkb4
DEFAULT_MOVEMENT_RANGES = {#'hkb_4': [-0.200, 0.200], 
                           #'vkb_4': [-0.200, 0.200], 
                           'hkb_4': [-20, 20],
                           'vkb_4': [-20, 20],
                           'hkb_3': [-0.02, 0.02], # in mrad
                           'vkb_3': [-0.02, 0.02], # in mrad
                           'hkb_1': [-30, 30], 
                           'hkb_2': [-30, 30],  
                           'vkb_1': [-30, 30],  
                           'vkb_2': [-30, 30]  
                           }

# I am adding this because the focusing system interface does not currently contain resolution
# values for hkb_q, vkb_q, hkb_1_2, and vkb_1_2 motors.
DEFAULT_MOTOR_RESOLUTIONS = {'hkb_4': motor_resolutions.get_motor_resolution("hkb_motor_4_translation")[0],
                             'vkb_4': motor_resolutions.get_motor_resolution("vkb_motor_4_translation")[0],
                             'hkb_3': motor_resolutions.get_motor_resolution("hkb_motor_3_pitch")[0],
                             'vkb_3': motor_resolutions.get_motor_resolution("vkb_motor_3_pitch")[0],
                             'hkb_q': 0.1, # in mm
                             'vkb_q': 0.1, # in mm
                             'hkb_1': motor_resolutions.get_motor_resolution("hkb_motor_1_2_bender")[0],
                             'vkb_1': motor_resolutions.get_motor_resolution("vkb_motor_1_2_bender")[0],
                             'hkb_2': motor_resolutions.get_motor_resolution("hkb_motor_1_2_bender")[0],
                             'vkb_2': motor_resolutions.get_motor_resolution("vkb_motor_1_2_bender")[0],
                             }

DEFAULT_MOTOR_TOLERANCES = DEFAULT_MOTOR_RESOLUTIONS


# These values only apply for the simulation with 50k simulated beams
DEFAULT_LOSS_TOLERANCES = {'centroid': 2e-4,
                           'fwhm': 2e-4,
                           'peak_intensity': -np.inf,
                           'sigma': 2e-4}
