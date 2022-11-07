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

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import (
    AngularUnits,
    DistanceUnits,
    MotorResolutionRegistry,
)

motor_resolutions = MotorResolutionRegistry.getInstance().get_motor_resolution_set("28-ID-B")

motors = {}

DEFAULT_DISTANCE_UNIT = DistanceUnits.MICRON
DEFAULT_ANGLE_UNIT = AngularUnits.MILLIRADIANS
DEFAULT_ACTUATOR_UNIT = DistanceUnits.OTHER  # This should be volts

# not sure about the movement ranges for hkb4 and vkb4
DEFAULT_MOVEMENT_RANGES = {
    "hb_1": [-50, 50],
    "hb_2": [-50, 50],
    "hb_pitch": [-0.02, 0.02],  # in mrad
    "hb_trans": [-20, 20],  # in mrad
    "vb_bender": [-30, 30],
    "vb_pitch": [-0.02, 0.02],
    "vb_trans": [-30, 30],
}
# These are shorthands for the longer names in the focusing system interface.
# The units for the bender motors are Volts.
DEFAULT_MOTOR_RESOLUTIONS = {
    "hb_1": motor_resolutions.get_motor_resolution("h_bendable_mirror_motor_bender", DEFAULT_ACTUATOR_UNIT)[0],
    "hb_2": motor_resolutions.get_motor_resolution("h_bendable_mirror_motor_bender", DEFAULT_ACTUATOR_UNIT)[0],
    "hb_pitch": motor_resolutions.get_motor_resolution("h_bendable_mirror_motor_pitch", DEFAULT_ANGLE_UNIT)[0],
    "hb_trans": motor_resolutions.get_motor_resolution("h_bendable_mirror_motor_translation", DEFAULT_DISTANCE_UNIT)[0],
    "vb_bender": motor_resolutions.get_motor_resolution("v_bimorph_mirror_motor_bender", DEFAULT_ACTUATOR_UNIT),
    "vb_pitch": motor_resolutions.get_motor_resolution("v_bimorph_mirror_motor_pitch", DEFAULT_ANGLE_UNIT)[0],
    "vb_trans": motor_resolutions.get_motor_resolution("v_bimorph_mirror_motor_translation", DEFAULT_DISTANCE_UNIT)[0],
}

DEFAULT_MOTOR_TOLERANCES = DEFAULT_MOTOR_RESOLUTIONS

# These values only apply for the simulation with 50k simulated beams
DEFAULT_LOSS_TOLERANCES = {"centroid": 2e-4, "fwhm": 2e-4, "peak_intensity": -np.inf, "sigma": 2e-4}

DEFAULT_CONSTRAINT_OPTIONS = {"centroid", "fwhm", "sigma", "peak_intensity", "sum_intensity"}
