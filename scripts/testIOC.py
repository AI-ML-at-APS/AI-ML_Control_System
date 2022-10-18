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

from ai.beamline34IDC.facade import focusing_optics_factory, ExecutionMode
from ai.beamline34IDC.facade.focusing_optics_interface import AngularUnits, DistanceUnits, Movement
from ai.beamline34IDC.hardware.facade import Implementors, Beamline

focusing_optics = focusing_optics_factory.focusing_optics_factory_method(execution_mode=ExecutionMode.HARDWARE, implementor=Implementors.EPICS, beamline=Beamline.VIRTUAL)
focusing_optics.initialize()

focusing_optics.move_hkb_motor_4_translation(300, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

print("COH-SLITS", focusing_optics.get_coherence_slits_parameters())

print("VKB, bender", focusing_optics.get_vkb_motor_1_2_bender(units=DistanceUnits.MICRON))
print("VKB, pitch", focusing_optics.get_vkb_motor_3_pitch(units=AngularUnits.MILLIRADIANS))
print("VKB, translation", focusing_optics.get_vkb_motor_4_translation(units=DistanceUnits.MICRON))

print("HKB, bender", focusing_optics.get_hkb_motor_1_2_bender(units=DistanceUnits.MICRON))
print("HKB, pitch", focusing_optics.get_hkb_motor_3_pitch(units=AngularUnits.MILLIRADIANS))
print("HKB, translation", focusing_optics.get_hkb_motor_4_translation(units=DistanceUnits.MICRON))

focusing_optics.move_hkb_motor_4_translation(300, movement=Movement.RELATIVE, units=DistanceUnits.MICRON)

print("HKB, translation", focusing_optics.get_hkb_motor_4_translation(units=DistanceUnits.MICRON))
