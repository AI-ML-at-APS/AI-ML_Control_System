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

import numpy, time

from epics import caget, caput

from beamline34IDC.facade.focusing_optics_interface import AngularUnits, DistanceUnits, Movement
from beamline34IDC.hardware.facade.focusing_optics_interface import AbstractHardwareFocusingOptics, Directions

def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(**kwargs)

class __EpicsFocusingOptics(AbstractHardwareFocusingOptics):
    def __init__(self, **kwargs):
        pass

    def initialize(self, **kwargs):
        pass

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None): pass
    def get_coherence_slits_parameters(self): pass # center x, center z, aperture x, aperture z

    # V-KB -----------------------

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): pass
    def get_vkb_motor_4_translation(self): pass
    def move_vkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_vkb_motor_1_2_bender(self, units=DistanceUnits.MICRON): pass

    # H-KB -----------------------

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): pass
    def get_hkb_motor_4_translation(self): pass
    def move_hkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_hkb_motor_1_2_bender(self, units=DistanceUnits.MICRON): pass

    # get radiation characteristics ------------------------------
    def get_beam_scan(self, direction=Directions.HORIZONTAL): pass

    @classmethod
    def __scan(cls, motor_name, first, final, steps):
        current = caget(motor_name)
        stepsize = (final - first) / float(steps)
        first = current + first

        data = numpy.zeros((steps, 2), float)

        caput('34idc:FastShutterState', 1)
        caput('34idcTIM2:cam1:AcquireTime', 0.3)

        for i in range(steps):
            caput(motor_name, first + i * stepsize)
            caput('34idcTIM2:cam1:Acquire', 1)

            time.sleep(0.2)

            while (caget('34idcTIM2:cam1:Acquire') != 0): time.sleep(0.1)
    
            data[i, 0] = i * stepsize + first
            data[i, 1] = caget('34idcTIM2:Stats5:Total_RBV')

        id = numpy.argmax(data[:, 1])

        caput(motor_name, data[id, 0])

        return data

