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

from oasys.util.oasys_util import get_sigma, get_fwhm, get_average
from orangecontrib.ml.util.data_structures import DictionaryWrapper
from beamline34IDC.util.gaussian_fit import calculate_1D_gaussian_fit
from beamline34IDC.facade.focusing_optics_interface import AngularUnits, DistanceUnits, Movement
from beamline34IDC.hardware.facade.focusing_optics_interface import AbstractHardwareFocusingOptics, Directions

def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(**kwargs)

class Motors:
    # TBD: COHERENCE SLITS

    VKB_MOTOR_1 = '34idc:m58:c1:m3.VAL'
    VKB_MOTOR_2 = '34idc:m58:c1:m4.VAL'
    VKB_MOTOR_3 = ""
    VKB_MOTOR_4 = ""

    HKB_MOTOR_1 = '34idc:m58:c1:m7.VAL'
    HKB_MOTOR_2 = '34idc:m58:c1:m8.VAL'
    HKB_MOTOR_3 = ""
    HKB_MOTOR_4 = ""

    SAMPLE_STAGE_X   = '34idc:lab:m1.VAL'
    SAMPLE_STAGE_Z   = '34idc:lab:m3.VAL'
    SAMPLE_STAGE_Z_2 = '34idc:mxv:c0:m1.VAL' # for scanning purposes, it appeared in M.C. code: to be verified

class __EpicsFocusingOptics(AbstractHardwareFocusingOptics):

    def __init__(self, **kwargs):
        pass

    def initialize(self, **kwargs): pass

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None): pass
    def get_coherence_slits_parameters(self): pass # center x, center z, aperture x, aperture z

    # V-KB -----------------------

    def move_vkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(Motors.VKB_MOTOR_1, Motors.VKB_MOTOR_2, pos_upstream, pos_downstream, movement, units)

    def get_vkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(Motors.VKB_MOTOR_1, Motors.VKB_MOTOR_2, units)

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): pass
    def get_vkb_motor_4_translation(self): pass

    # H-KB -----------------------

    def move_hkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(Motors.HKB_MOTOR_1, Motors.HKB_MOTOR_2, pos_upstream, pos_downstream, movement, units)

    def get_hkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(Motors.HKB_MOTOR_1, Motors.HKB_MOTOR_2, units)

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE): pass
    def get_hkb_motor_4_translation(self): pass

    # PRIVATE METHODS

    @classmethod
    def __move_motor_1_2_bender(cls, motor_1, motor_2, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MILLIMETERS:
            pos_upstream *= 1e3
            pos_downstream *= 1e3

        if movement == Movement.RELATIVE:
            if pos_upstream   != 0.0: caput(motor_1, caget(motor_1) + pos_upstream)
            if pos_downstream != 0.0: caput(motor_2, caget(motor_2) + pos_downstream)
        elif movement == Movement.ABSOLUTE:
            if pos_upstream   != caget(motor_1): caput(motor_1, pos_upstream)
            if pos_downstream != caget(motor_2): caput(motor_2, pos_downstream)

    @classmethod
    def __get_motor_1_2_bender(cls, motor_1, motor_2, units=DistanceUnits.MICRON):
        factor = 1e-3 if units == DistanceUnits.MILLIMETERS else 1.0

        return factor * caget(motor_1), factor * caget(motor_2)

    # get radiation characteristics ------------------------------
    def get_beam_scan(self, direction=Directions.HORIZONTAL, parameters=[-2, 2, 40]):
        if direction==Directions.HORIZONTAL: data = self.__scan(Motors.SAMPLE_STAGE_X, parameters[0], parameters[1], parameters[2])
        elif direction==Directions.VERTICAL: data = self.__scan(Motors.SAMPLE_STAGE_Z, parameters[0], parameters[1], parameters[2])

        fwhm, _, _  = get_fwhm(data[1], data[0])
        sigma       = get_sigma(data[1], data[0])
        centroid    = get_average(data[1], data[0])

        peak_intensity     = numpy.average(data[1][numpy.where(data[1] >= numpy.max(data[1]) * 0.90)])
        integral_intensity = numpy.sum(data[1])

        gaussian_fit = calculate_1D_gaussian_fit(data_1D=data[1], x=data[0])

        return data, \
               DictionaryWrapper(sigma=sigma,
                                 fwhm=fwhm,
                                 centroid=centroid,
                                 integral_intensity=integral_intensity,
                                 peak_intensity=peak_intensity,
                                 gaussian_fit=gaussian_fit)

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

        # put the motor on the peak!
        caput(motor_name, data[numpy.argmax(data[:, 1]), 0])

        return data
