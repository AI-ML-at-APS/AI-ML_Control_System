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

import os, numpy, time

from epics import caget, caput

from aps_ai.common.util.initializer import AlreadyInitializedError, register_ini_instance, IniMode
from aps_ai.common.facade.parameters import AngularUnits, DistanceUnits, Movement
from aps_ai.common.hardware.facade.parameters import Beamline, Directions

from aps_ai.common.hardware.epics.optics import AbstractEpicsOptics
from aps_ai.beamline34IDC.hardware.facade.focusing_optics_interface import AbstractHardwareFocusingOptics

def epics_focusing_optics_factory_method(**kwargs):
    try: register_ini_instance(ini_mode=IniMode.LOCAL_FILE, application_name="motors configuration", ini_file_name="motors_configuration.ini")
    except AlreadyInitializedError: pass

    return __EpicsFocusingOptics(**kwargs)

class Scan:
    SHUTTER  = {Beamline.REAL : '34idc:FastShutterState',     Beamline.VIRTUAL : '34idSim:FastShutterState'}
    DETECTOR = {Beamline.REAL : '34idcTIM2:cam1',             Beamline.VIRTUAL : '34idSimTIM2:cam1'}
    COUNTS   = {Beamline.REAL : '34idcTIM2:Stats5:Total_RBV', Beamline.VIRTUAL : '34idSimTIM2:Stats5:Total_RBV'}


class Motors:
    COH_SLITS_H_CENTER   = {Beamline.REAL : '34idc:m58:c2:m5', Beamline.VIRTUAL : '34idSim:m58:c2:m5'}
    COH_SLITS_H_APERTURE = {Beamline.REAL : '34idc:m58:c2:m6', Beamline.VIRTUAL : '34idSim:m58:c2:m6'}
    COH_SLITS_V_CENTER   = {Beamline.REAL : '34idc:m58:c2:m7', Beamline.VIRTUAL : '34idSim:m58:c2:m7'}
    COH_SLITS_V_APERTURE = {Beamline.REAL : '34idc:m58:c2:m8', Beamline.VIRTUAL : '34idSim:m58:c2:m8'}

    VKB_MOTOR_1 = {Beamline.REAL : '34idc:m58:c1:m3', Beamline.VIRTUAL : '34idSim:m58:c1:m3'} # upstream force micron
    VKB_MOTOR_2 = {Beamline.REAL : '34idc:m58:c1:m4', Beamline.VIRTUAL : '34idSim:m58:c1:m4'} # downstream force micron
    VKB_MOTOR_3 = {Beamline.REAL : '34idc:m58:c1:m2', Beamline.VIRTUAL : '34idSim:m58:c1:m2'} # pitch mrad
    VKB_MOTOR_4 = {Beamline.REAL : '34idc:m58:c1:m1', Beamline.VIRTUAL : '34idSim:m58:c1:m1'} # translation micron

    HKB_MOTOR_1 = {Beamline.REAL : '34idc:m58:c1:m7', Beamline.VIRTUAL : '34idSim:m58:c1:m7'}
    HKB_MOTOR_2 = {Beamline.REAL : '34idc:m58:c1:m8', Beamline.VIRTUAL : '34idSim:m58:c1:m8'}
    HKB_MOTOR_3 = {Beamline.REAL : '34idc:m58:c1:m6', Beamline.VIRTUAL : '34idSim:m58:c1:m6'}
    HKB_MOTOR_4 = {Beamline.REAL : '34idc:m58:c1:m5', Beamline.VIRTUAL : '34idSim:m58:c1:m5'}

    SAMPLE_STAGE_X        = {Beamline.REAL : '34idc:lab:m1'   , Beamline.VIRTUAL : '34idSim:lab:m1'   }
    SAMPLE_STAGE_Y        = {Beamline.REAL : '34idc:lab:m2'   , Beamline.VIRTUAL : '34idSim:lab:m2'   }
    SAMPLE_STAGE_Z        = {Beamline.REAL : '34idc:lab:m3'   , Beamline.VIRTUAL : '34idSim:lab:m3'   } # fine Z motion
    SAMPLE_STAGE_Z_COARSE = {Beamline.REAL : '34idc:mxv:c0:m1', Beamline.VIRTUAL : '34idSim:mxv:c0:m1'} # coarse Z motion

class __EpicsFocusingOptics(AbstractEpicsOptics, AbstractHardwareFocusingOptics):
    
    def __init__(self, **kwargs):
        try:    beamline = kwargs["beamline"]
        except: beamline = Beamline.REAL
        
        self.__beamline = beamline

    def initialize(self, **kwargs):
        os.environ["PATH"] = os.environ["PATH"] + ":" + "/Users/lrebuffi/Documents/Workspace/External_Codes/EPICS/epics-base/bin/darwin-x86/"

        if self.__beamline   == Beamline.VIRTUAL: os.environ["EPICS_CA_ADDR_LIST"] = "164.54.138.190"
        elif self.__beamline == Beamline.REAL:    os.environ["EPICS_CA_ADDR_LIST"] = "boh"

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MICRON:        factor = 1.0
        elif units == DistanceUnits.MILLIMETERS: factor = 1e3
        else: raise ValueError("Distance units not recognized")

        if not coh_slits_h_center is None:   caput(Motors.COH_SLITS_H_CENTER[self.__beamline] + ".VAL",   factor*coh_slits_h_center)
        if not coh_slits_v_center is None:   caput(Motors.COH_SLITS_V_CENTER[self.__beamline] + ".VAL",   factor*coh_slits_v_center)
        if not coh_slits_h_aperture is None: caput(Motors.COH_SLITS_H_APERTURE[self.__beamline] + ".VAL", factor*coh_slits_h_aperture)
        if not coh_slits_v_aperture is None: caput(Motors.COH_SLITS_V_APERTURE[self.__beamline] + ".VAL", factor*coh_slits_v_aperture)

    def get_coherence_slits_parameters(self, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MICRON:        factor = 1.0
        elif units == DistanceUnits.MILLIMETERS: factor = 1e-3
        else: raise ValueError("Distance units not recognized")

        return factor*caget(Motors.COH_SLITS_H_CENTER[self.__beamline] + ".VAL"), \
               factor*caget(Motors.COH_SLITS_V_CENTER[self.__beamline] + ".VAL"), \
               factor*caget(Motors.COH_SLITS_H_APERTURE[self.__beamline] + ".VAL"), \
               factor*caget(Motors.COH_SLITS_V_APERTURE[self.__beamline] + ".VAL")

    # V-KB -----------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translational_motor(Motors.VKB_MOTOR_1[self.__beamline], pos_upstream, movement, units)

    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self._get_translational_motor_position(Motors.VKB_MOTOR_1[self.__beamline], units)

    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translational_motor(Motors.VKB_MOTOR_2[self.__beamline], pos_downstream, movement, units)

    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self._get_translational_motor_position(Motors.VKB_MOTOR_2[self.__beamline], units)

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): 
        self._move_rotational_motor(Motors.VKB_MOTOR_3[self.__beamline], angle, movement, units)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): 
        return self._get_rotational_motor_angle(Motors.VKB_MOTOR_3[self.__beamline], units)
    
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): 
        self._move_translational_motor(Motors.VKB_MOTOR_4[self.__beamline], translation, movement, units)
        
    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):  
        return self._get_translational_motor_position(Motors.VKB_MOTOR_4[self.__beamline], units)

    # H-KB -----------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_translational_motor(Motors.HKB_MOTOR_1[self.__beamline], pos_upstream, movement, units)

    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self._get_translational_motor_position(Motors.HKB_MOTOR_1[self.__beamline], units)

    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_translational_motor(Motors.HKB_MOTOR_2[self.__beamline], pos_downstream, movement, units)

    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON): 
        return self._get_translational_motor_position(Motors.HKB_MOTOR_2[self.__beamline], units)

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_rotational_motor(Motors.HKB_MOTOR_3[self.__beamline], angle, movement, units)
        
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): 
        return self.__get_rotational_motor_angle(Motors.HKB_MOTOR_3[self.__beamline], units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): 
        self._move_translational_motor(Motors.HKB_MOTOR_4[self.__beamline], translation, movement, units)
    
    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON): 
        return self._get_translational_motor_position(Motors.HKB_MOTOR_4[self.__beamline], units)

        
    # get radiation characteristics ------------------------------
    
    def get_photon_beam(self, **kwargs): 
        try:    direction = kwargs["direction"]
        except: direction = Directions.BOTH
        try:    parameters = kwargs["parameters"]
        except: parameters = [[-2, 2, 40], [-2, 2, 40]] if direction==Directions.BOTH else [-2, 2, 40]
        
        data_h = None
        data_v = None
        
        if direction == Directions.HORIZONTAL: data_h = self.__scan(Motors.SAMPLE_STAGE_X[self.__beamline], parameters[0], parameters[1], parameters[2])
        elif direction == Directions.VERTICAL: data_v = self.__scan(Motors.SAMPLE_STAGE_Z[self.__beamline], parameters[0], parameters[1], parameters[2])
        elif direction == Directions.BOTH:
            data_h = self.__scan(Motors.SAMPLE_STAGE_X[self.__beamline], parameters[0][0], parameters[0][1], parameters[0][2])
            data_v = self.__scan(Motors.SAMPLE_STAGE_Z[self.__beamline], parameters[1][0], parameters[1][1], parameters[1][2])
        
        return data_h, data_v

    def __scan(self, motor_name, first, final, steps):
        current = caget(motor_name + ".VAL")
        stepsize = (final - first) / float(steps)
        first = current + first

        data = numpy.zeros((steps, 2), float)

        DETECTOR = Scan.DETECTOR[self.__beamline]
        COUNTS   = Scan.COUNTS[self.__beamline]

        caput(Scan.SHUTTER[self.__beamline], 1)
        caput(DETECTOR + ':AcquireTime', 0.3)

        for i in range(steps):
            caput(motor_name + ".VAL", first + i * stepsize)
            caput(DETECTOR + ':Acquire', 1)

            time.sleep(0.2)

            while (caget(DETECTOR + ':Acquire') != 0): time.sleep(0.1)
    
            data[i, 0] = i * stepsize + first
            data[i, 1] = caget(COUNTS)

        # put the motor on the peak!
        caput(motor_name + ".VAL", data[numpy.argmax(data[:, 1]), 0])

        return data
