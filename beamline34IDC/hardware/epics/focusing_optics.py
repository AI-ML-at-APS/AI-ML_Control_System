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
from beamline34IDC.hardware.facade import Beamline
from beamline34IDC.hardware.facade.focusing_optics_interface import AbstractHardwareFocusingOptics, Directions

def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(**kwargs)

class Motors:
    COH_SLITS_H_CENTER   = {Beamline.REAL : '34idc:m58:c2:m5.VAL', Beamline.VIRTUAL : '34idcsim:m58:c2:m5.VAL'}
    COH_SLITS_H_APERTURE = {Beamline.REAL : '34idc:m58:c2:m6.VAL', Beamline.VIRTUAL : '34idcsim:m58:c2:m6.VAL'}
    COH_SLITS_V_CENTER   = {Beamline.REAL : '34idc:m58:c2:m7.VAL', Beamline.VIRTUAL : '34idcsim:m58:c2:m7.VAL'}
    COH_SLITS_V_APERTURE = {Beamline.REAL : '34idc:m58:c2:m8.VAL', Beamline.VIRTUAL : '34idcsim:m58:c2:m8.VAL'}

    VKB_MOTOR_1 = {Beamline.REAL : '34idc:m58:c1:m3.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m3.VAL'} # upstream force micron
    VKB_MOTOR_2 = {Beamline.REAL : '34idc:m58:c1:m4.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m4.VAL'} # downstream force micron
    VKB_MOTOR_3 = {Beamline.REAL : '34idc:m58:c1:m2.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m2.VAL'} # pitch mrad
    VKB_MOTOR_4 = {Beamline.REAL : '34idc:m58:c1:m1.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m1.VAL'} # translation micron

    HKB_MOTOR_1 = {Beamline.REAL : '34idc:m58:c1:m7.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m7.VAL'}
    HKB_MOTOR_2 = {Beamline.REAL : '34idc:m58:c1:m8.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m8.VAL'}
    HKB_MOTOR_3 = {Beamline.REAL : '34idc:m58:c1:m6.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m6.VAL'}
    HKB_MOTOR_4 = {Beamline.REAL : '34idc:m58:c1:m5.VAL', Beamline.VIRTUAL : '34idcsim:m58:c1:m5.VAL'}

    SAMPLE_STAGE_X        = {Beamline.REAL : '34idc:lab:m1.VAL'   , Beamline.VIRTUAL : '34idcsim:lab:m1.VAL'   }
    SAMPLE_STAGE_Y        = {Beamline.REAL : '34idc:lab:m2.VAL'   , Beamline.VIRTUAL : '34idcsim:lab:m2.VAL'   }
    SAMPLE_STAGE_Z        = {Beamline.REAL : '34idc:lab:m3.VAL'   , Beamline.VIRTUAL : '34idcsim:lab:m3.VAL'   }    # fine Z motion
    SAMPLE_STAGE_Z_COARSE = {Beamline.REAL : '34idc:mxv:c0:m1.VAL', Beamline.VIRTUAL : '34idcsim:mxv:c0:m1.VAL'} # coarse Z motion

class __EpicsFocusingOptics(AbstractHardwareFocusingOptics):
    
    def __init__(self, **kwargs):
        try:    beamline = kwargs["beamline"]
        except: beamline = Beamline.REAL
        
        self.__beamline = beamline

    def initialize(self, **kwargs): pass

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system
    # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None):
        if not coh_slits_h_center is None:   caput(Motors.COH_SLITS_H_CENTER[self.__beamline], coh_slits_h_center)
        if not coh_slits_v_center is None:   caput(Motors.COH_SLITS_V_CENTER[self.__beamline], coh_slits_v_center)
        if not coh_slits_h_aperture is None: caput(Motors.COH_SLITS_H_APERTURE[self.__beamline], coh_slits_h_aperture)
        if not coh_slits_v_aperture is None: caput(Motors.COH_SLITS_V_APERTURE[self.__beamline], coh_slits_v_aperture)

    def get_coherence_slits_parameters(self): 
        return caget(Motors.COH_SLITS_H_CENTER[self.__beamline]), \
               caget(Motors.COH_SLITS_V_CENTER[self.__beamline]), \
               caget(Motors.COH_SLITS_H_APERTURE[self.__beamline]), \
               caget(Motors.COH_SLITS_V_APERTURE[self.__beamline])

    # V-KB -----------------------

    def move_vkb_motor_1_2_bender(self, pos_upstream=None, pos_downstream=None, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(Motors.VKB_MOTOR_1[self.__beamline], Motors.VKB_MOTOR_2[self.__beamline], pos_upstream, pos_downstream, movement, units)

    def get_vkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(Motors.VKB_MOTOR_1[self.__beamline], Motors.VKB_MOTOR_2[self.__beamline], units)

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): 
        self.__move_motor_3_pitch(Motors.VKB_MOTOR_3[self.__beamline], angle, movement, units)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): 
        return self.__get_motor_3_pitch(Motors.VKB_MOTOR_3[self.__beamline], units)
    
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): 
        self.__move_motor_4_transation(Motors.VKB_MOTOR_4[self.__beamline], translation, movement, units)
        
    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):  
        return self.__get_motor_4_translation(Motors.VKB_MOTOR_4[self.__beamline], units)

    # H-KB -----------------------

    def move_hkb_motor_1_2_bender(self, pos_upstream=None, pos_downstream=None, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(Motors.HKB_MOTOR_1[self.__beamline], Motors.HKB_MOTOR_2[self.__beamline], pos_upstream, pos_downstream, movement, units)

    def get_hkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(Motors.HKB_MOTOR_1[self.__beamline], Motors.HKB_MOTOR_2, units)

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self.__move_motor_3_pitch(Motors.HKB_MOTOR_3[self.__beamline], angle, movement, units)
        
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): 
        return self.__get_motor_3_pitch(Motors.HKB_MOTOR_3[self.__beamline], units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): 
        self.__move_motor_4_transation(Motors.HKB_MOTOR_4[self.__beamline], translation, movement, units)
    
    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON): 
        return self.__get_motor_4_translation(Motors.HKB_MOTOR_4[self.__beamline], units)

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

    @classmethod
    def __move_motor_3_pitch(cls, motor, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        if units == AngularUnits.MILLIRADIANS: pass
        elif units == AngularUnits.DEGREES:    angle = 1e3 * numpy.radians(angle)
        elif units == AngularUnits.RADIANS:    angle = 1e3 * angle
        else: raise ValueError("Angular units not recognized")

        if movement == Movement.ABSOLUTE:   caput(motor, angle)
        elif movement == Movement.RELATIVE: caput(motor, caget(motor) + angle)
        else:  raise ValueError("Movement not recognized")
    
    @classmethod
    def __move_motor_4_transation(cls, motor, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        if units == DistanceUnits.MILLIMETERS: translation *= 1e3
        
        if movement == Movement.ABSOLUTE:   caput(motor, translation)
        elif movement == Movement.RELATIVE: caput(motor, caget(motor) + translation)
        else: raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_3_pitch(cls, motor, units=AngularUnits.MILLIRADIANS):
        angle = caget(motor)

        if units == AngularUnits.MILLIRADIANS:  return angle
        elif units == AngularUnits.DEGREES:     return numpy.degrees(angle*1e-3)
        elif units == AngularUnits.RADIANS:     return angle*1e-3
        else: raise ValueError("Angular units not recognized")

    @classmethod
    def __get_motor_4_translation(cls, motor, units=DistanceUnits.MICRON):
        translation = caget(motor)

        if units == DistanceUnits.MICRON:        return translation
        elif units == DistanceUnits.MILLIMETERS: return translation*1e-3
        else: raise ValueError("Distance units not recognized")
        
    # get radiation characteristics ------------------------------
    
    def get_photon_beam(self, **kwargs): 
        try:    direction = kwargs["direction"]
        except: direction = Directions.BOTH
        try:    parameters = kwargs["parameters"]
        except: parameters = [[-2, 2, 40],[-2, 2, 40]] if direction==Directions.BOTH else [-2, 2, 40]
        
        data_h = None
        data_v = None
        
        if direction == Directions.HORIZONTAL: data_h = self.__scan(Motors.SAMPLE_STAGE_X[self.__beamline], parameters[0], parameters[1], parameters[2])
        elif direction == Directions.VERTICAL: data_v = self.__scan(Motors.SAMPLE_STAGE_Z[self.__beamline], parameters[0], parameters[1], parameters[2])
        elif direction == Directions.BOTH:
            data_h = self.__scan(Motors.SAMPLE_STAGE_X[self.__beamline], parameters[0][0], parameters[0][1], parameters[0][2])
            data_v = self.__scan(Motors.SAMPLE_STAGE_Z[self.__beamline], parameters[1][0], parameters[1][1], parameters[1][2])
        
        return data_h, data_v
        
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
