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
import os

import numpy
from epics import PV

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance
from aps.common.measurment.beamline.image_processor import ImageProcessor as ImageProcessorCommon
from aps.common.measurment.beamline.image_collector import ImageCollector

from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, Movement, AngularUnits
from aps.ai.autoalignment.common.hardware.epics.optics import AbstractEpicsOptics
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics, DISTANCE_V_MOTORS

APPLICATION_NAME = "IMAGE-PROCESSOR"

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="image_processor.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

ENERGY                = ini_file.get_float_from_ini(section="Execution", key="Energy",                default=20000.0)
SOURCE_DISTANCE_V     = ini_file.get_float_from_ini(section="Execution", key="Source-Distance-V",     default=1.5)
SOURCE_DISTANCE_H     = ini_file.get_float_from_ini(section="Execution", key="Source-Distance-H",     default=1.5)
IMAGE_TRANSFER_MATRIX = ini_file.get_list_from_ini( section="Execution", key="Image-Transfer-Matrix", default=[0, 1, 0], type=int)

ini_file.set_value_at_ini(section="Execution",   key="Energy",                value=ENERGY)
ini_file.set_value_at_ini(section="Execution",   key="Source-Distance-V",     value=SOURCE_DISTANCE_V)
ini_file.set_value_at_ini(section="Execution",   key="Source-Distance-H",     value=SOURCE_DISTANCE_H)
ini_file.set_list_at_ini( section="Execution",   key="Image-Transfer-Matrix", values_list=IMAGE_TRANSFER_MATRIX)

ini_file.push()

def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(kwargs)


# translation h: 28idb:m23
# pitch h:       28idb:m24
# bender 2 pv names

# V: pitch 2 + 1 motors
#    translation 3 motors together


class Motors:
    # Horizontal mirror:
    TRANSLATION_H = PV(pvname='28idb:m23')
    PITCH_H       = PV(pvname='28idb:m24')
    BENDER_H_1    = PV(pvname='28idb:xxx')
    BENDER_H_2    = PV(pvname='28idb:xxx')

    TRANSLATION_VO = PV(pvname='1bmopt:m13')
    TRANSLATION_DI = PV(pvname='1bmopt:m12')
    TRANSLATION_DO = PV(pvname='1bmopt:m14')
    LATERAL_V      = PV(pvname='1bmopt:m15')
    BENDER_V       = PV(pvname='simJTEC:E4')

class ImageProcessor(ImageProcessorCommon):
    def __init__(self, data_collection_directory):
        super(ImageProcessor, self).__init__(data_collection_directory=data_collection_directory,
                         energy=ENERGY,
                         source_distance=[SOURCE_DISTANCE_H, SOURCE_DISTANCE_V],
                         image_transfer_matrix=IMAGE_TRANSFER_MATRIX)


    def generate_simulated_mask(self, image_index_for_mask=1, verbose=False):
        image_transfer_matrix = super(ImageProcessor, self).generate_simulated_mask(image_index_for_mask, verbose)

        ini_file = get_registered_ini_instance(APPLICATION_NAME)
        ini_file.set_list_at_ini(section="Execution", key="Image-Transfer-Matrix", values_list=image_transfer_matrix)
        ini_file.push()

class __EpicsFocusingOptics(AbstractEpicsOptics, AbstractFocusingOptics):

    def __init__(self, **kwargs):
        super().__init__(translational_units=DistanceUnits.MILLIMETERS, angular_units=AngularUnits.DEGREES)
        
        try: measurement_directory = kwargs["measurement_directory"]
        except: measurement_directory = os.curdir

        self.__image_collector = ImageCollector(measurement_directory=measurement_directory)
        self.__image_processor = ImageProcessor(data_collection_directory=measurement_directory)

    def get_photon_beam(self, **kwargs): pass
    
    def initialize(self, **kwargs): pass
    
    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:   Motors.BENDER_V.put(actuator_value)
        elif movement == Movement.RELATIVE: Motors.BENDER_V.put(Motors.BENDER_V.get() + actuator_value)
        else: raise ValueError("Movement not recognized")

        
    def get_v_bimorph_mirror_motor_bender(self): 
        return Motors.BENDER_V.get()
    
    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREE):
        if units == AngularUnits.MILLIRADIANS: angle *= 1e-3
        elif units == AngularUnits.RADIANS: pass
        elif units == AngularUnits.DEGREES: angle = numpy.radians(angle)

        pos = 0.5*Motors.DISTANCE_V_MOTORS*numpy.sin(angle)

        if movement==Movement.ABSOLUTE:
            zero_pos = 0.5 * (self._get_translational_motor_position(Motors.TRANSLATION_VO, units=units) +
                              self._get_translational_motor_position(Motors.TRANSLATION_DO, units=units))

            self._move_translational_motor(Motors.TRANSLATION_VO, zero_pos+pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO, zero_pos-pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI, zero_pos-pos, movement=movement, units=DistanceUnits.MILLIMETERS)
        elif movement==Movement.RELATIVE:
            self._move_translational_motor(Motors.TRANSLATION_VO,  pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO, -pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI, -pos, movement=movement, units=DistanceUnits.MILLIMETERS)

    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.DEGREE):
        pos = (self._get_translational_motor_position(Motors.TRANSLATION_VO, units=DistanceUnits.MILLIMETERS) -
               self._get_translational_motor_position(Motors.TRANSLATION_DO, units=DistanceUnits.MILLIMETERS))
        angle = numpy.arcsin(pos/Motors.DISTANCE_V_MOTORS)

        if units == AngularUnits.MILLIRADIANS: angle *= 1e3
        elif units == AngularUnits.RADIANS: pass
        elif units == AngularUnits.DEGREES: angle = numpy.degrees(angle)

        return angle

    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS):
        self._move_translational_motor(Motors.TRANSLATION_VO, translation, movement=movement, units=units)
        self._move_translational_motor(Motors.TRANSLATION_DO, translation, movement=movement, units=units)
        self._move_translational_motor(Motors.TRANSLATION_DI, translation, movement=movement, units=units)        
        
    def get_v_bimorph_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS):
        return 0.5*(self._get_translational_motor_position(Motors.TRANSLATION_VO, units=units) +
                    self._get_translational_motor_position(Motors.TRANSLATION_DO, units=units))

    # H-KB -----------------------

    def move_h_bendable_mirror_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:   Motors.BENDER_H_1.put(pos_upstream)
        elif movement == Movement.RELATIVE: Motors.BENDER_H_1.put(Motors.BENDER_H_1.get() + pos_upstream)
        else: raise ValueError("Movement not recognized")

    def get_h_bendable_mirror_motor_1_bender(self): 
        return Motors.BENDER_H_1.get()

    def move_h_bendable_mirror_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:   Motors.BENDER_H_2.put(pos_downstream)
        elif movement == Movement.RELATIVE: Motors.BENDER_H_2.put(Motors.BENDER_H_2.get() + pos_downstream)
        else: raise ValueError("Movement not recognized")
    
    def get_h_bendable_mirror_motor_2_bender(self): 
        return Motors.BENDER_H_2.get()

    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREE): 
        self._move_rotational_motor(Motors.PITCH_H, angle, movement, units)

    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.DEGREE):
        return self._get_rotational_motor_angle(Motors.PITCH_H, units)

    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS):
        self._move_rotational_motor(Motors.TRANSLATION_H, translation, movement, units)

    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS):
        return self._get_translational_motor_position(Motors.TRANSLATION_H, units)
