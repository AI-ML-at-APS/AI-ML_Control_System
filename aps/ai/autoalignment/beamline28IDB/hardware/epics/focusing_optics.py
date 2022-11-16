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
import time

import numpy
from epics import PV

from aps.common.measurment.beamline.image_processor import IMAGE_SIZE_PIXEL_HxV, PIXEL_SIZE
from aps.common.measurment.beamline.image_collector import ImageCollector

from aps.ai.autoalignment.common.measurement.image_processor import ImageProcessor
from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, Movement, AngularUnits
from aps.ai.autoalignment.common.hardware.epics.optics import AbstractEpicsOptics
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics, DISTANCE_V_MOTORS

def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(**kwargs)

class Motors:
    # Horizontal mirror:
    TRANSLATION_H    = PV(pvname='28idb:m23')
    PITCH_H          = PV(pvname='28idb:m24')
    BENDER_H_1       = PV(pvname='zoomkb:pid1')
    BENDER_H_2       = PV(pvname='zoomkb:pid2')
    BENDER_H_1_RB    = PV(pvname='zoomkb:pid1.CVAL')
    BENDER_H_2_RB    = PV(pvname='zoomkb:pid2.CVAL')
    BENDER_H_1_FB    = PV(pvname='zoomkb:pid1.FBON')
    BENDER_H_2_FB    = PV(pvname='zoomkb:pid2.FBON')
    BENDER_THRESHOLD = 0.05

    TRANSLATION_VO = PV(pvname='1bmopt:m13')
    TRANSLATION_DI = PV(pvname='1bmopt:m12')
    TRANSLATION_DO = PV(pvname='1bmopt:m14')
    LATERAL_V      = PV(pvname='1bmopt:m15')
    BENDER_V       = PV(pvname='simJTEC:E4')

class __EpicsFocusingOptics(AbstractEpicsOptics, AbstractFocusingOptics):

    def __init__(self, **kwargs):
        super().__init__(translational_units=DistanceUnits.MILLIMETERS, angular_units=AngularUnits.DEGREES)
        
        try:    measurement_directory = kwargs["measurement_directory"]
        except: measurement_directory = os.curdir

        try:    self.__physical_boundaries = kwargs["physical_boundaries"]
        except: self.__physical_boundaries = None

        #TODO: ADD CHECK OF PHYSICAL BOuNDARIES

        self.__image_collector = ImageCollector(measurement_directory=measurement_directory)
        self.__image_processor = ImageProcessor(data_collection_directory=measurement_directory)

    def get_photon_beam(self, **kwargs):
        try: from_raw_image    = kwargs["from_raw_image"]
        except: from_raw_image = False

        try: self.__image_collector.restore_status()
        except: pass

        try:
            self.__image_collector.collect_single_shot_image(index=1)

            raw_image, crop_region, cropped_image = self.__image_processor.get_image_data(image_index=1)

            output = {}

            if from_raw_image:
                output["h_coord"] = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[0]/2, IMAGE_SIZE_PIXEL_HxV[0]/2, IMAGE_SIZE_PIXEL_HxV[0])*PIXEL_SIZE*1e3
                output["v_coord"] = numpy.linspace(-IMAGE_SIZE_PIXEL_HxV[1]/2, IMAGE_SIZE_PIXEL_HxV[1]/2, IMAGE_SIZE_PIXEL_HxV[1])*PIXEL_SIZE*1e3
                output["image"]   = raw_image
            else:
                output["width"]      = (crop_region[1]-crop_region[0])*PIXEL_SIZE*1e3
                output["height"]     = (crop_region[3]-crop_region[2])*PIXEL_SIZE*1e3
                output["centroid_h"] = (crop_region[0] + 0.5*output["length_h"])*PIXEL_SIZE*1e3
                output["centroid_v"] = (crop_region[2] + 0.5*output["length_v"])*PIXEL_SIZE*1e3
                output["h_coord"]    = numpy.linspace(crop_region[0], crop_region[1], cropped_image.shape[0])*PIXEL_SIZE*1e3
                output["v_coord"]    = numpy.linspace(crop_region[2], crop_region[3], cropped_image.shape[1])*PIXEL_SIZE*1e3
                output["image"]      = cropped_image

            try: self.__image_collector.save_status()
            except: pass

            return output
        except Exception as e:
            try: self.__image_collector.save_status()
            except: pass

            raise e
    
    def initialize(self, **kwargs): pass
    
    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:   Motors.BENDER_V.put(actuator_value)
        elif movement == Movement.RELATIVE: Motors.BENDER_V.put(Motors.BENDER_V.get() + actuator_value)
        else: raise ValueError("Movement not recognized")
        
    def get_v_bimorph_mirror_motor_bender(self): 
        return Motors.BENDER_V.get()
    
    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES):
        if units == AngularUnits.MILLIRADIANS: angle *= 1e-3
        elif units == AngularUnits.RADIANS: pass
        elif units == AngularUnits.DEGREES: angle = numpy.radians(angle)

        pos = 0.5*DISTANCE_V_MOTORS*numpy.sin(angle)

        if movement==Movement.ABSOLUTE:
            zero_pos = self.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

            self._move_translational_motor(Motors.TRANSLATION_VO, zero_pos-pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO, zero_pos+pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI, zero_pos+pos, movement=movement, units=DistanceUnits.MILLIMETERS)
        elif movement==Movement.RELATIVE:
            self._move_translational_motor(Motors.TRANSLATION_VO, -pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO,  pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI,  pos, movement=movement, units=DistanceUnits.MILLIMETERS)

    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.DEGREES):
        zero_pos = self.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

        pos = self._get_translational_motor_position(Motors.TRANSLATION_DO, units=DistanceUnits.MILLIMETERS) - zero_pos

        angle = numpy.arcsin(pos/(0.5*DISTANCE_V_MOTORS))

        if units == AngularUnits.MILLIRADIANS: angle *= 1e3
        elif units == AngularUnits.RADIANS: pass
        elif units == AngularUnits.DEGREES: angle = numpy.degrees(angle)

        return angle

    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS):
        if movement == Movement.RELATIVE:
            self._move_translational_motor(Motors.TRANSLATION_VO, translation, movement=movement, units=units)
            self._move_translational_motor(Motors.TRANSLATION_DO, translation, movement=movement, units=units)
            self._move_translational_motor(Motors.TRANSLATION_DI, translation, movement=movement, units=units)
        elif movement == Movement.ABSOLUTE:
            zero_pos = self.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

            difference = translation - zero_pos

            self._move_translational_motor(Motors.TRANSLATION_VO, difference, movement=Movement.RELATIVE, units=units)
            self._move_translational_motor(Motors.TRANSLATION_DO, difference, movement=Movement.RELATIVE, units=units)
            self._move_translational_motor(Motors.TRANSLATION_DI, difference, movement=Movement.RELATIVE, units=units)
        
    def get_v_bimorph_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS):
        return 0.5*(self._get_translational_motor_position(Motors.TRANSLATION_VO, units=units) +
                    self._get_translational_motor_position(Motors.TRANSLATION_DO, units=units))

    # H-KB -----------------------

    def move_h_bendable_mirror_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE):
        self.__move_h_bendable_mirror_motor_bender(motor=Motors.BENDER_H_1,
                                                   feeback=Motors.BENDER_H_1_FB,
                                                   readback=Motors.BENDER_H_1_RB,
                                                   pos=pos_upstream, movement=movement)

    def get_h_bendable_mirror_motor_1_bender(self):
        return Motors.BENDER_H_1.get()

    def move_h_bendable_mirror_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE):
        self.__move_h_bendable_mirror_motor_bender(motor=Motors.BENDER_H_2,
                                                   feeback=Motors.BENDER_H_2_FB,
                                                   readback=Motors.BENDER_H_2_RB,
                                                   pos=pos_downstream, movement=movement)
    
    def get_h_bendable_mirror_motor_2_bender(self): 
        return Motors.BENDER_H_2.get()

    def move_h_bendable_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES):
        self._move_rotational_motor(Motors.PITCH_H, angle, movement, units)

    def get_h_bendable_mirror_motor_pitch(self, units=AngularUnits.DEGREES):
        return self._get_rotational_motor_angle(Motors.PITCH_H, units)

    def move_h_bendable_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS):
        self._move_translational_motor(Motors.TRANSLATION_H, translation, movement, units)

    def get_h_bendable_mirror_motor_translation(self, units=DistanceUnits.MILLIMETERS):
        return self._get_translational_motor_position(Motors.TRANSLATION_H, units)

    def __move_h_bendable_mirror_motor_bender(self, motor, feeback, readback, pos, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:   desired_position = pos
        elif movement == Movement.RELATIVE: desired_position = motor.get() + pos
        else: raise ValueError("Movement not recognized")

        feeback.put(1)  # set feedback on
        motor.put(desired_position)

        # cycle until the readback is close enough to the desired position
        while (numpy.abs(readback.get() - desired_position) > Motors.BENDER_THRESHOLD): time.sleep(0.2)
