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
import sys
import time

import numpy
from epics import PV

from aps.common.measurment.beamline.image_processor import IMAGE_SIZE_PIXEL_HxV, PIXEL_SIZE
from aps.common.measurment.beamline.image_collector import ImageCollector

from aps.ai.autoalignment.common.measurement.image_processor import ImageProcessor
from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, Movement, AngularUnits
from aps.ai.autoalignment.common.hardware.epics.optics import AbstractEpicsOptics
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_interface import AbstractFocusingOptics, \
    DISTANCE_V_MOTORS


def epics_focusing_optics_factory_method(**kwargs):
    return __EpicsFocusingOptics(**kwargs)


class Motors:
    # Horizontal mirror:
    TRANSLATION_H = PV(pvname='28idb:m23')
    PITCH_H = PV(pvname='28idb:m24')
    BENDER_H_1 = PV(pvname='zoomkb:pid1')
    BENDER_H_2 = PV(pvname='zoomkb:pid2')
    BENDER_H_1_RB = PV(pvname='zoomkb:pid1.CVAL')
    BENDER_H_2_RB = PV(pvname='zoomkb:pid2.CVAL')
    BENDER_H_1_FB = PV(pvname='zoomkb:pid1.FBON')
    BENDER_H_2_FB = PV(pvname='zoomkb:pid2.FBON')
    BENDER_THRESHOLD = 0.05

    TRANSLATION_VO = PV(pvname='1bmopt:m13')
    TRANSLATION_DI = PV(pvname='1bmopt:m12')
    TRANSLATION_DO = PV(pvname='1bmopt:m14')
    LATERAL_V = PV(pvname='1bmopt:m15')
    BENDER_V = PV(pvname='simJTEC:E4')

    SURFACE_ACTUATORS_V = [PV(pvname='simJTEC:A1'),
                           PV(pvname='simJTEC:A2'),
                           PV(pvname='simJTEC:A3'),
                           PV(pvname='simJTEC:A4'),

                           PV(pvname='simJTEC:B1'),
                           PV(pvname='simJTEC:B2'),
                           PV(pvname='simJTEC:B3'),
                           PV(pvname='simJTEC:B4'),

                           PV(pvname='simJTEC:C1'),
                           PV(pvname='simJTEC:C2'),
                           PV(pvname='simJTEC:C3'),
                           PV(pvname='simJTEC:C4'),

                           PV(pvname='simJTEC:D1'),
                           PV(pvname='simJTEC:D2'),
                           PV(pvname='simJTEC:D3'),
                           PV(pvname='simJTEC:D4'),

                           PV(pvname='simJTEC:E1'),
                           PV(pvname='simJTEC:E2'),
                           PV(pvname='simJTEC:E3')]


class __EpicsFocusingOptics(AbstractEpicsOptics, AbstractFocusingOptics):

    def __init__(self, **kwargs):
        super().__init__(translational_units=DistanceUnits.MILLIMETERS, angular_units=AngularUnits.DEGREES)

        try:
            measurement_directory = kwargs["measurement_directory"]
        except:
            measurement_directory = os.curdir
        # TODO: ADD CHECK OF PHYSICAL BOuNDARIES
        try:
            self.__physical_boundaries = kwargs["physical_boundaries"]
        except:
            self.__physical_boundaries = None
        try:
            self.__bender_threshold = kwargs["bender_threshold"]
        except:
            self.__bender_threshold = Motors.BENDER_THRESHOLD
        try:
            self.__n_bender_threshold_check = kwargs["n_bender_threshold_check"]
        except:
            self.__n_bender_threshold_check = 1

        self.__image_collector = ImageCollector(measurement_directory=measurement_directory)
        self.__image_processor = ImageProcessor(data_collection_directory=measurement_directory)

    def get_photon_beam(self, **kwargs):
        try:
            from_raw_image = kwargs["from_raw_image"]
        except:
            from_raw_image = True

        try:
            self.__image_collector.restore_status()
        except:
            pass

        try:
            self.__image_collector.collect_single_shot_image(index=1)

            image, h_coord, v_coord = self.__image_processor.get_image_data(image_index=1)

            image_denoised = image - numpy.average(image[0:10, 0:10])
            image_denoised[numpy.where(image_denoised < 0)] = 0.0

            output = {}
            output["h_coord"] = h_coord
            output["v_coord"] = v_coord
            output["image"] = image
            output["image_denoised"] = image_denoised

            if not from_raw_image:
                from scipy.ndimage.measurements import center_of_mass

                footprint = numpy.ones(image.shape) * (image > 160)

                #from matplotlib import pyplot as plt
                #plt.imshow(footprint.T)
                #plt.show()

                center = center_of_mass(footprint)
                center_x, center_y = int(center[0]), int(center[1])

                # find the boundary
                n_width = 50

                strip_x = numpy.array(numpy.sum(footprint[:, center_y - n_width: center_y + n_width], axis=1))
                strip_y = numpy.flip(numpy.array(numpy.sum(footprint[center_x - n_width: center_x + n_width, :], axis=0)))

                #from matplotlib import pyplot as plt
                #plt.plot(strip_x, 'b-')
                #plt.plot(strip_y, 'r-')
                #plt.show()

                left_x  = numpy.amin(numpy.where(strip_x > 20))
                right_x = numpy.amax(numpy.where(strip_x > 20))
                up_y    = numpy.amin(numpy.where(strip_y > 20))
                down_y  = numpy.amax(numpy.where(strip_y > 20))

                center_x = h_coord[center_x]
                center_y = v_coord[IMAGE_SIZE_PIXEL_HxV[1] - center_y]
                width_x = (right_x - left_x)*PIXEL_SIZE * 1e3
                width_y = (down_y - up_y)*PIXEL_SIZE * 1e3

                print("Crop Region: Center (HxV) =", round(center_x, 4), round(center_y, 4),
                      "mm, Dimension (HxV) =", round(width_x, 3), round(width_y, 3), "mm")

                output["width"] = width_x
                output["height"] = width_y
                output["centroid_h"] = center_x
                output["centroid_v"] = center_y

            try:    self.__image_collector.end_collection()
            except: pass
            try:    self.__image_collector.save_status()
            except: pass


            return output
        except Exception as e:
            try:    self.__image_collector.end_collection()
            except: pass
            try:    self.__image_collector.save_status()
            except: pass

            raise e

    def initialize(self, **kwargs):
        pass

    def set_surface_actuators_to_baseline(self, baseline=500):
        for actuator in Motors.SURFACE_ACTUATORS_V: actuator.put(baseline)

    def move_v_bimorph_mirror_motor_bender(self, actuator_value, movement=Movement.ABSOLUTE):
        if movement == Movement.ABSOLUTE:
            Motors.BENDER_V.put(actuator_value)
        elif movement == Movement.RELATIVE:
            Motors.BENDER_V.put(Motors.BENDER_V.get() + actuator_value)
        else:
            raise ValueError("Movement not recognized")
        time.sleep(2)

    def get_v_bimorph_mirror_motor_bender(self):
        return Motors.BENDER_V.get()

    def move_v_bimorph_mirror_motor_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES):
        if units == AngularUnits.MILLIRADIANS:
            angle *= 1e-3
        elif units == AngularUnits.RADIANS:
            pass
        elif units == AngularUnits.DEGREES:
            angle = numpy.radians(angle)

        pos = 0.5 * DISTANCE_V_MOTORS * numpy.sin(angle)

        if movement == Movement.ABSOLUTE:
            zero_pos = self.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

            self._move_translational_motor(Motors.TRANSLATION_VO, zero_pos - pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO, zero_pos + pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI, zero_pos + pos, movement=movement, units=DistanceUnits.MILLIMETERS)
        elif movement == Movement.RELATIVE:
            self._move_translational_motor(Motors.TRANSLATION_VO, -pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DO, pos, movement=movement, units=DistanceUnits.MILLIMETERS)
            self._move_translational_motor(Motors.TRANSLATION_DI, pos, movement=movement, units=DistanceUnits.MILLIMETERS)

    def get_v_bimorph_mirror_motor_pitch(self, units=AngularUnits.DEGREES):
        zero_pos = self.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

        pos = self._get_translational_motor_position(Motors.TRANSLATION_DO, units=DistanceUnits.MILLIMETERS) - zero_pos

        angle = numpy.arcsin(pos / (0.5 * DISTANCE_V_MOTORS))

        if units == AngularUnits.MILLIRADIANS: angle *= 1e3
        elif units == AngularUnits.RADIANS:    pass
        elif units == AngularUnits.DEGREES:    angle = numpy.degrees(angle)

        return angle

    def move_v_bimorph_mirror_motor_translation(self, translation, movement=Movement.ABSOLUTE,
                                                units=DistanceUnits.MILLIMETERS):
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
        return 0.5 * (self._get_translational_motor_position(Motors.TRANSLATION_VO, units=units) +
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

        n_consecutive_positive_check = 0
        # cycle until the readback is close enough to the desired position
        while (n_consecutive_positive_check < self.__n_bender_threshold_check):
            if (numpy.abs(readback.get() - desired_position) <= self.__bender_threshold):
                n_consecutive_positive_check += 1
                # print("H Bender ok:" + str(n_consecutive_positive_check))
            else:
                n_consecutive_positive_check = 0

            time.sleep(0.1)
