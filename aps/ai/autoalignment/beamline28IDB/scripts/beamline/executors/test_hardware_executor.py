#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
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

from aps.common.measurment.beamline.image_processor import IMAGE_SIZE_PIXEL_HxV, PIXEL_SIZE
from aps.common.scripts.abstract_script import AbstractScript

from aps.ai.autoalignment.common.util.common import AspectRatio, ColorMap, get_info, plot_2D
from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, Movement, AngularUnits
from aps.ai.autoalignment.common.hardware.facade.parameters import Implementors

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import ExecutionMode, focusing_optics_factory_method

class HardwareTestParameters:
    test_h_pitch = True
    h_pitch_absolute_move = 0.17189
    h_pitch_relative_move = -0.01
    test_h_translation = True
    h_translation_absolute_move = 0.100
    h_translation_relative_move = 0.005
    test_h_bender_1 = True
    h_bender_1_absolute_move = -175
    h_bender_1_relative_move = 20
    test_h_bender_2 = True
    h_bender_2_absolute_move = -170
    h_bender_2_relative_move = -10
    test_v_pitch = True
    v_pitch_absolute_move = 0.17189
    v_pitch_relative_move = -0.01
    test_v_translation = True
    v_translation_absolute_move = 0.100
    v_translation_relative_move = 0.005
    test_v_bender = True
    v_bender_absolute_move = 445
    v_bender_relative_move = -20
    test_detector = True

class PlotParameters(object):
    def __init__(self):
        nbins_h = IMAGE_SIZE_PIXEL_HxV[0]
        nbins_v = IMAGE_SIZE_PIXEL_HxV[1]

        detector_x = nbins_h * PIXEL_SIZE*1e3 # mm
        detector_y = nbins_v * PIXEL_SIZE*1e3 # mm

        xrange = [-detector_x / 2, detector_x / 2]
        yrange = [-detector_y / 2, detector_y / 2]

        xcoord = xrange[0] + numpy.arange(0, nbins_h)*PIXEL_SIZE*1e3 # mm
        ycoord = yrange[0] + numpy.arange(0, nbins_v)*PIXEL_SIZE*1e3 # mm

        self.params = {
            "xrange": xrange,
            "yrange": yrange,
            "xcoord" : xcoord,
            "ycoord" : ycoord,
            "nbins_h": nbins_h,
            "nbins_v": nbins_v,
            "do_gaussian_fit": False
        }

class TestHardwareScript(AbstractScript):

    def __init__(self, root_directory, energy, hardware_test_parameters : HardwareTestParameters):
        self.__root_directory           = root_directory
        self.__energy                   = energy
        self.__hardware_test_parameters = hardware_test_parameters

        self.__focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.HARDWARE,
                                                                implementor=Implementors.EPICS,
                                                                measurement_directory=os.path.join(self.__root_directory, "Tests"))
        self.__focusing_system.initialize()

    def execute_script(self, **kwargs):
        if self.__hardware_test_parameters.test_h_pitch:
            print("\nHorizontal Mirror - TEST OF THE PITCH MOTOR")
            initial_value = self.__focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(initial_value))
            time.sleep(1)
            print("Absolute Pitch Movement: " + str(self.__hardware_test_parameters.h_pitch_absolute_move) + " deg")
            self.__focusing_system.move_h_bendable_mirror_motor_pitch(angle=self.__hardware_test_parameters.h_pitch_absolute_move, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES)))
            time.sleep(1)
            print("Relative Pitch Movement: " + str(self.__hardware_test_parameters.h_pitch_relative_move) + " deg")
            self.__focusing_system.move_h_bendable_mirror_motor_pitch(angle=self.__hardware_test_parameters.h_pitch_relative_move, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES)))
            time.sleep(1)
            print("Restore Pitch to initial position: " + str(initial_value) + " deg")
            self.__focusing_system.move_h_bendable_mirror_motor_pitch(angle=initial_value, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
            print("Final Pitch Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES)))

        if self.__hardware_test_parameters.test_v_pitch:
            print("\nVertical Mirror - TEST OF THE PITCH MOTOR")
            initial_value = self.__focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(initial_value))
            time.sleep(1)
            print("Absolute Pitch Movement: " + str(self.__hardware_test_parameters.v_pitch_absolute_move) + " deg")
            self.__focusing_system.move_v_bimorph_mirror_motor_pitch(angle=self.__hardware_test_parameters.v_pitch_absolute_move, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES)))
            time.sleep(1)
            print("Relative Pitch Movement: " + str(self.__hardware_test_parameters.v_pitch_relative_move) + " deg")
            self.__focusing_system.move_v_bimorph_mirror_motor_pitch(angle=self.__hardware_test_parameters.v_pitch_relative_move, movement=Movement.RELATIVE, units=AngularUnits.DEGREES)
            print("Current Pitch Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES)))
            time.sleep(1)
            print("Restore Pitch to initial position: " + str(initial_value) + " deg")
            self.__focusing_system.move_v_bimorph_mirror_motor_pitch(angle=initial_value, movement=Movement.ABSOLUTE, units=AngularUnits.DEGREES)
            print("Final Pitch Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES)))

        if self.__hardware_test_parameters.test_h_translation:
            print("\nHorizontal Mirror - TEST OF THE TRANSLATION MOTOR")
            initial_value = self.__focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(initial_value) + " mm")
            time.sleep(1)
            print("Absolute Translation Movement: " + str(self.__hardware_test_parameters.h_translation_absolute_move))
            self.__focusing_system.move_h_bendable_mirror_motor_translation(translation=self.__hardware_test_parameters.h_translation_absolute_move, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))
            time.sleep(1)
            print("Relative Translation Movement: " + str(self.__hardware_test_parameters.h_translation_relative_move) + " mm")
            self.__focusing_system.move_h_bendable_mirror_motor_translation(translation=self.__hardware_test_parameters.h_translation_relative_move, movement=Movement.RELATIVE, units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))
            time.sleep(1)
            print("Restore Translation to initial position: " + str(initial_value) + " mm")
            self.__focusing_system.move_h_bendable_mirror_motor_translation(translation=initial_value, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
            print("Final Translation Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))

        if self.__hardware_test_parameters.test_v_translation:
            print("\nVertical Mirror - TEST OF THE TRANSLATION MOTOR")
            initial_value = self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(initial_value))
            time.sleep(2)
            print("Absolute Translation Movement: " + str(self.__hardware_test_parameters.v_translation_absolute_move) + " mm")
            self.__focusing_system.move_v_bimorph_mirror_motor_translation(translation=self.__hardware_test_parameters.v_translation_absolute_move, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))
            time.sleep(2)
            print("Absolute Translation Movement: " + str(2*self.__hardware_test_parameters.v_translation_absolute_move) + " mm")
            self.__focusing_system.move_v_bimorph_mirror_motor_translation(translation=2*self.__hardware_test_parameters.v_translation_absolute_move, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))
            time.sleep(2)
            print("Relative Translation Movement: " + str(self.__hardware_test_parameters.v_translation_relative_move) + " mm")
            self.__focusing_system.move_v_bimorph_mirror_motor_translation(translation=self.__hardware_test_parameters.v_translation_relative_move, movement=Movement.RELATIVE, units=DistanceUnits.MILLIMETERS)
            print("Current Translation Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))
            time.sleep(2)
            print("Restore Translation to initial position: " + str(initial_value) + " mm")
            self.__focusing_system.move_v_bimorph_mirror_motor_translation(translation=initial_value, movement=Movement.ABSOLUTE, units=DistanceUnits.MILLIMETERS)
            print("Final Translation Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)))

        if self.__hardware_test_parameters.test_h_bender_1:
            print("\nHorizontal Mirror - TEST OF THE BENDER MOTOR 1")
            initial_value = self.__focusing_system.get_h_bendable_mirror_motor_1_bender()
            print("Current Bender 1 Value: " + str(initial_value))
            time.sleep(1)
            print("Absolute Bender 1 Movement: " + str(self.__hardware_test_parameters.h_bender_1_absolute_move))
            self.__focusing_system.move_h_bendable_mirror_motor_1_bender(pos_upstream=self.__hardware_test_parameters.h_bender_1_absolute_move, movement=Movement.ABSOLUTE)
            print("Current Bender 1 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_1_bender()))
            time.sleep(1)
            print("Relative Bender 1 Movement: " + str(self.__hardware_test_parameters.h_bender_1_relative_move))
            self.__focusing_system.move_h_bendable_mirror_motor_1_bender(pos_upstream=self.__hardware_test_parameters.h_bender_1_relative_move, movement=Movement.RELATIVE)
            print("Current Bender 1 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_1_bender()))
            time.sleep(1)
            print("Restore Bender 1 to initial position: " + str(initial_value))
            self.__focusing_system.move_h_bendable_mirror_motor_1_bender(pos_upstream=initial_value, movement=Movement.ABSOLUTE)
            print("Final Bender 1 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_1_bender()))

        if self.__hardware_test_parameters.test_h_bender_2:
            print("\nHorizontal Mirror - TEST OF THE BENDER MOTOR 2")
            initial_value = self.__focusing_system.get_h_bendable_mirror_motor_2_bender()
            print("Current Bender 2 Value: " + str(initial_value))
            time.sleep(1)
            print("Absolute Bender 2 Movement: " + str(self.__hardware_test_parameters.h_bender_2_absolute_move))
            self.__focusing_system.move_h_bendable_mirror_motor_2_bender(pos_downstream=self.__hardware_test_parameters.h_bender_2_absolute_move, movement=Movement.ABSOLUTE)
            print("Current Bender 2 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_2_bender()))
            time.sleep(1)
            print("Relative Bender 2 Movement: " + str(self.__hardware_test_parameters.h_bender_2_relative_move) + " deg")
            self.__focusing_system.move_h_bendable_mirror_motor_2_bender(pos_downstream=self.__hardware_test_parameters.h_bender_2_relative_move, movement=Movement.RELATIVE)
            print("Current Bender 2 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_2_bender()))
            time.sleep(1)
            print("Restore Bender 2 to initial position: " + str(initial_value) + " deg")
            self.__focusing_system.move_h_bendable_mirror_motor_2_bender(pos_downstream=initial_value, movement=Movement.ABSOLUTE)
            print("Final Bender 2 Value: " + str(self.__focusing_system.get_h_bendable_mirror_motor_2_bender()))

        if self.__hardware_test_parameters.test_v_bender:
            print("\nVertical Mirror - TEST OF THE BENDER MOTOR")
            initial_value = self.__focusing_system.get_v_bimorph_mirror_motor_bender()
            print("Current Bender Value: " + str(initial_value))
            time.sleep(1)
            print("Absolute Bender Movement: " + str(self.__hardware_test_parameters.v_bender_absolute_move))
            self.__focusing_system.move_v_bimorph_mirror_motor_bender(actuator_value=self.__hardware_test_parameters.v_bender_absolute_move, movement=Movement.ABSOLUTE)
            print("Current Bender Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_bender()))
            time.sleep(1)
            print("Relative Bender Movement: " + str(self.__hardware_test_parameters.v_bender_relative_move) + " deg")
            self.__focusing_system.move_v_bimorph_mirror_motor_bender(actuator_value=self.__hardware_test_parameters.v_bender_relative_move, movement=Movement.RELATIVE)
            print("Current Bender Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_bender()))
            time.sleep(1)
            print("Restore Bender to initial position: " + str(initial_value) + " deg")
            self.__focusing_system.move_v_bimorph_mirror_motor_bender(actuator_value=initial_value, movement=Movement.ABSOLUTE)
            print("Final Bender Value: " + str(self.__focusing_system.get_v_bimorph_mirror_motor_bender()))

        if self.__hardware_test_parameters.test_detector:
            print("\nTEST OF THE DETECTOR")
            photon_beam = self.__focusing_system.get_photon_beam(from_raw_image=True)

            plot_2D(x_array=photon_beam["h_coord"],
                    y_array=photon_beam["v_coord"],
                    z_array=photon_beam["image"],
                    title="Raw Image from detector",
                    color_map=ColorMap.GRAY,
                    aspect_ratio=AspectRatio.CARTESIAN)

            _, dictionary = get_info(x_array=photon_beam["h_coord"],
                                     y_array=photon_beam["v_coord"],
                                     z_array=photon_beam["image"].T,
                                     do_gaussian_fit=False)

            print("Beam Infos:")
            print(dictionary)

    def manage_keyboard_interrupt(self):
        print("\nTest Motors script interrupted by user")
