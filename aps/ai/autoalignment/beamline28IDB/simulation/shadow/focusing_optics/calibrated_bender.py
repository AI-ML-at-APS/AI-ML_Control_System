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

from oasys.widgets import congruence
from orangecontrib.ml.util.mocks import MockWidget

from aps.common.initializer import get_registered_ini_instance


class OneMotorCalibratedBenderManager():
    __P0 = 0.0
    __P1 = 0.0

    # 1/q = p0x + p1
    # x = (1/q - p1)/p0

    def __init__(self, shadow_oe=None, verbose=False):
        self._shadow_oe = shadow_oe
        self._verbose = verbose

    def load_calibration(self, key):
        ini = get_registered_ini_instance(application_name="benders calibration")

        self.__P0 = ini.get_float_from_ini(key, "p0")
        self.__P1 = ini.get_float_from_ini(key, "p1")

        if self._verbose: print(key + ", focus bender voltage from calibration : ", self.get_voltage())

    def get_q_distance(self):
        return self._shadow_oe._oe.SIMAG

    def get_voltage(self):
        return (1/(self._shadow_oe._oe.SIMAG*1e-3) - self.__P1)/self.__P0

    def get_voltage_for_focus(self, q_distance):
        return (1/(q_distance*1e-3) - self.__P1)/self.__P0

    def set_voltage(self, voltage):
        self._shadow_oe._oe.SIMAG = (1e3 / (self.__P0 * voltage + self.__P1))

    def remove_bender_file(self):
        if os.path.exists(self._shadow_oe.output_file_name_full):   os.remove(self._shadow_oe.output_file_name_full)

class TwoMotorsCalibratedBenderManager():
    __P0_upstream = 0.0
    __P0_downstream = 0.0
    __P1_upstream = 0.0
    __P1_downstream = 0.0

    q_upstream_previous   = 0.0
    q_downstream_previous = 0.0

    def __init__(self, kb_raytracing=None, kb_upstream=None, kb_downstream=None, verbose=False):
        self._kb_raytracing = kb_raytracing
        self._kb_upstream   = kb_upstream
        self._kb_downstream = kb_downstream
        self._verbose = verbose

    def load_calibration(self, key):
        ini = get_registered_ini_instance(application_name="benders calibration")

        self.__P0_upstream   = ini.get_float_from_ini(key, "p0_up")
        self.__P0_downstream = ini.get_float_from_ini(key, "p0_down")
        self.__P1_upstream   = ini.get_float_from_ini(key, "p1_up")
        self.__P1_downstream = ini.get_float_from_ini(key, "p1_down")

        if self._verbose: print(key + ", focus bender voltages from calibration (up, down): ", self.get_voltages())

    def get_q_distances(self):
        return self._kb_upstream.get_q_distance(), self._kb_downstream.get_q_distance()

    def get_voltages(self):
            return (1/(self._kb_upstream.get_q_distance()*1e-3)   - self.__P1_upstream)/self.__P0_upstream, \
                   (1/(self._kb_downstream.get_q_distance()*1e-3) - self.__P1_downstream)/self.__P0_downstream

    def get_voltages_for_focus(self, q_distance):
        return (1/(q_distance*1e-3) - self.__P1_upstream)/self.__P0_upstream, \
               (1/(q_distance*1e-3) - self.__P1_downstream)/self.__P0_downstream

    def set_voltages(self, pos_upstream, pos_downstream):
        self._kb_upstream.set_q_distance(1e3/(self.__P0_upstream * pos_upstream + self.__P1_upstream))
        self._kb_downstream.set_q_distance(1e3/(self.__P0_downstream * pos_downstream + self.__P1_downstream))

        if not self._kb_raytracing is None: self._kb_raytracing.set_q_distance(0.5*(self._kb_upstream.get_q_distance() + self._kb_downstream.get_q_distance()))

    def remove_bender_files(self):
        if os.path.exists(self._kb_upstream.output_file_name_full):   os.remove(self._kb_upstream.output_file_name_full)
        if os.path.exists(self._kb_downstream.output_file_name_full): os.remove(self._kb_downstream.output_file_name_full)

class HKBMockWidget(MockWidget):
    dim_x_minus = 0.0
    dim_x_plus = 0.0
    dim_y_minus = 0.0
    dim_y_plus = 0.0
    object_side_focal_distance = 0.0
    image_side_focal_distance = 0.0
    incidence_angle_respect_to_normal = 0.0

    modified_surface = 1
    ms_type_of_defect = 2
    ms_defect_file_name = "error_profile.dat"

    bender_bin_x = 50
    bender_bin_y = 500

    E = 131000
    h = 12.0

    kind_of_bender = 1 # double momentum
    shape          = 0 # trapezium

    output_file_name_full = "mirror_bender.dat"

    which_length = 0  # 0 - full length, 1 - partial length
    n_fit_steps = 5

    M1    = 0.0
    ratio = 0.5
    e     = 0.532524807056229

    M1_out    = 0.0
    ratio_out = 0.0
    e_out     = 0.0

    M1_fixed    = False
    ratio_fixed = False
    e_fixed     = True

    M1_min    = 0.0
    ratio_min = 0.0
    e_min     = 0.0

    M1_max    = 1000.0
    ratio_max = 10.0
    e_max     = 1.0

    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super(HKBMockWidget, self).__init__(verbose=verbose, workspace_units=workspace_units)
        self._shadow_oe = shadow_oe

        self.dim_x_minus = shadow_oe._oe.RWIDX2
        self.dim_x_plus = shadow_oe._oe.RWIDX1
        self.dim_y_minus = shadow_oe._oe.RLEN2
        self.dim_y_plus = shadow_oe._oe.RLEN1

        self.object_side_focal_distance        = shadow_oe._oe.SSOUR
        self.image_side_focal_distance         = None
        self.incidence_angle_respect_to_normal = shadow_oe._oe.THETA

        self.modified_surface = int(shadow_oe._oe.F_RIPPLE)
        self.ms_type_of_defect = int(shadow_oe._oe.F_G_S)
        self.ms_defect_file_name = shadow_oe._oe.FILE_RIP.decode('utf-8')

        self.initialize_bender_parameters(label)

    def manage_acceptance_slits(self, shadow_oe):
        pass  # do nothing

    def set_q_distance(self, q_distance):
        self._shadow_oe._oe.SIMAG      = q_distance
        self.image_side_focal_distance = q_distance

    def get_q_distance(self):
        return self._shadow_oe._oe.SIMAG

    def initialize_bender_parameters(self, label):
        self.output_file_name_full = congruence.checkFileName(("" if label is None else (label + "_")) + "HKB_bender_profile.dat")
        self.M1_out    = self.M1    = 500
        self.ratio_out = self.ratio = 1.3
