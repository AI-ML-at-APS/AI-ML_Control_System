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

import os, numpy

from oasys.widgets import congruence
from orangecontrib.ml.util.mocks import MockWidget

from oasys.widgets.abstract.benders.double_rod_bendable_ellispoid_mirror import calculate_W0, calculate_taper_factor

from aps_ai.common.util.initializer import get_registered_ini_instance

class CalibratedBenderManager():
    __P0_upstream = 0.0
    __P0_downstream = 0.0
    __P1_upstream = 0.0
    __P1_downstream = 0.0
    __P2_upstream = 0.0
    __P2_downstream = 0.0
    __power = 1

    q_upstream_previous   = 0.0
    q_downstream_previous = 0.0

    def __init__(self, kb_raytracing=None, kb_upstream=None, kb_downstream=None, verbose=False):
        self._kb_raytracing = kb_raytracing
        self._kb_upstream   = kb_upstream
        self._kb_downstream = kb_downstream
        self._verbose = verbose

    def load_calibration(self, key, power=1):
        ini = get_registered_ini_instance(application_name="benders calibration")
        if power==1:
            self.__P0_upstream   = ini.get_float_from_ini(key, "p0_up")
            self.__P0_downstream = ini.get_float_from_ini(key, "p0_down")
            self.__P1_upstream   = ini.get_float_from_ini(key, "p1_up")
            self.__P1_downstream = ini.get_float_from_ini(key, "p1_down")
        elif power==2:
            self.__P0_upstream   = ini.get_float_from_ini(key + "-2", "p0_up")
            self.__P0_downstream = ini.get_float_from_ini(key + "-2", "p0_down")
            self.__P1_upstream   = ini.get_float_from_ini(key + "-2", "p1_up")
            self.__P1_downstream = ini.get_float_from_ini(key + "-2", "p1_down")
            self.__P2_upstream   = ini.get_float_from_ini(key + "-2", "p2_up")
            self.__P2_downstream = ini.get_float_from_ini(key + "-2", "p2_down")
        else:
            raise ValueError("Polynomial grade not supported")

        self.__power = power

        if self._verbose: print(key + ", focus bender positions from calibration (up, down): ", self.get_positions())

    def get_q_distances(self):
        return self._kb_upstream.get_q_distance(), self._kb_downstream.get_q_distance()

    def get_positions(self):
        if self.__power == 1:
            return (self._kb_upstream.get_q_distance()   - self.__P1_upstream)/self.__P0_upstream, \
                   (self._kb_downstream.get_q_distance() - self.__P1_downstream)/self.__P0_downstream
        elif self.__power == 2:
            return (-self.__P1_upstream   + numpy.sqrt(self.__P1_upstream**2   - 4*self.__P0_upstream*(  self.__P2_upstream   - self._kb_upstream.get_q_distance())))  /(2*self.__P0_upstream), \
                   (-self.__P1_downstream + numpy.sqrt(self.__P1_downstream**2 - 4*self.__P0_downstream*(self.__P2_downstream - self._kb_downstream.get_q_distance())))/(2*self.__P0_downstream)

    def get_position_for_focus(self, q_distance):
        if self.__power == 1:
            return (q_distance - self.__P1_upstream)/self.__P0_upstream, \
                   (q_distance - self.__P1_downstream)/self.__P0_downstream
        elif self.__power == 2:
            return (-self.__P1_upstream + numpy.sqrt(self.__P1_upstream**2 - 4*self.__P0_upstream*(self.__P2_upstream-q_distance)))/(2*self.__P0_upstream)
            return (-self.__P1_downstream + numpy.sqrt(self.__P1_downstream**2 - 4*self.__P0_downstream*(self.__P2_downstream-q_distance)))/(2*self.__P0_downstream)

    def set_positions(self, pos_upstream, pos_downstream):
        if self.__power==1:
            self._kb_upstream.set_q_distance(self.__P0_upstream*pos_upstream + self.__P1_upstream)
            self._kb_downstream.set_q_distance(self.__P0_downstream*pos_downstream + self.__P1_downstream)
        elif self.__power == 2:
            self._kb_upstream.set_q_distance(self.__P0_upstream*pos_upstream**2 + self.__P1_upstream*pos_upstream + self.__P2_upstream)
            self._kb_downstream.set_q_distance(self.__P0_downstream*pos_downstream*22 + self.__P1_downstream*pos_downstream + self.__P2_downstream)

        self._kb_upstream.calculate_bender_quantities()
        self._kb_downstream.calculate_bender_quantities()

        if not self._kb_raytracing is None: self._kb_raytracing.set_q_distance(0.5*(self._kb_upstream.get_q_distance() + self._kb_downstream.get_q_distance()))

    def remove_bender_files(self):
        if os.path.exists(self._kb_upstream.output_file_name_full):   os.remove(self._kb_upstream.output_file_name_full)
        if os.path.exists(self._kb_downstream.output_file_name_full): os.remove(self._kb_downstream.output_file_name_full)

class _KBMockWidget(MockWidget):
    shadow_oe = None

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

    bender_bin_x = 10
    bender_bin_y = 200

    E = 131000
    h = 10.0
    r = 14.62
    l = 70.18
    output_file_name_full = "mirror_bender.dat"
    which_length = 1  # 0 - full length, 1 - partial length
    optimized_length = 72.0  # only optically active surface
    n_fit_steps = 5

    R0 = 45
    eta = 0.25
    W2 = 40.0
    R0_fixed = False
    eta_fixed = True
    W2_fixed = True
    R0_min = 20.0
    eta_min = 0.0
    W2_min = 1.0
    R0_max = 300.0
    eta_max = 2.0
    W2_max = 42.0

    R0_out = 0.0
    eta_out = 0.0
    W2_out = 0.0
    alpha = 0.0
    W0 = 0.0

    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super(_KBMockWidget, self).__init__(verbose=verbose, workspace_units=workspace_units)
        self.shadow_oe = shadow_oe

        self.dim_x_minus = shadow_oe._oe.RWIDX2
        self.dim_x_plus = shadow_oe._oe.RWIDX1
        self.dim_y_minus = shadow_oe._oe.RLEN2
        self.dim_y_plus = shadow_oe._oe.RLEN1

        self.object_side_focal_distance = shadow_oe._oe.SSOUR
        self.image_side_focal_distance = None
        self.incidence_angle_respect_to_normal = shadow_oe._oe.THETA

        self.modified_surface = int(shadow_oe._oe.F_RIPPLE)
        self.ms_type_of_defect = int(shadow_oe._oe.F_G_S)
        self.ms_defect_file_name = shadow_oe._oe.FILE_RIP.decode('utf-8')

        self.initialize_bender_parameters(label)
        self.calculate_bender_quantities()

        self.R0_out = self.R0

    def manage_acceptance_slits(self, shadow_oe):
        pass  # do nothing

    def initialize_bender_parameters(self, label):
        pass

    def set_q_distance(self, q_distance):
        self.shadow_oe._oe.SIMAG = q_distance
        self.image_side_focal_distance = q_distance

    def get_q_distance(self):
        return self.shadow_oe._oe.SIMAG

    def calculate_bender_quantities(self):
        W1 = self.dim_x_plus + self.dim_x_minus
        L = self.dim_y_plus + self.dim_y_minus

        p = self.object_side_focal_distance
        q = self.image_side_focal_distance
        grazing_angle = numpy.radians(90 - self.incidence_angle_respect_to_normal)

        if not q is None:
            self.alpha = calculate_taper_factor(W1, self.W2, L, p, q, grazing_angle)
            self.W0    = calculate_W0(W1, self.alpha, L, p, q, grazing_angle)  # W at the center
        else:
            self.W0 = (self.W2 + W1) / 2


class VKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units, label=label)

    def initialize_bender_parameters(self, label):
        self.output_file_name_full = congruence.checkFileName(("" if label is None else (label + "_")) + "VKB_bender_profile.dat")
        self.R0  = 152.76
        self.eta = 0.3257
        self.W2  = 21.0


class HKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units, label=label)

    def initialize_bender_parameters(self, label):
        self.output_file_name_full = congruence.checkFileName(("" if label is None else (label + "_")) + "HKB_bender_profile.dat")
        self.R0  = 81.63
        self.eta = 0.329
        self.W2  = 2.5
