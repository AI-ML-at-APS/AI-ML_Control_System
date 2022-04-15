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

import numpy

from oasys.widgets import congruence
from orangecontrib.ml.util.mocks import MockWidget

from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.double_rod_bendable_ellispoid_mirror_bl import calculate_W0, calculate_taper_factor

from beamline34IDC.util.initializer import get_registered_ini_instance

class BenderManager():
    F_upstream = 0.0
    F_downstream = 0.0
    C_upstream = 0.0
    C_downstream = 0.0
    K_upstream = 0.0
    K_downstream = 0.0

    def __init__(self, kb_upstream, kb_downstream, verbose=False):
        self._kb_upstream = kb_upstream
        self._kb_downstream = kb_downstream
        self._verbose = verbose

    def load_calibration(self, key):
        ini = get_registered_ini_instance(application_name="benders calibration")

        # F = C + KX, with X in micron
        #
        # -> K = (Fa-Fb)/(Xa-Xb)
        # -> C = F - KX

        pos_1_focus = ini.get_float_from_ini(key, "motor_1_focus")
        pos_2_focus = ini.get_float_from_ini(key, "motor_2_focus")
        pos_1_out_focus = ini.get_float_from_ini(key, "motor_1_+4mm")
        pos_2_out_focus = ini.get_float_from_ini(key, "motor_2_+4mm")

        force_1_focus = ini.get_float_from_ini(key, "force_1_focus")
        force_2_focus = ini.get_float_from_ini(key, "force_2_focus")
        force_1_out_focus = ini.get_float_from_ini(key, "force_1_+4mm")
        force_2_out_focus = ini.get_float_from_ini(key, "force_2_+4mm")

        self.K_upstream = (force_1_focus - force_1_out_focus) / (pos_1_focus - pos_1_out_focus)
        self.K_downstream = (force_2_focus - force_2_out_focus) / (pos_2_focus - pos_2_out_focus)

        self.C_upstream = force_1_focus - self.K_upstream * pos_1_focus
        self.C_downstream = force_2_focus - self.K_downstream * pos_2_focus

        self.F_upstream = force_1_focus
        self.F_downstream = force_2_focus

        if self._verbose: print(key + ", focus bender positions from calibration (up, down): ", self.get_positions())

    def get_positions(self):
        return (self.F_upstream - self.C_upstream) / self.K_upstream, (self.F_downstream - self.C_downstream) / self.K_downstream

    def set_positions(self, pos_upstream, pos_downstream):
        self.F_upstream   = self.C_upstream + pos_upstream * self.K_upstream
        self.F_downstream = self.C_downstream + pos_downstream * self.K_downstream

        try:
            self.set_q_from_forces(self.F_upstream, self.F_downstream)
        except:
            if self._verbose: print("Q values not initialized")

    def set_q_from_forces(self, F_upstream, F_downstream):
        # f  = R0 * sin(alpha) / 2
        # (1/p + 1/q) = 2 / R0 * sin(alpha)
        # 1/R0 = (1/p + 1/q) * sin(alpha) / 2
        #
        # -> M0 = E * I0 / R0 = E * I0 * (1/p + 1/q) * sin(alpha) / 2
        #
        # F{u/d}   = M0 / r ] * [1 -+ eta * (L + 2r) / 2*q]
        # F{u/d}   = [E * I0 * (1/p + 1/q) * sin(alpha) / 2r ] * [1 -+ eta * (L + 2r) / 2*q]

        #  2 * F{u/d} * r / (E * I0 * sin(alpha)) = (1/p + 1/q) * [1 -+ (1/q) eta * (L + 2r) / 2]

        # A = 2 * r / (E * I0 * sin(alpha))
        # B = eta * (L + 2r) / 2

        # A * F{u/d} = (1/p + 1/q) * (1 -+ B * (1/q) ] = 1/p -+ (B/p) * (1/q) + (1/q) -+ B *(1/q**2)
        # -+ B (1/q**2) + (1 -+ B/p)* (1/q) - A * F{u/d} + 1/p = 0

        def calculate_q(kb, F, side=0):
            grazing_angle = numpy.radians(90 - kb.incidence_angle_respect_to_normal)
            p = kb.object_side_focal_distance

            A = 2 * kb.r / (kb.E * I0 * numpy.sin(grazing_angle))
            B = kb.eta * (L + 2 * kb.r) / 2

            if side == 0:
                sign = -1  # upstream
            else:
                sign = 1

            a = sign * B
            b = 1 + sign * B / p
            c = 1 / p - A * F

            gamma = (-b + numpy.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

            return 1 / gamma

        L = self._kb_upstream.dim_y_plus + self._kb_upstream.dim_y_minus
        W0 = self._kb_upstream.W0 / self._kb_upstream.workspace_units_to_mm
        I0 = (W0 * self._kb_upstream.h ** 3) / 12

        self._kb_upstream.set_q_distance(calculate_q(self._kb_upstream, F_upstream, side=0))
        self._kb_upstream.calculate_bender_quantities()

        self._kb_downstream.set_q_distance(calculate_q(self._kb_downstream, F_downstream, side=1))
        self._kb_downstream.calculate_bender_quantities()


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
    h = 10
    r = 10
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
    F_upstream = 0.0  # output of bender calculation
    F_downstream = 0.0  # output of bender calculation

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

    def calculate_bender_quantities(self):
        W1 = self.dim_x_plus + self.dim_x_minus
        L = self.dim_y_plus + self.dim_y_minus

        p = self.object_side_focal_distance
        q = self.image_side_focal_distance
        grazing_angle = numpy.radians(90 - self.incidence_angle_respect_to_normal)

        if not q is None:
            self.alpha = calculate_taper_factor(W1, self.W2, L, p, q, grazing_angle)
            self.W0 = calculate_W0(W1, self.alpha, L, p, q, grazing_angle)  # W at the center
        else:
            self.W0 = (self.W2 + W1) / 2


class VKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units, label=label)

    def initialize_bender_parameters(self, label):
        self.output_file_name_full = congruence.checkFileName(("" if label is None else (label + "_")) + "VKB_bender_profile.dat")
        self.R0 = 146.36857
        self.eta = 0.39548
        self.W2 = 21.0


class HKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2, label=None):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units, label=label)

    def initialize_bender_parameters(self, label):
        self.output_file_name_full = congruence.checkFileName(("" if label is None else (label + "_")) + "HKB_bender_profile.dat")
        self.R0 = 79.57061
        self.eta = 0.36055
        self.W2 = 2.5
