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

from beamline34IDC.facade.focusing_optics_interface import Movement, DistanceUnits, AngularUnits
from beamline34IDC.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics, get_default_input_features

def srw_focusing_optics_factory_method(**kwargs):
    try:
        if kwargs["bender"] == True: return __BendableFocusingOptics()
        else:                        return __IdealFocusingOptics()
    except: return __IdealFocusingOptics()

class _FocusingOpticsCommon(AbstractSimulatedFocusingOptics):

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None): pass
    def restore_input_photon_beam(self): pass

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None, units=DistanceUnits.MICRON): pass
    def get_coherence_slits_parameters(self, units=DistanceUnits.MICRON): pass # center x, center z, aperture x, aperture z


class __IdealFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        pass

    def initialize(self, input_photon_beam, input_features=get_default_input_features(), **kwargs):
        try: rewrite_reflectivity_files    = kwargs["rewrite_reflectivity_files"]
        except: rewrite_reflectivity_files = False
        try: rewrite_height_error_profile_files    = kwargs["rewrite_height_error_profile_files"]
        except: rewrite_height_error_profile_files = False

    # V-KB -----------------------

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_vkb_q_distance(self): pass

    # H-KB -----------------------

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE): pass
    def get_hkb_q_distance(self): pass

class __BendableFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        pass
    
    def initialize(self, input_photon_beam, input_features=get_default_input_features(), **kwargs): pass


    # V-KB -----------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON): pass

    # H-KB -----------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): raise NotImplementedError()
    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON): raise NotImplementedError()
    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS): pass
    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS): pass
    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): pass
    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON): pass

    def get_photon_beam(self, **kwargs): pass
