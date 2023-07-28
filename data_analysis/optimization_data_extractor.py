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
import os, numpy
from aps.ai.autoalignment.beamline28IDB.optimization.analysis_utils import load_histograms_from_files
from aps.ai.autoalignment.common.util.common import plot_2D, ColorMap
from aps.ai.autoalignment.beamline28IDB.hardware.epics.focusing_optics import DISTANCE_V_MOTORS

def get_v_bimorph_mirror_motor_pitch(di, do, u):
    zero_pos = 0.5 * (u + do)
    pos      = di - zero_pos

    angle = numpy.arcsin(pos / (0.5 * DISTANCE_V_MOTORS))

    return numpy.degrees(angle)

def print_positions(title, motor_reference, motors=None):
    print(title + " Motors Positions:")
    print("hb up    (V)  :" + str(motor_reference['hb_1'     ] + (0.0 if motors is None else motors['hb_1'     ])))
    print("hb down  (V)  :" + str(motor_reference['hb_2'     ] + (0.0 if motors is None else motors['hb_2'     ])))
    print("hb pitch (deg):" + str(motor_reference['hb_pitch' ] + (0.0 if motors is None else motors['hb_pitch' ])))
    print("hb trans (deg):" + str(motor_reference['hb_trans' ] + (0.0 if motors is None else motors['hb_trans' ])))
    print("vb       (V)  :" + str(motor_reference['vb_bender'] + (0.0 if motors is None else motors['vb_bender'])))
    print("vb pitch (deg):" + str(motor_reference['vb_pitch' ] + (0.0 if motors is None else motors['vb_pitch' ])))
    print("vb trans (deg):" + str(motor_reference['vb_trans' ] + (0.0 if motors is None else motors['vb_trans' ])))

def plot_trial(title, histo):
    xx = histo.hh
    yy = histo.vv
    zz = histo.data_2D

    plot_2D(x_array=xx, y_array=yy, z_array=zz, title=title, xrange=[-0.1, 0.1], yrange=[-0.1, 0.1], color_map=ColorMap.SUNBURST)


import joblib

'''
directory = "/Users/lrebuffi/Library/CloudStorage/Box-Box/Luca_Documents/AI-ML/AXO/28-ID/Experiment-Nov-2022/AI/autofocusing/peak_fwhm/"
target_trial = 23

histo_dir     = os.path.join(directory, "peak_fwhm_ref_150_2022-11-18_steps")
final_output  = os.path.join(directory, "peak_fwhm_ref_optimization_final_150_2022-11-18_01-23.pkl")

histos = load_histograms_from_files(n_steps=150, hists_dir=histo_dir, extension="pkl")
'''
directory = "/Users/lrebuffi/Library/CloudStorage/Box-Box/Luca_Documents/AI-ML/AXO/28-ID/Experiment-Nov-2022/AI/autofocusing/peak_fwhm_nlpi/all_motors_coma_denoised/"
target_trial = 54

histo_dir     = os.path.join(directory, "peak_fwhm_nlpi_moo_100_2022-11-21_steps")
final_output  = os.path.join(directory, "peak_fwhm_nlpi_moo_optimization_final_101_2022-11-21_23-54.gz")

histos = load_histograms_from_files(n_steps=100, hists_dir=histo_dir, extension="gz")

trials = joblib.load(final_output)

motors_reference = {
    'hb_1' : -170.0,
    'hb_2' : -155.0,
    'hb_pitch' : 0.17164,
    'hb_trans' : 0.0,
    'vb_bender' : 384.0,
    'vb_pitch' : get_v_bimorph_mirror_motor_pitch(0.54126, 0.54126, -0.54126),
    'vb_trans' : 0.0,
}

print_positions("Reference", motors_reference)
print_positions("Initial", motors_reference, trials[0].params)
print_positions("Target", motors_reference, trials[target_trial].params)

plot_trial("Initial", histos[0])
plot_trial("Target",  histos[target_trial])


