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
import Shadow
import numpy
import os, glob, sys

from orangecontrib.ml.util.mocks import MockWidget

# OASYS + HYBRID library, to add correction for diffraction and error profiles interference effects.
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement
from orangecontrib.shadow_advanced_tools.widgets.sources.ow_hybrid_undulator import HybridUndulatorAttributes
import orangecontrib.shadow_advanced_tools.widgets.sources.bl.hybrid_undulator_bl as hybrid_undulator_bl

def run_geometrical_source(n_rays=500000, random_seed=5676561, aperture=[0.03, 0.07], distance=50500):
    #####################################################
    # SHADOW 3 INITIALIZATION

    #
    # initialize shadow3 source (oe0) and beam
    #
    oe0 = Shadow.Source()

    div = [0.5*aperture[0]/distance, 0.5*aperture[1]/distance]

    oe0.FDISTR = 3
    oe0.F_COLOR = 3
    oe0.F_PHOT = 0
    oe0.HDIV1 = div[0]
    oe0.HDIV2 = div[0]
    oe0.IDO_VX = 0
    oe0.IDO_VZ = 0
    oe0.IDO_X_S = 0
    oe0.IDO_Y_S = 0
    oe0.IDO_Z_S = 0
    oe0.ISTAR1 = random_seed
    oe0.NPOINT = n_rays
    oe0.PH1 = 4999.0
    oe0.PH2 = 5001.0
    oe0.SIGDIX = 1.6e-05
    oe0.SIGDIZ = 7e-06
    oe0.SIGMAX = 0.281
    oe0.SIGMAZ = 0.014
    oe0.VDIV1 = div[1]
    oe0.VDIV2 = div[1]

    # WEIRD MEMORY INITIALIZATION BY FORTRAN. JUST A FIX.
    def fix_Intensity(beam_out, polarization=0):
        if polarization == 0:
            beam_out._beam.rays[:, 15] = 0
            beam_out._beam.rays[:, 16] = 0
            beam_out._beam.rays[:, 17] = 0
        return beam_out

    shadow_source = ShadowSource.create_src()
    shadow_source.set_src(src=oe0)

    # Run SHADOW to create the source + BUT WE USE OASYS LIBRARY TO BE ABLE TO RUN HYBRYD
    source_beam = fix_Intensity(ShadowBeam.traceFromSource(shadow_source))

    return source_beam

class MockUndulatorHybrid(MockWidget, HybridUndulatorAttributes):
    def __init__(self, verbose=False,  workspace_units=2):
        MockWidget.__init__(self, verbose, workspace_units)

        self.distribution_source = 0  # SRW
        self.optimize_source = 0
        self.polarization = 1
        self.coherent_beam = 0
        self.phase_diff = 0.0
        self.polarization_degree = 1.0
        self.max_number_of_rejected_rays = 0

        self.use_harmonic = 2
        self.energy = 4999
        self.energy_to = 5001
        self.energy_points = 3#21

        self.number_of_periods = 72  # Number of ID Periods (without counting for terminations
        self.undulator_period = 0.033  # Period Length [m]
        self.horizontal_central_position = 0.0
        self.vertical_central_position = 0.0
        self.longitudinal_central_position = 0.0

        self.Kv = 1.907944
        self.Kh = 0.0
        self.magnetic_field_from = 0
        self.initial_phase_vertical = 0.0
        self.initial_phase_horizontal = 0.0
        self.symmetry_vs_longitudinal_position_vertical = 1
        self.symmetry_vs_longitudinal_position_horizontal = 0

        self.electron_energy_in_GeV = 7.0
        self.electron_energy_spread = 0.00098
        self.ring_current = 0.1
        self.electron_beam_size_h = 0.0002805
        self.electron_beam_size_v = 1.02e-05
        self.electron_beam_divergence_h = 1.18e-05
        self.electron_beam_divergence_v = 3.4e-06

        self.type_of_initialization = 0

        self.source_dimension_wf_h_slit_gap = 0.005
        self.source_dimension_wf_v_slit_gap = 0.001
        self.source_dimension_wf_h_slit_points = 500
        self.source_dimension_wf_v_slit_points = 100
        self.source_dimension_wf_distance = 10.0

        self.horizontal_range_modification_factor_at_resizing = 0.5
        self.horizontal_resolution_modification_factor_at_resizing = 5.0
        self.vertical_range_modification_factor_at_resizing = 0.5
        self.vertical_resolution_modification_factor_at_resizing = 5.0

        self.auto_expand = 0
        self.auto_expand_rays = 0

        self.kind_of_sampler = 1
        self.save_srw_result = 0

def run_hybrid_undulator_source(n_rays=500000, random_seed=5676561):
    widget = MockUndulatorHybrid(verbose=True)

    widget.number_of_rays = n_rays
    widget.seed = random_seed

    source_beam, _ = hybrid_undulator_bl.run_hybrid_undulator_simulation(widget)

    return source_beam

def run_hybrid_undulator_source_through_aperture(n_rays=500000, aperture=[0.001, 0.001], distance=10000.0, target_good_rays=50000):

    def run_beam_through_aperture(n_rays, aperture, distance):
        oe1 = Shadow.OE()

        # WB SLITS
        oe1.DUMMY = 0.1
        oe1.FWRITE = 3
        oe1.F_REFRAC = 2
        oe1.F_SCREEN = 1
        oe1.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        oe1.N_SCREEN = 1
        oe1.RX_SLIT = numpy.array([aperture[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        oe1.RZ_SLIT = numpy.array([aperture[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        oe1.T_IMAGE = 0.0
        oe1.T_INCIDENCE = 0.0
        oe1.T_REFLECTION = 180.0
        oe1.T_SOURCE = distance

        output_beam = ShadowBeam.traceFromOE(run_hybrid_undulator_source(n_rays, 0), # seed to 0 to ensure a new beam every time
                                             ShadowOpticalElement(oe1),
                                             widget_class_name="ScreenSlits")
        # good only
        good_only = numpy.where(output_beam._beam.rays[:, 9] == 1)
        output_beam._beam.rays = output_beam._beam.rays[good_only]

        output_beam._beam.retrace(-distance)

        return output_beam

    source_beam = ShadowBeam()

    while(source_beam.get_number_of_rays() < target_good_rays):
        temp_beam = run_beam_through_aperture(n_rays, aperture, distance)

        print("HYBRID UNDULATOR: ", temp_beam.get_number_of_rays(), " good rays")
        source_beam._beam.rays = numpy.append(source_beam._beam.rays,
                                              temp_beam._beam.rays)
        print("TOTAL: ", source_beam.get_number_of_rays(), " good rays on ", target_good_rays)

    return source_beam

def save_source_beam(source_beam, file_name="begin.dat"):
    source_beam.writeToFile(file_name)

def load_source_beam(file_name="begin.dat"):
    source_beam = ShadowBeam()
    source_beam.loadFromFile(file_name)

    return source_beam
