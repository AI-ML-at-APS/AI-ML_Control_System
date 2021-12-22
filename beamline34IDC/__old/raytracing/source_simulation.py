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

from orangecontrib.ml.util.mocks import MockWidget

# OASYS + HYBRID library, to add correction for diffraction and error profiles interference effects.
from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOEHistoryItem, ShadowOpticalElement
from orangecontrib.shadow_advanced_tools.widgets.sources.attributes.hybrid_undulator_attributes import HybridUndulatorAttributes
import orangecontrib.shadow_advanced_tools.widgets.sources.bl.hybrid_undulator_bl as hybrid_undulator_bl

class SourceType:
    GEOMETRICAL = 0
    HYBRID_UNDULATOR = 1
    HYBRID_UNDULATOR_APERTURE = 2

def run_geometrical_source(n_rays=500000, random_seed=5676561, aperture=[0.03, 0.07], distance=50500):
    #####################################################
    # SHADOW 3 INITIALIZATION

    #
    # initialize shadow3 source (oe0) and beam
    #
    oe0 = Shadow.Source()

    div = [0.5*aperture[0]/distance, 0.5*aperture[1]/distance]

    print("Run geometrical source with limited divergence: ", div, "rad")

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
        self.energy_points = 11

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
        output_beam._beam.retrace(0.0) # back to source position

        # good only
        good_only = numpy.where(output_beam._beam.rays[:, 9] == 1)
        output_beam._beam.rays = output_beam._beam.rays[good_only]

        return output_beam

    current_good_rays = 0

    while(current_good_rays < target_good_rays):
        temp_beam = run_beam_through_aperture(n_rays*5, aperture, distance)

        print("HYBRID UNDULATOR: ", temp_beam.get_number_of_rays(), " good rays")

        if current_good_rays == 0: source_beam = temp_beam
        else:                      source_beam = ShadowBeam.mergeBeams(source_beam, temp_beam)

        current_good_rays = source_beam.get_number_of_rays()

        print("TOTAL: ", current_good_rays, " good rays on ", target_good_rays)

    return source_beam

def __save_source_beam(source_beam, file_name="begin.dat"):
    source_beam.getOEHistory(0)._shadow_source_end.src.write("source_" + file_name)
    source_beam.writeToFile(file_name)

def __load_source_beam(file_name="begin.dat"):
    source_beam = ShadowBeam()
    source_beam.loadFromFile(file_name)

    shadow_source = ShadowSource.create_src_from_file("source_" + file_name)

    source_beam.history.append(ShadowOEHistoryItem(shadow_source_start=shadow_source,
                                                   shadow_source_end=shadow_source,
                                                   widget_class_name="UndeterminedSource"))

    return source_beam

def __run_source(type=SourceType.GEOMETRICAL, n_rays=50000, random_seed=5676561, aperture=[0.03, 0.07], distance=50500, target_good_rays=500000):
    if type == SourceType.GEOMETRICAL:
        source_beam = run_geometrical_source(n_rays, random_seed, aperture, distance)
    elif type == SourceType.HYBRID_UNDULATOR:
        source_beam = run_hybrid_undulator_source(n_rays, random_seed)
    elif type == SourceType.HYBRID_UNDULATOR_APERTURE:
        source_beam = run_hybrid_undulator_source_through_aperture(n_rays, aperture, distance, target_good_rays)

    return source_beam

def __run_and_save_source(type=SourceType.GEOMETRICAL, n_rays=50000, random_seed=5676561, aperture=[0.03, 0.07], distance=50500, target_good_rays=500000, file_name="begin.dat"):
    source_beam = __run_source(type, n_rays, random_seed, aperture, distance, target_good_rays)
    __save_source_beam(source_beam, file_name)

    return source_beam

def __parse_arguments(arguments):
    type = SourceType.GEOMETRICAL
    n_rays = 100000
    target_good_rays=500000
    random_seed = 0
    aperture = [0.03, 0.07]
    distance = 50500
    file_name = "begin.dat"

    if not arguments is None:
        if arguments[0] == "--h":
            raise Exception("Sintax:\n" + \
                            "python -m beamline34IDC.raytracing <action> -t<source type> -n<nr. rays> -r<random seed> -a<aperture> -N<target good rays> -f<file name>\n\n" + \
                            "action          : run, save, load\n" + \
                            "source type     : 0 (geometrical), 1 (hybrid undulator), 2 (hybrid undulator with aperture); action=run,save\n" + \
                            "nr. rays        : number of rays (integer>0, action=run,save)\n" +
                            "random seed     : shadow seed (integer>0, action=run,save)\n" + \
                            "aperture        : horizontal,vertical,distance (float>0, action=run,save; t=0,2) \n" + \
                            "target good rays: target total good rays (integer>0, action=run,save; t=2) \n"  \
                            "file name       : file name with the initial source raytracing (action=save,load)\n\n" + \
                            "Examples: \n\n" + \
                            "Run the simulation                         : python -m beamline34IDC.raytracing run -t1 -n1000000 -r0 -a0.03,0.07,50500\n" + \
                            "Run the simulation and save the source beam: python -m beamline34IDC.raytracing save -t2 -n500000 -r0 -a0.03,0.07,50500 -N100000 -fbegin_hybrid.dat\n" + \
                            "Load the source beam and run the simulation: python -m beamline34IDC.raytracing load -fbegin_hybrid.dat\n")
        elif len(arguments) > 1:
            for i in range(1, len(arguments)):
                if "-t" == arguments[i][:2]:
                    type = int(arguments[i][2:])
                elif "-n" == arguments[i][:2]:
                    n_rays = int(arguments[i][2:])
                elif "-N" == arguments[i][:2]:
                    target_good_rays = int(arguments[i][2:])
                elif "-r" == arguments[i][:2]:
                    random_seed = int(arguments[i][2:])
                elif "-a" == arguments[i][:2]:
                    values = arguments[i][2:].split(sep=",")
                    aperture = [float(values[0]), float(values[1])]
                    distance = float(values[2])
                elif "-f" == arguments[i][:2]:
                    file_name = arguments[i][2:]

    return type, n_rays, random_seed, aperture, distance, target_good_rays, file_name

def get_source_beam(arguments):
    type, n_rays, random_seed, aperture, distance, target_good_rays, file_name = __parse_arguments(arguments)

    if arguments[0] == "save":
        source_beam = __run_and_save_source(type, n_rays, random_seed, aperture, distance, target_good_rays, file_name)
    else:
        if arguments[0] == "load":
            source_beam = __load_source_beam(file_name=file_name)
        elif arguments[0] == "run":
            source_beam = __run_source(type, n_rays, random_seed, aperture, distance, target_good_rays)
        else:
            raise ValueError("Calculation not recognized")

    return source_beam

import sys

if __name__=="__main__":
    arguments = sys.argv[1:]

    try:
        get_source_beam(arguments)
    except Exception as e:
        print(e)
