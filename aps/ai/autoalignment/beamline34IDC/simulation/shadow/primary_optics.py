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
import Shadow

from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowOpticalElement
from orangecontrib.shadow.util.shadow_util import ShadowPhysics

from aps.ai.autoalignment.common.simulation.facade.primary_optics_interface import AbstractPrimaryOptics
from aps.ai.autoalignment.common.util.shadow.common import write_bragg_file, write_reflectivity_file, PreProcessorFiles, TTYInibitor, rotate_axis_system

def shadow_primary_optics_factory_method():
    return __PrimaryOptics()

class __PrimaryOptics(AbstractPrimaryOptics):
    def __init__(self):
        self.__source_beam = None
        self.__optical_system = None

    def initialize(self, source_photon_beam, **kwargs):
        try:    rewrite_preprocessor_files = kwargs["rewrite_preprocessor_files"]
        except: rewrite_preprocessor_files = PreProcessorFiles.YES_SOURCE_RANGE

        self.__source_beam = source_photon_beam

        energies = ShadowPhysics.getEnergyFromShadowK(self.__source_beam._beam.rays[:, 10])

        central_energy = numpy.average(energies)
        energy_range = [max(numpy.min(energies)-100.0, 0.0), numpy.max(energies)+100.0]

        if rewrite_preprocessor_files==PreProcessorFiles.YES_FULL_RANGE:
            reflectivity_file = write_reflectivity_file()
            bragg_file        = write_bragg_file()
        elif rewrite_preprocessor_files==PreProcessorFiles.YES_SOURCE_RANGE:
            reflectivity_file = write_reflectivity_file(energy_range=energy_range)
            bragg_file        = write_bragg_file(energy_range=energy_range)
        elif rewrite_preprocessor_files==PreProcessorFiles.NO:
            reflectivity_file = "Pt.dat"
            bragg_file        = "Si111.dat"

        print("System initialized to central energy: " + str(central_energy))

        #####################################################
        # SHADOW 3 INITIALIZATION

        #
        # Define variables. See meaning of variables in:
        #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
        #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
        #

        # WB SLITS
        white_beam_slits = Shadow.OE()
        white_beam_slits.DUMMY = 0.1
        white_beam_slits.FWRITE = 3
        white_beam_slits.F_REFRAC = 2
        white_beam_slits.F_SCREEN = 1
        white_beam_slits.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        white_beam_slits.N_SCREEN = 1
        white_beam_slits.RX_SLIT = numpy.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        white_beam_slits.RZ_SLIT = numpy.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        white_beam_slits.T_IMAGE = 0.0
        white_beam_slits.T_INCIDENCE = 0.0
        white_beam_slits.T_REFLECTION = 180.0
        white_beam_slits.T_SOURCE = 26800.0

        # PLANE MIRROR
        mirror_1 = Shadow.OE()
        mirror_1.DUMMY = 0.1
        mirror_1.FHIT_C = 1
        mirror_1.FILE_REFL = reflectivity_file.encode()
        mirror_1.FWRITE = 1
        mirror_1.F_REFLEC = 1
        mirror_1.RLEN1 = 250.0
        mirror_1.RLEN2 = 250.0
        mirror_1.RWIDX1 = 10.0
        mirror_1.RWIDX2 = 10.0
        mirror_1.T_IMAGE = 0.0
        mirror_1.T_INCIDENCE = 89.7135211024
        mirror_1.T_REFLECTION = 89.7135211024
        mirror_1.T_SOURCE = 2800.0

        # DCM-1
        dcm_1 = Shadow.OE()
        dcm_1.DUMMY = 0.1
        dcm_1.FHIT_C = 1
        dcm_1.FILE_REFL = bragg_file.encode()
        dcm_1.FWRITE = 1
        dcm_1.F_CENTRAL = 1
        dcm_1.F_CRYSTAL = 1
        dcm_1.F_PHOT_CENT = 0
        dcm_1.PHOT_CENT = central_energy
        dcm_1.RLEN1 = 50.0
        dcm_1.RLEN2 = 50.0
        dcm_1.RWIDX1 = 50.0
        dcm_1.RWIDX2 = 50.0
        dcm_1.T_IMAGE = 0.0
        dcm_1.T_INCIDENCE = 0.0
        dcm_1.T_REFLECTION = 0.0
        dcm_1.T_SOURCE = 15400.0

        # DCM-2
        dcm_2 = Shadow.OE()
        dcm_2.ALPHA = 180.0
        dcm_2.DUMMY = 0.1
        dcm_2.FHIT_C = 1
        dcm_2.FILE_REFL = bragg_file.encode()
        dcm_2.FWRITE = 1
        dcm_2.F_CENTRAL = 1
        dcm_2.F_CRYSTAL = 1
        dcm_2.F_PHOT_CENT = 0
        dcm_2.PHOT_CENT = central_energy
        dcm_2.RLEN1 = 50.0
        dcm_2.RLEN2 = 50.0
        dcm_2.RWIDX1 = 50.0
        dcm_2.RWIDX2 = 50.0
        dcm_2.T_IMAGE = 5490.0
        dcm_2.T_INCIDENCE = 0.0
        dcm_2.T_REFLECTION = 0.0
        dcm_2.T_SOURCE = 10

        self.__optical_system = [[ShadowOpticalElement(white_beam_slits), "ScreenSlits", False],
                                 [ShadowOpticalElement(mirror_1), "PlaneMirror", False],
                                 [ShadowOpticalElement(dcm_1), "PlaneCrystal", False],
                                 [ShadowOpticalElement(dcm_2), "PlaneCrystal", True]]

    def get_photon_beam(self, **kwargs):
        try:    verbose = kwargs["verbose"]
        except: verbose = False

        if self.__source_beam is None: raise ValueError("Primary Optical System is not initialized")

        input_beam = self.__source_beam.duplicate()

        if not verbose:
            fortran_suppressor = TTYInibitor()
            fortran_suppressor.start()

        output_beam = None

        try:
            for optical_element_data in self.__optical_system:
                optical_element   = optical_element_data[0]
                widget_class_name = optical_element_data[1]
                is_last_element   = optical_element_data[2]

                output_beam = ShadowBeam.traceFromOE(input_beam, optical_element, widget_class_name=widget_class_name, recursive_history=False)

                if not is_last_element: input_beam = output_beam.duplicate()
        except Exception as e:
            if not verbose:
                try: fortran_suppressor.stop()
                except: pass

            raise e
        else:
            if not verbose:
                try: fortran_suppressor.stop()
                except: pass

        output_beam = rotate_axis_system(output_beam, rotation_angle=180.0)

        return output_beam
