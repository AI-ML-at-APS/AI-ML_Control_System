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
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowPreProcessor
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control

from aps.ai.autoalignment.common.simulation.facade.primary_optics_interface import AbstractPrimaryOptics
from aps.ai.autoalignment.common.util.shadow.common import TTYInibitor, rotate_axis_system, HybridFailureException, get_hybrid_input_parameters

def shadow_primary_optics_factory_method():
    return __PrimaryOptics()

class __PrimaryOptics(AbstractPrimaryOptics):
    def __init__(self):
        self.__source_beam = None
        self.__optical_system = None

    def initialize(self, source_photon_beam, **kwargs):
        try: relative_source_position = kwargs["relative_source_position"]
        except: relative_source_position = 0.0
        self.__source_beam = source_photon_beam

        energies = ShadowPhysics.getEnergyFromShadowK(self.__source_beam._beam.rays[:, 10])

        central_energy = numpy.average(energies)

        print("System initialized to central energy: " + str(central_energy))

        #####################################################
        # SHADOW 3 INITIALIZATION
        #
        # Define variables. See meaning of variables in:
        #  https://raw.githubusercontent.com/srio/shadow3/master/docs/source.nml
        #  https://raw.githubusercontent.com/srio/shadow3/master/docs/oe.nml
        #

        white_beam_slits = Shadow.OE()
        white_beam_slits.DUMMY = 0.1
        white_beam_slits.FWRITE = 3
        white_beam_slits.F_REFRAC = 2
        white_beam_slits.F_SCREEN = 1
        white_beam_slits.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        white_beam_slits.N_SCREEN = 1
        white_beam_slits.RX_SLIT = numpy.array([0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        white_beam_slits.RZ_SLIT = numpy.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        white_beam_slits.T_IMAGE = 0.0
        white_beam_slits.T_INCIDENCE = 0.0
        white_beam_slits.T_REFLECTION = 180.0
        white_beam_slits.T_SOURCE = 27000.0 + relative_source_position # only 1 undulator is running

        primary_mirror_1 = Shadow.OE()
        primary_mirror_1.ALPHA = 90.0
        primary_mirror_1.DUMMY = 0.1
        primary_mirror_1.FHIT_C = 1
        primary_mirror_1.FILE_RIP = b'M1.dat'
        primary_mirror_1.FWRITE = 1
        primary_mirror_1.F_G_S = 2
        primary_mirror_1.F_RIPPLE = 1
        primary_mirror_1.RLEN1 = 250.0
        primary_mirror_1.RLEN2 = 250.0
        primary_mirror_1.RWIDX1 = 15.0
        primary_mirror_1.RWIDX2 = 15.0
        primary_mirror_1.T_IMAGE = 685.0
        primary_mirror_1.T_INCIDENCE = 89.8567605512
        primary_mirror_1.T_REFLECTION = 89.8567605512
        primary_mirror_1.T_SOURCE = 1684.5

        primary_mirror_2 = Shadow.OE()
        primary_mirror_2.ALPHA = 180.0
        primary_mirror_2.DUMMY = 0.1
        primary_mirror_2.FHIT_C = 1
        primary_mirror_2.FILE_RIP = b'M2.dat'
        primary_mirror_2.FWRITE = 1
        primary_mirror_2.F_G_S = 2
        primary_mirror_2.F_RIPPLE = 1
        primary_mirror_2.RLEN1 = 250.0
        primary_mirror_2.RLEN2 = 250.0
        primary_mirror_2.RWIDX1 = 15.0
        primary_mirror_2.RWIDX2 = 15.0
        primary_mirror_2.T_IMAGE = 0.0
        primary_mirror_2.T_INCIDENCE = 89.8567605512
        primary_mirror_2.T_REFLECTION = 89.8567605512
        primary_mirror_2.T_SOURCE = 0.0

        ml_mono_1 = Shadow.OE()
        ml_mono_1.DUMMY = 0.1
        ml_mono_1.FHIT_C = 1
        ml_mono_1.FWRITE = 1
        ml_mono_1.F_ANGLE = 1
        ml_mono_1.RLEN1 = 80.0
        ml_mono_1.RLEN2 = 80.0
        ml_mono_1.RWIDX1 = 15.0
        ml_mono_1.RWIDX2 = 15.0
        ml_mono_1.T_IMAGE = 100.0
        ml_mono_1.T_INCIDENCE = 89.3942
        ml_mono_1.T_REFLECTION = 89.3942
        ml_mono_1.T_SOURCE = 1930.5

        ml_mono_2 = Shadow.OE()
        ml_mono_2.ALPHA = 180.0
        ml_mono_2.DUMMY = 0.1
        ml_mono_2.FHIT_C = 1
        ml_mono_2.FWRITE = 1
        ml_mono_2.F_ANGLE = 1
        ml_mono_2.RLEN1 = 80.0
        ml_mono_2.RLEN2 = 80.0
        ml_mono_2.RWIDX1 = 15.0
        ml_mono_2.RWIDX2 = 15.0
        ml_mono_2.T_IMAGE = 0.0
        ml_mono_2.T_INCIDENCE = 89.3942
        ml_mono_2.T_REFLECTION = 89.3942
        ml_mono_2.T_SOURCE = 100.0

        slits_screen = Shadow.OE() # restore lab system
        slits_screen.ALPHA = 90.0
        slits_screen.DUMMY = 0.1
        slits_screen.FWRITE = 3
        slits_screen.F_REFRAC = 2
        slits_screen.F_SCREEN = 1
        slits_screen.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        slits_screen.N_SCREEN = 1
        slits_screen.RX_SLIT = numpy.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        slits_screen.RZ_SLIT = numpy.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        slits_screen.T_IMAGE = 0.0
        slits_screen.T_INCIDENCE = 0.0
        slits_screen.T_REFLECTION = 180.0
        slits_screen.T_SOURCE = 31000.0

        self.__white_beam_slits = ShadowOpticalElement(white_beam_slits)
        self.__primary_mirror_1 = ShadowOpticalElement(primary_mirror_1)
        self.__primary_mirror_2 = ShadowOpticalElement(primary_mirror_2)
        self.__ml_mono_1 = ShadowOpticalElement(ml_mono_1)
        self.__ml_mono_2 = ShadowOpticalElement(ml_mono_2)
        self.__slits_screen = ShadowOpticalElement(slits_screen)

    def get_photon_beam(self, **kwargs):
        try:    verbose = kwargs["verbose"]
        except: verbose = False
        try:    random_seed = kwargs["random_seed"]
        except: random_seed = None

        if self.__source_beam is None: raise ValueError("Primary Optical System is not initialized")

        input_beam = self.__source_beam.duplicate()

        if not verbose:
            fortran_suppressor = TTYInibitor()
            fortran_suppressor.start()

        try:
            if verbose: print("White Beam Slits:")
            output_beam = ShadowBeam.traceFromOE(input_beam, self.__white_beam_slits, widget_class_name="ScreenSlits", recursive_history=False)

            # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
            try:
                if verbose: print("Hybrid calculation")
                output_beam =  hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=1,  # Sagittal
                                                                                calcType=1,  # Diffraction by Simple Aperture
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 100))).ff_beam
            except Exception:
                raise HybridFailureException(oe="White Beam Slits")


            if verbose: print("Primary Mirror 1:")
            output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), self.__primary_mirror_1, widget_class_name="PlaneMirror", recursive_history=False)

            # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
            try:
                if verbose: print("Hybrid calculation")
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 200))).ff_beam
            except Exception:
                raise HybridFailureException(oe="Primary Mirror 1")

            if verbose: print("Primary Mirror 2:")
            output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), self.__primary_mirror_2, widget_class_name="PlaneMirror", recursive_history=False)

            # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
            try:
                if verbose: print("Hybrid calculation")
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 300))).ff_beam
            except Exception:
                raise HybridFailureException(oe="Primary Mirror 2")

            if verbose: print("ML Mono 1:")
            output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), self.__ml_mono_1, widget_class_name="PlaneMirror", recursive_history=False)
            output_beam = ShadowPreProcessor.apply_user_reflectivity(2, 1, 0, "xoppy_HR_multilayer_mono_reflectivity.dat", output_beam)

            if verbose: print("ML Mono 2:")
            output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), self.__ml_mono_2, widget_class_name="PlaneMirror", recursive_history=False)
            output_beam = ShadowPreProcessor.apply_user_reflectivity(2, 1, 0, "xoppy_HR_multilayer_mono_reflectivity.dat", output_beam)

            if verbose: print("Experimental-Hutch Slits:")
            output_beam = ShadowBeam.traceFromOE(output_beam.duplicate(), self.__slits_screen, widget_class_name="ScreenSlits", recursive_history=False)

            # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
            try:
                if verbose: print("Hybrid calculation")

                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=3,  # Both
                                                                                calcType=1,  # Diffraction by Simple Aperture
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 400))).ff_beam
            except Exception as e:
                print(e)
                raise HybridFailureException(oe="Experimental-Hutch Slits")

        except Exception as e:
            if not verbose:
                try: fortran_suppressor.stop()
                except: pass

            raise e
        else:
            if not verbose:
                try: fortran_suppressor.stop()
                except: pass

        if verbose: print("Finished")

        return output_beam
