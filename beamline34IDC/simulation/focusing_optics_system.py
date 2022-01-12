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
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowMath
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control
from beamline34IDC.util.common import PreProcessorFiles, write_reflectivity_file, write_dabam_file, \
    rotate_axis_system, get_hybrid_input_parameters, plot_shadow_beam_spatial_distribution
from orangecontrib.ml.util.data_structures import DictionaryWrapper

class Movement:
    ABSOLUTE = 0
    RELATIVE = 1

def get_default_input_features():
    return DictionaryWrapper(coh_slits_h_aperture=0.03,
                             coh_slits_h_center=0.0,
                             coh_slits_v_aperture=0.07,
                             coh_slits_v_center=0.0,
                             vkb_q_distance=221,
                             vkb_motor_4_translation=0.0,
                             vkb_motor_3_pitch_angle=0.003,
                             vkb_motor_3_delta_pitch_angle=0.0,
                             hkb_q_distance=120,
                             hkb_motor_4_translation=0.0,
                             hkb_motor_3_pitch_angle=0.003,
                             hkb_motor_3_delta_pitch_angle=0.0)

class FocusingOpticsSystem():

    def __init__(self):
        self.__input_beam = None
        self.__slits_beam = None
        self.__vkb_beam = None
        self.__hkb_beam = None

        self.__coherence_slits = None
        self.__vkb = None
        self.__hkb = None

        self.__modified_elements = None

    def initialize(self,
                   input_beam,
                   input_features=get_default_input_features(),
                   rewrite_preprocessor_files=PreProcessorFiles.YES_SOURCE_RANGE,
                   rewrite_height_error_profile_files=False,
                   **kwargs):
        self.__input_beam         = input_beam.duplicate()
        self.__initial_input_beam = input_beam.duplicate()

        energies = ShadowPhysics.getEnergyFromShadowK(self.__input_beam._beam.rays[:, 10])

        central_energy = numpy.average(energies)
        energy_range = [numpy.min(energies), numpy.max(energies)]

        if rewrite_preprocessor_files==PreProcessorFiles.YES_FULL_RANGE:
            reflectivity_file = write_reflectivity_file()
        elif rewrite_preprocessor_files==PreProcessorFiles.YES_SOURCE_RANGE:
            reflectivity_file = write_reflectivity_file(energy_range=energy_range)
        elif rewrite_preprocessor_files==PreProcessorFiles.NO:
            reflectivity_file = "Pt.dat"

        if rewrite_height_error_profile_files==True:
            vkb_error_profile_file = write_dabam_file(dabam_entry_number=20, heigth_profile_file_name="VKB.dat", seed=8787)
            hkb_error_profile_file = write_dabam_file(dabam_entry_number=62, heigth_profile_file_name="HKB.dat", seed=2345345)
        else:
            vkb_error_profile_file = "VKB.dat"
            hkb_error_profile_file = "HKB.dat"

        coherence_slits = Shadow.OE()

        # COHERENCE SLITS
        coherence_slits.DUMMY = 0.1
        coherence_slits.FWRITE = 3
        coherence_slits.F_REFRAC = 2
        coherence_slits.F_SCREEN = 1
        coherence_slits.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        coherence_slits.N_SCREEN = 1
        coherence_slits.CX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        coherence_slits.CZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_center"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        coherence_slits.RX_SLIT = numpy.array([input_features.get_parameter("coh_slits_h_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        coherence_slits.RZ_SLIT = numpy.array([input_features.get_parameter("coh_slits_v_aperture"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        coherence_slits.T_IMAGE = 0.0
        coherence_slits.T_INCIDENCE = 0.0
        coherence_slits.T_REFLECTION = 180.0
        coherence_slits.T_SOURCE = 0.0

        # V-KB
        vkb = Shadow.OE()

        vkb_motor_3_pitch_angle             = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow              = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle       = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
        vkb_pitch_angle_displacement_shadow = numpy.degrees(vkb_motor_3_delta_pitch_angle)

        vkb_motor_4_translation = input_features.get_parameter("vkb_motor_4_translation")

        vkb.ALPHA = 180.0
        vkb.DUMMY = 0.1
        vkb.FCYL = 1
        vkb.FHIT_C = 1
        vkb.FILE_REFL = reflectivity_file.encode()
        vkb.FILE_RIP = vkb_error_profile_file.encode()
        vkb.FMIRR = 2
        vkb.FWRITE = 1
        vkb.F_DEFAULT = 0
        vkb.F_G_S = 2
        vkb.F_REFLEC = 1
        vkb.F_RIPPLE = 1
        vkb.RLEN1 = 50.0
        vkb.RLEN2 = 50.0
        vkb.RWIDX1 = 10.0
        vkb.RWIDX2 = 10.0
        vkb.SIMAG = input_features.get_parameter("vkb_q_distance")
        vkb.SSOUR = 50667.983
        vkb.THETA = vkb_pitch_angle_shadow
        vkb.T_IMAGE = 101.0
        vkb.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb.T_REFLECTION = vkb_pitch_angle_shadow
        vkb.T_SOURCE = 150.0
        # DISPLACEMENTS
        vkb.F_MOVE = 1
        vkb.OFFY = vkb_motor_4_translation*numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.OFFZ = vkb_motor_4_translation*numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.X_ROT = vkb_pitch_angle_displacement_shadow

        # H-KB
        hkb = Shadow.OE()

        hkb_motor_3_pitch_angle             = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow              = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle       = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
        hkb_pitch_angle_displacement_shadow = numpy.degrees(hkb_motor_3_delta_pitch_angle)

        hkb_motor_4_translation = input_features.get_parameter("hkb_motor_4_translation")

        hkb.ALPHA = 90.0
        hkb.DUMMY = 0.1
        hkb.FCYL = 1
        hkb.FHIT_C = 1
        hkb.FILE_REFL = reflectivity_file.encode()
        hkb.FILE_RIP = hkb_error_profile_file.encode()
        hkb.FMIRR = 2
        hkb.FWRITE = 1
        hkb.F_DEFAULT = 0
        hkb.F_G_S = 2
        hkb.F_REFLEC = 1
        hkb.F_RIPPLE = 1
        hkb.RLEN1 = 50.0
        hkb.RLEN2 = 50.0
        hkb.RWIDX1 = 10.0
        hkb.RWIDX2 = 10.0
        hkb.SIMAG = input_features.get_parameter("hkb_q_distance")
        hkb.SSOUR = 50768.983
        hkb.THETA = hkb_pitch_angle_shadow
        hkb.T_IMAGE = 120.0
        hkb.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb.T_REFLECTION = hkb_pitch_angle_shadow
        hkb.T_SOURCE = 0.0
        # DISPLACEMENT
        hkb.F_MOVE = 1
        hkb.X_ROT = hkb_pitch_angle_displacement_shadow
        hkb.OFFY  = hkb_motor_4_translation*numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb.OFFZ  = hkb_motor_4_translation*numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        self.__coherence_slits = ShadowOpticalElement(coherence_slits)
        self.__vkb             =  ShadowOpticalElement(vkb)
        self.__hkb             =  ShadowOpticalElement(hkb)

        self.__modified_elements = [self.__coherence_slits, self.__vkb, self.__hkb]

    def perturbate_input_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None):
        if self.__input_beam is None: raise ValueError("Focusing Optical System is not initialized")

        good_only = numpy.where(self.__input_beam._beam.rays[:, 9] == 1)

        if not shift_h is None: self.__input_beam._beam.rays[good_only, 0] += shift_h
        if not shift_v is None: self.__input_beam._beam.rays[good_only, 2] += shift_v

        v_out = [self.__input_beam._beam.rays[good_only, 3],
                 self.__input_beam._beam.rays[good_only, 4],
                 self.__input_beam._beam.rays[good_only, 5]]

        if not rotation_h is None: v_out = ShadowMath.vector_rotate([0, 0, 1], rotation_h, v_out)
        if not rotation_v is None: v_out = ShadowMath.vector_rotate([1, 0, 0], rotation_v, v_out)

        if not (rotation_h is None and rotation_v is None):
            self.__input_beam._beam.rays[good_only, 3] = v_out[0]
            self.__input_beam._beam.rays[good_only, 4] = v_out[1]
            self.__input_beam._beam.rays[good_only, 5] = v_out[2]

    def restore_input_beam(self):
        if self.__input_beam is None: raise ValueError("Focusing Optical System is not initialized")
        self.__input_beam = self.__initial_input_beam.duplicate()

    #####################################################################################
    # This methods represent the run-time interface, to interact with the optical system 
    # in real time, like in the real beamline
    
    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None):
        if self.__coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        if not coh_slits_h_center   is None: self.__coherence_slits._oe.CX_SLIT = numpy.array([coh_slits_h_center, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_center   is None: self.__coherence_slits._oe.CZ_SLIT = numpy.array([coh_slits_v_center, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_h_aperture is None: self.__coherence_slits._oe.RX_SLIT = numpy.array([coh_slits_h_aperture, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_aperture is None: self.__coherence_slits._oe.RZ_SLIT = numpy.array([coh_slits_v_aperture, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if not self.__coherence_slits in self.__modified_elements: self.__modified_elements.append(self.__coherence_slits)
        if not self.__vkb             in self.__modified_elements: self.__modified_elements.append(self.__vkb)
        if not self.__hkb             in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_coherence_slits_parameters(self): # center x, center z, aperture x, aperture z
        if self.__coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        return self.__coherence_slits._oe.CX_SLIT, self.__coherence_slits._oe.CZ_SLIT, self.__coherence_slits._oe.RX_SLIT, self.__coherence_slits._oe.RZ_SLIT

    # V-KB -----------------------

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__move_motor_3_pitch(self.__vkb, angle, movement)
        
        if not self.__vkb in self.__modified_elements: self.__modified_elements.append(self.__vkb)
        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_vkb_motor_3_pitch(self):
        return FocusingOpticsSystem.__get_motor_3_pitch(self.__vkb)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__move_motor_4_transation(self.__vkb, translation, movement)

        if not self.__vkb in self.__modified_elements: self.__modified_elements.append(self.__vkb)
        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_vkb_motor_4_translation(self):
        return FocusingOpticsSystem.__get_motor_4_translation(self.__vkb)

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__change_shape(self.__vkb, q_distance, movement)

        if not self.__vkb in self.__modified_elements: self.__modified_elements.append(self.__vkb)
        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_vkb_q_distance(self):
        return FocusingOpticsSystem.__get_q_distance(self.__vkb)

    # H-KB -----------------------

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__move_motor_3_pitch(self.__hkb, angle, movement)

        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_hkb_motor_3_pitch(self):
        return FocusingOpticsSystem.__get_motor_3_pitch(self.__hkb)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__move_motor_4_transation(self.__hkb, translation, movement)

        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_hkb_motor_4_translation(self):
        return FocusingOpticsSystem.__get_motor_4_translation(self.__hkb)

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        FocusingOpticsSystem.__change_shape(self.__hkb, q_distance, movement)

        if not self.__hkb in self.__modified_elements: self.__modified_elements.append(self.__hkb)

    def get_hkb_q_distance(self):
        return FocusingOpticsSystem.__get_q_distance(self.__hkb)

    # PRIVATE -----------------------

    @classmethod
    def __move_motor_3_pitch(cls, element, angle, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if movement == Movement.ABSOLUTE:
            delta_pitch_angle = numpy.degrees(angle - numpy.radians(90 - element._oe.T_INCIDENCE))
        elif movement == Movement.RELATIVE:
            delta_pitch_angle = numpy.degrees(angle)
        else:
            raise ValueError("Movement not recognized")

        element._oe.X_ROT = delta_pitch_angle

    @classmethod
    def __move_motor_4_transation(cls, element, translation, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + element._oe.X_ROT)

        if movement == Movement.ABSOLUTE:
            element._oe.OFFY = translation * numpy.sin(total_pitch_angle)
            element._oe.OFFZ = translation * numpy.cos(total_pitch_angle)
        elif movement == Movement.RELATIVE:
            element._oe.OFFY += translation * numpy.sin(total_pitch_angle)
            element._oe.OFFZ += translation * numpy.cos(total_pitch_angle)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __change_shape(cls, element, q_distance, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if movement == Movement.ABSOLUTE:   element._oe.SIMAG = q_distance
        elif movement == Movement.RELATIVE: element._oe.SIMAG += q_distance
        else: raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_3_pitch(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        return numpy.radians(90 - element._oe.T_INCIDENCE), numpy.radians(element._oe.X_ROT)

    def __get_motor_4_translation(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + element._oe.X_ROT)

        return numpy.average([element._oe.OFFY/numpy.sin(total_pitch_angle), element._oe.OFFZ/numpy.cos(total_pitch_angle)])

    def __get_q_distance(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        return element._oe.SIMAG

    #####################################################################################
    # Run the simulation
    
    def get_beam(self, verbose=False, **kwargs):
        if self.__input_beam is None: raise ValueError("Focusing Optical System is not initialized")

        try: debug_mode = kwargs["debug_mode"]
        except: debug_mode = False

        try: near_field_calculation = kwargs["near_field_calculation"]
        except: near_field_calculation = False

        run_all = self.__modified_elements == [] or len(self.__modified_elements) == 3

        if run_all or self.__coherence_slits in self.__modified_elements:
            # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
            output_beam = ShadowBeam.traceFromOE(self.__input_beam.duplicate(), self.__coherence_slits.duplicate(), widget_class_name="ScreenSlits")
            output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                            diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                            calcType=1,  # Diffraction by Simple Aperture
                                                                            verbose=verbose)).ff_beam


            if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="Coherence Slits", xrange=None, yrange=None)

            self.__slits_beam = output_beam.duplicate()

        if run_all or self.__vkb in self.__modified_elements:
            output_beam = ShadowBeam.traceFromOE(self.__slits_beam.duplicate(), self.__vkb.duplicate(), widget_class_name="EllypticalMirror")

            if not near_field_calculation:
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                verbose=verbose)).ff_beam
            else:
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                nf=1,
                                                                                focal_length=self.__vkb._oe.SIMAG, # at focus
                                                                                image_distance=self.__vkb._oe.SIMAG, # at focus
                                                                                verbose=verbose)).nf_beam
                output_beam._beam.retrace(self.__vkb._oe.T_IMAGE - self.__vkb._oe.SIMAG)

            if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="VKB", xrange=None, yrange=None)

            self.__vkb_beam = output_beam

        if run_all or self.__hkb in self.__modified_elements:
            output_beam = ShadowBeam.traceFromOE(self.__vkb_beam.duplicate(), self.__hkb.duplicate(), widget_class_name="EllypticalMirror")

            if not near_field_calculation:
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                verbose=verbose)).ff_beam
            else:
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                nf=1,
                                                                                verbose=verbose)).nf_beam

            if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="HKB", xrange=None, yrange=None)

            self.__hkb_beam = output_beam

        # after every run, we assume to start again from scratch
        self.__modified_elements = []

        return rotate_axis_system(output_beam, rotation_angle=270.0)

from beamline34IDC.util.common import PreProcessorFiles, plot_shadow_beam_spatial_distribution, load_shadow_beam
from beamline34IDC.util import clean_up

if __name__ == "__main__":

    clean_up()

    input_beam = load_shadow_beam("primary_optics_system_beam.dat")

    # Focusing Optics System -------------------------

    focusing_system = FocusingOpticsSystem()

    focusing_system.initialize(input_beam=input_beam,
                               rewrite_preprocessor_files=PreProcessorFiles.NO,
                               rewrite_height_error_profile_files=False)

    focusing_system.perturbate_input_beam(shift_h=0.01, shift_v=0.01)

    output_beam = focusing_system.get_beam(verbose=False, near_field_calculation=False, debug_mode=False)

    focusing_system.move_vkb_motor_3_pitch(1e-4, movement=Movement.RELATIVE)

    output_beam = focusing_system.get_beam(verbose=False)

    plot_shadow_beam_spatial_distribution(output_beam, xrange=None, yrange=None)

    clean_up()
