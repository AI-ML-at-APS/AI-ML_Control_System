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
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowMath, ShadowCongruence
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control
from orangecontrib.ml.util.mocks import MockWidget
from beamline34IDC.util.shadow.common import TTYInibitor, EmptyBeamException, PreProcessorFiles, write_reflectivity_file, write_dabam_file, rotate_axis_system, get_hybrid_input_parameters, plot_shadow_beam_spatial_distribution
from beamline34IDC.simulation.facade.focusing_optics_interface import AbstractFocusingOptics, Movement, AngularUnits, DistanceUnits, get_default_input_features, MotorResolution

def shadow_focusing_optics_factory_method(**kwargs):
    try:
        if kwargs["bender"] == True: return __FocusingOpticsWithBender()
        else:                        return __FocusingOptics()
    except: return __FocusingOptics()

class _FocusingOpticsCommon(AbstractFocusingOptics):
    def __init__(self):
        self._input_beam = None
        self._slits_beam = None
        self._vkb_beam = None
        self._hkb_beam = None
        self._coherence_slits = None
        self._vkb = None
        self._hkb = None
        self._modified_elements = None

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):
        try:    rewrite_preprocessor_files = kwargs["rewrite_preprocessor_files"]
        except: rewrite_preprocessor_files = PreProcessorFiles.YES_SOURCE_RANGE
        try:    rewrite_height_error_profile_files = kwargs["rewrite_height_error_profile_files"]
        except: rewrite_height_error_profile_files = False

        self._input_beam = input_photon_beam.duplicate()
        self.__initial_input_beam = input_photon_beam.duplicate()

        energies     = ShadowPhysics.getEnergyFromShadowK(self._input_beam._beam.rays[:, 10])
        energy_range = [numpy.min(energies), numpy.max(energies)]

        if rewrite_preprocessor_files == PreProcessorFiles.YES_FULL_RANGE:     reflectivity_file = write_reflectivity_file()
        elif rewrite_preprocessor_files == PreProcessorFiles.YES_SOURCE_RANGE: reflectivity_file = write_reflectivity_file(energy_range=energy_range)
        elif rewrite_preprocessor_files == PreProcessorFiles.NO:               reflectivity_file = "Pt.dat"

        if rewrite_height_error_profile_files == True:
            vkb_error_profile_file = write_dabam_file(dabam_entry_number=92, heigth_profile_file_name="VKB-LTP_shadow.dat", seed=8787)
            hkb_error_profile_file = write_dabam_file(dabam_entry_number=93, heigth_profile_file_name="HKB-LTP_shadow.dat", seed=2345345)
        else:
            vkb_error_profile_file = "VKB-LTP_shadow.dat"
            hkb_error_profile_file = "HKB-LTP_shadow.dat"

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

        vkb_motor_3_pitch_angle = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
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
        vkb.RWIDX1 = 20.95
        vkb.RWIDX2 = 20.95
        vkb.SIMAG = input_features.get_parameter("vkb_q_distance")
        vkb.SSOUR = 50667.983
        vkb.THETA = vkb_pitch_angle_shadow
        vkb.T_IMAGE = 101.0
        vkb.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb.T_REFLECTION = vkb_pitch_angle_shadow
        vkb.T_SOURCE = 150.0
        # DISPLACEMENTS
        vkb.F_MOVE = 1
        vkb.OFFY = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.OFFZ = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb.X_ROT = vkb_pitch_angle_displacement_shadow

        # H-KB
        hkb = Shadow.OE()

        hkb_motor_3_pitch_angle = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
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
        hkb.RWIDX1 = 24.75
        hkb.RWIDX2 = 24.75
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
        hkb.OFFY = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb.OFFZ = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        self._coherence_slits = ShadowOpticalElement(coherence_slits)
        self._vkb = ShadowOpticalElement(vkb)
        self._hkb = ShadowOpticalElement(hkb)

        self._modified_elements = [self._coherence_slits, self._vkb, self._hkb]

    def perturbate_input_photon_beam(self, shift_h=None, shift_v=None, rotation_h=None, rotation_v=None):
        if self._input_beam is None: raise ValueError("Focusing Optical System is not initialized")

        good_only = numpy.where(self._input_beam._beam.rays[:, 9] == 1)

        if not shift_h is None: self._input_beam._beam.rays[good_only, 0] += shift_h
        if not shift_v is None: self._input_beam._beam.rays[good_only, 2] += shift_v

        v_out = [self._input_beam._beam.rays[good_only, 3],
                 self._input_beam._beam.rays[good_only, 4],
                 self._input_beam._beam.rays[good_only, 5]]

        if not rotation_h is None: v_out = ShadowMath.vector_rotate([0, 0, 1], rotation_h, v_out)
        if not rotation_v is None: v_out = ShadowMath.vector_rotate([1, 0, 0], rotation_v, v_out)

        if not (rotation_h is None and rotation_v is None):
            self._input_beam._beam.rays[good_only, 3] = v_out[0]
            self._input_beam._beam.rays[good_only, 4] = v_out[1]
            self._input_beam._beam.rays[good_only, 5] = v_out[2]

    def restore_input_photon_beam(self):
        if self._input_beam is None: raise ValueError("Focusing Optical System is not initialized")
        self._input_beam = self.__initial_input_beam.duplicate()

        #####################################################################################
        # This methods represent the run-time interface, to interact with the optical system
        # in real time, like in the real beamline

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None):
        if self._coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        if not coh_slits_h_center is None: self._coherence_slits._oe.CX_SLIT = numpy.array([coh_slits_h_center, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_center is None: self._coherence_slits._oe.CZ_SLIT = numpy.array([coh_slits_v_center, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_h_aperture is None: self._coherence_slits._oe.RX_SLIT = numpy.array([coh_slits_h_aperture, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_aperture is None: self._coherence_slits._oe.RZ_SLIT = numpy.array([coh_slits_v_aperture, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if not self._coherence_slits in self._modified_elements: self._modified_elements.append(self._coherence_slits)
        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_coherence_slits_parameters(self):  # center x, center z, aperture x, aperture z
        if self._coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        return self._coherence_slits._oe.CX_SLIT, self._coherence_slits._oe.CZ_SLIT, self._coherence_slits._oe.RX_SLIT, self._coherence_slits._oe.RZ_SLIT

        # V-KB -----------------------

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self.__move_motor_3_pitch(self._vkb, angle, movement, units,
                                  round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution()[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self.__get_motor_3_pitch(self._vkb, units)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE):
        self.__move_motor_4_transation(self._vkb, translation, movement,
                                       round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution()[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_4_translation(self):
        return self.__get_motor_4_translation(self._vkb)

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        self.__change_shape(self._vkb, q_distance, movement)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_q_distance(self):
        return self.__get_q_distance(self._vkb)

        # H-KB -----------------------

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self.__move_motor_3_pitch(self._hkb, angle, movement, units,
                                  round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution()[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self.__get_motor_3_pitch(self._hkb, units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE):
        self.__move_motor_4_transation(self._hkb, translation, movement,
                                       round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution()[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_4_translation(self):
        return self.__get_motor_4_translation(self._hkb)

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        self.__change_shape(self._hkb, q_distance, movement)

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_q_distance(self):
        return self.__get_q_distance(self._hkb)

        # PRIVATE -----------------------

    @classmethod
    def __move_motor_3_pitch(cls, element, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, round_digit=4):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == AngularUnits.MILLIRADIANS:
            angle = numpy.degrees(angle * 1e-3)
        elif units == AngularUnits.DEGREES:
            pass
        elif units == AngularUnits.RADIANS:
            angle = numpy.degrees(angle)
        else:
            raise ValueError("Angular units not recognized")

        if movement == Movement.ABSOLUTE:
            element._oe.X_ROT = round(angle - (90 - element._oe.T_INCIDENCE), round_digit)
        elif movement == Movement.RELATIVE:
            element._oe.X_ROT += round(angle, round_digit)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __move_motor_4_transation(cls, element, translation, movement=Movement.ABSOLUTE, round_digit=3):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + element._oe.X_ROT)

        if movement == Movement.ABSOLUTE:
            element._oe.OFFY = round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ = round(translation, round_digit) * numpy.cos(total_pitch_angle)
        elif movement == Movement.RELATIVE:
            element._oe.OFFY += round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ += round(translation, round_digit) * numpy.cos(total_pitch_angle)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __change_shape(cls, element, q_distance, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if movement == Movement.ABSOLUTE:
            element._oe.SIMAG = q_distance
        elif movement == Movement.RELATIVE:
            element._oe.SIMAG += q_distance
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_3_pitch(cls, element, units):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        angle = 90 - element._oe.T_INCIDENCE + element._oe.X_ROT

        if units == AngularUnits.MILLIRADIANS:
            return 1000 * numpy.radians(angle)
        elif units == AngularUnits.DEGREES:
            return angle
        elif units == AngularUnits.RADIANS:
            return numpy.radians(angle)
        else:
            raise ValueError("Angular units not recognized")

    @classmethod
    def __get_motor_4_translation(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + element._oe.X_ROT)

        return numpy.average([element._oe.OFFY / numpy.sin(total_pitch_angle), element._oe.OFFZ / numpy.cos(total_pitch_angle)])

    @classmethod
    def __get_q_distance(cls, element):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        return element._oe.SIMAG

    #####################################################################################
    # Run the simulation

    def get_photon_beam(self, near_field_calculation=False, remove_lost_rays=True, **kwargs):
        try:    verbose = kwargs["verbose"]
        except: verbose = False
        try:    debug_mode = kwargs["debug_mode"]
        except: debug_mode = False
        try:    random_seed = kwargs["random_seed"]
        except: random_seed = None

        if self._input_beam is None: raise ValueError("Focusing Optical System is not initialized")

        self._check_beam(self._input_beam, "Primary Optical System", remove_lost_rays)

        if not verbose:
            fortran_suppressor = TTYInibitor()
            fortran_suppressor.start()

        output_beam = None

        try:
            run_all = self._modified_elements == [] or len(self._modified_elements) == 3

            if run_all or self._coherence_slits in self._modified_elements:
                # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
                output_beam = self._trace_coherence_slits(remove_lost_rays)

                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                                calcType=1,  # Diffraction by Simple Aperture
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 100))).ff_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="Coherence Slits", xrange=None, yrange=None)

                self._slits_beam = output_beam.duplicate()

            if run_all or self._vkb in self._modified_elements:
                output_beam = self._trace_vkb(remove_lost_rays)

                # NOTE: Near field not possible for vkb (beam is untraceable)
                output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                diffraction_plane=2,  # Tangential
                                                                                calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                verbose=verbose,
                                                                                random_seed=None if random_seed is None else (random_seed + 200))).ff_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="VKB", xrange=None, yrange=None)

                self._vkb_beam = output_beam

            if run_all or self._hkb in self._modified_elements:
                output_beam = self._trace_hkb(remove_lost_rays)

                if not near_field_calculation:
                    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                    diffraction_plane=2,  # Tangential
                                                                                    calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                    verbose=verbose,
                                                                                    random_seed=None if random_seed is None else (random_seed + 300))).ff_beam
                else:
                    output_beam = hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                                    diffraction_plane=2,  # Tangential
                                                                                    calcType=3,  # Diffraction by Mirror Size + Errors
                                                                                    nf=1,
                                                                                    verbose=verbose,
                                                                                    random_seed=None if random_seed is None else (random_seed + 300))).nf_beam

                output_beam = rotate_axis_system(output_beam, rotation_angle=270.0)

                if debug_mode: plot_shadow_beam_spatial_distribution(output_beam, title="HKB", xrange=None, yrange=None)

                self._hkb_beam = output_beam

            # after every run, we assume to start again from scratch
            self._modified_elements = []

        except Exception as e:
            if not verbose:
                try:    fortran_suppressor.stop()
                except: pass

            raise e
        else:
            if not verbose:
                try:    fortran_suppressor.stop()
                except: pass

        return output_beam

    def _trace_coherence_slits(self, remove_lost_rays):
        return self._trace_oe(input_beam=self._input_beam,
                              shadow_oe=self._coherence_slits,
                              widget_class_name="ScreenSlits",
                              oe_name="Coherence Slits",
                              remove_lost_rays=remove_lost_rays)

    def _trace_vkb(self, remove_lost_rays):
        return self._trace_oe(input_beam=self._slits_beam,
                              shadow_oe=self._vkb,
                              widget_class_name="EllypticalMirror",
                              oe_name="V-KB",
                              remove_lost_rays=remove_lost_rays)



    def _trace_hkb(self, remove_lost_rays):
        return self._trace_oe(input_beam=self._vkb_beam,
                              shadow_oe=self._hkb,
                              widget_class_name="EllypticalMirror",
                              oe_name="H-KB",
                              remove_lost_rays=remove_lost_rays)

    def _trace_oe(self, input_beam, shadow_oe, widget_class_name, oe_name, remove_lost_rays):
        return self._check_beam(ShadowBeam.traceFromOE(input_beam.duplicate(),
                                                       shadow_oe.duplicate(),
                                                       widget_class_name=widget_class_name),
                                oe_name, remove_lost_rays)

    def _check_beam(self, output_beam, oe, remove_lost_rays):
        if ShadowCongruence.checkEmptyBeam(output_beam):
            if ShadowCongruence.checkGoodBeam(output_beam):
                if remove_lost_rays: output_beam._beam.rays = output_beam._beam.rays[numpy.where(output_beam._beam.rays[:, 9] == 1)]
                return output_beam
            else: raise EmptyBeamException(oe)
        else: raise EmptyBeamException(oe)


class __FocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        super(_FocusingOpticsCommon, self).__init__()

from oasys.widgets import congruence
from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.double_rod_bendable_ellispoid_mirror_bl import *

class _KBMockWidget(MockWidget):
    dim_x_minus = 0.0
    dim_x_plus = 0.0
    dim_y_minus = 0.0
    dim_y_plus = 0.0
    object_side_focal_distance        = 0.0
    image_side_focal_distance         = 0.0
    incidence_angle_respect_to_normal = 0.0

    modified_surface    = 1
    ms_type_of_defect   = 2
    ms_defect_file_name = "error_profile.dat"

    bender_bin_x = 10
    bender_bin_y = 200

    E = 131000
    h = 10
    r = 10
    output_file_name_full = "mirror_bender.dat"
    which_length     = 1 # 0 - full length, 1 - partial length
    optimized_length = 72.0 # only optically active surface
    n_fit_steps      = 5

    R0         = 45
    eta        = 0.25
    W2         = 40.0
    R0_fixed   = False
    eta_fixed  = True
    W2_fixed   = True
    R0_min     = 20.0
    eta_min    = 0.0
    W2_min     = 1.0
    R0_max     = 300.0
    eta_max    = 2.0
    W2_max     = 42.0

    R0_out  = 0.0
    eta_out = 0.0
    W2_out  = 0.0
    alpha   = 0.0
    W0      = 0.0
    F_upstream        = 0.0 # output of bender calculation
    F_downstream      = 0.0 # output of bender calculation

    F_upstream_apparent   = 0.0 # for positioning and repeatability
    F_downstream_apparent = 0.0
    C_upstream            = 0.0
    C_downstream          = 0.0
    K_upstream            = 0.0
    K_downstream          = 0.0

    def __init__(self, shadow_oe, verbose=False, workspace_units=2):
        super(_KBMockWidget, self).__init__(verbose=verbose, workspace_units=workspace_units)
        self.dim_x_minus = shadow_oe._oe.RWIDX2
        self.dim_x_plus  = shadow_oe._oe.RWIDX1
        self.dim_y_minus = shadow_oe._oe.RLEN2
        self.dim_y_plus  = shadow_oe._oe.RLEN1

        self.object_side_focal_distance        = shadow_oe._oe.SSOUR
        self.image_side_focal_distance         = shadow_oe._oe.SIMAG
        self.incidence_angle_respect_to_normal = shadow_oe._oe.THETA

        self.modified_surface    = int(shadow_oe._oe.F_RIPPLE)
        self.ms_type_of_defect   = int(shadow_oe._oe.F_G_S)
        self.ms_defect_file_name = shadow_oe._oe.FILE_RIP.decode('utf-8')

        self.initialize_bender_parameters()
        self.calculate_bender_quantities()

        self.R0_out = self.R0

    def manage_acceptance_slits(self, shadow_oe): pass # do nothing
    def initialize_bender_parameters(self): pass

    def calculate_bender_quantities(self):
        W1 = self.dim_x_plus + self.dim_x_minus
        L = self.dim_y_plus + self.dim_y_minus

        p = self.object_side_focal_distance
        q = self.image_side_focal_distance
        grazing_angle = numpy.radians(90 - self.incidence_angle_respect_to_normal)

        self.alpha = calculate_taper_factor(W1, self.W2, L, p, q, grazing_angle)
        self.W0 = calculate_W0(W1, self.alpha, L, p, q, grazing_angle)  # W at the center

    def get_positions(self): 
        return (self.F_upstream_apparent - self.C_upstream)/self.K_upstream, (self.F_downstream_apparent - self.C_downstream)/self.K_downstream
    
    def set_positions(self, pos_upstream, pos_downstream):
        self.F_upstream_apparent   = self.C_upstream   + pos_upstream * self.K_upstream
        self.F_downstream_apparent = self.C_downstream + pos_downstream * self.K_downstream
    
class VKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units)

    def initialize_bender_parameters(self):
        self.output_file_name_full = congruence.checkFileName("VKB_bender_profile.dat")
        self.R0  = 146.36857
        self.eta = 0.39548
        self.W2  = 21.0

        # F = C + KX - with X in micron!
        #
        # from beamtime:
        # q = 221.0
        # X1 = 142.5000 = (209.379473 - C_upstream)/K_upstream
        # X2 = 299.5000 = (259.750158 - C_downstream)/K_downstream
        #
        # q = 225.0
        # X1 = 139.0000 = (205.946389 - C_upstream)/K_upstream
        # X2 = 296.0000 = (254.506535 - C_downstream)/K_downstream

        # -> K = (Fa-Fb)/(Xa-Xb)
        # -> C = F - KX

        self.C_upstream   = 69.60391014
        self.C_downstream = -188.954153
        self.K_upstream   = 0.980881143
        self.K_downstream = 1.498178

        self.set_positions(142.5, 299.5) # from beamline calibration

class HKBMockWidget(_KBMockWidget):
    def __init__(self, shadow_oe, verbose=False, workspace_units=2):
        super().__init__(shadow_oe=shadow_oe, verbose=verbose, workspace_units=workspace_units)

    def initialize_bender_parameters(self):
        self.output_file_name_full = congruence.checkFileName("HKB_bender_profile.dat")
        self.R0  = 79.57061
        self.eta = 0.36055
        self.W2  = 2.5

        # F = C + KX, with X in micron
        #
        # from beamtime:
        # q = 120.0
        # X1 = 250.0515 = (292.400729 - C_upstream)/K_upstream
        # X2 = 157.0341 = (421.011757 - C_downstream)/k2
        #
        # q = 124.0
        # X1 = 248.0515 = (284.169317 - C_upstream)/K_upstream
        # X2 = 155.0341 = (404.275779 - C_downstream)/k2

        # -> K = (Fa-Fb)/(Xa-Xb)
        # -> C = F - KX

        self.C_upstream    = -736.7377299
        self.C_downstream  = -893.0478644
        self.K_upstream    = 4.115706
        self.K_downstream  = 8.367989

        self.set_positions(250.0515, 157.0341) # from beamline calibration

class __FocusingOpticsWithBender(_FocusingOpticsCommon):
    def __init__(self):
        super(_FocusingOpticsCommon, self).__init__()

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):

        super().initialize(input_photon_beam, input_features, **kwargs)

        self.__vkb_widget = VKBMockWidget(self._vkb)
        self.__hkb_widget = HKBMockWidget(self._hkb)

    def _trace_vkb(self, remove_lost_rays):
        return self.__trace_kb(widget=self.__vkb_widget,
                               input_beam=self._slits_beam,
                               shadow_oe=self._vkb,
                               widget_class_name="DoubleRodBenderEllypticalMirror",
                               oe_name="V-KB",
                               remove_lost_rays=remove_lost_rays)

    def _trace_hkb(self, remove_lost_rays):
        return self.__trace_kb(widget=self.__hkb_widget,
                               input_beam=self._vkb_beam,
                               shadow_oe=self._hkb,
                               widget_class_name="DoubleRodBenderEllypticalMirror",
                               oe_name="H-KB",
                               remove_lost_rays=remove_lost_rays)

    def __trace_kb(self, widget, input_beam, shadow_oe, widget_class_name, oe_name, remove_lost_rays):
        widget.R0              = widget.R0_out  # use last fit result
        shadow_oe._oe.FILE_RIP = bytes(widget.ms_defect_file_name, 'utf-8') # restore original error profile

        apply_bender_surface(widget=widget, shadow_oe=shadow_oe, input_beam=input_beam.duplicate())

        # Redo raytracing with the bender correction as error profile
        return self._trace_oe(input_beam=input_beam,
                              shadow_oe=shadow_oe,
                              widget_class_name=widget_class_name,
                              oe_name=oe_name,
                              remove_lost_rays=remove_lost_rays)

    def move_vkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_widget, self._vkb, pos_upstream, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_resolution()[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def move_hkb_motor_1_2_bender(self, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_widget, self._hkb, pos_upstream, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_resolution()[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    @classmethod
    def __move_motor_1_2_bender(cls, widget, element, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=2):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MILLIMETERS:
            pos_upstream   = round(pos_upstream,   round_digit)*1e3
            pos_downstream = round(pos_downstream, round_digit)*1e3
        else:
            pos_upstream   = round(pos_upstream,   round_digit - 3)
            pos_downstream = round(pos_downstream, round_digit - 3)

        if movement == Movement.ABSOLUTE:
            widget.set_positions(pos_upstream, pos_downstream)
        elif movement == Movement.RELATIVE:
            current_pos_upstream, current_pos_downstream = widget.get_positions()
            widget.set_positions(current_pos_upstream + pos_upstream, current_pos_downstream + pos_downstream)
        else:
            raise ValueError("Movement not recognized")

        set_q_from_forces(widget, widget.F_upstream_apparent, widget.F_downstream_apparent)

        widget.image_side_focal_distance = round(widget.image_side_focal_distance, int(2*widget.workspace_units_to_mm))

        element._oe.SIMAG = widget.image_side_focal_distance

    def get_vkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_widget, units)

    def get_hkb_motor_1_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_widget, units)

    @classmethod
    def __get_motor_1_2_bender(cls, widget, units=DistanceUnits.MICRON):
        if widget is None: raise ValueError("Initialize Focusing Optics System first")

        return widget.get_positions()
