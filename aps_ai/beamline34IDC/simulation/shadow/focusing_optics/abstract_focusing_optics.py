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

import os.path

import numpy
import Shadow

from orangecontrib.shadow.util.shadow_objects import ShadowOpticalElement, ShadowBeam
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowMath, ShadowCongruence
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control

from aps_ai.common.util.shadow.common import TTYInibitor, HybridFailureException, EmptyBeamException, PreProcessorFiles, write_reflectivity_file, write_dabam_file, get_hybrid_input_parameters, plot_shadow_beam_spatial_distribution
from aps_ai.beamline34IDC.facade.focusing_optics_interface import Movement, MotorResolution, AngularUnits, DistanceUnits
from aps_ai.beamline34IDC.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics, get_default_input_features

class FocusingOpticsCommon(AbstractSimulatedFocusingOptics):
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
            vkb_error_profile_file = "VKB-LTP-3_shadow.dat"
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

        self._coherence_slits = ShadowOpticalElement(coherence_slits)

        self._initialize_kb(input_features, reflectivity_file, vkb_error_profile_file, hkb_error_profile_file)

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

    def modify_coherence_slits(self, coh_slits_h_center=None, coh_slits_v_center=None, coh_slits_h_aperture=None, coh_slits_v_aperture=None, units=DistanceUnits.MICRON):
        if self._coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON:        factor = 1e-3
        elif units == DistanceUnits.MILLIMETERS: factor = 1.0
        else: raise ValueError("Distance units not recognized")

        round_digit = MotorResolution.getInstance().get_coh_slits_motors_resolution(units=DistanceUnits.MILLIMETERS)

        if not coh_slits_h_center   is None: self._coherence_slits._oe.CX_SLIT = numpy.array([round(factor*coh_slits_h_center,   round_digit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_center   is None: self._coherence_slits._oe.CZ_SLIT = numpy.array([round(factor*coh_slits_v_center,   round_digit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_h_aperture is None: self._coherence_slits._oe.RX_SLIT = numpy.array([round(factor*coh_slits_h_aperture, round_digit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not coh_slits_v_aperture is None: self._coherence_slits._oe.RZ_SLIT = numpy.array([round(factor*coh_slits_v_aperture, round_digit), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if not self._coherence_slits in self._modified_elements: self._modified_elements.append(self._coherence_slits)
        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_coherence_slits_parameters(self, units=DistanceUnits.MICRON):  # center x, center z, aperture x, aperture z
        if self._coherence_slits is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON:        factor = 1e3
        elif units == DistanceUnits.MILLIMETERS: factor = 1.0
        else: raise ValueError("Distance units not recognized")

        return factor*self._coherence_slits._oe.CX_SLIT, \
               factor*self._coherence_slits._oe.CZ_SLIT, \
               factor*self._coherence_slits._oe.RX_SLIT, \
               factor*self._coherence_slits._oe.RZ_SLIT

        # V-KB -----------------------

    # PROTECTED GENERIC MOTOR METHODS
    @classmethod
    def _move_motor_3_pitch(cls, element, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, round_digit=4, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == AngularUnits.MILLIRADIANS: angle = numpy.degrees(angle * 1e-3)
        elif units == AngularUnits.DEGREES:    pass
        elif units == AngularUnits.RADIANS:    angle = numpy.degrees(angle)
        else: raise ValueError("Angular units not recognized")

        sign = -1 if invert else 1

        if movement == Movement.ABSOLUTE:   element._oe.X_ROT =  sign*round(angle - (90 - element._oe.T_INCIDENCE), round_digit)
        elif movement == Movement.RELATIVE: element._oe.X_ROT += sign*round(angle, round_digit)
        else:  raise ValueError("Movement not recognized")

    @classmethod
    def _move_motor_4_transation(cls, element, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=3, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON: translation *= 1e-3
        elif units == DistanceUnits.MILLIMETERS: pass
        else: raise ValueError("Distance units not recognized")

        sign = -1 if invert else 1

        total_pitch_angle = numpy.radians(90 - element._oe.T_INCIDENCE + sign*element._oe.X_ROT)

        if movement == Movement.ABSOLUTE:
            element._oe.OFFY = round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ = round(translation, round_digit) * numpy.cos(total_pitch_angle)
        elif movement == Movement.RELATIVE:
            element._oe.OFFY += round(translation, round_digit) * numpy.sin(total_pitch_angle)
            element._oe.OFFZ += round(translation, round_digit) * numpy.cos(total_pitch_angle)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def _get_motor_3_pitch(cls, element, units=AngularUnits.MILLIRADIANS, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        sign = -1 if invert else 1

        angle = (90 - element._oe.T_INCIDENCE) + sign*element._oe.X_ROT

        if units == AngularUnits.MILLIRADIANS:  return 1000 * numpy.radians(angle)
        elif units == AngularUnits.DEGREES:     return angle
        elif units == AngularUnits.RADIANS:     return numpy.radians(angle)
        else: raise ValueError("Angular units not recognized")

    @classmethod
    def _get_motor_4_translation(cls, element, units=DistanceUnits.MICRON, invert=False):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        pitch_angle = cls._get_motor_3_pitch(element, units=AngularUnits.RADIANS, invert=invert)

        translation = numpy.average([element._oe.OFFY / numpy.sin(pitch_angle), element._oe.OFFZ / numpy.cos(pitch_angle)])

        if units == DistanceUnits.MICRON:        return translation*1e3
        elif units == DistanceUnits.MILLIMETERS: return translation
        else: raise ValueError("Distance units not recognized")

    @classmethod
    def _get_q_distance(cls, element):
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
                self._slits_beam = self._trace_coherence_slits(random_seed, remove_lost_rays, verbose)
                output_beam      = self._slits_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(self._slits_beam, title="Coherence Slits", xrange=None, yrange=None)

            if run_all or self._vkb in self._modified_elements:
                self._vkb_beam = self._trace_vkb(random_seed, remove_lost_rays, verbose)
                output_beam    = self._vkb_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(self._vkb_beam, title="VKB", xrange=None, yrange=None)

            if run_all or self._hkb in self._modified_elements:
                self._hkb_beam = self._trace_hkb(near_field_calculation, random_seed, remove_lost_rays, verbose)
                output_beam    = self._hkb_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(self._hkb_beam, title="HKB", xrange=None, yrange=None)

            # after every run, the list of modified elements must be empty
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

        return output_beam.duplicate(history=False)

    def _trace_coherence_slits(self, random_seed, remove_lost_rays, verbose):
        output_beam = self._trace_oe(input_beam=self._input_beam,
                                     shadow_oe=self._coherence_slits,
                                     widget_class_name="ScreenSlits",
                                     oe_name="Coherence Slits",
                                     remove_lost_rays=remove_lost_rays)

        # HYBRID CORRECTION TO CONSIDER DIFFRACTION FROM SLITS
        try:
            return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                     diffraction_plane=4,  # BOTH 1D+1D (3 is 2D)
                                                                     calcType=1,  # Diffraction by Simple Aperture
                                                                     verbose=verbose,
                                                                     random_seed=None if random_seed is None else (random_seed + 100))).ff_beam
        except Exception:
            raise HybridFailureException(oe="Coherence Slits")

    def _trace_vkb(self, random_seed, remove_lost_rays, verbose): raise NotImplementedError()
    def _trace_hkb(self, near_field_calculation, random_seed, remove_lost_rays, verbose): raise NotImplementedError()

    def _trace_oe(self, input_beam, shadow_oe, widget_class_name, oe_name, remove_lost_rays, history=True):
        return self._check_beam(ShadowBeam.traceFromOE(input_beam, #.duplicate(history=history),
                                                       shadow_oe.duplicate(),
                                                       widget_class_name=widget_class_name,
                                                       history=history,
                                                       recursive_history=False),
                                oe_name, remove_lost_rays)

    def _check_beam(self, output_beam, oe, remove_lost_rays):
        if ShadowCongruence.checkEmptyBeam(output_beam):
            if ShadowCongruence.checkGoodBeam(output_beam):
                if remove_lost_rays: output_beam._beam.rays = output_beam._beam.rays[numpy.where(output_beam._beam.rays[:, 9] == 1)]
                return output_beam
            else: raise EmptyBeamException(oe)
        else: raise EmptyBeamException(oe)