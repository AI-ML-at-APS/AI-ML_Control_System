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
import os.path

import numpy
import Shadow

from orangecontrib.shadow.util.shadow_objects import ShadowOpticalElement, ShadowBeam
from orangecontrib.shadow.util.shadow_util import ShadowPhysics, ShadowMath, ShadowCongruence
from orangecontrib.shadow.widgets.special_elements.bl import hybrid_control

from beamline34IDC.util.shadow.common import TTYInibitor, HybridFailureException, EmptyBeamException, PreProcessorFiles, write_reflectivity_file, write_dabam_file, rotate_axis_system, get_hybrid_input_parameters, plot_shadow_beam_spatial_distribution
from beamline34IDC.facade.focusing_optics_interface import Movement, MotorResolution, AngularUnits, DistanceUnits
from beamline34IDC.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics, get_default_input_features

def shadow_focusing_optics_factory_method(**kwargs):
    try:
        if kwargs["bender"] == True: return __BendableFocusingOptics()
        else:                        return __IdealFocusingOptics()
    except: return __IdealFocusingOptics()

class _FocusingOpticsCommon(AbstractSimulatedFocusingOptics):
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
    def _move_motor_3_pitch(cls, element, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS, round_digit=4):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == AngularUnits.MILLIRADIANS: angle = numpy.degrees(angle * 1e-3)
        elif units == AngularUnits.DEGREES:    pass
        elif units == AngularUnits.RADIANS:    angle = numpy.degrees(angle)
        else: raise ValueError("Angular units not recognized")

        if movement == Movement.ABSOLUTE:   element._oe.X_ROT = round(angle - (90 - element._oe.T_INCIDENCE), round_digit)
        elif movement == Movement.RELATIVE: element._oe.X_ROT += round(angle, round_digit)
        else:  raise ValueError("Movement not recognized")

    @classmethod
    def _move_motor_4_transation(cls, element, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=3):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if units == DistanceUnits.MICRON: translation *= 1e-3
        elif units == DistanceUnits.MILLIMETERS: pass
        else: raise ValueError("Distance units not recognized")

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
    def _get_motor_3_pitch(cls, element, units=AngularUnits.MILLIRADIANS):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        angle = 90 - element._oe.T_INCIDENCE + element._oe.X_ROT

        if units == AngularUnits.MILLIRADIANS:  return 1000 * numpy.radians(angle)
        elif units == AngularUnits.DEGREES:     return angle
        elif units == AngularUnits.RADIANS:     return numpy.radians(angle)
        else: raise ValueError("Angular units not recognized")

    @classmethod
    def _get_motor_4_translation(cls, element, units=DistanceUnits.MICRON):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        pitch_angle = cls._get_motor_3_pitch(element, units=AngularUnits.RADIANS)

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

class __IdealFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        super(_FocusingOpticsCommon, self).__init__()

    def _initialize_kb(self, input_features, reflectivity_file, vkb_error_profile_file, hkb_error_profile_file):
        # V-KB

        vkb_motor_3_pitch_angle = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
        vkb_pitch_angle_displacement_shadow = numpy.degrees(vkb_motor_3_delta_pitch_angle)
        vkb_motor_4_translation = input_features.get_parameter("vkb_motor_4_translation")

        vkb = Shadow.OE()
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
        hkb_motor_3_pitch_angle = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
        hkb_pitch_angle_displacement_shadow = numpy.degrees(hkb_motor_3_delta_pitch_angle)
        hkb_motor_4_translation = input_features.get_parameter("hkb_motor_4_translation")

        hkb = Shadow.OE()
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

        self._vkb = ShadowOpticalElement(vkb)
        self._hkb = ShadowOpticalElement(hkb)

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._vkb, angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self._get_motor_3_pitch(self._vkb, units)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._vkb, translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        return self._get_motor_4_translation(self._vkb, units)

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._hkb, angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        return self._get_motor_3_pitch(self._hkb, units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._hkb, translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        return self._get_motor_4_translation(self._hkb, units)

    def change_vkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        self.__change_shape(self._vkb, q_distance, movement)

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_q_distance(self):
        return self._get_q_distance(self._vkb)

        # H-KB -----------------------

    def change_hkb_shape(self, q_distance, movement=Movement.ABSOLUTE):
        self.__change_shape(self._hkb, q_distance, movement)

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_q_distance(self):
        return self._get_q_distance(self._hkb)

    # IMPLEMENTATION OF PROTECTED METHODS FROM SUPERCLASS

    def _trace_vkb(self, random_seed, remove_lost_rays, verbose):
        output_beam =  self._trace_oe(input_beam=self._slits_beam,
                                      shadow_oe=self._vkb,
                                      widget_class_name="EllypticalMirror",
                                      oe_name="V-KB",
                                      remove_lost_rays=remove_lost_rays)

        # NOTE: Near field not possible for vkb (beam is untraceable)
        try:
            return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                     diffraction_plane=2,  # Tangential
                                                                     calcType=3,  # Diffraction by Mirror Size + Errors
                                                                     verbose=verbose,
                                                                     random_seed=None if random_seed is None else (random_seed + 200))).ff_beam
        except Exception:
            raise HybridFailureException(oe="V-KB")

    def _trace_hkb(self, near_field_calculation, random_seed, remove_lost_rays, verbose):
        output_beam = self._trace_oe(input_beam=self._vkb_beam,
                              shadow_oe=self._hkb,
                              widget_class_name="EllypticalMirror",
                              oe_name="H-KB",
                              remove_lost_rays=remove_lost_rays)
        try:
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
        except Exception:
            raise HybridFailureException(oe="H-KB")

        output_beam = rotate_axis_system(output_beam, rotation_angle=270.0)

        return output_beam

    # PRIVATE -----------------------

    @classmethod
    def __change_shape(cls, element, q_distance, movement=Movement.ABSOLUTE):
        if element is None: raise ValueError("Initialize Focusing Optics System first")

        if movement == Movement.ABSOLUTE:   element._oe.SIMAG = q_distance
        elif movement == Movement.RELATIVE: element._oe.SIMAG += q_distance
        else: raise ValueError("Movement not recognized")

from beamline34IDC.simulation.shadow.bender import BenderManager, HKBMockWidget, VKBMockWidget
from orangecontrib.shadow_advanced_tools.widgets.optical_elements.bl.double_rod_bendable_ellispoid_mirror_bl import apply_bender_surface

class __BendableFocusingOptics(_FocusingOpticsCommon):
    def __init__(self):
        super(_FocusingOpticsCommon, self).__init__()

    def initialize(self,
                   input_photon_beam,
                   input_features=get_default_input_features(),
                   **kwargs):

        super().initialize(input_photon_beam, input_features, **kwargs)
        
        self.__vkb_bender_manager = BenderManager(kb_upstream=VKBMockWidget(self._vkb[0], verbose=True, label="Upstream"),
                                                  kb_downstream=VKBMockWidget(self._vkb[1], verbose=True, label="Downstream"))
        self.__vkb_bender_manager.load_calibration("V-KB")
        self.__vkb_bender_manager.set_positions(input_features.get_parameter("vkb_motor_1_bender_position"),
                                                input_features.get_parameter("vkb_motor_2_bender_position"))
        self.__vkb_bender_manager.remove_bender_files()

        self.__hkb_bender_manager = BenderManager(kb_upstream=HKBMockWidget(self._hkb[0], verbose=True, label="Upstream"),
                                                  kb_downstream=HKBMockWidget(self._hkb[1], verbose=True, label="Downstream"))
        self.__hkb_bender_manager.load_calibration("H-KB")
        self.__hkb_bender_manager.set_positions(input_features.get_parameter("hkb_motor_1_bender_position"),
                                                input_features.get_parameter("hkb_motor_2_bender_position"))
        self.__hkb_bender_manager.remove_bender_files()

    def _initialize_kb(self, input_features, reflectivity_file, vkb_error_profile_file, hkb_error_profile_file):
        # V-KB --------------------
        vkb_motor_3_pitch_angle             = input_features.get_parameter("vkb_motor_3_pitch_angle")
        vkb_pitch_angle_shadow              = 90 - numpy.degrees(vkb_motor_3_pitch_angle)
        vkb_motor_3_delta_pitch_angle       = input_features.get_parameter("vkb_motor_3_delta_pitch_angle")
        vkb_pitch_angle_displacement_shadow = numpy.degrees(vkb_motor_3_delta_pitch_angle)
        vkb_motor_4_translation             = input_features.get_parameter("vkb_motor_4_translation")

        vkb_up = Shadow.OE()
        vkb_up.ALPHA = 180.0
        vkb_up.DUMMY = 0.1
        vkb_up.FCYL = 1
        vkb_up.FHIT_C = 1
        vkb_up.FILE_REFL = reflectivity_file.encode()
        vkb_up.FILE_RIP = vkb_error_profile_file.encode()
        vkb_up.FMIRR = 2
        vkb_up.FWRITE = 1
        vkb_up.F_DEFAULT = 0
        vkb_up.F_G_S = 2
        vkb_up.F_REFLEC = 1
        vkb_up.F_RIPPLE = 1
        vkb_up.RLEN1 = 50.0
        vkb_up.RLEN2 = 50.0
        vkb_up.RWIDX1 = 20.95
        vkb_up.RWIDX2 = 20.95
        vkb_up.SIMAG = -999
        vkb_up.SSOUR = 50667.983
        vkb_up.THETA = vkb_pitch_angle_shadow
        vkb_up.T_IMAGE = 101.0
        vkb_up.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb_up.T_REFLECTION = vkb_pitch_angle_shadow
        vkb_up.T_SOURCE = 150.0
        
        # DISPLACEMENTS
        vkb_up.F_MOVE = 1
        vkb_up.OFFY   = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_up.OFFZ   = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_up.X_ROT  = vkb_pitch_angle_displacement_shadow

        vkb_down = Shadow.OE()
        vkb_down.ALPHA = 180.0
        vkb_down.DUMMY = 0.1
        vkb_down.FCYL = 1
        vkb_down.FHIT_C = 1
        vkb_down.FILE_REFL = reflectivity_file.encode()
        vkb_down.FILE_RIP = vkb_error_profile_file.encode()
        vkb_down.FMIRR = 2
        vkb_down.FWRITE = 1
        vkb_down.F_DEFAULT = 0
        vkb_down.F_G_S = 2
        vkb_down.F_REFLEC = 1
        vkb_down.F_RIPPLE = 1
        vkb_down.RLEN1 = 50.0
        vkb_down.RLEN2 = 50.0
        vkb_down.RWIDX1 = 20.95
        vkb_down.RWIDX2 = 20.95
        vkb_down.SIMAG = -999
        vkb_down.SSOUR = 50667.983
        vkb_down.THETA = vkb_pitch_angle_shadow
        vkb_down.T_IMAGE = 101.0
        vkb_down.T_INCIDENCE = vkb_pitch_angle_shadow
        vkb_down.T_REFLECTION = vkb_pitch_angle_shadow
        vkb_down.T_SOURCE = 150.0

        # DISPLACEMENTS
        vkb_down.F_MOVE = 1
        vkb_down.OFFY   = vkb_motor_4_translation * numpy.sin(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_down.OFFZ   = vkb_motor_4_translation * numpy.cos(vkb_motor_3_pitch_angle + vkb_motor_3_delta_pitch_angle)
        vkb_down.X_ROT  = vkb_pitch_angle_displacement_shadow

        # H-KB --------------------

        hkb_motor_3_pitch_angle             = input_features.get_parameter("hkb_motor_3_pitch_angle")
        hkb_pitch_angle_shadow              = 90 - numpy.degrees(hkb_motor_3_pitch_angle)
        hkb_motor_3_delta_pitch_angle       = input_features.get_parameter("hkb_motor_3_delta_pitch_angle")
        hkb_pitch_angle_displacement_shadow = numpy.degrees(hkb_motor_3_delta_pitch_angle)
        hkb_motor_4_translation             = input_features.get_parameter("hkb_motor_4_translation")

        hkb_up = Shadow.OE()
        hkb_up.ALPHA = 90.0
        hkb_up.DUMMY = 0.1
        hkb_up.FCYL = 1
        hkb_up.FHIT_C = 1
        hkb_up.FILE_REFL = reflectivity_file.encode()
        hkb_up.FILE_RIP  = hkb_error_profile_file.encode()
        hkb_up.FMIRR = 2
        hkb_up.FWRITE = 1
        hkb_up.F_DEFAULT = 0
        hkb_up.F_G_S = 2
        hkb_up.F_REFLEC = 1
        hkb_up.F_RIPPLE = 1
        hkb_up.RLEN1 = 50.0   # dim plus
        hkb_up.RLEN2 = 50.0  # dim minus
        hkb_up.RWIDX1 = 24.75
        hkb_up.RWIDX2 = 24.75
        hkb_up.SIMAG = -999
        hkb_up.SSOUR = 50768.983
        hkb_up.THETA = hkb_pitch_angle_shadow
        hkb_up.T_IMAGE = 120.0
        hkb_up.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb_up.T_REFLECTION = hkb_pitch_angle_shadow
        hkb_up.T_SOURCE = 0.0
        
        # DISPLACEMENT
        hkb_up.F_MOVE = 1
        hkb_up.X_ROT  = hkb_pitch_angle_displacement_shadow
        hkb_up.OFFY   = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb_up.OFFZ   = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)

        # H-KB
        hkb_down = Shadow.OE()
        hkb_down.ALPHA = 90.0
        hkb_down.DUMMY = 0.1
        hkb_down.FCYL = 1
        hkb_down.FHIT_C = 1
        hkb_down.FILE_REFL = reflectivity_file.encode()
        hkb_down.FILE_RIP = hkb_error_profile_file.encode()
        hkb_down.FMIRR = 2
        hkb_down.FWRITE = 1
        hkb_down.F_DEFAULT = 0
        hkb_down.F_G_S = 2
        hkb_down.F_REFLEC = 1
        hkb_down.F_RIPPLE = 1
        hkb_down.RLEN1 = 50.0  # dim plus
        hkb_down.RLEN2 = 50.0   # dim minus
        hkb_down.RWIDX1 = 24.75
        hkb_down.RWIDX2 = 24.75
        hkb_down.SIMAG = -999
        hkb_down.SSOUR = 50768.983
        hkb_down.THETA = hkb_pitch_angle_shadow
        hkb_down.T_IMAGE = 120.0
        hkb_down.T_INCIDENCE = hkb_pitch_angle_shadow
        hkb_down.T_REFLECTION = hkb_pitch_angle_shadow
        hkb_down.T_SOURCE = 0.0

        # DISPLACEMENT
        hkb_down.F_MOVE = 1
        hkb_down.X_ROT  = hkb_pitch_angle_displacement_shadow
        hkb_down.OFFY   = hkb_motor_4_translation * numpy.sin(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        hkb_down.OFFZ   = hkb_motor_4_translation * numpy.cos(hkb_motor_3_pitch_angle + hkb_motor_3_delta_pitch_angle)
        
        self._vkb = [ShadowOpticalElement(vkb_up), ShadowOpticalElement(vkb_down)]
        self._hkb = [ShadowOpticalElement(hkb_up), ShadowOpticalElement(hkb_down)]

    # ---- H-KB ---------------------------------------------------------

    def move_vkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_1_bender(self, units=DistanceUnits.MICRON): 
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[0]
    
    def move_vkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON): 
        self.__move_motor_1_2_bender(self.__vkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_vkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__vkb_bender_manager, units)[1]


    def get_vkb_q_distance(self):
        return self._get_q_distance(self._vkb[0]), self._get_q_distance(self._vkb[1])

    def move_vkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._vkb[0], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])
        self._move_motor_3_pitch(self._vkb[1], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_vkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._vkb[0], units)

    def move_vkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._vkb[0], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])
        self._move_motor_4_transation(self._vkb[1], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_vkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._vkb in self._modified_elements: self._modified_elements.append(self._vkb)
        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_vkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._vkb[0], units)

    # ---- H-KB ---------------------------------------------------------

    def move_hkb_motor_1_bender(self, pos_upstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, pos_upstream, None, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_1_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[0]

    def move_hkb_motor_2_bender(self, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self.__move_motor_1_2_bender(self.__hkb_bender_manager, None, pos_downstream, movement, units,
                                     round_digit=MotorResolution.getInstance().get_hkb_motor_1_2_bender_resolution(units=DistanceUnits.MICRON)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_2_bender(self, units=DistanceUnits.MICRON):
        return self.__get_motor_1_2_bender(self.__hkb_bender_manager, units)[1]

    def get_hkb_q_distance(self):
        return self._get_q_distance(self._hkb[0]), self._get_q_distance(self._hkb[1])

    def move_hkb_motor_3_pitch(self, angle, movement=Movement.ABSOLUTE, units=AngularUnits.MILLIRADIANS):
        self._move_motor_3_pitch(self._hkb[0], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])
        self._move_motor_3_pitch(self._hkb[1], angle, movement, units,
                                 round_digit=MotorResolution.getInstance().get_hkb_motor_3_pitch_resolution(units=AngularUnits.DEGREES)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_3_pitch(self, units=AngularUnits.MILLIRADIANS):
        # motor 3/4 are identical for the two sides
        return self._get_motor_3_pitch(self._hkb[0], units)

    def move_hkb_motor_4_translation(self, translation, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON):
        self._move_motor_4_transation(self._hkb[0], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])
        self._move_motor_4_transation(self._hkb[1], translation, movement, units,
                                      round_digit=MotorResolution.getInstance().get_hkb_motor_4_translation_resolution(units=DistanceUnits.MILLIMETERS)[1])

        if not self._hkb in self._modified_elements: self._modified_elements.append(self._hkb)

    def get_hkb_motor_4_translation(self, units=DistanceUnits.MICRON):
        # motor 3/4 are identical for the two sides
        return self._get_motor_4_translation(self._hkb[0], units)

    # IMPLEMENTATION OF PROTECTED METHODS FROM SUPERCLASS

    def _trace_vkb(self, random_seed, remove_lost_rays, verbose):
        output_beam_upstream, cursor_upstream, output_beam_downstream, cursor_downstream =  \
            self.__trace_kb(bender_manager=self.__vkb_bender_manager,
                            input_beam=self._slits_beam,
                            widget_class_name="DoubleRodBenderEllypticalMirror",
                            oe_name="V-KB",
                            remove_lost_rays=remove_lost_rays)

        def run_hybrid(output_beam, increment):
            # NOTE: Near field not possible for vkb (beam is untraceable)
            try:
                return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                         diffraction_plane=2,  # Tangential
                                                                         calcType=3,  # Diffraction by Mirror Size + Errors
                                                                         verbose=verbose,
                                                                         random_seed=None if random_seed is None else (random_seed + increment))).ff_beam
            except Exception:
                raise HybridFailureException(oe="V-KB")

        output_beam_upstream = run_hybrid(output_beam_upstream, increment=200)
        output_beam_upstream._beam.rays = output_beam_upstream._beam.rays[cursor_upstream]

        output_beam_downstream = run_hybrid(output_beam_downstream, increment=201)
        output_beam_downstream._beam.rays = output_beam_downstream._beam.rays[cursor_downstream]

        return ShadowBeam.mergeBeams(output_beam_upstream, output_beam_downstream, which_flux=3, merge_history=0)

    def _trace_hkb(self, near_field_calculation, random_seed, remove_lost_rays, verbose):
        output_beam_upstream, cursor_upstream, output_beam_downstream, cursor_downstream = \
            self.__trace_kb(bender_manager=self.__hkb_bender_manager,
                            input_beam=self._vkb_beam,
                            widget_class_name="DoubleRodBenderEllypticalMirror",
                            oe_name="H-KB",
                            remove_lost_rays=remove_lost_rays)

        def run_hybrid(output_beam, increment):
            try:
                if not near_field_calculation:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment))).ff_beam
                else:
                    return hybrid_control.hy_run(get_hybrid_input_parameters(output_beam,
                                                                             diffraction_plane=2,  # Tangential
                                                                             calcType=3,  # Diffraction by Mirror Size + Errors
                                                                             nf=1,
                                                                             verbose=verbose,
                                                                             random_seed=None if random_seed is None else (random_seed + increment))).nf_beam
            except Exception:
                raise HybridFailureException(oe="H-KB")

        output_beam_upstream = run_hybrid(output_beam_upstream, increment=300)
        output_beam_upstream._beam.rays = output_beam_upstream._beam.rays[cursor_upstream]

        output_beam_downstream = run_hybrid(output_beam_downstream, increment=301)
        output_beam_downstream._beam.rays = output_beam_downstream._beam.rays[cursor_downstream]

        output_beam = ShadowBeam.mergeBeams(output_beam_upstream, output_beam_downstream, which_flux=3, merge_history=0)

        return rotate_axis_system(output_beam, rotation_angle=270.0)

    # PRIVATE METHODS

    def __trace_kb(self, bender_manager, input_beam, widget_class_name, oe_name, remove_lost_rays):
        upstream_widget   = bender_manager._kb_upstream
        downstream_widget = bender_manager._kb_downstream

        upstream_oe   = upstream_widget.shadow_oe.duplicate()
        downstream_oe = downstream_widget.shadow_oe.duplicate()

        upstream_oe._oe.RLEN1   = 0.0 # no positive part
        downstream_oe._oe.RLEN2 = 0.0 # no negative part

        #trace both sides separately and get the beams:
        upstream_beam_cursor = numpy.where(self._trace_oe(input_beam=input_beam,
                                                          shadow_oe=upstream_oe,
                                                          widget_class_name=widget_class_name,
                                                          oe_name=oe_name + "_UPSTREAM",
                                                          remove_lost_rays=False,
                                                          history=False)._beam.rays[:, 9] == 1)

        downstream_beam_cursor = numpy.where(self._trace_oe(input_beam=input_beam,
                                                            shadow_oe=downstream_oe,
                                                            widget_class_name=widget_class_name,
                                                            oe_name=oe_name + "_DOWNSTREAM",
                                                            remove_lost_rays=False,
                                                            history=False)._beam.rays[:, 9] == 1)

        
        # this make HYBRID FAIL! we have to do it after the hybrid calculation
        #upstream_input_beam   = input_beam.duplicate()
        #downstream_input_beam = input_beam.duplicate()
        #upstream_input_beam._beam.rays   = upstream_input_beam._beam.rays[upstream_beam_cursor]
        #downstream_input_beam._beam.rays = downstream_input_beam._beam.rays[downstream_beam_cursor]

        def calculate_bender(input_beam, widget, do_calculation=True):
            widget.R0                     = widget.R0_out                              # use last fit result

            if do_calculation:
                widget.shadow_oe._oe.FILE_RIP = bytes(widget.ms_defect_file_name, 'utf-8') # restore original error profile

                apply_bender_surface(widget=widget, shadow_oe=widget.shadow_oe, input_beam=input_beam)
            else:
                widget.shadow_oe._oe.F_RIPPLE = 1
                widget.shadow_oe._oe.F_G_S = 2
                widget.shadow_oe._oe.FILE_RIP = bytes(widget.output_file_name_full, 'utf-8')

        if (bender_manager.F_upstream == bender_manager.F_upstream_previous) and (bender_manager.F_downstream == bender_manager.F_downstream_previous) and \
                os.path.exists(upstream_widget.output_file_name_full) and os.path.exists(downstream_widget.output_file_name_full):
            # trace both the beam on the whole bender widget
            calculate_bender(input_beam, upstream_widget, do_calculation=False)
            calculate_bender(input_beam, downstream_widget, do_calculation=False)
        else:
            calculate_bender(input_beam, upstream_widget)
            calculate_bender(input_beam, downstream_widget)

        bender_manager.F_upstream_previous   = bender_manager.F_upstream
        bender_manager.F_downstream_previous = bender_manager.F_downstream

        # Redo raytracing with the bender correction as error profile
        return self._trace_oe(input_beam=input_beam,
                              shadow_oe=upstream_widget.shadow_oe,
                              widget_class_name=widget_class_name,
                              oe_name=oe_name,
                              remove_lost_rays=remove_lost_rays), \
               upstream_beam_cursor, \
               self._trace_oe(input_beam=input_beam,
                              shadow_oe=downstream_widget.shadow_oe,
                              widget_class_name=widget_class_name,
                              oe_name=oe_name,
                              remove_lost_rays=remove_lost_rays), \
               downstream_beam_cursor

    @classmethod
    def __move_motor_1_2_bender(cls, bender_manager, pos_upstream, pos_downstream, movement=Movement.ABSOLUTE, units=DistanceUnits.MICRON, round_digit=2):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        current_pos_upstream, current_pos_downstream = bender_manager.get_positions()

        def check_pos(pos, current_pos):
            if not pos is None:
                if units == DistanceUnits.MILLIMETERS: return round(pos*1e3, round_digit)
                elif units == DistanceUnits.MICRON:    return round(pos,     round_digit)
                else: raise ValueError("Distance units not recognized")
            else:
                return 0.0 if movement == Movement.RELATIVE else current_pos

        pos_upstream   = check_pos(pos_upstream,   current_pos_upstream)
        pos_downstream = check_pos(pos_downstream, current_pos_downstream)

        if movement == Movement.ABSOLUTE:
            bender_manager.set_positions(pos_upstream, pos_downstream)
        elif movement == Movement.RELATIVE:
            current_pos_upstream, current_pos_downstream = bender_manager.get_positions()
            bender_manager.set_positions(current_pos_upstream + pos_upstream, current_pos_downstream + pos_downstream)
        else:
            raise ValueError("Movement not recognized")

    @classmethod
    def __get_motor_1_2_bender(cls, bender_manager, units=DistanceUnits.MICRON):
        if bender_manager is None: raise ValueError("Initialize Focusing Optics System first")

        pos_upstream, pos_downstream = bender_manager.get_positions()

        if units == DistanceUnits.MILLIMETERS:
            pos_upstream   *= 1e-3
            pos_downstream *= 1e-3
        elif units == DistanceUnits.MICRON: pass
        else: raise ValueError("Distance units not recognized")

        return pos_upstream, pos_downstream
