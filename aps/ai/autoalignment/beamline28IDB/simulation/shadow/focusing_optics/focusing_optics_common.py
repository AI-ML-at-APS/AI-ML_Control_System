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
import numpy

from orangecontrib.shadow.util.shadow_util import ShadowPhysics

from aps.ai.autoalignment.common.util.shadow.common import TTYInibitor, PreProcessorFiles, write_reflectivity_file, plot_shadow_beam_spatial_distribution
from aps.ai.autoalignment.common.simulation.shadow.focusing_optics import AbstractShadowFocusingOptics
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import AbstractSimulatedFocusingOptics, get_default_input_features, Layout
from aps.ai.autoalignment.common.facade.parameters import MotorResolutionRegistry

class FocusingOpticsCommonAbstract(AbstractShadowFocusingOptics, AbstractSimulatedFocusingOptics):
    def __init__(self):
        super(FocusingOpticsCommonAbstract, self).__init__()

        self._h_bendable_mirror_beam = None
        self._v_bimorph_mirror_beam = None
        self._h_bendable_mirror = None
        self._v_bimorph_mirror = None

    def initialize(self, **kwargs):
        super(FocusingOpticsCommonAbstract, self).initialize(**kwargs)

        try:    self._input_features = kwargs["input_features"]
        except: self._input_features = get_default_input_features()

        try:    layout = kwargs["layout"]
        except: layout = Layout.AUTO_ALIGNMENT

        self._layout           = layout
        self._shift_horizontal_mirror = 0.0 if self._layout == Layout.AUTO_ALIGNMENT else 325.0
        self._shift_detector          = 0.0 if self._layout == Layout.AUTO_ALIGNMENT else -710.0

        self._motor_resolution = MotorResolutionRegistry.getInstance().get_motor_resolution_set("28-ID-B")

        try:    rewrite_preprocessor_files = kwargs["rewrite_preprocessor_files"]
        except: rewrite_preprocessor_files = PreProcessorFiles.YES_SOURCE_RANGE

        energies     = ShadowPhysics.getEnergyFromShadowK(self._input_beam._beam.rays[:, 10])
        energy_range = [numpy.min(energies), numpy.max(energies)]

        if rewrite_preprocessor_files   == PreProcessorFiles.YES_FULL_RANGE:   reflectivity_file = write_reflectivity_file()
        elif rewrite_preprocessor_files == PreProcessorFiles.YES_SOURCE_RANGE: reflectivity_file = write_reflectivity_file(energy_range=energy_range)
        elif rewrite_preprocessor_files == PreProcessorFiles.NO:               reflectivity_file = "Pt.dat"

        h_bendable_mirror_error_profile_file = "H-Bendable-Mirror_shadow.dat"

        self._initialize_mirrors(self._input_features, reflectivity_file, h_bendable_mirror_error_profile_file)

        self._modified_elements = [self._h_bendable_mirror, self._v_bimorph_mirror]

        #####################################################################################
        # This methods represent the run-time interface, to interact with the optical system
        # in real time, like in the real beamline

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

        self._input_beam = self._check_beam(self._input_beam, "Primary Optical System", remove_lost_rays)

        if not verbose:
            fortran_suppressor = TTYInibitor()
            fortran_suppressor.start()

        output_beam = None

        try:
            run_all = self._modified_elements == [] or len(self._modified_elements) == 2

            if run_all or self._h_bendable_mirror in self._modified_elements:
                self._h_bendable_mirror_beam = self._trace_h_bendable_mirror(False, random_seed, remove_lost_rays, verbose)
                if near_field_calculation: self._h_bendable_mirror_beam_nf = self._trace_h_bendable_mirror(True, random_seed, remove_lost_rays, verbose)
                else:                      self._h_bendable_mirror_beam_nf = None
                output_beam    = self._h_bendable_mirror_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(self._h_bendable_mirror_beam, title="H-Bendable-Mirror", xrange=None, yrange=None)

            if run_all or self._v_bimorph_mirror in self._modified_elements:
                if near_field_calculation:
                    self._v_bimorph_mirror_beam = self.__generate_v_bimorph_mirror_beam_nf(remove_lost_rays, random_seed, verbose)
                else:
                    self._v_bimorph_mirror_beam = self._trace_v_bimorph_mirror(near_field_calculation, random_seed, remove_lost_rays, verbose)

                output_beam = self._v_bimorph_mirror_beam

                if debug_mode: plot_shadow_beam_spatial_distribution(self._v_bimorph_mirror_beam, title="V-Bimorph-Mirror", xrange=None, yrange=None)

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

    def __generate_v_bimorph_mirror_beam_nf(self, remove_lost_rays, random_seed, verbose):
        v_bimorph_mirror_beam, go_orig = self._trace_v_bimorph_mirror(True, random_seed, False, verbose)
        go = numpy.where(v_bimorph_mirror_beam._beam.rays[:, 9] == 1)

        if   len(go[0]) < len(go_orig[0]): go_orig = go_orig[0][0 : len(go[0])]
        elif len(go[0]) > len(go_orig[0]): go      = go[0][0 : len(go_orig[0])]

        v_bimorph_mirror_beam._beam.rays[go, 0] = self._h_bendable_mirror_beam_nf._beam.rays[go_orig, 2]
        v_bimorph_mirror_beam._beam.rays[go, 3] = self._h_bendable_mirror_beam_nf._beam.rays[go_orig, 5]

        if remove_lost_rays:
            v_bimorph_mirror_beam._beam.rays = v_bimorph_mirror_beam._beam.rays[go]
            v_bimorph_mirror_beam._beam.rays[:, 11] = numpy.arange(1, v_bimorph_mirror_beam._beam.rays.shape[0] + 1, 1)

        return v_bimorph_mirror_beam

    def _trace_h_bendable_mirror(self, near_field_calculation, random_seed, remove_lost_rays, verbose): raise NotImplementedError()
    def _trace_v_bimorph_mirror(self,  near_field_calculation, random_seed, remove_lost_rays, verbose): raise NotImplementedError()
    def _initialize_mirrors(self, input_features, reflectivity_file, h_bendable_mirror_error_profile_file): raise NotImplementedError()
