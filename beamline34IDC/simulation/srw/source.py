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

from wofrysrw.propagator.wavefront2D.srw_wavefront import WavefrontParameters, WavefrontPrecisionParameters, PolarizationComponent
from wofrysrw.storage_ring.srw_electron_beam import SRWElectronBeam
from wofrysrw.storage_ring.light_sources.srw_undulator_light_source import SRWUndulatorLightSource
from wofrysrw.storage_ring.magnetic_structures.srw_undulator import SRWUndulator
from wofrysrw.beamline.srw_beamline import SRWBeamline

from beamline34IDC.simulation.facade.source_interface import AbstractSource, Sources, StorageRing, ElectronBeamAPS_U, ElectronBeamAPS
from beamline34IDC.util.shadow.common import codata, m2ev

def srw_source_factory_method(kind_of_source=Sources.GAUSSIAN):
    if kind_of_source == Sources.GAUSSIAN:    raise ValueError("SRW simulation doesn't provide this kind of source")
    elif kind_of_source == Sources.UNDULATOR: return __SRWUndulatorSource()
    else: raise ValueError("Kind of Source not recognized")

#############################################################################
# Undulator SOURCE
#

class __SRWUndulatorSource(AbstractSource):

    def __init__(self):
        self.__srw_source = None
        self.__wavefront_energy = 5000.0

    def initialize(self, storage_ring=StorageRing.APS, **kwargs):
        try: verbose = kwargs["verbose"]
        except: verbose = False

        if storage_ring == StorageRing.APS:
            electron_beam = SRWElectronBeam(energy_in_GeV=ElectronBeamAPS.energy_in_GeV,
                                            energy_spread=ElectronBeamAPS.energy_spread,
                                            current=ElectronBeamAPS.ring_current)

            electron_beam.set_sigmas_all(sigma_x=ElectronBeamAPS.sigma_x,
                                         sigma_y=ElectronBeamAPS.sigma_z,
                                         sigma_xp=ElectronBeamAPS.sigdi_x,
                                         sigma_yp=ElectronBeamAPS.sigdi_z)
        elif storage_ring == StorageRing.APS_U:
            electron_beam = SRWElectronBeam(energy_in_GeV=ElectronBeamAPS_U.energy_in_GeV,
                                            energy_spread=ElectronBeamAPS_U.energy_spread,
                                            current=ElectronBeamAPS_U.ring_current)

            electron_beam.set_sigmas_all(sigma_x=ElectronBeamAPS_U.sigma_x,
                                         sigma_y=ElectronBeamAPS_U.sigma_z,
                                         sigma_xp=ElectronBeamAPS_U.sigdi_x,
                                         sigma_yp=ElectronBeamAPS_U.sigdi_z)

        period_length = 0.033
        number_of_periods = 72

        electron_beam._moment_x = 0.0
        electron_beam._moment_y = 0.0
        electron_beam._moment_z = - 0.5 * period_length * (number_of_periods + 8)
        electron_beam._moment_xp = 0.0
        electron_beam._moment_yp = 0.0

        undulator_magnetic_structure=SRWUndulator(horizontal_central_position = 0.0,
                                                  vertical_central_position = 0.0,
                                                  longitudinal_central_position=0.0,
                                                  K_vertical=self.__K_from_magnetic_field(harmonic_energy=5000.0,
                                                                                          harmonic_number=1,
                                                                                          electron_energy_in_GeV=electron_beam._energy_in_GeV,
                                                                                          undulator_period=period_length),
                                                  K_horizontal=0.0,
                                                  period_length=period_length,
                                                  number_of_periods=number_of_periods,
                                                  initial_phase_horizontal=0.0,
                                                  initial_phase_vertical=0.0,
                                                  symmetry_vs_longitudinal_position_horizontal=1,
                                                  symmetry_vs_longitudinal_position_vertical=-1)

        self.__srw_source = SRWUndulatorLightSource(electron_beam=electron_beam,
                                                    undulator_magnetic_structure=undulator_magnetic_structure)



    def set_angular_acceptance(self, divergence=[1e-4, 1e-4]):
        raise ValueError("This operation is not necessary on wave optics calculation")

    def set_angular_acceptance_from_aperture(self, aperture=[0.03, 0.07], distance=50500):
        raise ValueError("This operation is not necessary on wave optics calculation")

    def set_energy(self, energy=5000, **kwargs):
        try: harmonic_number = kwargs["harmonic_number"]
        except: harmonic_number = 1

        self.__wavefront_energy = energy / harmonic_number

    def get_source_beam(self, **kwargs):
        wf_parameters = WavefrontParameters(photon_energy_min = self.__wavefront_energy,
                                            photon_energy_max = self.__wavefront_energy,
                                            photon_energy_points=1,
                                            h_slit_gap = 0.002,
                                            v_slit_gap = 0.002,
                                            h_slit_points=400,
                                            v_slit_points=400,
                                            distance = 26.8,
                                            electric_field_units = 1,
                                            wavefront_precision_parameters=WavefrontPrecisionParameters(sr_method=1,
                                                                                                        relative_precision=0.01,
                                                                                                        start_integration_longitudinal_position=0.0,
                                                                                                        end_integration_longitudinal_position=0.0,
                                                                                                        number_of_points_for_trajectory_calculation=50000,
                                                                                                        use_terminating_terms=1,
                                                                                                        sampling_factor_for_adjusting_nx_ny=0.0))
        return self.__srw_source.get_SRW_Wavefront(source_wavefront_parameters=wf_parameters)

    @classmethod
    def __gamma(cls, electron_energy_in_GeV):
        return 1e9*electron_energy_in_GeV / (codata.m_e *  codata.c**2 / codata.e)

    @classmethod
    def __K_from_magnetic_field(cls, harmonic_energy, harmonic_number, electron_energy_in_GeV, undulator_period):
        wavelength = harmonic_number*m2ev/harmonic_energy

        return round(numpy.sqrt(2*(((wavelength*2*cls.__gamma(electron_energy_in_GeV)**2)/undulator_period) - 1)), 6)


