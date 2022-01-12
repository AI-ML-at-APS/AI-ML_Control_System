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

from beamline34IDC.util.common import fix_Intensity, m2ev
from orangecontrib.ml.util.mocks import MockWidget

from orangecontrib.shadow.util.shadow_objects import ShadowBeam, ShadowSource, ShadowOpticalElement
from orangecontrib.shadow_advanced_tools.widgets.sources.attributes.hybrid_undulator_attributes import HybridUndulatorAttributes
import orangecontrib.shadow_advanced_tools.widgets.sources.bl.hybrid_undulator_bl as HU

from beamline34IDC.simulation.interfaces.source_interface import AbstractSource, Sources, StorageRing, ElectronBeamAPS_U, ElectronBeamAPS

def shadow_source_factory_method(kind_of_source=Sources.GAUSSIAN):
    if kind_of_source == Sources.GAUSSIAN:
        return __ShadowGaussianUndulatorSource()
    elif kind_of_source == Sources.UNDULATOR:
        return __ShadowHybridUndulatorSource()
    else:
        raise ValueError("Kind of Source not recognized")

class __ShadowGaussianUndulatorSource(AbstractSource):

    class PhotonEnergyDistributions:
        SINGLE_LINE = 1
        UNIFORM = 3

    def __init__(self):
        self.__shadow_source = None

    def initialize(self, storage_ring=StorageRing.APS, **kwargs):
        try: n_rays = kwargs["n_rays"]
        except: n_rays = 500000
        try: random_seed = kwargs["random_seed"]
        except: random_seed = 5676561
        try: undulator_length = kwargs["undulator_length"]
        except: undulator_length = 2.376

        #####################################################
        # SHADOW 3 INITIALIZATION
        #
        # initialize shadow3 source (oe0) and beam
        #
        self.__shadow_source = ShadowSource.create_undulator_gaussian_src()
        self.__shadow_source.src.ISTAR1 = random_seed
        self.__shadow_source.src.NPOINT = n_rays

        self.__undulator_length = undulator_length
        self.__storage_ring = storage_ring

        self.set_angular_acceptance_from_aperture() # defaults
        self.set_energy() # defaults

    def set_angular_acceptance(self, divergence=[1e-4, 1e-4]):
        print("Initializa geometrical source with limited divergence: ", divergence, "rad")
        self.__shadow_source.src.FDISTR = 3 # gaussian

        self.__shadow_source.src.HDIV1 = divergence[0]/2
        self.__shadow_source.src.HDIV2 = divergence[0]/2
        self.__shadow_source.src.VDIV1 = divergence[1]/2
        self.__shadow_source.src.VDIV2 = divergence[1]/2

    def set_angular_acceptance_from_aperture(self, aperture=[0.03, 0.07], distance=50500):
        self.set_angular_acceptance(divergence=[aperture[0] / distance, aperture[1] / distance])

    def set_energy(self, energy_range=[4999.0, 5001.0], **kwargs):
        try: self.__shadow_source.src.F_COLOR = kwargs["photon_energy_distribution"]
        except: self.__shadow_source.src.F_COLOR = 3

        self.__shadow_source.src.PH1 = energy_range[0]
        if self.__shadow_source.src.F_COLOR==3: self.__shadow_source.src.PH2 = energy_range[1]

        self.__set_photon_sizes()

    def get_source_beam(self):
        return fix_Intensity(ShadowBeam.traceFromSource(self.__shadow_source))

    def __set_photon_sizes(self):
        if self.__storage_ring == StorageRing.APS:
            sigma_x = ElectronBeamAPS.sigma_x
            sigma_z = ElectronBeamAPS.sigma_z
            sigdi_x = ElectronBeamAPS.sigdi_x
            sigdi_z = ElectronBeamAPS.sigdi_z
        elif self.__storage_ring == StorageRing.APS_U:
            sigma_x = ElectronBeamAPS_U.sigma_x
            sigma_z = ElectronBeamAPS_U.sigma_z
            sigdi_x = ElectronBeamAPS_U.sigdi_x
            sigdi_z = ElectronBeamAPS_U.sigdi_z

        if self.__shadow_source.src.F_COLOR == 3: harmonic_energy = 0.5*(self.__shadow_source.src.PH2 + self.__shadow_source.src.PH1)
        else:                                     harmonic_energy = self.__shadow_source.src.PH1

        harmonic_wavelength = m2ev / harmonic_energy

        # calculate sizes of the photon undulator beam
        # see formulas 25 & 30 in Elleaume (Onaki & Elleaume)
        s_phot = 2.740 / (4e0 * numpy.pi) * numpy.sqrt(self.__undulator_length * harmonic_wavelength)
        sp_phot = 0.69 * numpy.sqrt(harmonic_wavelength / self.__undulator_length)

        self.__shadow_source.src.F_OPD = 1
        self.__shadow_source.src.F_SR_TYPE = 0
        self.__shadow_source.src.SIGMAX = numpy.sqrt(numpy.power(sigma_x, 2) + numpy.power(s_phot, 2)) * 1e3 # to mm
        self.__shadow_source.src.SIGMAZ = numpy.sqrt(numpy.power(sigma_z, 2) + numpy.power(s_phot, 2)) * 1e3 # to mm
        self.__shadow_source.src.SIGDIX = numpy.sqrt(numpy.power(sigdi_x, 2) + numpy.power(sp_phot, 2))
        self.__shadow_source.src.SIGDIZ = numpy.sqrt(numpy.power(sigdi_z, 2) + numpy.power(sp_phot, 2))

#############################################################################
# HYBRID SOURCE
#

class __ShadowHybridUndulatorSource(AbstractSource):
    class KDirection:
        VERTICAL = 1
        HORIZONTAL = 2
        BOTH = 3

    class PhotonEnergyDistributions:
        ON_HARMONIC = 0
        SINGLE_ENERGY = 1
        RANGE = 2

    def __init__(self):
        self.__widget = None
        self.__aperture = None
        self.__distance = None

    def initialize(self, storage_ring=StorageRing.APS, **kwargs):
        try: n_rays = kwargs["n_rays"]
        except: n_rays = 500000
        try: random_seed = kwargs["random_seed"]
        except: random_seed = 5676561
        try: verbose = kwargs["verbose"]
        except: verbose = False

        self.__widget = self.__MockUndulatorHybrid(storage_ring=storage_ring, verbose=verbose)
        self.__widget.number_of_rays = n_rays
        self.__widget.seed = random_seed

    def set_angular_acceptance(self, divergence=[1e-4, 1e-4]):
        if divergence is None:
            self.__aperture = None
            self.__distance = None
        else:
            self.__distance = 10
            self.__aperture = [self.__distance*numpy.tan(divergence[0]), self.__distance*numpy.tan(divergence[0])]

    def set_angular_acceptance_from_aperture(self, aperture=[0.03, 0.07], distance=50500):
        self.__aperture = aperture
        self.__distance = distance

    def set_K_on_specific_harmonic(self, harmonic_energy, harmonic_number, which=KDirection.VERTICAL):
        wavelength = harmonic_number*m2ev/harmonic_energy
        K = round(numpy.sqrt(2*(((wavelength*2*HU.gamma(self.__widget)**2)/self.__widget.undulator_period)-1)), 6)

        if which == self.KDirection.VERTICAL:
            self.__widget.Kv = K
            self.__widget.Kh = 0.0
        elif which == self.KDirection.HORIZONTAL:
            self.__widget.Kh = K
            self.__widget.Kv = 0.0
        elif which == self.KDirection.BOTH:
            Kboth = round(K / numpy.sqrt(2), 6)
            self.__widget.Kv = Kboth
            self.__widget.Kh = Kboth

    def set_energy(self, energy_range=[4999.0, 5001.0], **kwargs):
        try: photon_energy_distribution = kwargs["photon_energy_distribution"]
        except: photon_energy_distribution = 2

        try: harmonic_number = kwargs["harmonic_number"]
        except: harmonic_number = 1

        try: energy_points = kwargs["energy_points"]
        except: energy_points = 11

        if photon_energy_distribution == self.PhotonEnergyDistributions.ON_HARMONIC:
            self.__widget.use_harmonic = 0
            self.__widget.harmonic_number = harmonic_number
        elif photon_energy_distribution == self.PhotonEnergyDistributions.SINGLE_ENERGY:
            self.__widget.use_harmonic = 1
            self.__widget.energy = energy_range[0]
        elif photon_energy_distribution == self.PhotonEnergyDistributions.RANGE:
            self.__widget.use_harmonic = 2
            self.__widget.energy = energy_range[0]
            self.__widget.energy_to = energy_range[1]
            self.__widget.energy_points = energy_points

    def get_source_beam(self, **kwargs):
        if self.__widget is None: raise ValueError("Source has not been initialized")

        try: ignore_aperture = kwargs["ignore_aperture"]
        except: ignore_aperture = False

        if self.__aperture is None or ignore_aperture:
            source_beam, _ = HU.run_hybrid_undulator_simulation(self.__widget)
        else:
            if self.__distance is None: raise ValueError("Aperture distance must be specified")

            current_good_rays = 0

            while (current_good_rays < self.__widget.number_of_rays):
                temp_beam = self.__run_beam_through_aperture(self.__widget.number_of_rays, self.__aperture, self.__distance)

                if self.__widget.is_verbose(): print("HYBRID UNDULATOR: ", temp_beam.get_number_of_rays(), " good rays")

                if current_good_rays == 0: source_beam = temp_beam
                else:                      source_beam = ShadowBeam.mergeBeams(source_beam, temp_beam)

                current_good_rays = source_beam.get_number_of_rays()

                if self.__widget.is_verbose(): print("TOTAL: ", current_good_rays, " good rays on ", self.__widget.number_of_rays)

            source_beam._beam.rays = source_beam._beam.rays[:self.__widget.number_of_rays, :]

        return source_beam

    def __run_beam_through_aperture(self, n_rays, aperture, distance):
        slits_oe = Shadow.OE()
        slits_oe.DUMMY = 0.1 # mm
        slits_oe.FWRITE = 3
        slits_oe.F_REFRAC = 2
        slits_oe.F_SCREEN = 1
        slits_oe.I_SLIT = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        slits_oe.N_SCREEN = 1
        slits_oe.RX_SLIT = numpy.array([aperture[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        slits_oe.RZ_SLIT = numpy.array([aperture[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        slits_oe.T_IMAGE = 0.0
        slits_oe.T_INCIDENCE = 0.0
        slits_oe.T_REFLECTION = 180.0
        slits_oe.T_SOURCE = distance

        random_seed = self.__widget.seed

        self.__widget.seed = 0 # seed to 0 to ensure a new beam every time
        source_beam = self.get_source_beam(ignore_aperture=True)
        self.__widget.seed = random_seed # restore user seed

        slits_beam = ShadowBeam.traceFromOE(source_beam, ShadowOpticalElement(slits_oe), widget_class_name="ScreenSlits")

        # good only
        good_only = numpy.where(slits_beam._beam.rays[:, 9] == 1)

        source_beam._beam.rays = source_beam._beam.rays[good_only]

        return source_beam

    class __MockUndulatorHybrid(MockWidget, HybridUndulatorAttributes):
        def __init__(self, storage_ring=StorageRing.APS, verbose=False):
            MockWidget.__init__(self, verbose, workspace_units=2)

            self.distribution_source = 0  # SRW
            self.optimize_source = 0
            self.polarization = 1
            self.coherent_beam = 0
            self.phase_diff = 0.0
            self.polarization_degree = 1.0
            self.max_number_of_rejected_rays = 0

            self.use_harmonic = 2
            self.harmonic_number = 1
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

            if storage_ring == StorageRing.APS:
                self.electron_energy_in_GeV     = ElectronBeamAPS.energy_in_GeV
                self.electron_energy_spread     = ElectronBeamAPS.energy_spread
                self.ring_current               = ElectronBeamAPS.ring_current
                self.electron_beam_size_h       = ElectronBeamAPS.sigma_x
                self.electron_beam_size_v       = ElectronBeamAPS.sigma_z
                self.electron_beam_divergence_h = ElectronBeamAPS.sigdi_x
                self.electron_beam_divergence_v = ElectronBeamAPS.sigdi_z

                self.source_dimension_wf_h_slit_gap = 0.005
                self.source_dimension_wf_v_slit_gap = 0.001
                self.source_dimension_wf_h_slit_points = 500
                self.source_dimension_wf_v_slit_points = 100
                self.source_dimension_wf_distance = 10.0

            elif storage_ring == StorageRing.APS_U:
                self.electron_energy_in_GeV     = ElectronBeamAPS_U.energy_in_GeV
                self.electron_energy_spread     = ElectronBeamAPS_U.energy_spread
                self.ring_current               = ElectronBeamAPS_U.ring_current
                self.electron_beam_size_h       = ElectronBeamAPS_U.sigma_x
                self.electron_beam_size_v       = ElectronBeamAPS_U.sigma_z
                self.electron_beam_divergence_h = ElectronBeamAPS_U.sigdi_x
                self.electron_beam_divergence_v = ElectronBeamAPS_U.sigdi_z

                self.source_dimension_wf_h_slit_gap = 0.001
                self.source_dimension_wf_v_slit_gap = 0.001
                self.source_dimension_wf_h_slit_points = 100
                self.source_dimension_wf_v_slit_points = 100
                self.source_dimension_wf_distance = 10.0

            self.type_of_initialization = 0

            self.horizontal_range_modification_factor_at_resizing = 0.5
            self.horizontal_resolution_modification_factor_at_resizing = 5.0
            self.vertical_range_modification_factor_at_resizing = 0.5
            self.vertical_resolution_modification_factor_at_resizing = 5.0

            self.auto_expand = 0
            self.auto_expand_rays = 0

            self.kind_of_sampler = 1
            self.save_srw_result = 0
