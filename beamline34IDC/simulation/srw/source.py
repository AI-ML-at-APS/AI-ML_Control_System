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

from beamline34IDC.util.common import m2ev

from beamline34IDC.simulation.interfaces.source_interface import AbstractSource, Sources, StorageRing, ElectronBeamAPS_U, ElectronBeamAPS

def srw_source_factory_method(kind_of_source=Sources.GAUSSIAN):
    if kind_of_source == Sources.GAUSSIAN:
        raise ValueError("SRW simulation doesn't provide this kind of source")
    elif kind_of_source == Sources.UNDULATOR:
        return __SRWUndulatorSource()
    else:
        raise ValueError("Kind of Source not recognized")

#############################################################################
# Undulator SOURCE
#

class __SRWUndulatorSource(AbstractSource):
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

    def initialize(self, torage_ring=StorageRing.APS, **kwargs):
        try: verbose = kwargs["verbose"]
        except: verbose = False

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
        return  None
