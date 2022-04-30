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

from beamline34IDC.simulation.facade.primary_optics_interface import AbstractPrimaryOptics

from wofrysrw.beamline.srw_beamline import SRWBeamline
from wofrysrw.propagator.wavefront2D.srw_wavefront import WavefrontParameters, WavefrontPrecisionParameters

def srw_primary_optics_factory_method():
    return __PrimaryOpticsSystem()

class __PrimaryOpticsSystem(AbstractPrimaryOptics):
    def __init__(self):
        self.__beamline = None
        self.__source_wavefront = None

    def initialize(self, source_photon_beam, **kwargs):
        self.__source_wavefront = source_photon_beam

        self.__beamline = SRWBeamline()

    def get_photon_beam(self, **kwargs): pass


'''
####################################################
# BEAMLINE

srw_oe_array = []
srw_pp_array = []

oe_0=SRWLOptA(_shape='r',
               _ap_or_ob='a',
               _Dx=0.0001,
               _Dy=0.01,
               _x=0.0,
               _y=0.0)

pp_oe_0 = [0,0,1.0,0,0,0.2,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(oe_0)
srw_pp_array.append(pp_oe_0)

drift_before_oe_1 = SRWLOptD(2.8)
pp_drift_before_oe_1 = [0,0,1.0,2,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(drift_before_oe_1)
srw_pp_array.append(pp_drift_before_oe_1)

acceptance_slits_oe_1=SRWLOptA(_shape='r',
               _ap_or_ob='a',
               _Dx=0.02,
               _Dy=0.0024999895836482406,
               _x=0.0,
               _y=0.0)

oe_1 = SRWLOptMirPl(_size_tang=0.5,
                     _size_sag=0.02,
                     _ap_shape='r',
                     _sim_meth=2,
                     _treat_in_out=1,
                     _nvx=0,
                     _nvy=0.9999875000260386,
                     _nvz=-0.004999979167296481,
                     _tvx=0,
                     _tvy=-0.004999979167296481,
                     _x=0.0,
                     _y=0.0)
oe_1.set_dim_sim_meth(_size_tang=0.5,
                      _size_sag=0.02,
                      _ap_shape='r',
                      _sim_meth=2,
                      _treat_in_out=1)
oe_1.set_orient(_nvx=0,
                 _nvy=0.9999875000260386,
                 _nvz=-0.004999979167296481,
                 _tvx=0,
                 _tvy=-0.004999979167296481,
                 _x=0.0,
                 _y=0.0)


pp_acceptance_slits_oe_1 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]
pp_oe_1 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(acceptance_slits_oe_1)
srw_pp_array.append(pp_acceptance_slits_oe_1)

srw_oe_array.append(oe_1)
srw_pp_array.append(pp_oe_1)

drift_before_oe_2 = SRWLOptD(20.902583)
pp_drift_before_oe_2 = [0,0,1.0,2,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(drift_before_oe_2)
srw_pp_array.append(pp_drift_before_oe_2)


oe_3=SRWLOptA(_shape='r',
               _ap_or_ob='a',
               _Dx=3e-05,
               _Dy=7e-05,
               _x=0.0,
               _y=0.0)

pp_oe_3 = [0,0,1.0,0,0,0.1,5.0,0.1,5.0,0,0.0,0.0]

srw_oe_array.append(oe_3)
srw_pp_array.append(pp_oe_3)

drift_before_oe_4 = SRWLOptD(0.15)
pp_drift_before_oe_4 = [0,0,1.0,2,0,1.0,2.0,1.0,2.0,0,0.0,0.0]

srw_oe_array.append(drift_before_oe_4)
srw_pp_array.append(pp_drift_before_oe_4)


acceptance_slits_oe_5=SRWLOptA(_shape='r',
               _ap_or_ob='a',
               _Dx=0.02,
               _Dy=0.00029999954993170305,
               _x=0.0,
               _y=0.0)

oe_5 = SRWLOptMirEl(_size_tang=0.1,
                     _size_sag=0.02,
                     _p=50.66798299999999,
                     _q=0.22100000000000364,
                     _ang_graz=0.0029999999993150024,
                     _ap_shape='r',
                     _sim_meth=2,
                     _treat_in_out=1,
                     _nvx=0,
                     _nvy=0.999995500003377,
                     _nvz=-0.0029999954993170305,
                     _tvx=0,
                     _tvy=-0.0029999954993170305,
                     _x=0.0,
                     _y=0.0)
oe_5.set_dim_sim_meth(_size_tang=0.1,
                      _size_sag=0.02,
                      _ap_shape='r',
                      _sim_meth=2,
                      _treat_in_out=1)
oe_5.set_orient(_nvx=0,
                 _nvy=0.999995500003377,
                 _nvz=-0.0029999954993170305,
                 _tvx=0,
                 _tvy=-0.0029999954993170305,
                 _x=0.0,
                 _y=0.0)


height_profile_data = srwl_uti_read_data_cols('VKB-LTP_srw.dat',
                                              _str_sep='\t')
optTrEr_oe_5 = srwl_opt_setup_surf_height_2d(_height_prof_data=height_profile_data,
                                                        _ang=0.0029999999993150024,
                                                        _dim='y',
                                                        _amp_coef=1.0)

pp_acceptance_slits_oe_5 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]
pp_oe_5 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(acceptance_slits_oe_5)
srw_pp_array.append(pp_acceptance_slits_oe_5)

srw_oe_array.append(oe_5)
srw_pp_array.append(pp_oe_5)

srw_oe_array.append(optTrEr_oe_5)
srw_pp_array.append([0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0])

drift_after_oe_5 = SRWLOptD(0.101)
pp_drift_after_oe_5 = [0,0,1.0,2,0,1.0,2.0,1.0,2.0,0,0.0,0.0]

srw_oe_array.append(drift_after_oe_5)
srw_pp_array.append(pp_drift_after_oe_5)

acceptance_slits_oe_6=SRWLOptA(_shape='r',
               _ap_or_ob='a',
               _Dx=0.00029999954993170305,
               _Dy=0.02,
               _x=0.0,
               _y=0.0)

oe_6 = SRWLOptMirEl(_size_tang=0.1,
                     _size_sag=0.02,
                     _p=50.76898299999999,
                     _q=0.12000000000000455,
                     _ang_graz=0.0029999999993150024,
                     _ap_shape='r',
                     _sim_meth=2,
                     _treat_in_out=1,
                     _nvx=-0.999995500003377,
                     _nvy=0,
                     _nvz=-0.0029999954993170305,
                     _tvx=0.0029999954993170305,
                     _tvy=0,
                     _x=0.0,
                     _y=0.0)
oe_6.set_dim_sim_meth(_size_tang=0.1,
                      _size_sag=0.02,
                      _ap_shape='r',
                      _sim_meth=2,
                      _treat_in_out=1)
oe_6.set_orient(_nvx=-0.999995500003377,
                 _nvy=0,
                 _nvz=-0.0029999954993170305,
                 _tvx=0.0029999954993170305,
                 _tvy=0,
                 _x=0.0,
                 _y=0.0)


height_profile_data = srwl_uti_read_data_cols('HKB-LTP_srw.dat',
                                              _str_sep='\t')
optTrEr_oe_6 = srwl_opt_setup_surf_height_2d(_height_prof_data=height_profile_data,
                                                        _ang=0.0029999999993150024,
                                                        _dim='x',
                                                        _amp_coef=1.0)

pp_acceptance_slits_oe_6 = [0,0,1.0,0,0,1.0,2.0,1.0,2.0,0,0.0,0.0]
pp_oe_6 = [0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(acceptance_slits_oe_6)
srw_pp_array.append(pp_acceptance_slits_oe_6)

srw_oe_array.append(oe_6)
srw_pp_array.append(pp_oe_6)

srw_oe_array.append(optTrEr_oe_6)
srw_pp_array.append([0,0,1.0,0,0,1.0,1.0,1.0,1.0,0,0.0,0.0])

drift_before_oe_7 = SRWLOptD(0.12)
pp_drift_before_oe_7 = [0,0,1.0,1,0,1.0,1.0,1.0,1.0,0,0.0,0.0]

srw_oe_array.append(drift_before_oe_7)
srw_pp_array.append(pp_drift_before_oe_7)



'''
