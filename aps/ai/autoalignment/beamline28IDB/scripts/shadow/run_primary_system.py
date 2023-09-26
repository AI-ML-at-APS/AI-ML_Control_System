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
# be clearly marked, so as not to confuse it witn available   #
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
import os

import aps
from aps.ai.autoalignment.beamline28IDB.simulation.facade.primary_optics_factory import primary_optics_factory_method

from aps.ai.autoalignment.common.simulation.facade.source_factory import Implementors
from aps.ai.autoalignment.common.util.shadow.common import load_source_beam, save_shadow_beam, plot_shadow_beam_spatial_distribution, PlotMode
from aps.ai.autoalignment.common.util import clean_up

import Shadow
from pathlib import Path

if __name__ == "__main__":
    verbose = True

    os.chdir(Path(aps.__file__).parent.parent / "work_directory/28-ID")

    clean_up()

    # Source -------------------------
    source_beam = load_source_beam("undulator_source.dat")

    # Primary Optics System -------------------------
    primary_system = primary_optics_factory_method(implementor=Implementors.SHADOW)
    primary_system.initialize(source_photon_beam=source_beam, relative_source_position=1300)

    output_beam = primary_system.get_photon_beam(verbose=verbose)

    save_shadow_beam(output_beam, "primary_optics_system_beam.dat")

    plot_shadow_beam_spatial_distribution(output_beam, plot_mode=PlotMode.BOTH)#, xrange=[-0.2, 0.2], yrange=[-0.2, 0.2])

    Shadow.ShadowTools.histo1(output_beam._beam, 11, nolost=1, ref=23, xrange=[19650, 20150])

    clean_up()
