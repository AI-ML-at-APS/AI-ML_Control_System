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
from datetime import date

from aps.common.registry import AlreadyInitializedError
from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

import scipy.constants as codata
M2EV = codata.c*codata.h/codata.e  # lambda(m)  = m2eV / energy(eV)

APPLICATION_NAME = "AA-28ID-DEFAULT-VALUES"

try:
    register_ini_instance(IniMode.LOCAL_FILE,
                          ini_file_name="default-values.ini",
                          application_name=APPLICATION_NAME,
                          verbose=False)
except AlreadyInitializedError: pass

ini_file = get_registered_ini_instance(APPLICATION_NAME)

today = date.today()
year  = today.year
month = today.month-1
months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]

root_directory = "/data/Beamline28IDB_" + months[month] + str(year) + "/"

_ROOT_DIRECTORY = ini_file.get_string_from_ini( section="Directories", key="Root-Directory", default=root_directory)
_ENERGY         = ini_file.get_float_from_ini(  section="Execution",   key="Energy",         default=20000)
_PERIOD         = ini_file.get_float_from_ini(  section="Execution",   key="Period",         default=30)
_N_CYCLES       = ini_file.get_float_from_ini(  section="Execution",   key="N-Cycles",       default=100)

ini_file.set_value_at_ini(section="Directories", key="Root-Directory", value=_ROOT_DIRECTORY)
ini_file.set_value_at_ini(section="Execution",   key="Energy",         value=_ENERGY)
ini_file.set_value_at_ini(section="Execution",   key="Period",         value=_PERIOD)
ini_file.set_value_at_ini(section="Execution",   key="N-Cycles",       value=_N_CYCLES)
ini_file.push()

class DefaultValues:
    ROOT_DIRECTORY = _ROOT_DIRECTORY
    ENERGY         = _ENERGY
    WAVELENGTH     = M2EV/_ENERGY
    PERIOD         = _PERIOD
    N_CYCLES       = _N_CYCLES
