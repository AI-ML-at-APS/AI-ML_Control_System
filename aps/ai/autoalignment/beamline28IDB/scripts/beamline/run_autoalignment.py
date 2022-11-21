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

import os, sys
from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance
from aps.common.scripts.script_registry import get_registered_running_script_instance, register_running_script_instance

from aps.ai.autoalignment.beamline28IDB.scripts.beamline.executors.autoalignment_executor import AutoalignmentScript
from aps.ai.autoalignment.beamline28IDB.util.beamline.default_values import DefaultValues

APPLICATION_NAME = "RUN-AUTOALIGNMENT"

def __get_input_parameters(sys_argv):
    register_ini_instance(IniMode.LOCAL_FILE,
                          ini_file_name="run_autoalignment.ini",
                          application_name=APPLICATION_NAME,
                          verbose=False)

    ini_file = get_registered_ini_instance(APPLICATION_NAME)

    root_directory    = ini_file.get_string_from_ini( section="Directories", key="Root-Directory",    default=DefaultValues.ROOT_DIRECTORY)
    energy            = ini_file.get_float_from_ini(  section="Execution",   key="Energy",            default=DefaultValues.ENERGY)
    period            = ini_file.get_float_from_ini(  section="Execution",   key="Period",            default=DefaultValues.PERIOD)
    n_cycles          = ini_file.get_float_from_ini(  section="Execution",   key="N-Cycles",          default=DefaultValues.N_CYCLES)
    get_new_reference = ini_file.get_boolean_from_ini(section="Execution",   key="Get-New-Reference", default=False)
    simulation_mode   = False
    mocking_mode      = False
    regenerate_ini    = False
    exit_script       = False

    if len(sys_argv) > 2:
        for i in range(2, len(sys_argv)):
            if "-pd"     == sys_argv[i][:3]: period = int(sys_argv[i][3:])
            elif "-nc"   == sys_argv[i][:3]: n_cycles = int(sys_argv[i][3:])
            elif "-nr"   == sys_argv[i][:3]: get_new_reference = int(sys_argv[i][3:])==1
            elif "-ri"   == sys_argv[i][:3]: exit_script = regenerate_ini   = True
            elif "-sim"  == sys_argv[i][:4]: simulation_mode = True
            elif "-mock" == sys_argv[i][:5]: mocking_mode = True
            elif "--h"   == sys_argv[i][:3]:
                print("Run Autolignment\n\npython -m aps.ai.autolignment 28ID AA <options>\n\n" +
                      "Options: -pd <period in minutes (int)>\n" +
                      "         -nc <number of cycles>\n" +
                      "         -nr <get new reference image 0(No)/1(Yes)>\n" +
                      "         -sim (run optimizer on simulation)\n" +
                      "         -mock (fake execution, for test purposes)\n" +
                      "         -ri (to regenerate ini file with default value)>")
                exit_script = True

    ini_file = get_registered_ini_instance(APPLICATION_NAME)
    if regenerate_ini:
        ini_file.set_value_at_ini(section="Directories", key="Root-Directory",    value=DefaultValues.ROOT_DIRECTORY)
        ini_file.set_value_at_ini(section="Execution",   key="Energy",            value=DefaultValues.ENERGY)
        ini_file.set_value_at_ini(section="Execution",   key="Period",            value=DefaultValues.PERIOD)
        ini_file.set_value_at_ini(section="Execution",   key="N-Cycles",          value=DefaultValues.N_CYCLES)
        ini_file.set_value_at_ini(section="Execution",   key="Get-New-Reference", value=False)

        print("File ini regenerated with default values in\n" + os.path.abspath(os.curdir))
    else:
        ini_file.set_value_at_ini(section="Directories", key="Root-Directory",    value=root_directory)
        ini_file.set_value_at_ini(section="Execution",   key="Energy",            value=energy)
        ini_file.set_value_at_ini(section="Execution",   key="Period",            value=period)
        ini_file.set_value_at_ini(section="Execution",   key="N-Cycles",          value=n_cycles)
        ini_file.set_value_at_ini(section="Execution",   key="Get-New-Reference", value=get_new_reference)

    ini_file.push()

    try: print("Variable QT_QPA_PLATFORM=", os.environ["QT_QPA_PLATFORM"])
    except: print("Variable QT_QPA_PLATFORM not defined")

    if exit_script: sys.exit(0)

    return root_directory, energy, period, n_cycles, get_new_reference, mocking_mode, simulation_mode


def run_script(sys_argv):
    if "linux" in sys.platform: os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    root_directory, energy, period, n_cycles, get_new_reference, mocking_mode, simulation_mode = __get_input_parameters(sys_argv)

    script = AutoalignmentScript(root_directory=root_directory,
                                 energy=energy,
                                 period=period,
                                 get_new_reference=get_new_reference,
                                 mocking_mode=mocking_mode,
                                 simulation_mode=simulation_mode
                                 )
    register_running_script_instance(script)

    script.execute_script()

# ===================================================================================================
# ===================================================================================================
# ===================================================================================================

if __name__ == "__main__":
    try:
        run_script(sys.argv)
    except KeyboardInterrupt:
        running_script = get_registered_running_script_instance()
        if not running_script is None: running_script.manage_keyboard_interrupt()
        else: print("\nScript interrupted by user")
