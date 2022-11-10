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
from aps.common.scripts.script_registry import get_registered_running_script_instance

from aps.ai.autoalignment.beamline28IDB.scripts.beamline.run_autoalignment import run_script as aa_run_script
from aps.ai.autoalignment.beamline28IDB.scripts.beamline.run_autofocusing  import run_script as af_run_script

def run_script(sys_argv):
    def show_help(error=False):
        print("")
        if error:
            print("*************************************************************")
            print("********              Command not valid!             ********")
            print("*************************************************************\n")
        else:
            print("=============================================================")
            print("  WELCOME TO AUTO-ALIGNMENT/AUTOFOCUS AI-DRIVEN CONTROLLER   ")
            print("                  28-ID-B BEAMLINE SCRIPTS                   ")
            print("=============================================================\n")
        print("To launch a script:         python -m aps.ai.autoalignment 28ID <script id> <options>")
        print("To show help of a script:   python -m aps.ai.autoalignment 28ID <script id> --h")
        print("To show this help:          python -m aps.ai.autoalignment 28ID --h")
        print("* Available scripts:\n" +
              "    1) Auto-alignment, id: AA\n" +
              "    2) Auto-focusing,  id: AF\n")

    if len(sys_argv) == 2 or sys_argv[2] == "--h":
        show_help()
    else:
        if sys_argv[2]   == "AA": aa_run_script(sys_argv)
        elif sys_argv[2] == "AF": af_run_script(sys_argv)
        else: show_help(error=True)

# ===================================================================================================
# ===================================================================================================
# ===================================================================================================

import sys
if __name__=="__main__":
    try:
        run_script(sys.argv)
    except KeyboardInterrupt as e:
        running_script = get_registered_running_script_instance()
        if not running_script is None: running_script.manage_keyboard_interrupt()
        else: print("\nScript interrupted by user")
