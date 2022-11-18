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


'''

           self.__focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.HARDWARE, implementor=HW_Implementors.EPICS)
            self.__focusing_system.initialize()

'''

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

from aps.ai.autoalignment.beamline28IDB.util.beamline.default_values import DefaultValues
from aps.ai.autoalignment.beamline28IDB.scripts.beamline.executors.test_hardware_executor import TestHardwareScript, HardwareTestParameters

APPLICATION_NAME = "TEST-HARDWARE"

def __get_input_parameters(sys_argv):
    register_ini_instance(IniMode.LOCAL_FILE,
                          ini_file_name="test_motors.ini",
                          application_name=APPLICATION_NAME,
                          verbose=False)

    ini_file = get_registered_ini_instance(APPLICATION_NAME)
    
    hardware_test_parameters = HardwareTestParameters()
    
    root_directory  = ini_file.get_string_from_ini( section="Directories", key="Root-Directory", default=DefaultValues.ROOT_DIRECTORY)
    energy          = ini_file.get_float_from_ini(  section="Execution",   key="Energy",         default=DefaultValues.ENERGY)

    test_h_pitch                = ini_file.get_boolean_from_ini(section="Tests",   key="test-h-pitch",                default=hardware_test_parameters.test_h_pitch               )
    h_pitch_absolute_move       = ini_file.get_float_from_ini(  section="Tests",   key="h-pitch-absolute-move",       default=hardware_test_parameters.h_pitch_absolute_move      )
    h_pitch_relative_move       = ini_file.get_float_from_ini(  section="Tests",   key="h-pitch-relative-move",       default=hardware_test_parameters.h_pitch_relative_move      )
    test_h_translation          = ini_file.get_boolean_from_ini(section="Tests",   key="test-h-translation",          default=hardware_test_parameters.test_h_translation         )
    h_translation_absolute_move = ini_file.get_float_from_ini(  section="Tests",   key="h-translation-absolute-move", default=hardware_test_parameters.h_translation_absolute_move)
    h_translation_relative_move = ini_file.get_float_from_ini(  section="Tests",   key="h-translation-relative-move", default=hardware_test_parameters.h_translation_relative_move)
    test_h_bender_1             = ini_file.get_boolean_from_ini(section="Tests",   key="test-h-bender-1",             default=hardware_test_parameters.test_h_bender_1            )
    h_bender_1_absolute_move    = ini_file.get_float_from_ini(  section="Tests",   key="h-bender-1-absolute-move",    default=hardware_test_parameters.h_bender_1_absolute_move   )
    h_bender_1_relative_move    = ini_file.get_float_from_ini(  section="Tests",   key="h-bender-1-relative-move",    default=hardware_test_parameters.h_bender_1_relative_move   )
    test_h_bender_2             = ini_file.get_boolean_from_ini(section="Tests",   key="test-h-bender-2",             default=hardware_test_parameters.test_h_bender_2            )
    h_bender_2_absolute_move    = ini_file.get_float_from_ini(  section="Tests",   key="h-bender-2-absolute-move",    default=hardware_test_parameters.h_bender_2_absolute_move   )
    h_bender_2_relative_move    = ini_file.get_float_from_ini(  section="Tests",   key="h-bender-2-relative-move",    default=hardware_test_parameters.h_bender_2_relative_move   )
    test_v_pitch                = ini_file.get_boolean_from_ini(section="Tests",   key="test-v-pitch",                default=hardware_test_parameters.test_v_pitch               )
    v_pitch_absolute_move       = ini_file.get_float_from_ini(  section="Tests",   key="v-pitch-absolute-move", default=hardware_test_parameters.v_pitch_absolute_move)
    v_pitch_relative_move       = ini_file.get_float_from_ini(  section="Tests",   key="v-pitch-relative-move", default=hardware_test_parameters.v_pitch_relative_move)
    test_v_translation          = ini_file.get_boolean_from_ini(section="Tests",   key="test-v-translation",          default=hardware_test_parameters.test_v_translation         )
    v_translation_absolute_move = ini_file.get_float_from_ini(  section="Tests",   key="v-translation-absolute-move", default=hardware_test_parameters.v_translation_absolute_move)
    v_translation_relative_move = ini_file.get_float_from_ini(  section="Tests",   key="v-translation-relative-move", default=hardware_test_parameters.v_translation_relative_move)
    test_v_bender               = ini_file.get_boolean_from_ini(section="Tests",   key="test-v-bender",               default=hardware_test_parameters.test_v_bender              )
    v_bender_absolute_move      = ini_file.get_float_from_ini(  section="Tests",   key="v-bender-absolute-move",      default=hardware_test_parameters.v_bender_absolute_move     )
    v_bender_relative_move      = ini_file.get_float_from_ini(  section="Tests",   key="v-bender-relative-move",      default=hardware_test_parameters.v_bender_relative_move     )
    test_detector               = ini_file.get_boolean_from_ini(section="Tests",   key="test-detector",               default=hardware_test_parameters.test_detector              )
    plot_motors                 = ini_file.get_boolean_from_ini(section="Tests",   key="plot-motors",                 default=hardware_test_parameters.plot_motors              )

    regenerate_ini                 = False
    exit_script                    = False

    if len(sys_argv) > 2:
        for i in range(2, len(sys_argv)):
            if "-ri"   == sys_argv[i][:3]: exit_script = regenerate_ini   = True
            elif "--h"   == sys_argv[i][:3]:
                print("Test Hardware\n\npython -m aps.ai.autolignment 28ID TH <options>\n\n" +
                      "Options: -ri (to regenerate ini file with default value)>")
                exit_script = True

    ini_file = get_registered_ini_instance(APPLICATION_NAME)

    if regenerate_ini:
        ini_file.set_value_at_ini(section="Directories", key="Root-Directory",  value=DefaultValues.ROOT_DIRECTORY)
        ini_file.set_value_at_ini(section="Execution",   key="Energy",          value=DefaultValues.ENERGY)
        
        ini_file.set_value_at_ini(section="Tests", key="test-h-pitch",                value=hardware_test_parameters.test_h_pitch               )
        ini_file.set_value_at_ini(section="Tests", key="h-pitch-absolute-move",       value=hardware_test_parameters.h_pitch_absolute_move      )
        ini_file.set_value_at_ini(section="Tests", key="h-pitch-relative-move",       value=hardware_test_parameters.h_pitch_relative_move      )
        ini_file.set_value_at_ini(section="Tests", key="test-h-translation",          value=hardware_test_parameters.test_h_translation         )
        ini_file.set_value_at_ini(section="Tests", key="h-translation-absolute-move", value=hardware_test_parameters.h_translation_absolute_move)
        ini_file.set_value_at_ini(section="Tests", key="h-translation-relative-move", value=hardware_test_parameters.h_translation_relative_move)
        ini_file.set_value_at_ini(section="Tests", key="test-h-bender-1",             value=hardware_test_parameters.test_h_bender_1            )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-1-absolute-move",    value=hardware_test_parameters.h_bender_1_absolute_move   )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-1-relative-move",    value=hardware_test_parameters.h_bender_1_relative_move   )
        ini_file.set_value_at_ini(section="Tests", key="test-h-bender-2",             value=hardware_test_parameters.test_h_bender_2            )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-2-absolute-move",    value=hardware_test_parameters.h_bender_2_absolute_move   )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-2-relative-move",    value=hardware_test_parameters.h_bender_2_relative_move   )
        ini_file.set_value_at_ini(section="Tests", key="test-v-pitch",                value=hardware_test_parameters.test_v_pitch               )
        ini_file.set_value_at_ini(section="Tests", key="v-pitch-absolute-move",       value=hardware_test_parameters.v_pitch_absolute_move)
        ini_file.set_value_at_ini(section="Tests", key="v-pitch-relative-move",       value=hardware_test_parameters.v_pitch_relative_move)
        ini_file.set_value_at_ini(section="Tests", key="test-v-translation",          value=hardware_test_parameters.test_v_translation         )
        ini_file.set_value_at_ini(section="Tests", key="v-translation-absolute-move", value=hardware_test_parameters.v_translation_absolute_move)
        ini_file.set_value_at_ini(section="Tests", key="v-translation-relative-move", value=hardware_test_parameters.v_translation_relative_move)
        ini_file.set_value_at_ini(section="Tests", key="test-v-bender",               value=hardware_test_parameters.test_v_bender              )
        ini_file.set_value_at_ini(section="Tests", key="v-bender-absolute-move",      value=hardware_test_parameters.v_bender_absolute_move     )
        ini_file.set_value_at_ini(section="Tests", key="v-bender-relative-move",      value=hardware_test_parameters.v_bender_relative_move     )
        ini_file.set_value_at_ini(section="Tests", key="test-detector",               value=hardware_test_parameters.test_detector              )
        ini_file.set_value_at_ini(section="Tests", key="plot-motors",                 value=hardware_test_parameters.plot_motors              )

        print("File ini regenerated with default values in\n" + os.path.abspath(os.curdir))
    else:
        ini_file.set_value_at_ini(section="Directories", key="Root-Directory",                 value=root_directory)
        ini_file.set_value_at_ini(section="Execution",   key="Energy",                         value=energy)

        ini_file.set_value_at_ini(section="Tests", key="test-h-pitch",                value=test_h_pitch               )
        ini_file.set_value_at_ini(section="Tests", key="h-pitch-absolute-move",       value=h_pitch_absolute_move      )
        ini_file.set_value_at_ini(section="Tests", key="h-pitch-relative-move",       value=h_pitch_relative_move      )
        ini_file.set_value_at_ini(section="Tests", key="test-h-translation",          value=test_h_translation         )
        ini_file.set_value_at_ini(section="Tests", key="h-translation-absolute-move", value=h_translation_absolute_move)
        ini_file.set_value_at_ini(section="Tests", key="h-translation-relative-move", value=h_translation_relative_move)
        ini_file.set_value_at_ini(section="Tests", key="test-h-bender-1",             value=test_h_bender_1            )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-1-absolute-move",    value=h_bender_1_absolute_move   )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-1-relative-move",    value=h_bender_1_relative_move   )
        ini_file.set_value_at_ini(section="Tests", key="test-h-bender-2",             value=test_h_bender_2            )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-2-absolute-move",    value=h_bender_2_absolute_move   )
        ini_file.set_value_at_ini(section="Tests", key="h-bender-2-relative-move",    value=h_bender_2_relative_move   )
        ini_file.set_value_at_ini(section="Tests", key="test-v-pitch",                value=test_v_pitch               )
        ini_file.set_value_at_ini(section="Tests", key="v-pitch-absolute-move",       value=v_pitch_absolute_move      )
        ini_file.set_value_at_ini(section="Tests", key="v-pitch-relative-move",       value=v_pitch_relative_move      )
        ini_file.set_value_at_ini(section="Tests", key="test-v-translation",          value=test_v_translation         )
        ini_file.set_value_at_ini(section="Tests", key="v-translation-absolute-move", value=v_translation_absolute_move)
        ini_file.set_value_at_ini(section="Tests", key="v-translation-relative-move", value=v_translation_relative_move)
        ini_file.set_value_at_ini(section="Tests", key="test-v-bender",               value=test_v_bender              )
        ini_file.set_value_at_ini(section="Tests", key="v-bender-absolute-move",      value=v_bender_absolute_move     )
        ini_file.set_value_at_ini(section="Tests", key="v-bender-relative-move",      value=v_bender_relative_move     )
        ini_file.set_value_at_ini(section="Tests", key="test-detector",               value=test_detector              )
        ini_file.set_value_at_ini(section="Tests", key="plot-motors",                 value=plot_motors              )

    ini_file.push()

    try: print("Variable QT_QPA_PLATFORM=", os.environ["QT_QPA_PLATFORM"])
    except: print("Variable QT_QPA_PLATFORM not defined")

    if exit_script: sys.exit(0)

    hardware_test_parameters.test_h_pitch                = test_h_pitch
    hardware_test_parameters.h_pitch_absolute_move       = h_pitch_absolute_move
    hardware_test_parameters.h_pitch_relative_move       = h_pitch_relative_move
    hardware_test_parameters.test_h_translation          = test_h_translation
    hardware_test_parameters.h_translation_absolute_move = h_translation_absolute_move
    hardware_test_parameters.h_translation_relative_move = h_translation_relative_move
    hardware_test_parameters.test_h_bender_1             = test_h_bender_1
    hardware_test_parameters.h_bender_1_absolute_move    = h_bender_1_absolute_move
    hardware_test_parameters.h_bender_1_relative_move    = h_bender_1_relative_move
    hardware_test_parameters.test_h_bender_2             = test_h_bender_2
    hardware_test_parameters.h_bender_2_absolute_move    = h_bender_2_absolute_move
    hardware_test_parameters.h_bender_2_relative_move    = h_bender_2_relative_move
    hardware_test_parameters.test_v_pitch                = test_v_pitch
    hardware_test_parameters.v_pitch_absolute_move       = v_pitch_absolute_move
    hardware_test_parameters.v_pitch_relative_move       = v_pitch_relative_move
    hardware_test_parameters.test_v_translation          = test_v_translation
    hardware_test_parameters.v_translation_absolute_move = v_translation_absolute_move
    hardware_test_parameters.v_translation_relative_move = v_translation_relative_move
    hardware_test_parameters.test_v_bender               = test_v_bender
    hardware_test_parameters.v_bender_absolute_move      = v_bender_absolute_move
    hardware_test_parameters.v_bender_relative_move      = v_bender_relative_move
    hardware_test_parameters.test_detector               = test_detector
    hardware_test_parameters.plot_motors                 = plot_motors

    return root_directory, energy, hardware_test_parameters


def run_script(sys_argv):
    if "linux" in sys.platform: os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    root_directory, energy, hardware_test_parameters = __get_input_parameters(sys_argv)

    script = TestHardwareScript(root_directory=root_directory,
                                energy=energy,
                                hardware_test_parameters=hardware_test_parameters)
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
