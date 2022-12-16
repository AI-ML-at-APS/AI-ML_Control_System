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

from aps.ai.autoalignment.beamline28IDB.scripts.beamline.executors.generic_executor import GenericScript, OptimizationParameters
import aps.ai.autoalignment.beamline28IDB.optimization.configs as configs
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import Layout
from aps.ai.autoalignment.beamline28IDB.optimization.common import OptimizationCriteria, SelectionAlgorithm

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

DEFAULT_RANDOM_SEED = numpy.random.randint(100001)

APPLICATION_NAME = "AUTOALIGNMENT"

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="autoalignment.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

hb_pitch = ini_file.get_list_from_ini(section="Motor-Ranges", key="HKB-Pitch",       default=configs.DEFAULT_MOVEMENT_RANGES["hb_pitch"], type=float)  # in degrees
hb_trans = ini_file.get_list_from_ini(section="Motor-Ranges", key="HKB-Translation", default=configs.DEFAULT_MOVEMENT_RANGES["hb_trans"], type=float)  # in mm
vb_pitch = ini_file.get_list_from_ini(section="Motor-Ranges", key="VKB-Pitch",       default=configs.DEFAULT_MOVEMENT_RANGES["vb_pitch"], type=float)  # in degrees
vb_trans = ini_file.get_list_from_ini(section="Motor-Ranges", key="VKB-Translation", default=configs.DEFAULT_MOVEMENT_RANGES["vb_trans"], type=float)  # in mm

bound_hb_pitch = ini_file.get_list_from_ini(section="Motor-Boundaries", key="Boundaries-HKB-Pitch",       default=[-0.2, 0.2], type=float)  # in degrees
bound_hb_trans = ini_file.get_list_from_ini(section="Motor-Boundaries", key="Boundaries-HKB-Translation", default=[-5.0, 5.0], type=float)  # in mm
bound_vb_pitch = ini_file.get_list_from_ini(section="Motor-Boundaries", key="Boundaries-VKB-Pitch",       default=[-0.2, 0.2], type=float)  # in degrees
bound_vb_trans = ini_file.get_list_from_ini(section="Motor-Boundaries", key="Boundaries-VKB-Translation", default=[-5.0, 5.0], type=float)  # in mm

crop_threshold   = ini_file.get_float_from_ini(section="Hardware-Setup", key="Crop-Threshold",   default=None)
crop_strip_width = ini_file.get_int_from_ini(  section="Hardware-Setup", key="Crop-Strip-Width", default=50)

pitch_only                    = ini_file.get_boolean_from_ini(section="Optimization-Parameters", key="Pitch-Only",                    default=True)
sum_intensity_soft_constraint = ini_file.get_float_from_ini(  section="Optimization-Parameters", key="Sum-Intensity-Soft-Constraint", default=7e3)
sum_intensity_hard_constraint = ini_file.get_float_from_ini(  section="Optimization-Parameters", key="Sum-Intensity-Hard-Constraint", default=6.5e3)
loss_parameters               = ini_file.get_list_from_ini(   section="Optimization-Parameters", key="Loss-Parameters",               default=[OptimizationCriteria.CENTROID], type=str)
reference_position            = ini_file.get_list_from_ini(   section="Optimization-Parameters", key="Reference-Position",            default=[0.0, 0.0], type=float)
reference_size                = ini_file.get_list_from_ini(   section="Optimization-Parameters", key="Reference-Size",                default=[0.0, 0.0], type=float)
moo_thresholds                = ini_file.get_list_from_ini(   section="Optimization-Parameters", key="Moo-Thresholds",                default=[OptimizationCriteria.CENTROID], type=str)
moo_threshold_position        = ini_file.get_float_from_ini(  section="Optimization-Parameters", key="Moo-Thresholds-Position",       default=0.2)
moo_threshold_size            = ini_file.get_float_from_ini(  section="Optimization-Parameters", key="Moo-Thresholds-Size",           default=0.2)
multi_objective_optimization  = ini_file.get_boolean_from_ini(section="Optimization-Parameters", key="Multi-Objective-Optimization",  default=False)
selection_algorithm           = ini_file.get_string_from_ini( section="Optimization-Parameters", key="Selection-Algorithm",           default=SelectionAlgorithm.NASH_EQUILIBRIUM)
n_trials                      = ini_file.get_int_from_ini(    section="Optimization-Parameters", key="N-Trials",                      default=100)

save_images                          =  ini_file.get_boolean_from_ini(section="Calculation-Parameters", key="Save-Images",                   default=False)
every_n_images                       =  ini_file.get_int_from_ini(    section="Calculation-Parameters", key="Every-N-Images",                default=5)
add_noise                            =  ini_file.get_boolean_from_ini(section="Calculation-Parameters", key="Add-Noise",                     default=False)
noise                                =  ini_file.get_float_from_ini(  section="Calculation-Parameters", key="Noise",                         default=None)
percentage_fluctutation              =  ini_file.get_float_from_ini(  section="Calculation-Parameters", key="Percentage-Fluctuation",        default=10.0)
calculate_over_noise                 =  ini_file.get_boolean_from_ini(section="Calculation-Parameters", key="Calculate-Over-Noise",          default=True)
noise_threshold                      =  ini_file.get_float_from_ini(  section="Calculation-Parameters", key="Noise-Threshold",               default=1.5)

ini_file.set_list_at_ini(section="Motor-Ranges", key="HKB-Pitch", values_list=hb_pitch)
ini_file.set_list_at_ini(section="Motor-Ranges", key="HKB-Translation", values_list=hb_trans)
ini_file.set_list_at_ini(section="Motor-Ranges", key="VKB-Pitch", values_list=vb_pitch)
ini_file.set_list_at_ini(section="Motor-Ranges", key="VKB-Translation", values_list=vb_trans)

ini_file.set_list_at_ini(section="Motor-Boundaries", key="Boundaries-HKB-Pitch", values_list=bound_hb_pitch)
ini_file.set_list_at_ini(section="Motor-Boundaries", key="Boundaries-HKB-Translation", values_list=bound_hb_trans)
ini_file.set_list_at_ini(section="Motor-Boundaries", key="Boundaries-VKB-Pitch", values_list=bound_vb_pitch)
ini_file.set_list_at_ini(section="Motor-Boundaries", key="Boundaries-VKB-Translation", values_list=bound_vb_trans)

ini_file.set_value_at_ini(section="Hardware-Setup", key="Crop-Threshold",   value=crop_threshold)
ini_file.set_value_at_ini(section="Hardware-Setup", key="Crop-Strip-Width", value=crop_strip_width)

ini_file.set_value_at_ini(section="Optimization-Parameters", key="Pitch-Only", value=pitch_only)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Sum-Intensity-Soft-Constraint", value=sum_intensity_soft_constraint)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Sum-Intensity-Hard-Constraint", value=sum_intensity_hard_constraint)
ini_file.set_list_at_ini(section="Optimization-Parameters", key="Loss-Parameters", values_list=loss_parameters)
ini_file.set_list_at_ini(section="Optimization-Parameters", key="Reference-Position", values_list=reference_position)
ini_file.set_list_at_ini(section="Optimization-Parameters", key="Reference-Size", values_list=reference_size)
ini_file.set_list_at_ini(section="Optimization-Parameters", key="Moo-Thresholds", values_list=moo_thresholds)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Moo-Thresholds-Position", value=moo_threshold_position)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Moo-Thresholds-Size", value=moo_threshold_size)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Multi-Objective-Optimization", value=multi_objective_optimization)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="Selection-Algorithm", value=selection_algorithm)
ini_file.set_value_at_ini(section="Optimization-Parameters", key="N-Trials", value=n_trials)

ini_file.set_value_at_ini(section="Calculation-Parameters", key="Save-Images",                   value=save_images)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Every-N-Images",                value=every_n_images)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Add-Noise",                     value=add_noise)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Noise",                         value=noise)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Percentage-Fluctuation",        value=percentage_fluctutation)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Calculate-Over-Noise",          value=calculate_over_noise)
ini_file.set_value_at_ini(section="Calculation-Parameters", key="Noise-Threshold",               value=noise_threshold)

ini_file.push()

class AAOptimizationParameters(OptimizationParameters):
    def __init__(self):
        super(AAOptimizationParameters, self).__init__()
        
        self.move_motors_ranges["hb_pitch"] = hb_pitch
        self.move_motors_ranges["hb_trans"] = hb_trans
        self.move_motors_ranges["vb_pitch"] = vb_pitch
        self.move_motors_ranges["vb_trans"] = vb_trans
        
        self.move_motors_boundaries["bound_hb_pitch"] = bound_hb_pitch
        self.move_motors_boundaries["bound_hb_trans"] = bound_hb_trans
        self.move_motors_boundaries["bound_vb_pitch"] = bound_vb_pitch
        self.move_motors_boundaries["bound_vb_trans"] = bound_vb_trans

        self.params["pitch_only"]                    =  pitch_only
        self.params["sum_intensity_soft_constraint"] =  sum_intensity_soft_constraint
        self.params["sum_intensity_hard_constraint"] =  sum_intensity_hard_constraint
        self.params["reference_parameters_h_v"]      =  self._get_reference_parameters_h_v(loss_parameters, reference_position, reference_size)
        self.params["loss_parameters"]               =  loss_parameters
        self.params["moo_thresholds"]                =  self._get_moo_thresholds_dict(moo_thresholds, moo_threshold_position, moo_threshold_size, None)
        self.params["multi_objective_optimization"]  =  multi_objective_optimization
        self.params["selection_algorithm"]           =  selection_algorithm
        self.params["n_trials"]                      =  n_trials


class AutoalignmentScript(GenericScript):
    def __init__(self, root_directory, energy, period, n_cycles, get_new_reference, test_mode, mocking_mode, simulation_mode):
        super(AutoalignmentScript, self).__init__(root_directory,
                                                  energy, period,
                                                  n_cycles,
                                                  test_mode,
                                                  mocking_mode,
                                                  simulation_mode,
                                                  save_images,
                                                  every_n_images,
                                                  False,
                                                  add_noise,
                                                  noise,
                                                  percentage_fluctutation,
                                                  calculate_over_noise,
                                                  noise_threshold,
                                                  Layout.AUTO_FOCUSING,
                                                  crop_threshold=crop_threshold,
                                                  crop_strip_width=crop_strip_width)


        self.__get_new_reference = get_new_reference

    def _get_script_name(self):             return "Autoalignment"
    def _get_optimization_parameters(self): return AAOptimizationParameters()

    def _run_preliminary_operations(self, current_cycle):
        if not self._simulation_mode:
            self._parameters.params.focusing_system.set_surface_actuators_to_baseline(baseline=500)
            if current_cycle == 1 and self.__get_new_reference: self.__set_reference()

    def _get_optimizer_moo_thresholds_and_contraints(self, opt_trial):
        # Setting up the optimizer
        constraints = {"sum_intensity": self._optimization_parameters.params["sum_intensity_soft_constraint"]}

        return moo_thresholds, constraints

    def _run_optimization(self, opt_trial):
        n = self._optimization_parameters.params["n_trials"]
        print(f"Optimizing all motors together for {n} trials.")

        if self._optimization_parameters.params["pitch_only"]: opt_trial.trials(n, trial_motor_types=["hb_pitch", "vb_pitch"])
        else:                                                  opt_trial.trials(n)

        return n

    def __set_reference(self):
        reference_beam = self._parameters.params.focusing_system.get_photon_beam(from_raw_image=False)

        position = [reference_beam["centroid_h"], reference_beam["centroid_v"]]
        size     = [reference_beam["width"], reference_beam["height"]]

        ini_file = get_registered_ini_instance(APPLICATION_NAME)

        if OptimizationCriteria.CENTROID in self._optimization_parameters.params["loss_parameters"]:
            self._optimization_parameters.params["reference_parameters_h_v"][OptimizationCriteria.CENTROID] = position
            ini_file.set_list_at_ini(section="Optimization-Parameters", key="Reference-Position", values_list=position)

        if OptimizationCriteria.SIGMA in self._optimization_parameters.params["loss_parameters"]:
            self._optimization_parameters.params["reference_parameters_h_v"][OptimizationCriteria.SIGMA] = size
            ini_file.set_list_at_ini(section="Optimization-Parameters", key="Reference-Size", values_list=size)

        if OptimizationCriteria.FWHM in self._optimization_parameters.params["loss_parameters"]:
            self._optimization_parameters.params["reference_parameters_h_v"][OptimizationCriteria.FWHM] = size
            ini_file.set_list_at_ini(section="Optimization-Parameters", key="Reference-Size", values_list=size)

        ini_file.push()

