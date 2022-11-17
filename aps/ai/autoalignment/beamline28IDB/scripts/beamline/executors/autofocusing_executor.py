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
import os
import numpy
from datetime import datetime
import joblib
import warnings

from aps.common.measurment.beamline.image_processor import IMAGE_SIZE_PIXEL_HxV, PIXEL_SIZE

from aps.ai.autoalignment.beamline28IDB.scripts.beamline.executors.generic_executor import GenericScript

import aps.ai.autoalignment.beamline28IDB.optimization.common as opt_common
import aps.ai.autoalignment.beamline28IDB.optimization.movers as movers
import aps.ai.autoalignment.beamline28IDB.optimization.configs as configs

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import ExecutionMode, focusing_optics_factory_method
from aps.ai.autoalignment.beamline28IDB.optimization.optuna_botorch import OptunaOptimizer
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import Layout, get_default_input_features
from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors as Sim_Implementors
from aps.ai.autoalignment.common.hardware.facade.parameters import Implementors as HW_Implementors

from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import AspectRatio, ColorMap, PlotMode, get_info, plot_2D
from aps.ai.autoalignment.common.util.shadow.common import PreProcessorFiles, load_shadow_beam
from aps.ai.autoalignment.common.util.wrappers import plot_distribution
from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, AngularUnits

from aps.common.initializer import IniMode, register_ini_instance, get_registered_ini_instance

DEFAULT_RANDOM_SEED = numpy.random.randint(100000)

APPLICATION_NAME = "AUTOFOCUSING"


class CentroidSigmaThresholdDependency:
    STATIC = 0
    INITIAL_STRUCTURE = 1

register_ini_instance(IniMode.LOCAL_FILE,
                      ini_file_name="autofocusing.ini",
                      application_name=APPLICATION_NAME,
                      verbose=False)
ini_file = get_registered_ini_instance(APPLICATION_NAME)

hb_1      = ini_file.get_list_from_ini( section="Motor-Ranges", key="HKB-Bender-1",    default=configs.DEFAULT_MOVEMENT_RANGES["hb_1"],      type=float)
hb_2      = ini_file.get_list_from_ini( section="Motor-Ranges", key="HKB-Bender-2",    default=configs.DEFAULT_MOVEMENT_RANGES["hb_2"],      type=float)
hb_pitch  = ini_file.get_list_from_ini( section="Motor-Ranges", key="HKB-Pitch",       default=configs.DEFAULT_MOVEMENT_RANGES["hb_pitch"],  type=float)  # in degrees
hb_trans  = ini_file.get_list_from_ini( section="Motor-Ranges", key="HKB-Translation", default=configs.DEFAULT_MOVEMENT_RANGES["hb_trans"],  type=float)  # in mm
vb_bender = ini_file.get_list_from_ini( section="Motor-Ranges", key="VKB-Bender",      default=configs.DEFAULT_MOVEMENT_RANGES["vb_bender"], type=float)  # in volt
vb_pitch  = ini_file.get_list_from_ini( section="Motor-Ranges", key="VKB-Pitch",       default=configs.DEFAULT_MOVEMENT_RANGES["vb_pitch"],  type=float)  # in degrees
vb_trans  = ini_file.get_list_from_ini( section="Motor-Ranges", key="VKB-Translation", default=configs.DEFAULT_MOVEMENT_RANGES["vb_trans"],  type=float)  # in mm

bound_hb_1      = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-HKB-Bender-1",    default=[-200, -50],  type=float)
bound_hb_2      = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-HKB-Bender-2",    default=[-180, -50],  type=float)
bound_hb_pitch  = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-HKB-Pitch",       default=[-0.2, 0.2],  type=float)  # in degrees
bound_hb_trans  = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-HKB-Translation", default=[-5.0, 5.0],  type=float)  # in mm
bound_vb_bender = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-VKB-Bender",      default=[0, 600],     type=float)  # in volt
bound_vb_pitch  = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-VKB-Pitch",       default=[-0.2, 0.2],  type=float)  # in degrees
bound_vb_trans  = ini_file.get_list_from_ini( section="Motor-Boundaries", key="Boundaries-VKB-Translation", default=[-5.0, 5.0],  type=float)  # in mm

sum_intensity_soft_constraint        =  ini_file.get_float_from_ini( section="Optimization-Parameters",  key="Sum-Intensity-Soft-Constraint",        default=7e3)
sum_intensity_hard_constraint        =  ini_file.get_float_from_ini( section="Optimization-Parameters",  key="Sum-Intensity-Hard-Constraint",        default=6.5e3)
loss_parameters                      =  ini_file.get_list_from_ini(  section="Optimization-Parameters",  key="Loss-Parameters",                      default=["sigma", "centroid"], type=str)
reference_centroid                   =  ini_file.get_list_from_ini(  section="Optimization-Parameters",  key="Reference-Centroid",                   default=[0.0, 0.0], type=float)
reference_size                       =  ini_file.get_list_from_ini(  section="Optimization-Parameters",  key="Reference-Size",                       default=[0.0, 0.0], type=float)
multi_objective_optimization         =  ini_file.get_boolean_from_ini(section="Optimization-Parameters", key="Multi-Objective-Otimization",          default=True)
n_pitch_trans_motor_trials           =  ini_file.get_int_from_ini(    section="Optimization-Parameters", key="N-Pitch-Trans-Motor-Trials",           default=50)
n_all_motor_trials                   =  ini_file.get_int_from_ini(    section="Optimization-Parameters", key="N-All-Motor-Trials",                   default=100)

ini_file.set_list_at_ini( section="Motor-Ranges",   key="HKB-Bender-1",    values_list=hb_1     )
ini_file.set_list_at_ini( section="Motor-Ranges",   key="HKB-Bender-2",    values_list=hb_2     )
ini_file.set_list_at_ini( section="Motor-Ranges",   key="HKB-Pitch",       values_list=hb_pitch )
ini_file.set_list_at_ini( section="Motor-Ranges",   key="HKB-Translation", values_list=hb_trans )
ini_file.set_list_at_ini( section="Motor-Ranges",   key="VKB-Bender",      values_list=vb_bender)
ini_file.set_list_at_ini( section="Motor-Ranges",   key="VKB-Pitch",       values_list=vb_pitch )
ini_file.set_list_at_ini( section="Motor-Ranges",   key="VKB-Translation", values_list=vb_trans )

ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-HKB-Bender-1",    values_list=bound_hb_1     )
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-HKB-Bender-2",    values_list=bound_hb_2     )
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-HKB-Pitch",       values_list=bound_hb_pitch )
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-HKB-Translation", values_list=bound_hb_trans )
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-VKB-Bender",      values_list=bound_vb_bender)
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-VKB-Pitch",       values_list=bound_vb_pitch )
ini_file.set_list_at_ini( section="Motor-Boundaries",   key="Boundaries-VKB-Translation", values_list=bound_vb_trans )

ini_file.set_value_at_ini(section="Optimization-Parameters",   key="Sum-Intensity-Soft-Constraint",        value=sum_intensity_soft_constraint)
ini_file.set_value_at_ini(section="Optimization-Parameters",   key="Sum-Intensity-Hard-Constraint",        value=sum_intensity_hard_constraint)
ini_file.set_list_at_ini( section="Optimization-Parameters",   key="Loss-Parameters",                      values_list=loss_parameters        )
ini_file.set_list_at_ini( section="Optimization-Parameters",   key="Reference-Centroid",                   values_list=reference_centroid     )
ini_file.set_list_at_ini( section="Optimization-Parameters",   key="Reference-Dimensions",                 values_list=reference_size         )
ini_file.set_value_at_ini(section="Optimization-Parameters",   key="Multi-Objective-Otimization",          value=multi_objective_optimization )
ini_file.set_value_at_ini(section="Optimization-Parameters",   key="N-Pitch-Trans-Motor-Trials",           value=n_pitch_trans_motor_trials   )
ini_file.set_value_at_ini(section="Optimization-Parameters",   key="N-All-Motor-Trials",                   value=n_all_motor_trials           )

ini_file.push()

class OptimizationParameters:
    def __init__(self):
        self.move_motors_ranges = {
            "hb_1":      hb_1,
            "hb_2":      hb_2,
            "hb_pitch":  hb_pitch,
            "hb_trans":  hb_trans,
            "vb_bender": vb_bender,
            "vb_pitch":  vb_pitch,
            "vb_trans":  vb_trans
        }

        self.move_motors_boundaries = {
            "bound_hb_1":      bound_hb_1,
            "bound_hb_2":      bound_hb_2,
            "bound_hb_pitch":  bound_hb_pitch,
            "bound_hb_trans":  bound_hb_trans,
            "bound_vb_bender": bound_vb_bender,
            "bound_vb_pitch":  bound_vb_pitch,
            "bound_vb_trans":  bound_vb_trans

        }

        reference_parameters_h_v = {}
        for loss_parameter in loss_parameters:
            reference_parameters_h_v[loss_parameter] = reference_centroid if loss_parameter=="centroid" else reference_size

        self.params = {
            "sum_intensity_soft_constraint":        sum_intensity_soft_constraint,
            "sum_intensity_hard_constraint":        sum_intensity_hard_constraint,
            "reference_parameters_h_v":             reference_parameters_h_v,
            "loss_parameters":                      loss_parameters,
            "multi_objective_optimization":         multi_objective_optimization,
            "n_pitch_trans_motor_trials":           n_pitch_trans_motor_trials,
            "n_all_motor_trials":                   n_all_motor_trials
        }

    def analyze_motor_ranges(self, initial_positions):
        for motor in self.move_motors_ranges.keys():
            if initial_positions[motor] + self.move_motors_ranges[motor][0] < self.move_motors_boundaries["bound_" + motor][0]:
                self.move_motors_ranges[motor][0] = self.move_motors_boundaries["bound_" + motor][0] - initial_positions[motor]
            if initial_positions[motor] + self.move_motors_ranges[motor][1] > self.move_motors_boundaries["bound_" + motor][1]:
                self.move_motors_ranges[motor][1] = self.move_motors_boundaries["bound_" + motor][1] - initial_positions[motor]

class PlotParameters(object):
    def __init__(self):
        nbins_h = IMAGE_SIZE_PIXEL_HxV[0]
        nbins_v = IMAGE_SIZE_PIXEL_HxV[1]

        detector_x = nbins_h * PIXEL_SIZE*1e3 # mm
        detector_y = nbins_v * PIXEL_SIZE*1e3 # mm

        xrange = [-detector_x / 2, detector_x / 2]
        yrange = [-detector_y / 2, detector_y / 2]

        xcoord = xrange[0] + numpy.arange(0, nbins_h)*PIXEL_SIZE*1e3 # mm
        ycoord = yrange[0] + numpy.arange(0, nbins_v)*PIXEL_SIZE*1e3 # mm

        self.params = {
            "xrange": xrange,
            "yrange": yrange,
            "xcoord" : xcoord,
            "ycoord" : ycoord,
            "nbins_h": nbins_h,
            "nbins_v": nbins_v,
            "do_gaussian_fit": False
        }

class SimulationParameters(PlotParameters):
    def __init__(self):
        super(SimulationParameters, self).__init__()
        self.params["execution_mode"] = ExecutionMode.SIMULATION
        self.params["implementor"]    = Sim_Implementors.SHADOW
        self.params["random_seed"]    = DEFAULT_RANDOM_SEED

    def get_plot_params(self):
        return {
            "xrange": self.params["xrange"],
            "yrange": self.params["yrange"],
            "nbins_h": self.params["nbins_h"],
            "nbins_v": self.params["nbins_v"],
            "do_gaussian_fit": self.params["do_gaussian_fit"]
        }

class HardwareParameters(PlotParameters):
    def __init__(self):
        super(HardwareParameters, self).__init__()
        self.params["execution_mode"] = ExecutionMode.HARDWARE
        self.params["implementor"]    = HW_Implementors.EPICS
        self.params["from_raw_image"] = True

    def get_plot_params(self):
        return {
            "xrange": self.params["xrange"],
            "yrange": self.params["yrange"],
            "nbins_h": self.params["nbins_h"],
            "nbins_v": self.params["nbins_v"],
            "do_gaussian_fit": self.params["do_gaussian_fit"]
        }


input_beam_path = "primary_optics_system_beam.dat"

class AutofocusingScript(GenericScript):

    def __init__(self, root_directory, energy, period, n_cycles, mocking_mode, simulation_mode):
        super(AutofocusingScript, self).__init__(root_directory, energy, period, n_cycles, mocking_mode, simulation_mode)

        self.__data_directory = os.path.join(self._root_directory, "AI", "autoalignment")
        self.__plot_mode      = PlotMode.INTERNAL
        self.__aspect_ratio   = AspectRatio.AUTO
        self.__color_map      = ColorMap.GRAY

        if self._simulation_mode:
            self.__sim_params = SimulationParameters()
            print("Simulation parameters")
            print(self.__sim_params.__dict__)
            self.__setup_work_dir()
            clean_up()

            # Initializing the focused beam from simulation
            self.__focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.SIMULATION,
                                                                    implementor=self.__sim_params.params["implementor"],
                                                                    bender=True)

            self.__focusing_system.initialize(input_photon_beam=load_shadow_beam(input_beam_path),
                                              rewrite_preprocessor_files=PreProcessorFiles.NO,
                                              layout=Layout.AUTO_FOCUSING,
                                              input_features=get_default_input_features(layout=Layout.AUTO_FOCUSING))
        else:
            self.__focusing_system = focusing_optics_factory_method(execution_mode=ExecutionMode.HARDWARE,
                                                                    implementor=HW_Implementors.EPICS,
                                                                    measurement_directory=os.path.join(self._root_directory, "AI", "autoalignment"))
            self.__focusing_system.initialize()

        self.__opt_params = OptimizationParameters()
        self.__opt_params.analyze_motor_ranges(self.__get_initial_positions())

        print("Motors and movement ranges")
        print(self.__opt_params.move_motors_ranges)
        print("Optimization parameters")
        print(self.__opt_params.params)

    def __get_initial_positions(self):
        return {
            "hb_1":      self.__focusing_system.get_h_bendable_mirror_motor_1_bender(),
            "hb_2":      self.__focusing_system.get_h_bendable_mirror_motor_2_bender(),
            "hb_pitch":  self.__focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES),
            "hb_trans":  self.__focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS),
            "vb_bender": self.__focusing_system.get_v_bimorph_mirror_motor_bender(),
            "vb_pitch":  self.__focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES),
            "vb_trans":  self.__focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)
        }

    def _get_script_name(self):
        return "Autofocusing"

    def _execute_script_inner(self, **kwargs):
        def get_optimizer(params):
            opt_trial = OptunaOptimizer(
                self.__focusing_system,
                motor_types=list(self.__opt_params.move_motors_ranges.keys()),
                loss_parameters=self.__opt_params.params["loss_parameters"],
                reference_parameters_h_v=self.__opt_params.params["reference_parameters_h_v"],
                multi_objective_optimization=self.__opt_params.params["multi_objective_optimization"],
                **params,
            )

            # Setting up the optimizer
            constraints = {"sum_intensity": self.__opt_params.params["sum_intensity_soft_constraint"]}

            opt_trial.set_optimizer_options(
                motor_ranges=list(self.__opt_params.move_motors_ranges.values()),
                raise_prune_exception=True,
                use_discrete_space=True,
                sum_intensity_threshold=self.__opt_params.params["sum_intensity_hard_constraint"],
                constraints=constraints,
            )

            return opt_trial

        if self._simulation_mode:
            warnings.filterwarnings("ignore")

            beam, hist, dw = opt_common.get_beam_hist_dw(focusing_system=self.__focusing_system, photon_beam=None, **self.__sim_params.params)

            plot_distribution(
                beam=beam,
                title="Initial Beam",
                plot_mode=self.__plot_mode,
                aspect_ratio=self.__aspect_ratio,
                color_map=self.__color_map,
                **self.__sim_params.params,
            )

            centroid_ground, *_ = opt_common.get_centroid_distance(photon_beam=beam, **self.__sim_params.params)
            sigma_ground, *_    = opt_common.get_sigma(photon_beam=beam, **self.__sim_params.params)

            motors = list(self.__opt_params.move_motors_ranges.keys())
            initial_absolute_positions = {k: movers.get_absolute_positions(self.__focusing_system, k)[0] for k in motors}
            print("Focused absolute position are", initial_absolute_positions)
    
            # Adding random perturbation to the motor values
            initial_movement, self.__focusing_system, (beam_init, hist_init, dw_init) = opt_common.get_random_init(
                self.__focusing_system,
                motor_types_and_ranges=self.__opt_params.move_motors_ranges,
                intensity_sum_threshold=self.__opt_params.params["sum_intensity_hard_constraint"],
                **self.__sim_params.params,
            )
    
            centroid_init = opt_common._get_centroid_distance_from_dw(dw_init)
            sigma_init    = opt_common._get_sigma_from_dw(dw_init)
            print(f"Perturbed system centroid: {centroid_init:4.3e}, sigma: {sigma_init:4.3e}")
    
            plot_distribution(
                beam=beam_init,
                title="Perturbed Beam",
                plot_mode=self.__plot_mode,
                aspect_ratio=self.__aspect_ratio,
                color_map=self.__color_map,
                **self.__sim_params.params,
            )
    
            # Now the optimization
            opt_trial = get_optimizer(self.__sim_params.params)
        else:
            self.__hw_params = HardwareParameters()

            motors = list(self.__opt_params.move_motors_ranges.keys())
            initial_absolute_positions = {k: movers.get_absolute_positions(self.__focusing_system, k)[0] for k in motors}

            print("Focused absolute position are", initial_absolute_positions)

            # taking initial image of the beam

            beam, hist, dw_init = opt_common.get_beam_hist_dw(focusing_system=self.__focusing_system,
                                                              photon_beam=None
                                                              **self.__hw_params.params)

            centroid_init = opt_common._get_centroid_distance_from_dw(dw_init)
            sigma_init    = opt_common._get_sigma_from_dw(dw_init)

            print(f"Initial system centroid: {centroid_init:4.3e}, sigma: {sigma_init:4.3e}")

            plot_2D(x_array=beam["h_coord"],
                    y_array=beam["v_coord"],
                    z_array=beam["image"],
                    title="Initial beam",
                    color_map=self.__color_map,
                    aspect_ratio=self.__aspect_ratio)

            opt_trial = get_optimizer(self.__hw_params.params)

        n1 = self.__opt_params.params["n_pitch_trans_motor_trials"]
        print(f"First optimizing only the pitch and translation motors for {n1} trials.")

        opt_trial.trials(n1, trial_motor_types=["hb_pitch", "hb_trans", "vb_pitch", "vb_trans"])

        n2 = self.__opt_params.params["n_all_motor_trials"]
        print(f"Optimizing all motors together for {n2} trials.")
        opt_trial.trials(n2)

        print("Selecting the optimal parameters")
        optimal_params, values = opt_trial.select_best_trial_params(opt_trial.study.best_trials)

        print("Optimal parameters")
        print(optimal_params)
        print("Optimal values: (centroid, sigma)")
        print(values)

        print("Moving motor to optimal position")

        print("Moving motor to optimal position")
        opt_trial.study.enqueue_trial(optimal_params)
        opt_trial.trials(1)

        if self._simulation_mode:
            plot_distribution(
                beam=opt_trial.beam_state.photon_beam,
                title="Optimized beam",
                plot_mode=self.__plot_mode,
                aspect_ratio=self.__aspect_ratio,
                color_map=self.__color_map,
                **self.__sim_params.params,
            )

            clean_up()
        else:
            plot_2D(x_array=opt_trial.beam_state.photon_beam["h_coord"],
                    y_array=opt_trial.beam_state.photon_beam["v_coord"],
                    z_array=opt_trial.beam_state.photon_beam["image"],
                    title="Optimized beam",
                    color_map=self.__color_map,
                    aspect_ratio=self.__aspect_ratio)

        datetime_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M")
        chkpt_name = f"optimization_final_{n1 + n2}_{datetime_str}.pkl"
        joblib.dump(opt_trial.study.trials, chkpt_name)
        print(f"Saving all trials in {chkpt_name}")

    def __setup_work_dir(self):
        os.chdir(os.path.join(self.__data_directory, "simulation"))


