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
import time
import numpy
import optuna
import warnings
import json
import os
from datetime import datetime
import joblib

from optuna.visualization.matplotlib import plot_pareto_front, plot_optimization_history
from matplotlib import pyplot as plt

from aps.common.scripts.abstract_script import AbstractScript
from aps.common.traffic_light import get_registered_traffic_light_instance
from aps.common.measurment.beamline.image_processor import IMAGE_SIZE_PIXEL_HxV, PIXEL_SIZE

from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors as Sim_Implementors
from aps.ai.autoalignment.common.hardware.facade.parameters import Implementors as HW_Implementors
from aps.ai.autoalignment.common.util.common import AspectRatio, ColorMap, PlotMode

from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import plot_2D
from aps.ai.autoalignment.common.util.wrappers import plot_distribution
from aps.ai.autoalignment.common.facade.parameters import DistanceUnits, AngularUnits
from aps.ai.autoalignment.common.util.shadow.common import PreProcessorFiles, load_shadow_beam

from aps.ai.autoalignment.beamline28IDB.optimization.common import OptimizationCriteria, MooThresholds, CalculationParameters
import aps.ai.autoalignment.beamline28IDB.optimization.movers as movers
import aps.ai.autoalignment.beamline28IDB.optimization.common as opt_common

from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import ExecutionMode
from aps.ai.autoalignment.beamline28IDB.scripts.beamline import AA_28ID_BEAMLINE_SCRIPTS
from aps.ai.autoalignment.beamline28IDB.optimization.optuna_botorch import OptunaOptimizer


from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import focusing_optics_factory_method
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import get_default_input_features


DEFAULT_RANDOM_SEED = numpy.random.randint(100000)

class OptimizationParameters:
    def __init__(self):
        self.move_motors_ranges     = {}
        self.move_motors_boundaries = {}
        self.params                 = {}

    def analyze_motor_ranges(self, initial_positions):
        for motor in self.move_motors_ranges.keys():
            if initial_positions[motor] + self.move_motors_ranges[motor][0] < self.move_motors_boundaries["bound_" + motor][0]:
                self.move_motors_ranges[motor][0] = self.move_motors_boundaries["bound_" + motor][0] - initial_positions[motor]
            if initial_positions[motor] + self.move_motors_ranges[motor][1] > self.move_motors_boundaries["bound_" + motor][1]:
                self.move_motors_ranges[motor][1] = self.move_motors_boundaries["bound_" + motor][1] - initial_positions[motor]

    @classmethod
    def _get_reference_parameters_h_v(cls, loss_parameters, reference_position, reference_size):
        reference_parameters_h_v = {}
        for loss_parameter in loss_parameters:
            if   loss_parameter in [OptimizationCriteria.CENTROID,
                                    OptimizationCriteria.PEAK_DISTANCE]: reference_parameters_h_v[loss_parameter] = reference_position
            elif loss_parameter in [OptimizationCriteria.SIGMA,
                                    OptimizationCriteria.FWHM]:          reference_parameters_h_v[loss_parameter] = reference_size
        return reference_parameters_h_v

    @classmethod
    def _get_moo_thresholds_dict(cls, moo_thresholds, moo_threshold_position, moo_threshold_size, moo_threshold_intensity):
        moo_thresholds_dict = {}
        for moo_threshold in moo_thresholds:
            if   moo_threshold in [MooThresholds.CENTROID,
                                   MooThresholds.PEAK_DISTANCE]: moo_thresholds_dict[moo_threshold] = moo_threshold_position
            elif moo_threshold in [MooThresholds.SIGMA,
                                   MooThresholds.FWHM]:          moo_thresholds_dict[moo_threshold] = moo_threshold_size
            elif moo_threshold in [MooThresholds.PEAK_INTENSITY,
                                   MooThresholds.SUM_INTENSITY]: moo_thresholds_dict[moo_threshold] = moo_threshold_intensity
        return  moo_thresholds_dict


class PlotParameters(object):
    def __init__(self,
                 save_images,
                 every_n_images):
        nbins_h = IMAGE_SIZE_PIXEL_HxV[0]
        nbins_v = IMAGE_SIZE_PIXEL_HxV[1]
        detector_x = nbins_h * PIXEL_SIZE*1e3 # mm
        detector_y = nbins_v * PIXEL_SIZE*1e3 # mm
        xrange = [-detector_x / 2, detector_x / 2]
        yrange = [-detector_y / 2, detector_y / 2]
        xcoord = xrange[0] + numpy.arange(0, nbins_h)*PIXEL_SIZE*1e3 # mm
        ycoord = yrange[0] + numpy.arange(0, nbins_v)*PIXEL_SIZE*1e3 # mm

        self.params = CalculationParameters()
        self.params.xrange          = xrange
        self.params.yrange          = yrange
        self.params.xcoord          = xcoord
        self.params.ycoord          = ycoord
        self.params.nbins_h         = nbins_h
        self.params.nbins_v         = nbins_v
        self.params.do_gaussian_fit = False
        self.params.save_images     = save_images
        self.params.every_n_images  = every_n_images

    def as_kwargs(self):
        return vars(self.params)

class SimulationParameters(PlotParameters):
    def __init__(self,
                 save_images,
                 every_n_images,
                 use_denoised,
                 add_noise,
                 noise,
                 percentage_fluctutation,
                 calculate_over_noise,
                 noise_threshold):
        super(SimulationParameters, self).__init__(save_images, every_n_images)
        self.params.execution_mode          = ExecutionMode.SIMULATION
        self.params.implementor             = Sim_Implementors.SHADOW
        self.params.random_seed             = DEFAULT_RANDOM_SEED
        self.params.use_denoised            = use_denoised
        self.params.add_noise               = add_noise
        self.params.noise                   = noise
        self.params.percentage_fluctutation = percentage_fluctutation
        self.params.calculate_over_noise    = calculate_over_noise
        self.params.noise_threshold         = noise_threshold

class HardwareParameters(PlotParameters):
    def __init__(self,
                 save_images,
                 every_n_images,
                 use_denoised,
                 calculate_over_noise,
                 noise_threshold):
        super(HardwareParameters, self).__init__(save_images, every_n_images)
        self.params.execution_mode       = ExecutionMode.HARDWARE
        self.params.implementor          = HW_Implementors.EPICS
        self.params.from_raw_image       = True
        self.params.use_denoised         = use_denoised
        self.params.calculate_over_noise = calculate_over_noise
        self.params.noise_threshold      = noise_threshold

input_beam_path = "primary_optics_system_beam.dat"

class GenericScript(AbstractScript):
    def __init__(self,
                 root_directory,
                 energy, period,
                 n_cycles,
                 test_mode,
                 mocking_mode,
                 simulation_mode,
                 save_images,
                 every_n_images,
                 use_denoised,
                 add_noise,
                 noise,
                 percentage_fluctutation,
                 calculate_over_noise,
                 noise_threshold,
                 layout,
                 **kwargs):
        self._root_directory  = root_directory
        self._data_directory  = os.path.join(self._root_directory, "autoalignment")
        self._energy          = energy
        self._test_mode       = test_mode
        self._mocking_mode    = mocking_mode
        self._simulation_mode = simulation_mode
        self._period          = period * 60.0 # in seconds
        self._n_cycles        = n_cycles

        self.__traffic_light  = get_registered_traffic_light_instance(application_name=AA_28ID_BEAMLINE_SCRIPTS)

        self._optimization_parameters = None
        self._parameters              = None

        self._data_directory = os.path.join(self._root_directory, "AI", self._get_script_name().lower())
        
        self._plot_mode    = PlotMode.INTERNAL
        self._aspect_ratio = AspectRatio.AUTO
        self._color_map    = ColorMap.GRAY


        if mocking_mode: print(self._get_script_name() + " in Mocking Mode")
        else:
            if self._simulation_mode:
                self._initialize_simulation_parameters(save_images,
                                                       every_n_images,
                                                       use_denoised,
                                                       add_noise,
                                                       noise,
                                                       percentage_fluctutation,
                                                       calculate_over_noise,
                                                       noise_threshold,
                                                       layout)
            else:
                self._initialize_hardware_parameters(save_images,
                                                     every_n_images,
                                                     use_denoised,
                                                     calculate_over_noise,
                                                     noise_threshold,
                                                     **kwargs)
            self._parameters.params.photon_beam = None

            self._optimization_parameters = self._get_optimization_parameters()
            self._optimization_parameters.analyze_motor_ranges(self._get_initial_positions())

            print("Motors and movement ranges")
            print(self._optimization_parameters.move_motors_ranges)
            print("Optimization parameters")
            print(self._optimization_parameters.params)

    def execute_script(self, **kwargs):
        cycles = 0

        try:
            while(cycles < self._n_cycles):
                cycles += 1
                self.__traffic_light.request_red_light()

                print("Running " + self._get_script_name() + " #" + str(cycles))

                if self._mocking_mode:
                    print("Mocking Mode: do nothing and wait 10 second")
                    time.sleep(10)
                else:
                    self._execute_script_inner(current_cycle=cycles, **kwargs)

                self.__traffic_light.set_green_light()

                print(self._get_script_name() + " #" + str(cycles) + " completed.")

                if self._n_cycles > 1:
                    print("Pausing for " + str(self._period) + " seconds.")
                    time.sleep(self._period)
        except Exception as e:
            try:    self.__traffic_light.set_green_light()
            except: pass

            print("Script interrupted by the following exception:\n" + str(e))

            raise e

    def manage_keyboard_interrupt(self):
        print("\n" + self._get_script_name() + " interrupted by user")

        try:    self.__traffic_light.set_green_light()
        except: pass

    def _execute_script_inner(self, current_cycle, **kwargs):
        warnings.filterwarnings("ignore")

        self._run_preliminary_operations(current_cycle)

        if self._simulation_mode:
            beam, hist, dw = opt_common.get_beam_hist_dw(self._parameters.params, **kwargs)

            if self._test_mode:
                plot_distribution(beam=beam,
                                  title="Initial Beam",
                                  plot_mode=self._plot_mode,
                                  aspect_ratio=self._aspect_ratio,
                                  color_map=self._color_map,
                                  **self._parameters.as_kwargs())

            motors = list(self._optimization_parameters.move_motors_ranges.keys())
            initial_absolute_positions = {k: movers.get_absolute_positions(self._parameters.params.focusing_system, k)[0] for k in motors}
            print("Focused absolute position are", initial_absolute_positions)

            # Adding random perturbation to the motor values
            initial_movement, self._parameters.params.focusing_system, (beam_init, hist_init, dw_init) = \
                opt_common.get_random_init(cp=self._parameters.params,
                                           motor_types_and_ranges=self._optimization_parameters.move_motors_ranges,
                                           intensity_sum_threshold=self._optimization_parameters.params["sum_intensity_hard_constraint"],
                                           **kwargs)

            self._print_beam_attributes(hist_init, dw_init, "Perturbed")

            if self._test_mode:
                plot_distribution(beam=beam_init,
                                  title="Perturbed Beam",
                                  plot_mode=self._plot_mode,
                                  aspect_ratio=self._aspect_ratio,
                                  color_map=self._color_map,
                                  **self._parameters.as_kwargs())
        else:
            motors = list(self._optimization_parameters.move_motors_ranges.keys())
            initial_absolute_positions = {k: movers.get_absolute_positions(self._parameters.params.focusing_system, k)[0] for k in motors}

            print("Focused absolute position are", initial_absolute_positions)
            with open(os.path.join(self._data_directory, "initial_motor_positions.json"), 'w') as fp: json.dump(initial_absolute_positions, fp)

            # taking initial image of the beam
            beam, hist_init, dw_init = opt_common.get_beam_hist_dw(cp=self._parameters.params, **kwargs)

            self._print_beam_attributes(hist_init, dw_init, "Initial")

            if self._test_mode:
                plot_2D(x_array=beam["h_coord"],
                        y_array=beam["v_coord"],
                        z_array=beam["image"],
                        title="Initial beam",
                        color_map=self._color_map,
                        aspect_ratio=self._aspect_ratio,
                        save_image=True,
                        save_path=self._data_directory)
                if self._parameters.params.use_denoised:
                    plot_2D(x_array=beam["h_coord"],
                            y_array=beam["v_coord"],
                            z_array=beam["image_denoised"],
                            title="Initial beam denoised",
                            color_map=self._color_map,
                            aspect_ratio=self._aspect_ratio,
                            save_image=True,
                            save_path=self._data_directory)

        opt_trial = self._get_optimizer(self._parameters.params)

        n_trials = self._run_optimization(opt_trial)

        print("Selecting the optimal parameters, with algorithm: " + self._optimization_parameters.params["selection_algorithm"])
        optimal_params, values = opt_trial.select_best_trial_params(opt_trial.study.best_trials, algorithm=self._optimization_parameters.params["selection_algorithm"])

        print("Optimal parameters")
        print(optimal_params)
        print("Optimal values: " + str(self._optimization_parameters.params["loss_parameters"]))
        print(values)

        print("Moving motor to optimal position")
        opt_trial.study.enqueue_trial(optimal_params)
        opt_trial.trials(1)

        if self._simulation_mode:
            if self._test_mode:
                plot_distribution(beam=opt_trial.beam_state.photon_beam,
                                  title="Optimized beam",
                                  plot_mode=self._plot_mode,
                                  aspect_ratio=self._aspect_ratio,
                                  color_map=self._color_map,
                                  **self._parameters.as_kwargs())

            clean_up()
        else:
            if self._test_mode:
                plot_2D(x_array=opt_trial.beam_state.photon_beam["h_coord"],
                        y_array=opt_trial.beam_state.photon_beam["v_coord"],
                        z_array=opt_trial.beam_state.photon_beam["image"],
                        title="Optimized beam",
                        color_map=self._color_map,
                        aspect_ratio=self._aspect_ratio,
                        save_image=True,
                        save_path=self._data_directory)
                if self._parameters.params.use_denoised:
                    plot_2D(x_array=beam["h_coord"],
                            y_array=beam["v_coord"],
                            z_array=opt_trial.beam_state.photon_beam["image_denoised"],
                            title="Optimized beam denoised",
                            color_map=self._color_map,
                            aspect_ratio=self._aspect_ratio,
                            save_image=True,
                            save_path=self._data_directory)

        datetime_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M")
        chkpt_name = f"optimization_final_{n_trials}_{datetime_str}.gz"
        joblib.dump(opt_trial.study.trials, chkpt_name)
        print(f"Saving all trials in {chkpt_name}")

        if self._test_mode: self._postprocess_optimization(opt_trial.study.trials)

    def _get_script_name(self):                                        raise NotImplementedError()
    def _get_optimization_parameters(self):                            raise NotImplementedError()
    def _run_preliminary_operations(self, current_cycle):              pass
    def _get_optimizer_moo_thresholds_and_contraints(self, opt_trial): raise NotImplementedError()
    def _run_optimization(self, opt_trial):                            raise NotImplementedError()

    def _initialize_simulation_parameters(self,
                                          save_images,
                                          every_n_images,
                                          use_denoised,
                                          add_noise,
                                          noise,
                                          percentage_fluctutation,
                                          calculate_over_noise,
                                          noise_threshold,
                                          layout):
        self._parameters = SimulationParameters(save_images,
                                                every_n_images,
                                                use_denoised,
                                                add_noise,
                                                noise,
                                                percentage_fluctutation,
                                                calculate_over_noise,
                                                noise_threshold)
        print("Simulation parameters")
        print(self._parameters.as_kwargs())
        self.__setup_work_dir()
        clean_up()

        # Initializing the focused beam from simulation
        self._parameters.params.focusing_system = focusing_optics_factory_method(execution_mode=self._parameters.params.execution_mode,
                                                                                 implementor=self._parameters.params.implementor,
                                                                                 bender=True)

        self._parameters.params.focusing_system.initialize(input_photon_beam=load_shadow_beam(input_beam_path),
                                                           rewrite_preprocessor_files=PreProcessorFiles.NO,
                                                           layout=layout,
                                                           input_features=get_default_input_features(layout=layout))

    def _initialize_hardware_parameters(self,
                                        save_images,
                                        every_n_images,
                                        use_denoised,
                                        calculate_over_noise,
                                        noise_threshold,
                                        **kwargs):
        self._parameters = HardwareParameters(save_images,
                                              every_n_images,
                                              use_denoised,
                                              calculate_over_noise,
                                              noise_threshold)
        print("Hardware parameters")
        print(self._parameters.as_kwargs())

        self._parameters.params.focusing_system = focusing_optics_factory_method(execution_mode=self._parameters.params.execution_mode,
                                                                                 implementor=self._parameters.params.execution_mode,
                                                                                 measurement_directory=self._data_directory,
                                                                                 **kwargs)
        self._parameters.params.focusing_system.initialize()

    def _get_initial_positions(self):
        initial_positions = {}

        for motor in self._optimization_parameters.move_motors_ranges.keys():
            if motor == "hb1":         initial_positions[motor] = self._parameters.params.focusing_system.get_h_bendable_mirror_motor_1_bender()
            elif motor == "hb2":       initial_positions[motor] = self._parameters.params.focusing_system.get_h_bendable_mirror_motor_2_bender()
            elif motor == "hb_pitch":  initial_positions[motor] = self._parameters.params.focusing_system.get_h_bendable_mirror_motor_pitch(units=AngularUnits.DEGREES)
            elif motor == "hb_trans":  initial_positions[motor] = self._parameters.params.focusing_system.get_h_bendable_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)
            elif motor == "vb_bender": initial_positions[motor] = self._parameters.params.focusing_system.get_v_bimorph_mirror_motor_bender()
            elif motor == "vb_pitch":  initial_positions[motor] = self._parameters.params.focusing_system.get_v_bimorph_mirror_motor_pitch(units=AngularUnits.DEGREES)
            elif motor == "vb_trans":  initial_positions[motor] = self._parameters.params.focusing_system.get_v_bimorph_mirror_motor_translation(units=DistanceUnits.MILLIMETERS)

        return initial_positions

    def _print_beam_attributes(self, hist, dw, title):
        if OptimizationCriteria.PEAK_DISTANCE               in self._optimization_parameters.params["loss_parameters"]: print(title + f" system peak:     {opt_common._get_peak_distance_from_dw(dw):4.3e}")
        if OptimizationCriteria.CENTROID                    in self._optimization_parameters.params["loss_parameters"]: print(title + f" system centroid: {opt_common._get_centroid_distance_from_dw(dw):4.3e}")
        if OptimizationCriteria.SIGMA                       in self._optimization_parameters.params["loss_parameters"]: print(title + f" system sigma:    {opt_common._get_sigma_from_dw(dw):4.3e}")
        if OptimizationCriteria.FWHM                        in self._optimization_parameters.params["loss_parameters"]: print(title + f" system fwhm:     {opt_common._get_fwhm_from_dw(dw):4.3e}")
        if OptimizationCriteria.NEGATIVE_LOG_PEAK_INTENSITY in self._optimization_parameters.params["loss_parameters"]: print(title + f" system peak intensity: {opt_common._get_peak_intensity_from_dw(dw):8.1e}")
        if OptimizationCriteria.LOG_WEIGHTED_SUM_INTENSITY  in self._optimization_parameters.params["loss_parameters"]: print(title + f" system sum intensity:  {opt_common._get_weighted_sum_intensity_from_hist(hist):8.1e}")

    def _get_optimizer(self, params, **kwargs):
        opt_trial = OptunaOptimizer(calculation_parameters=params,
                                    motor_types=list(self._optimization_parameters.move_motors_ranges.keys()),
                                    loss_parameters=self._optimization_parameters.params["loss_parameters"],
                                    reference_parameters_h_v=self._optimization_parameters.params["reference_parameters_h_v"],
                                    multi_objective_optimization=self._optimization_parameters.params["multi_objective_optimization"],
                                    **kwargs)


        moo_thresholds, constraints = self._get_optimizer_moo_thresholds_and_contraints()

        opt_trial.set_optimizer_options(
            motor_ranges=list(self._optimization_parameters.move_motors_ranges.values()),
            raise_prune_exception=True,
            use_discrete_space=True,
            sum_intensity_threshold=self._optimization_parameters.params["sum_intensity_hard_constraint"],
            constraints=constraints,
            moo_thresholds=moo_thresholds
        )

        return opt_trial



    def _postprocess_optimization(self, trials):
        for t in trials:
            for td, tdval in t.distributions.items():
                tdval.step = None

        if self._optimization_parameters.params["multi_objective_optimization"]:
            study = optuna.create_study(directions=["minimize" for m in self._optimization_parameters.params["loss_parameters"]])  # For multiobjective optimization
        else:
            study = optuna.create_study(directions=["minimize"])

        study.add_trials(trials)

        if self._optimization_parameters.params["multi_objective_optimization"]:
            # Generating the pareto front for the multiobjective optimization
            plot_pareto_front(study, target_names=self._optimization_parameters.params["loss_parameters"])
            plt.tight_layout()
            try:    plt.savefig(os.path.join(self._data_directory, "pareto_front.png"))
            except: print("Image not saved")
            plt.show()

        for i in range(len(self._optimization_parameters.params["loss_parameters"])):
            plot_optimization_history(study,
                                      target=lambda t: t.values[i],
                                      target_name=self._optimization_parameters.params["loss_parameters"][i])
            plt.tight_layout()
            try:    plt.savefig(os.path.join(self._data_directory, "optimization_" + self._optimization_parameters.params["loss_parameters"][i] + ".png"))
            except: print("Image not saved")
            plt.show()

    def __setup_work_dir(self):
        os.chdir(os.path.join(self._data_directory, "simulation"))
