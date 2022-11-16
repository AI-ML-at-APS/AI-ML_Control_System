import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

import aps
import aps.ai.autoalignment.beamline28IDB.optimization.common as opt_common
import aps.ai.autoalignment.beamline28IDB.optimization.movers as movers
from aps.ai.autoalignment.beamline28IDB.facade.focusing_optics_factory import (
    ExecutionMode,
    focusing_optics_factory_method,
)
from aps.ai.autoalignment.beamline28IDB.optimization.optuna_botorch import OptunaOptimizer
from aps.ai.autoalignment.beamline28IDB.simulation.facade.focusing_optics_interface import (
    Layout,
    get_default_input_features,
)
from aps.ai.autoalignment.common.simulation.facade.parameters import Implementors
from aps.ai.autoalignment.common.util import clean_up
from aps.ai.autoalignment.common.util.common import AspectRatio, ColorMap, PlotMode
from aps.ai.autoalignment.common.util.shadow.common import (
    PreProcessorFiles,
    load_shadow_beam,
)
from aps.ai.autoalignment.common.util.wrappers import plot_distribution

DEFAULT_RANDOM_SEED = np.random.randint(100000)


class CentroidSigmaThresholdDependency:
    STATIC = 0
    INITIAL_STRUCTURE = 1


class OptimizationParameters:
    def __init__(self):
        self.move_motors_ranges = {
            "hb_1": [-100, 100],  # in V
            "hb_2": [-100, 100],  # in V
            "hb_pitch": [-0.02, 0.02],  # in mrad
            "hb_trans": [-30, 30],  # in microns
            "vb_bender": [-600, 600],  # in volt
            "vb_pitch": [-0.02, 0.02],  # in mrad
            "vb_trans": [-30, 30],  # in microns
        }

        self.params = {
            "sum_intensity_soft_constraint": 7e3,
            "sum_intensity_hard_constraint": 6.5e3,
            "centroid_sigma_threshold_dependency": CentroidSigmaThresholdDependency.INITIAL_STRUCTURE,
            "centroid_sigma_hard_thresholds_tuple": [0.01, 0.03],
            "n_pitch_trans_motor_trials": 50,
            "n_all_motor_trials": 100,
        }


class SimulationParameters:
    def __init__(self):
        detector_x = 2160 * 0.65 * 1e-3
        detector_y = 2560 * 0.65 * 1e-3

        xrange = [-detector_x / 2, detector_x / 2]
        yrange = [-detector_y / 2, detector_y / 2]

        self.params = {
            "xrange": xrange,
            "yrange": yrange,
            "nbins_h": 1024,
            "nbins_v": 1024,
            "do_gaussian_fit": False,
            "implementor": Implementors.SHADOW,
            "random_seed": DEFAULT_RANDOM_SEED,
        }


def setup_work_dir():
    root_dir = Path(aps.__path__[0]).parents[0]
    work_dir = root_dir / "work_directory/28-ID"
    os.chdir(work_dir)


if __name__ == "__main__":

    plot_mode = PlotMode.INTERNAL
    aspect_ratio = AspectRatio.AUTO
    color_map = ColorMap.VIRIDIS

    setup_work_dir()

    sim_params = SimulationParameters()
    print("Simulation parameters")
    print(sim_params.__dict__)

    opt_params = OptimizationParameters()

    print("Motors and movement ranges")
    print(opt_params.move_motors_ranges)

    print("Optimization parameters")
    print(opt_params.params)

    input_beam_path = "primary_optics_system_beam.dat"

    clean_up()

    # Initializing the focused beam from simulation
    input_features = get_default_input_features(layout=Layout.AUTO_FOCUSING)
    input_beam = load_shadow_beam(input_beam_path)
    focusing_system = focusing_optics_factory_method(
        execution_mode=ExecutionMode.SIMULATION,
        implementor=Implementors.SHADOW,
        bender=True,
    )

    focusing_system.initialize(
        input_photon_beam=input_beam,
        rewrite_preprocessor_files=PreProcessorFiles.NO,
        layout=Layout.AUTO_FOCUSING,
        input_features=input_features,
    )

    beam, hist, dw = opt_common.get_beam_hist_dw(focusing_system=focusing_system, photon_beam=None, **sim_params.params)
    plot_distribution(
        beam=beam,
        title="Initial Beam",
        plot_mode=plot_mode,
        aspect_ratio=aspect_ratio,
        color_map=color_map,
        **sim_params.params,
    )

    centroid_ground, *_ = opt_common.get_centroid_distance(photon_beam=beam, **sim_params.params)
    sigma_ground, *_ = opt_common.get_sigma(photon_beam=beam, **sim_params.params)

    print(f"centroid_ground: {centroid_ground:4.3e}, sigma_ground: {sigma_ground:4.3e}")

    mots = list(opt_params.move_motors_ranges.keys())
    initial_absolute_positions = {k: movers.get_absolute_positions(focusing_system, k)[0] for k in mots}
    print("Focused absolute position are", initial_absolute_positions)

    # Adding random perturbation to the motor values
    initial_movement, focusing_system, (beam_init, hist_init, dw_init) = opt_common.get_random_init(
        focusing_system,
        motor_types_and_ranges=opt_params.move_motors_ranges,
        intensity_sum_threshold=opt_params.params["sum_intensity_hard_constraint"],
        **sim_params.params,
    )

    centroid_init = opt_common._get_centroid_distance_from_dw(dw_init)
    sigma_init = opt_common._get_sigma_from_dw(dw_init)
    print(f"Perturbed system centroid: {centroid_init:4.3e}, sigma: {sigma_init:4.3e}")

    plot_distribution(
        beam=beam_init,
        title="Perturbed Beam",
        plot_mode=plot_mode,
        aspect_ratio=aspect_ratio,
        color_map=color_map,
        **sim_params.params,
    )

    # Now the optimization
    opt_trial = OptunaOptimizer(
        focusing_system,
        motor_types=list(opt_params.move_motors_ranges.keys()),
        loss_parameters=["centroid", "sigma"],
        multi_objective_optimization=True,
        **sim_params.params,
    )

    # Setting up the optimizer
    constraints = {"sum_intensity": opt_params.params["sum_intensity_soft_constraint"]}

    if opt_params.params["centroid_sigma_threshold_dependency"] == CentroidSigmaThresholdDependency.STATIC:
        moo_thresholds = {
            "centroid": opt_params.params["centroid_sigma_hard_thresholds_tuple"][0],
            "sigma": opt_params.params["centroid_sigma_hard_thresholds_tuple"][1],
        }
    elif opt_params.params["centroid_sigma_threshold_dependency"] == CentroidSigmaThresholdDependency.INITIAL_STRUCTURE:
        moo_thresholds = {"centroid": centroid_init, "sigma": sigma_init}
    else:
        raise ValueError

    opt_trial.set_optimizer_options(
        motor_ranges=list(opt_params.move_motors_ranges.values()),
        raise_prune_exception=True,
        use_discrete_space=True,
        sum_intensity_threshold=opt_params.params["sum_intensity_hard_constraint"],
        constraints=constraints,
        moo_thresholds=moo_thresholds,
    )

    n1 = opt_params.params["n_pitch_trans_motor_trials"]
    print(f"First optimizing only the pitch and translation motors for {n1} trials.")

    opt_trial.trials(n1, trial_motor_types=["hb_pitch", "hb_trans", "vb_pitch", "vb_trans"])

    datetime_str = datetime.strftime(datetime.now(), "%Y:%m:%d:%H:%M")
    chkpt_name = f"optimization_checkpoint_{n1}_{datetime_str}.pkl"
    joblib.dump(opt_trial.study.trials, chkpt_name)
    print(f"Saving a checkpoint in {chkpt_name}")

    n2 = opt_params.params["n_all_motor_trials"]
    print(f"Optimizing all motors together for {n2} trials.")
    opt_trial.trials(n2)

    print("Selecting the optimal parameters")
    optimal_params, values = opt_trial.select_best_trial_params(opt_trial.study.best_trials)

    print("Optimal parameters")
    print(optimal_params)
    print("Optimal values: (centroid, sigma)")
    print(values)

    print("Moving motor to optimal position")
    opt_trial._loss_fn_this(list(optimal_params.values()))
    plot_distribution(
        beam=opt_trial.beam_state.photon_beam,
        title="Optimized beam",
        plot_mode=plot_mode,
        aspect_ratio=aspect_ratio,
        color_map=color_map,
        **sim_params.params,
    )

    datetime_str = datetime.strftime(datetime.now(), "%Y:%m:%d:%H:%M")
    chkpt_name = f"optimization_final_{n1+n2}_{datetime_str}.pkl"
    joblib.dump(opt_trial.study.trials, chkpt_name)
    print(f"Saving all trials in {chkpt_name}")

    clean_up()