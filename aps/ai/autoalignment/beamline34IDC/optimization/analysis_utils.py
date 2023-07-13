from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import optuna
from aps.common.ml.data_structures import DictionaryWrapper
from aps.ai.autoalignment.common.util.common import get_info
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from aps.ai.autoalignment.common.util.common import Histogram, calculate_projections_over_noise
from aps.ai.autoalignment.common.util.gaussian_fit import calculate_2D_gaussian_fit
from aps.ai.autoalignment.beamline34IDC.optimization.common import CalculationParameters


def create_study_from_trials(
    fname: Union[str, Path], n_objectives: int = 2, directions: Sequence[str] = None
) -> "optuna.Study":
    if directions is None:
        print("Assuming minimization for all objectives.")
        directions = ["minimize"] * n_objectives
    else:
        print("Number of objectives is ignored if directions are supplied.")
        for dr in directions:
            if dr not in ["maximize", "minimize"]:
                raise ValueError("Direction should be minimize or maximize.")

    trials = joblib.load(fname)
    dist_mins = {}
    dist_maxes = {}
    for t in trials:
        for td, tdval in t.distributions.items():
            dist_mins.setdefault(td, tdval.low)
            dist_maxes.setdefault(td, tdval.high)
            dist_mins[td] = np.minimum(dist_mins[td], t.params[td])

            dist_maxes[td] = np.maximum(dist_maxes[td], t.params[td])

    for t in trials:
        for td, tdval in t.distributions.items():
            # print(td, tdval, t.params[td])
            tdval.step = None
            tdval.high = dist_maxes[td]
            tdval.low = dist_mins[td]

    study = optuna.create_study(directions=directions)
    study.add_trials(trials)
    return study


def load_histograms_from_files(n_steps: int, hists_dir: Union[str, Path], extension: str = "pkl") -> List[Histogram]:
    hists = []
    for ix in range(n_steps):
        fname_str = Path(hists_dir) / f"optimized_beam_histogram_{ix}.{extension}"
        hists.append(joblib.load(fname_str))
    return hists


def select_nash_equil_trial_from_pareto_front(study: optuna.Study) -> Tuple[FrozenTrial, int, Sequence[int]]:
    """This identifies the nash equilibrium = the trial that dominates the most number of trials"""
    n_dominated = calculate_dominated_trials(study.best_trials, study.trials, study.directions)
    ix = np.argmax(n_dominated)
    return study.best_trials[ix], ix, n_dominated


# The pareto front functions are adapted from "optuna/optuna/study/_multi_objective.py"
def get_pareto_front_trials(
    trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection], values_fns: Sequence[Callable] = None
) -> List[FrozenTrial]:
    """Return trials located at the pareto front in the study.
    A trial is located at the pareto front if there are no trials that dominate the trial.
    It's called that a trial ``t0`` dominates another trial ``t1`` if
    ``all(v0 <= v1) for v0, v1 in zip(t0.values, t1.values)`` and
    ``any(v0 < v1) for v0, v1 in zip(t0.values, t1.values)`` are held.
    Returns:
        A list of :class:`~optuna.multi_objective.trial.FrozenMultiObjectiveTrial` objects.
    """

    pareto_front = []
    trials = [t for t in trials if t.state == TrialState.COMPLETE]

    # TODO(ohta): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, directions, values_fns):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def calculate_dominated_trials(
    trial_set_1: List[FrozenTrial],
    trial_set_2: List[FrozenTrial],
    directions: Sequence[StudyDirection],
    values_fns: Sequence[Callable] = None,
) -> List[int]:
    n_dominated = []
    for t in trial_set_1:
        nd = 0
        for t1 in trial_set_2:
            if t1.state != optuna.trial.TrialState.COMPLETE:
                continue
            if _dominates(t, t1, directions, values_fns):
                nd += 1
        n_dominated.append(nd)
    return n_dominated


def _dominates(
    trial0: FrozenTrial,
    trial1: FrozenTrial,
    directions: Sequence[StudyDirection],
    values_fns: Sequence[Callable] = None,
) -> bool:
    def _get_values(trial: FrozenTrial, values_fns: Optional[Sequence[Callable]]):
        if values_fns is None:
            return trial.values
        if len(values_fns) != len(directions):
            return ValueError("Number of value functions must match number of directions.")
        values = []
        for vf in values_fns:
            values.append(vf(trial))
        return values

    def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
        if value is None:
            value = float("inf")

        if direction is StudyDirection.MAXIMIZE:
            value = -value
        return value

    values_t0 = _get_values(trial0, values_fns)
    values_t1 = _get_values(trial1, values_fns)

    assert values_t0 is not None
    assert values_t1 is not None

    if len(values_t0) != len(values_t1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values_t0) != len(directions):
        raise ValueError("The number of the values and the number of the objectives are mismatched.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values_t0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values_t1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))

def recalculate_info(cp: CalculationParameters, hist: Histogram, crop: int=0):
    x_array = hist.hh
    y_array = hist.vv
    z_array = hist.data_2D

    if crop > 0:
        x_array = x_array[crop:-crop]
        y_array = y_array[crop:-crop]
        z_array = z_array[crop:-crop, crop:-crop]
    
    hist, dw = get_info(x_array=x_array, y_array=y_array, z_array=z_array, xrange=cp.xrange, yrange=cp.yrange, 
                        do_gaussian_fit=cp.do_gaussian_fit, calculate_over_noise=cp.calculate_over_noise, noise_threshold=cp.noise_threshold)
    return hist, dw


def get_pareto_dataframe_from_study(study, loss_parameters):
    par_names = ["h_peak", "v_peak", "h_fwhm", "v_fwhm", "peak_intensity"]
    pars = {}
    for k in par_names:
        if k == "gaussian_fit":
            continue
        pars_this = []
        for t in study.trials:
            pars_this.append(t.user_attrs["dw"].get_parameter(k))
        pars[k] = pars_this

    df = study.trials_dataframe()
    df1 = df[["number"]].copy()
    for idx, l in enumerate(loss_parameters):
        df1[l] = df[f"values_{idx}"].copy()
    for k, v in pars.items():
        df1[k] = v.copy()

    best_nums = [t.number for t in study.best_trials]
    mask = df1["number"].isin(best_nums)
    df2 = df1[mask]
    return df2
