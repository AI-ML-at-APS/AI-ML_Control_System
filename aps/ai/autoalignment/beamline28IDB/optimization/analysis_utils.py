
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import optuna
import scipy
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from aps.ai.autoalignment.common.util.common import Histogram


def create_study_from_trials(fname: Union[str, Path], n_objectives: int = 2,
                             directions: Sequence[str] = None) -> "optuna.Study":
    if directions is None:
        print("Assuming minimization for all objectives.")
        directions = ["minimize"] * n_objectives
    else:
        print("Number of objectives is ignored if directions are supplied.")
        for dr in directions:
            if dr not in ["maximize", "minimize"]:
                raise ValueError("Direction should be minimize or maximize.")

    trials = joblib.load(fname)
    for t in trials:
        for td, tdval in t.distributions.items():
            tdval.step = None

    study = optuna.create_study(directions=directions)
    study.add_trials(trials)
    return study


def load_histograms_from_files(n_steps: int, hists_dir: Union[str, Path], extension: str='pkl') -> List[Histogram]:
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


def calculate_dominated_trials(trial_set_1: List[FrozenTrial],
                               trial_set_2: List[FrozenTrial],
                               directions: Sequence[StudyDirection]) -> List[int]:
    n_dominated = []
    for t in trial_set_1:
        nd = 0
        for t1 in trial_set_2:
            if t1.state != optuna.trial.TrialState.COMPLETE: continue
            if _dominates(t, t1, directions):
                nd += 1
        n_dominated.append(nd)
    return n_dominated


# This is adapted from "optuna/optuna/study/_multi_objective.py"
def _dominates(
        trial0: FrozenTrial, trial1: FrozenTrial,
        directions: Sequence[StudyDirection]
) -> bool:
    def _normalize_value(value: Optional[float], direction: StudyDirection) -> float:
        if value is None:
            value = float("inf")

        if direction is StudyDirection.MAXIMIZE:
            value = -value
        return value

    values0 = trial0.values
    values1 = trial1.values

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def calculate_weighted_sum(hist: Histogram, threshold: float = 500, crop: int = 700):
    img = hist.data_2D.T
    if crop > 0:
        img = img[crop:-crop, crop:-crop]
    #img[img < threshold] = 0
    img_filtered = scipy.ndimage.median_filter(img, 3)
    img_filtered[img_filtered < threshold] = 0

    hh, vv = hist.hh, hist.vv
    if crop > 0:
        hh = hist.hh[crop: -crop]
        vv = hist.vv[crop:-crop]
    radii = (hh ** 2 + vv[:, None] ** 2) ** 0.5
    return img_filtered.sum(), (img_filtered * radii ** 2).sum()