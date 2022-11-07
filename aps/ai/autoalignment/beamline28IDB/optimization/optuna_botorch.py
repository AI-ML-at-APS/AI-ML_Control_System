#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------- #
# Copyright (c) 2021, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2021. UChicago Argonne, LLC. This software was produced       #
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
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np
import optuna
import torch
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

from aps.ai.autoalignment.beamline28IDB.optimization import common, configs, movers


def qnei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
) -> "torch.Tensor":
    """Quasi MC-based batch Noisy Expected Improvement (qEI).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with single-objective optimization.

    Args:
        train_x:
            Previous parameter configurations. A ``torch.Tensor`` of shape
            ``(n_trials, n_params)``. ``n_trials`` is the number of already observed trials
            and ``n_params`` is the number of parameters. ``n_params`` may be larger than the
            actual number of parameters if categorical parameters are included in the search
            space, since these parameters are one-hot encoded.
            Values are not normalized.
        train_obj:
            Previously observed objectives. A ``torch.Tensor`` of shape
            ``(n_trials, n_objectives)``. ``n_trials`` is identical to that of ``train_x``.
            ``n_objectives`` is the number of objectives. Observations are not normalized.
        train_con:
            Objective constraints. A ``torch.Tensor`` of shape ``(n_trials, n_constraints)``.
            ``n_trials`` is identical to that of ``train_x``. ``n_constraints`` is the number of
            constraints. A constraint is violated if strictly larger than 0. If no constraints are
            involved in the optimization, this argument will be :obj:`None`.
        bounds:
            Search space bounds. A ``torch.Tensor`` of shape ``(2, n_params)``. ``n_params`` is
            identical to that of ``train_x``. The first and the second rows correspond to the
            lower and upper bounds for each parameter respectively.
        acq_func_kwargs:
            Extra keyword arguments.

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qNEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        constraints = []
        n_constraints = train_con.size(1)
        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])
        objective = ConstrainedMCObjective(
            objective=lambda Z: Z[..., 0],
            constraints=constraints,
        )
    else:
        train_y = train_obj

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.size(-1)))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(num_samples=256),
        objective=objective,
        prune_baseline=True,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


def qnehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    ref_point: List,
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Hypervolume Improvement (qnehvi).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        constraints = []
        n_constraints = train_con.size(1)

        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])
        additional_qnehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": constraints,
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qnehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    if ref_point is None:
        ref_point = train_obj.min(dim=0).values - 1e-8
        ref_point_list = list(ref_point)
    else:
        ref_point_list = list(ref_point)

    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(num_samples=256),
        **additional_qnehvi_kwargs,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return candidates


class OptunaOptimizer(common.OptimizationCommon):
    opt_platform = "optuna"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_params = None
        self.motor_ranges = None
        self.study = None
        self._constraints = None
        self._raise_prune_exception = None
        self._base_sampler = None
        self._sum_intensity_threshold = None
        self._loss_fn_this = None
        self._use_discrete_space = None

    def set_optimizer_options(
        self,
        motor_ranges: list = None,
        base_sampler: optuna.samplers.BaseSampler = None,
        directions: Dict = None,
        sum_intensity_threshold: float = 1e2,
        raise_prune_exception: bool = True,
        acquisition_function: Callable = None,
        moo_thresholds: Dict = None,
        botorch_seed: int = None,
        use_discrete_space: bool = True,
        constraints: Dict = None,
    ):

        self.motor_ranges = self._get_guess_ranges(motor_ranges)

        directions_list = self._check_directions(directions)
        # Creating the acquisition function
        if acquisition_function is None:
            if self.multi_objective_optimization:

                def acquisition_function(*args, **kwargs):
                    thresholds_list = self._check_thresholds(moo_thresholds, directions_list)
                    return qnehvi_candidates_func(*args, ref_point=thresholds_list, **kwargs)

            else:
                acquisition_function = qnei_candidates_func

        # Setting up the constraints
        self._constraints = OptunaOptimizer._check_constraints(constraints)

        # Initializing the sampler
        seed = self.random_seed if botorch_seed is None else botorch_seed
        if base_sampler is None:
            sampler_extra_options = {}
            if self._constraints is not None:
                sampler_extra_options = {"constraints_func": self._constraints_func}
            base_sampler = optuna.integration.BoTorchSampler(
                candidates_func=acquisition_function, seed=seed, **sampler_extra_options
            )
        self._base_sampler = base_sampler
        self._raise_prune_exception = raise_prune_exception

        self.study = optuna.create_study(sampler=self._base_sampler, directions=directions_list)
        self.study.enqueue_trial({mt: 0.0 for mt in self.motor_types})

        loss_fn_obj = self.TrialInstanceLossFunction(self, verbose=False)
        self._loss_fn_this = loss_fn_obj.loss
        self._sum_intensity_threshold = sum_intensity_threshold
        self._use_discrete_space = use_discrete_space

        self.best_params = {k: 0.0 for k in self.motor_types}

    def _check_directions(self, directions: Dict) -> List:

        if directions is None:
            return ["minimize" for k in self.loss_parameters]
        directions_list = []
        for k in self.loss_parameters:
            if k not in directions:
                raise ValueError
            if directions[k] not in ["minimize", "maximize"]:
                raise ValueError
            directions_list.append(directions[k])
        return directions_list

    def _check_thresholds(self, thresholds: Dict, directions_list: List) -> Union[List, None]:
        if thresholds is None:
            return None

        thresholds_list = []
        for i, k in enumerate(self.loss_parameters):
            if k not in thresholds:
                raise ValueError
            v = np.abs(thresholds[k])
            if directions_list[i] == "minimize":
                v *= -1
            thresholds_list.append(v)
        return thresholds_list

    @staticmethod
    def _check_constraints(constraints: Dict):
        if constraints is None:
            return None
        for constraint in constraints:
            if constraint not in configs.DEFAULT_CONSTRAINT_OPTIONS:
                raise ValueError
        return constraints

    def _constraints_func(self, trial):
        constraint_vals = []
        for constraint_type in self._constraints:
            constraint_vals.append(trial.user_attrs[f"{constraint_type}_constraint"])
        return constraint_vals

    def _set_trial_constraints(self, trial: object) -> NoReturn:
        if self._constraints is None:
            return
        minimize_constraint_fns = {
            "centroid": self.get_centroid_distance,
            "sigma": self.get_sigma,
            "fwhm": self.get_fwhm,
        }
        maximize_constraint_fns = {
            "sum_intensity": self.get_sum_intensity,
            "peak_intensity": self.get_peak_intensity,
        }
        for constraint, threshold in self._constraints.items():
            if constraint in minimize_constraint_fns:
                x = minimize_constraint_fns[constraint]()
                value = -2 * (x < threshold) + 1
            else:
                x = maximize_constraint_fns[constraint]()
                value = -2 * (x > threshold) + 1
            trial.set_user_attr(f"{constraint}_constraint", value)
        return

    def _objective(self, trial, step_scale: float = 1):
        current_params = []
        for mot, r in zip(self.motor_types, self.motor_ranges):
            if self._use_discrete_space:
                resolution = np.round(configs.DEFAULT_MOTOR_RESOLUTIONS[mot], 5)
                r_low = np.round(r[0], 5)
                r_high = ((r[1] - r[0]) // resolution) * resolution + r_low
                current_params.append(trial.suggest_float(mot, r_low, r_high, step=resolution * step_scale))
            else:
                current_params.append(trial.suggest_float(mot, r[0], r[1]))
        loss = self._loss_fn_this(current_params)
        self._set_trial_constraints(trial)
        if self.multi_objective_optimization:
            if np.nan in loss and self._raise_prune_exception:
                raise optuna.TrialPruned
            loss[np.isnan(loss)] = 1e4

            if self._sum_intensity_threshold is not None:
                if self.beam_state.hist.data_2D.sum() < self._sum_intensity_threshold:
                    if self._raise_prune_exception:
                        raise optuna.TrialPruned
                    else:
                        return [1e4] * len(self._loss_function_list)

            for k in ["sigma", "fwhm"]:
                if k in self.loss_parameters:
                    width_idx = self.loss_parameters.index(k)
                    if loss[width_idx] == 0:
                        loss[width_idx] = 1e4
            loss = list(loss)
        rads = (self.beam_state.hist.hh**2 + self.beam_state.hist.vv**2) ** 0.5
        weighted_sum_ints = np.sum(self.beam_state.hist.data_2D * rads)
        trial.set_user_attr("wsum", weighted_sum_ints)
        return loss

    def trials(self, n_trials: int, trial_motor_types: list = None, step_scale: float = 1):

        obj_this = lambda t: self._objective(t, step_scale=step_scale)

        if trial_motor_types is None:
            self.study.optimize(obj_this, n_trials)
        else:
            fixed_params = {k: self.best_params[k] for k in self.best_params if k not in trial_motor_types}
            partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, self._base_sampler)

            self.study.sampler = partial_sampler
            self.study.optimize(obj_this, n_trials=n_trials)

            self.study.sampler = self._base_sampler

        self.best_params.update(self.study.best_trials[0].params)

    # def trials(self, n_guesses = 1, verbose: bool = False, accept_all_solutions: bool = False):
    #    pass

    def _optimize(self):
        pass

    # def set_optimizer_options(self):
    #    pass
