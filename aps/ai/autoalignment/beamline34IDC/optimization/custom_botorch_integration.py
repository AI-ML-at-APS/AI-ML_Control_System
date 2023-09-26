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

import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import botorch.utils.multi_objective.pareto
import numpy
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
try: from botorch.sampling.samplers import SobolQMCNormalSampler
except: from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import manual_seed, sample_simplex
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler, IntersectionSearchSpace, RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY, _process_constraints_after_trial
from optuna.study import Study, StudyDirection
from optuna.trial import FrozenTrial, TrialState

_logger = logging.get_logger(__name__)


def qnei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    model_mean_module: Optional[object] = None,
    model_covar_module: Optional[object] = None,
) -> Tuple["SingleTaskGP", "torch.Tensor"]:
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

    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.size(-1)),
        mean_module=model_mean_module,
        covar_module=model_covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(256),
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

    return model, candidates


def qnehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    ref_point: List,
    model_mean_module: Optional[object] = None,
    model_covar_module: Optional[object] = None,
) -> Tuple[SingleTaskGP, "torch.Tensor"]:
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

    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1]),
        mean_module=model_mean_module,
        covar_module=model_covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    pareto_front = botorch.utils.multi_objective.pareto.is_non_dominated(train_y)
    pareto_front = train_y[pareto_front]

    ref_points_inferred = botorch.utils.multi_objective.hypervolume.infer_reference_point(pareto_front)

    if ref_point is None:
        ref_point = train_obj.min(dim=0).values - 1e-8
    else:
        ref_point = torch.tensor(ref_point)

    ref_point = torch.maximum(ref_points_inferred[:n_objectives], ref_point)
    ref_point_list = list(ref_point)

    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(256),
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

    return model, candidates


def qei_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    model_mean_module: Optional[object] = None,
    model_covar_module: Optional[object] = None,
) -> Tuple[SingleTaskGP, "torch.Tensor"]:
    """Quasi MC-based batch Expected Improvement (qEI).

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

    Returns:
        Next set of candidates. Usually the return value of BoTorch's ``optimize_acqf``.

    """

    if train_obj.size(-1) != 1:
        raise ValueError("Objective may only contain single values with qEI.")
    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        is_feas = (train_con <= 0).all(dim=-1)
        train_obj_feas = train_obj[is_feas]

        if train_obj_feas.numel() == 0:
            # TODO(hvy): Do not use 0 as the best observation.
            _logger.warning("No objective values are feasible. Using 0 as the best objective in qEI.")
            best_f = torch.zeros(())
        else:
            best_f = train_obj_feas.max()

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

        best_f = train_obj.max()

        objective = None  # Using the default identity objective.

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.size(-1)),
        mean_module=model_mean_module,
        covar_module=model_covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=best_f,
        sampler=SobolQMCNormalSampler(256),
        objective=objective,
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

    return model, candidates


def qehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    model_mean_module: Optional[object] = None,
    model_covar_module: Optional[object] = None,
) -> Tuple[SingleTaskGP, "torch.Tensor"]:
    """Quasi MC-based batch Expected Hypervolume Improvement (qEHVI).

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
        additional_qehvi_kwargs = {
            "objective": IdentityMCMultiOutputObjective(outcomes=list(range(n_objectives))),
            "constraints": constraints,
        }
    else:
        train_y = train_obj

        train_obj_feas = train_obj

        additional_qehvi_kwargs = {}

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1]),
        mean_module=model_mean_module,
        covar_module=model_covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    ref_point = train_obj.min(dim=0).values - 1e-8

    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj_feas, alpha=alpha)

    ref_point_list = ref_point.tolist()

    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        partitioning=partitioning,
        sampler=SobolQMCNormalSampler(256),
        **additional_qehvi_kwargs,
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

    return model, candidates


def qparego_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    model_mean_module: Optional[object] = None,
    model_covar_module: Optional[object] = None,
) -> Tuple[SingleTaskGP, "torch.Tensor"]:
    """Quasi MC-based extended ParEGO (qParEGO) for constrained multi-objective optimization.

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is larger than three.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)

    weights = sample_simplex(n_objectives).squeeze()
    scalarization = get_chebyshev_scalarization(weights=weights, Y=train_obj)

    if train_con is not None:
        train_y = torch.cat([train_obj, train_con], dim=-1)

        constraints = []
        n_constraints = train_con.size(1)

        for i in range(n_constraints):
            constraints.append(lambda Z, i=i: Z[..., -n_constraints + i])

        objective = ConstrainedMCObjective(
            objective=lambda Z: scalarization(Z[..., :n_objectives]),
            constraints=constraints,
        )
    else:
        train_y = train_obj

        objective = GenericMCObjective(scalarization)

    train_x = normalize(train_x, bounds=bounds)

    model = SingleTaskGP(
        train_x,
        train_y,
        outcome_transform=Standardize(m=train_y.size(-1)),
        mean_module=model_mean_module,
        covar_module=model_covar_module,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    acqf = qExpectedImprovement(
        model=model,
        best_f=objective(train_y).max(),
        sampler=SobolQMCNormalSampler(256),
        objective=objective,
    )

    standard_bounds = torch.zeros_like(bounds)
    standard_bounds[1] = 1

    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=standard_bounds,
        q=1,
        num_restarts=20,
        raw_samples=1024,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )

    candidates = unnormalize(candidates.detach(), bounds=bounds)

    return model, candidates


def _get_default_candidates_func(
    n_objectives: int,
) -> Callable[
    ["torch.Tensor", "torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", Optional[object], Optional[object]],
    Tuple[SingleTaskGP, "torch.Tensor"],
]:
    if n_objectives > 3:
        return qparego_candidates_func
    elif n_objectives > 1:
        return qehvi_candidates_func
    else:
        return qei_candidates_func


class BoTorchSampler(BaseSampler):
    """A sampler that uses BoTorch, a Bayesian optimization library built on top of PyTorch.

    This sampler allows using BoTorch's optimization algorithms from Optuna to suggest parameter
    configurations. Parameters are transformed to continuous space and passed to BoTorch, and then
    transformed back to Optuna's representations. Categorical parameters are one-hot encoded.

    .. seealso::
        See an `example <https://github.com/optuna/optuna-examples/blob/main/multi_objective/
        botorch_simple.py>`_ how to use the sampler.

    .. seealso::
        See the `BoTorch <https://botorch.org/>`_ homepage for details and for how to implement
        your own ``candidates_func``.

    .. note::
        An instance of this sampler *should not be used with different studies* when used with
        constraints. Instead, a new instance should be created for each new study. The reason for
        this is that the sampler is stateful keeping all the computed constraints.

    Args:
        candidates_func:
            An optional function that suggests the next candidates. It must take the training
            data, the objectives, the constraints, the search space bounds and return the next
            candidates. The arguments are of type ``torch.Tensor``. The return value must be a
            ``torch.Tensor``. However, if ``constraints_func`` is omitted, constraints will be
            :obj:`None`. For any constraints that failed to compute, the tensor will contain
            NaN.

            If omitted, it is determined automatically based on the number of objectives. If the
            number of objectives is one, Quasi MC-based batch Expected Improvement (qEI) is used.
            If the number of objectives is either two or three, Quasi MC-based
            batch Expected Hypervolume Improvement (qEHVI) is used. Otherwise, for larger number
            of objectives, the faster Quasi MC-based extended ParEGO (qParEGO) is used.

            The function should assume *maximization* of the objective.

            .. seealso::
                See :func:`optuna.integration.botorch.qei_candidates_func` for an example.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraint is violated. A value equal to or smaller than 0 is considered feasible.

            If omitted, no constraints will be passed to ``candidates_func`` nor taken into
            account during suggestion.
        n_startup_trials:
            Number of initial trials, that is the number of trials to resort to independent
            sampling.
        independent_sampler:
            An independent sampler to use for the initial trials and for parameters that are
            conditional.
        seed:
            Seed for random number generator.
    """

    def __init__(
        self,
        *,
        candidates_func: Optional[
            Callable[
                [
                    "torch.Tensor",
                    "torch.Tensor",
                    Optional["torch.Tensor"],
                    "torch.Tensor",
                ],
                "torch.Tensor",
            ]
        ] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        n_startup_trials: int = 10,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        model_mean_module: Optional[object] = None,
        model_covar_module: Optional[object] = None,
    ):
        self._candidates_func = candidates_func
        self._constraints_func = constraints_func
        self._independent_sampler = independent_sampler or RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self._model_mean_module = model_mean_module
        self._model_covar_module = model_covar_module

        self._study_id: Optional[int] = None
        self._search_space = IntersectionSearchSpace()
        self._model: Optional[SingleTaskGP] = None

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> Dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            # Note that the check below is meaningless when `InMemoryStorage` is used
            # because `InMemoryStorage.create_new_study` always returns the same study ID.
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")

        search_space: Dict[str, BaseDistribution] = OrderedDict()
        for name, distribution in self._search_space.calculate(study, ordered_dict=True).items():
            if distribution.single():
                # built-in `candidates_func` cannot handle distributions that contain just a
                # single value, so we skip them. Note that the parameter values for such
                # distributions are sampled in `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        assert isinstance(search_space, OrderedDict)

        if len(search_space) == 0:
            return {}

        trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))

        n_trials = len(trials)
        if n_trials < self._n_startup_trials:
            return {}

        trans = _SearchSpaceTransform(search_space)
        n_objectives = len(study.directions)
        values: Union[numpy.ndarray, torch.Tensor] = numpy.empty((n_trials, n_objectives), dtype=numpy.float64)
        params: Union[numpy.ndarray, torch.Tensor]
        con: Optional[Union[numpy.ndarray, torch.Tensor]] = None
        bounds: Union[numpy.ndarray, torch.Tensor] = trans.bounds
        params = numpy.empty((n_trials, trans.bounds.shape[0]), dtype=numpy.float64)
        for trial_idx, trial in enumerate(trials):
            params[trial_idx] = trans.transform(trial.params)
            assert len(study.directions) == len(trial.values)

            for obj_idx, (direction, value) in enumerate(zip(study.directions, trial.values)):
                assert value is not None
                if direction == StudyDirection.MINIMIZE:  # BoTorch always assumes maximization.
                    value *= -1
                values[trial_idx, obj_idx] = value

            if self._constraints_func is not None:
                constraints = study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY)
                if constraints is not None:
                    n_constraints = len(constraints)

                    if con is None:
                        con = numpy.full((n_trials, n_constraints), numpy.nan, dtype=numpy.float64)
                    elif n_constraints != con.shape[1]:
                        raise RuntimeError(f"Expected {con.shape[1]} constraints but received {n_constraints}.")

                    con[trial_idx] = constraints

        if self._constraints_func is not None:
            if con is None:
                warnings.warn(
                    "`constraints_func` was given but no call to it correctly computed "
                    "constraints. Constraints passed to `candidates_func` will be `None`."
                )
            elif numpy.isnan(con).any():
                warnings.warn(
                    "`constraints_func` was given but some calls to it did not correctly compute "
                    "constraints. Constraints passed to `candidates_func` will contain NaN."
                )

        values = torch.from_numpy(values)
        params = torch.from_numpy(params)
        if con is not None:
            con = torch.from_numpy(con)
        bounds = torch.from_numpy(bounds)

        if con is not None:
            if con.dim() == 1:
                con.unsqueeze_(-1)
        bounds.transpose_(0, 1)

        if self._candidates_func is None:
            self._candidates_func = _get_default_candidates_func(n_objectives=n_objectives)

        with manual_seed(self._seed):
            # `manual_seed` makes the default candidates functions reproducible.
            # `SobolQMCNormalSampler`'s constructor has a `seed` argument, but its behavior is
            # deterministic when the BoTorch's seed is fixed.
            self._model, candidates = self._candidates_func(
                params,
                values,
                con,
                bounds,
                model_mean_module=self._model_mean_module,
                model_covar_module=self._model_covar_module,
            )
            if self._seed is not None:
                self._seed += 1
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("Candidates must be a torch.Tensor.")
        if candidates.dim() == 2:
            if candidates.size(0) != 1:
                raise ValueError(
                    "Candidates batch optimization is not supported and the first dimension must "
                    "have size 1 if candidates is a two-dimensional tensor. Actual: "
                    f"{candidates.size()}."
                )
            # Batch size is one. Get rid of the batch dimension.
            candidates = candidates.squeeze(0)
        if candidates.dim() != 1:
            raise ValueError("Candidates must be one or two-dimensional.")
        if candidates.size(0) != bounds.size(1):
            raise ValueError(
                "Candidates size must match with the given bounds. Actual candidates: "
                f"{candidates.size(0)}, bounds: {bounds.size(1)}."
            )

        return trans.untransform(candidates.numpy())

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._seed is not None:
            self._seed = numpy.random.RandomState().randint(2**60)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)
