from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union
import warnings

import numpy

from optuna import logging
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_model
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import InputStandardize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import manual_seed
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import normalize
from botorch.utils.transforms import unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch


def saas_qnehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    acq_func_kwargs: Optional[Dict[Any, Any]],
) -> "torch.Tensor":
    """Quasi MC-based batch Expected Hypervolume Improvement (qnehvi).

    The default value of ``candidates_func`` in :class:`~optuna.integration.BoTorchSampler`
    with multi-objective optimization when the number of objectives is three or less.

    .. seealso::
        :func:`~optuna.integration.botorch.qei_candidates_func` for argument and return value
        descriptions.
    """

    n_objectives = train_obj.size(-1)
    if n_objectives > 1:
        raise ValueError("Does not work for multi-output GP.")
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

    model = SaasFullyBayesianSingleTaskGP(
        train_x,
        train_y,
        input_transform=InputStandardize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=train_y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_fully_bayesian_model_nuts(model)
    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    if "ref_point" not in acq_func_kwargs:
        ref_point = train_obj.min(dim=0).values - 1e-8
        ref_point_list = ref_point.tolist()
    else:
        ref_point_list = acq_func_kwargs.pop("ref_point")

    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        # partitioning=partitioning,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(num_samples=256),
        **additional_qnehvi_kwargs
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


def additive_kernel_qnehvi_candidates_func(
    train_x: "torch.Tensor",
    train_obj: "torch.Tensor",
    train_con: Optional["torch.Tensor"],
    bounds: "torch.Tensor",
    acq_func_kwargs: Optional[Dict[Any, Any]],
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
    fit_gpytorch_model(mll)

    # Approximate box decomposition similar to Ax when the number of objectives is large.
    # https://github.com/facebook/Ax/blob/master/ax/models/torch/botorch_moo_defaults
    if n_objectives > 2:
        alpha = 10 ** (-8 + n_objectives)
    else:
        alpha = 0.0

    if "ref_point" not in acq_func_kwargs:
        ref_point = train_obj.min(dim=0).values - 1e-8
        ref_point_list = ref_point.tolist()
    else:
        ref_point_list = acq_func_kwargs.pop("ref_point")

    acqf = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,
        X_baseline=train_x,
        # partitioning=partitioning,
        prune_baseline=True,
        sampler=SobolQMCNormalSampler(num_samples=256),
        **additional_qnehvi_kwargs
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
