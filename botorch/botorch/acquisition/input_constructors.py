#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A registry of helpers for generating inputs to acquisition function
constructors programmatically from a consistent input format.
"""

from __future__ import annotations
import math

from copy import deepcopy
from functools import partial
from botorch.sampling.pathwise_sampler import PathwiseSampler
from botorch.acquisition.prior_monte_carlo import (
    qPriorExpectedImprovement,
    qPriorLogExpectedImprovement,
    qPriorKnowledgeGradient,
    qPriorUpperConfidenceBound,
    qPriorJointEntropySearch,
    qPriorMaxValueEntropySearch
)
from botorch.acquisition.user_prior import PiBO

from botorch.acquisition.analytic import (
    AnalyticThompsonSampling,
    DiscreteThompsonSampling
)
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    TAU_MAX,
    TAU_RELU,
)
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import (
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    LogConstrainedExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    LogProbabilityOfImprovement,
    NoisyExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qKnowledgeGradient,
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.joint_entropy_search import (
    qJointEntropySearch,
    qExploitJointEntropySearch,
)
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    MCMultiOutputObjective,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.predictive_entropy_search import (
    qPredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.objective import (
    AnalyticMultiOutputObjective,
    IdentityAnalyticMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)

from botorch.acquisition.multi_objective.utils import get_default_partitioning_alpha
from botorch.acquisition.objective import (
    AcquisitionObjective,
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.acquisition.utils import (
    expand_trace_observations,
    project_to_target_fidelity,
)
from botorch.acquisition.utils import get_optimal_samples
from botorch.exceptions.errors import UnsupportedError
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.containers import BotorchContainer
from botorch.utils.datasets import BotorchDataset, SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
    NondominatedPartitioning,
)
from botorch.utils.prior import UserPrior
from botorch.utils.sample_pareto import sample_pareto_sets_and_fronts
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition

from torch import Tensor


ACQF_INPUT_CONSTRUCTOR_REGISTRY = {}

T = TypeVar("T")
MaybeDict = Union[T, Dict[Hashable, T]]

MAX_FEATURES = 2 ** 14

def _field_is_shared(
    datasets: Union[Iterable[BotorchDataset], Dict[Hashable, BotorchDataset]],
    fieldname: Hashable,
) -> bool:
    r"""Determines whether or not a given field is shared by all datasets."""
    if isinstance(datasets, dict):
        datasets = datasets.values()

    base = None
    for dataset in datasets:
        if not hasattr(dataset, fieldname):
            raise AttributeError(f"{type(dataset)} object has no field `{fieldname}`.")

        obj = getattr(dataset, fieldname)
        if base is None:
            base = obj
        elif base != obj:
            return False

    return True


def _get_dataset_field(
    dataset: MaybeDict[BotorchDataset],
    fieldname: str,
    transform: Optional[Callable[[BotorchContainer], Any]] = None,
    join_rule: Optional[Callable[[Sequence[Any]], Any]] = None,
    first_only: bool = False,
    assert_shared: bool = False,
) -> Any:
    r"""Convenience method for extracting a given field from one or more datasets."""
    if isinstance(dataset, dict):
        if assert_shared and not _field_is_shared(dataset, fieldname):
            raise ValueError(f"Field `{fieldname}` must be shared.")

        if not first_only:
            fields = (
                _get_dataset_field(d, fieldname, transform) for d in dataset.values()
            )
            return join_rule(tuple(fields)) if join_rule else tuple(fields)

        dataset = next(iter(dataset.values()))

    field = getattr(dataset, fieldname)
    return transform(field) if transform else field


def get_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., Dict[str, Any]]:
    r"""Get acqusition function input constructor from registry.

    Args:
        acqf_cls: The AcquisitionFunction class (not instance) for which
            to retrieve the input constructor.

    Returns:
        The input constructor associated with `acqf_cls`.

    """
    if acqf_cls not in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
        raise RuntimeError(
            f"Input constructor for acquisition class `{acqf_cls.__name__}` not "
            "registered. Use the `@acqf_input_constructor` decorator to register "
            "a new method."
        )
    return ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls]


def acqf_input_constructor(
    *acqf_cls: Type[AcquisitionFunction],
) -> Callable[..., AcquisitionFunction]:
    r"""Decorator for registering acquisition function input constructors.

    Args:
        acqf_cls: The AcquisitionFunction classes (not instances) for which
            to register the input constructor.
    """
    for acqf_cls_ in acqf_cls:
        if acqf_cls_ in ACQF_INPUT_CONSTRUCTOR_REGISTRY:
            raise ValueError(
                "Cannot register duplicate arg constructor for acquisition "
                f"class `{acqf_cls_.__name__}`"
            )

    def decorator(method):
        for acqf_cls_ in acqf_cls:
            _register_acqf_input_constructor(
                acqf_cls=acqf_cls_, input_constructor=method
            )
            ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls_] = method
        return method

    return decorator


def _register_acqf_input_constructor(
    acqf_cls: Type[AcquisitionFunction],
    input_constructor: Callable[..., Dict[str, Any]],
) -> None:
    ACQF_INPUT_CONSTRUCTOR_REGISTRY[acqf_cls] = input_constructor


# ------------------------- Deprecation Helpers ------------------------- #


def _deprecate_objective_arg(
    posterior_transform: Optional[PosteriorTransform] = None,
    objective: Optional[AcquisitionObjective] = None,
) -> Optional[PosteriorTransform]:
    if posterior_transform is not None:
        if objective is None:
            return posterior_transform
        else:
            raise RuntimeError(
                "Got both a non-MC objective (DEPRECATED) and a posterior "
                "transform. Use only a posterior transform instead."
            )
    elif objective is not None:
        warnings.warn(
            "The `objective` argument to AnalyticAcquisitionFunctions is deprecated "
            "and will be removed in the next version. Use `posterior_transform` "
            "instead.",
            DeprecationWarning,
        )
        if not isinstance(objective, ScalarizedObjective):
            raise UnsupportedError(
                "Analytic acquisition functions only support ScalarizedObjective "
                "(DEPRECATED) type objectives."
            )
        return ScalarizedPosteriorTransform(
            weights=objective.weights, offset=objective.offset
        )
    else:
        return None


# --------------------- Input argument constructors --------------------- #


@acqf_input_constructor(PosteriorMean)
def construct_inputs_analytic_base(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic analytic acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        posterior_transform: The posterior transform to be used in the
            acquisition function.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "posterior_transform": _deprecate_objective_arg(
            posterior_transform=posterior_transform,
            objective=kwargs.get("objective"),
        ),
    }


@acqf_input_constructor(
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    LogProbabilityOfImprovement,
)
def construct_inputs_best_f(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the acquisition functions requiring `best_f`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
            Used to determine default value for `best_f`.
        best_f: Threshold above (or below) which improvement is defined.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )
    if best_f is None:
        best_f = get_best_f_analytic(
            training_data=training_data,
            objective=kwargs.get("objective"),
            posterior_transform=posterior_transform,
        )

    return {**base_inputs, "best_f": best_f, "maximize": maximize}


@acqf_input_constructor(UpperConfidenceBound)
def construct_inputs_ucb(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    beta: Union[float, Tensor] = 0.2,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `UpperConfidenceBound`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
            representing the trade-off parameter between mean and covariance
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_analytic_base(
        model=model,
        training_data=training_data,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    return {**base_inputs, "beta": beta, "maximize": maximize}


@acqf_input_constructor(
    ConstrainedExpectedImprovement, LogConstrainedExpectedImprovement
)
def construct_inputs_constrained_ei(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_index: int,
    constraints: Dict[int, Tuple[Optional[float], Optional[float]]],
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ConstrainedExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective_index: The index of the objective.
        constraints: A dictionary of the form `{i: [lower, upper]}`, where
            `i` is the output index, and `lower` and `upper` are lower and upper
            bounds on that output (resp. interpreted as -Inf / Inf if None)
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Implement best point computation from training data
    # best_f =
    # return {
    #     "model": model,
    #     "best_f": best_f,
    #     "objective_index": objective_index,
    #     "constraints": constraints,
    #     "maximize": maximize,
    # }
    raise NotImplementedError  # pragma: nocover


@acqf_input_constructor(NoisyExpectedImprovement, LogNoisyExpectedImprovement)
def construct_inputs_noisy_ei(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    num_fantasies: int = 20,
    maximize: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `NoisyExpectedImprovement`.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        num_fantasies: The number of fantasies to generate. The higher this
            number the more accurate the model (at the expense of model
            complexity and performance).
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # TODO: Add prune_baseline functionality as for qNEI
    X = _get_dataset_field(training_data, "X", first_only=True, assert_shared=True)
    return {
        "model": model,
        "X_observed": X(),
        "num_fantasies": num_fantasies,
        "maximize": maximize,
    }


@acqf_input_constructor(qSimpleRegret)
def construct_inputs_mc_base(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic MC acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        "model": model,
        "objective": objective,
        "posterior_transform": posterior_transform,
        "X_pending": X_pending,
        "sampler": sampler,
    }


@acqf_input_constructor(qExpectedImprovement)
def construct_inputs_qEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    plot: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
        )

    return {**base_inputs, "best_f": best_f, "plot": plot}


@acqf_input_constructor(qExploitJointEntropySearch)
def construct_inputs_qJESe(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: Tensor,
    num_optima: int = 100,
    exploit_fraction: float = 0.1,
    maximize: bool = True,
    condition_noiseless: bool = True,
    hypercell_bounds: Tensor = None,
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
    path_dtype: str = 'float',
    sample_around_best_std: int = 0.05,
    best_X_frac: float = 0.2,
    **kwargs: Any,
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    dtype = model.train_targets.dtype
    sample_shape = torch.Size([num_optima])
    from copy import deepcopy
    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape)

    best_indices = torch.topk(model.train_targets[0], 1 + int(len(model.train_targets[0]) * best_X_frac)).indices
    cands = model.train_inputs[0][0, best_indices].to(path_dtype)
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=torch.as_tensor(bounds, dtype=path_dtype).T,
        sample_around_points=cands,
        sample_around_best_std=sample_around_best_std,
    )

    optimal_inputs, optimal_outputs = optimal_inputs.to(
        dtype), optimal_outputs.to(dtype)
    inputs = {
        'model': model,
        'optimal_inputs': optimal_inputs,
        'optimal_outputs': optimal_outputs,
        'condition_noiseless': condition_noiseless,
        'exploit_fraction': exploit_fraction,
        'maximize': maximize,
        'X_pending': X_pending,
        'estimation_type': estimation_type,
        'num_samples': num_samples,
        **kwargs
    }
    return inputs


@acqf_input_constructor(qPredictiveEntropySearch)
def construct_inputs_qPES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    num_optima: int = 8,
    sampling_type: Optional[Literal['pareto', 'pathwise', 'exact']] = 'pathwise',
    maximize: bool = True,
    X_pending: Optional[Tensor] = None,
    constraint_model: Optional[Model] = None,
    max_ep_iterations: Optional[int] = 250,
    ep_jitter: Optional[float] = 1e-4,
    test_jitter: Optional[float] = 1e-4,
    threshold: Optional[float] = 1e-2,
    phi_update: Optional[bool] = True,
    verbose: Optional[bool] = False,
    path_dtype: str = 'float',
    sample_around_best_std: int = 0.05,
    best_X_frac: float = 0.2,
    **kwargs: Any,
) -> Dict[str, Any]:

    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    dtype = model.train_targets.dtype
    sample_shape = torch.Size([num_optima])
    from copy import deepcopy
    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape)
    
    best_indices = torch.topk(model.train_targets[0], 1 + int(len(model.train_targets[0]) * best_X_frac)).indices
    cands = model.train_inputs[0][0, best_indices].to(path_dtype)
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=torch.as_tensor(bounds, dtype=path_dtype).T,
        sample_around_points=cands,
        sample_around_best_std=sample_around_best_std,
    )

    optimal_inputs, optimal_outputs = optimal_inputs.to(
        dtype), optimal_outputs.to(dtype)
    inputs = {
        'model': model,
        'optimal_inputs': optimal_inputs,
        'model': model,
        'maximize': maximize,
        'X_pending': X_pending,
        'max_ep_iterations': max_ep_iterations,
        'ep_jitter': ep_jitter,
        'test_jitter': test_jitter,
        'threshold': threshold,
        'phi_update': phi_update,
        'verbose': verbose,
        **kwargs
    }
    return inputs


@acqf_input_constructor(qLogExpectedImprovement)
def construct_inputs_qLogEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        best_f: Threshold above (or below) which improvement is defined.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.
        fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
            approximation to the ReLU.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    return {
        **construct_inputs_qEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            best_f=best_f,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
    }


@acqf_input_constructor(qLogNoisyExpectedImprovement)
def construct_inputs_qLogNEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: Optional[bool] = True,
    cache_root: Optional[bool] = True,
    constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
    eta: Union[Tensor, float] = 1e-3,
    fat: bool = True,
    tau_max: float = TAU_MAX,
    tau_relu: float = TAU_RELU,
    **kwargs,
):
    r"""Construct kwargs for the `qLogNoisyExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first `X`.
        prune_baseline: If True, remove points in `X_baseline` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.
        constraints: A list of constraint callables which map a Tensor of posterior
            samples of dimension `sample_shape x batch-shape x q x m`-dim to a
            `sample_shape x batch-shape x q`-dim Tensor. The associated constraints
            are considered satisfied if the output is less than zero.
        eta: Temperature parameter(s) governing the smoothness of the sigmoid
            approximation to the constraint indicators. For more details, on this
            parameter, see the docs of `compute_smoothed_feasibility_indicator`.
        fat: Toggles the logarithmic / linear asymptotic behavior of the smooth
            approximation to the ReLU.
        tau_max: Temperature parameter controlling the sharpness of the smooth
            approximations to max.
        tau_relu: Temperature parameter controlling the sharpness of the smooth
            approximations to ReLU.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    res = {
        **construct_inputs_qNEI(
            model=model,
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            sampler=sampler,
            X_baseline=X_baseline,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            constraints=constraints,
            eta=eta,
        ),
        "fat": fat,
        "tau_max": tau_max,
        "tau_relu": tau_relu,
    }
    return res


@acqf_input_constructor(qNoisyExpectedImprovement)
def construct_inputs_qNEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: Optional[bool] = False,
    cache_root: Optional[bool] = True,
    plot: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qNoisyExpectedImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        X_baseline: A `batch_shape x r x d`-dim Tensor of `r` design points
            that have already been observed. These points are considered as
            the potential best design point. If omitted, checks that all
            training_data have the same input features and take the first `X`.
        prune_baseline: If True, remove points in `X_baseline` that are
            highly unlikely to be the best point. This can significantly
            improve performance and is generally recommended.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending
    )
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            transform=lambda field: field(),
            assert_shared=True,
            first_only=True,
        )

    return {
        **base_inputs,
        "X_baseline": X_baseline,
        "prune_baseline": prune_baseline,
        "cache_root": cache_root,
        "plot": plot
    }


@acqf_input_constructor(qProbabilityOfImprovement)
def construct_inputs_qPI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    tau: float = 1e-3,
    best_f: Optional[Union[float, Tensor]] = None,
    plot: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qProbabilityOfImprovement` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        tau: The temperature parameter used in the sigmoid approximation
            of the step function. Smaller values yield more accurate
            approximations of the function, but result in gradients
            estimates with higher variance.
        best_f: The best objective value observed so far (assumed noiseless). Can
            be a `batch_shape`-shaped tensor, which in case of a batched model
            specifies potentially different values for each element of the batch.
    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    if best_f is None:
        best_f = get_best_f_mc(training_data=training_data, objective=objective)

    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )

    return {**base_inputs, "tau": tau, "best_f": best_f, "plot": plot}


@acqf_input_constructor(qUpperConfidenceBound)
def construct_inputs_qUCB(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    beta: float = 0.2,
    plot: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `qUpperConfidenceBound` constructor.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `m x d`-dim Tensor of `m` design points that have been
            submitted for function evaluation but have not yet been evaluated.
            Concatenated into X upon forward call.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.
        beta: Controls tradeoff between mean and standard deviation in UCB.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    beta = 0.2 * math.log(len(training_data.Y()) ** 2)

    return {**base_inputs, "beta": beta, "plot": plot}


def _get_sampler(mc_samples: int, qmc: bool) -> MCSampler:
    """Set up MC sampler for q(N)EHVI."""
    # initialize the sampler
    shape = torch.Size([mc_samples])
    if qmc:
        return SobolQMCNormalSampler(sample_shape=shape)
    return IIDNormalSampler(sample_shape=shape)


@acqf_input_constructor(ExpectedHypervolumeImprovement)
def construct_inputs_EHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[AnalyticMultiOutputObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `ExpectedHypervolumeImprovement` constructor."""
    num_objectives = objective_thresholds.shape[0]
    if kwargs.get("outcome_constraints") is not None:
        raise NotImplementedError("EHVI does not yet support outcome constraints.")

    X = _get_dataset_field(
        training_data,
        fieldname="X",
        transform=lambda field: field(),
        first_only=True,
        assert_shared=True,
    )

    alpha = kwargs.get(
        "alpha",
        get_default_partitioning_alpha(num_objectives=num_objectives),
    )
    # This selects the objectives (a subset of the outcomes) and set each
    # objective threhsold to have the proper optimization direction.
    if objective is None:
        objective = IdentityAnalyticMultiOutputObjective()
    if isinstance(objective, RiskMeasureMCObjective):
        pre_obj = objective.preprocessing_function
    else:
        pre_obj = objective
    ref_point = pre_obj(objective_thresholds)

    # Compute posterior mean (for ref point computation ref pareto frontier)
    # if one is not provided among arguments.
    Y_pmean = kwargs.get("Y_pmean")
    if Y_pmean is None:
        with torch.no_grad():
            Y_pmean = model.posterior(X).mean
    if alpha > 0:
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=pre_obj(Y_pmean),
            alpha=alpha,
        )
    else:
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pre_obj(Y_pmean),
        )

    return {
        "model": model,
        "ref_point": ref_point,
        "partitioning": partitioning,
        "objective": objective,
    }


@acqf_input_constructor(qExpectedHypervolumeImprovement)
def construct_inputs_qEHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qExpectedHypervolumeImprovement` constructor."""
    X = _get_dataset_field(
        training_data,
        fieldname="X",
        transform=lambda field: field(),
        first_only=True,
        assert_shared=True,
    )

    # compute posterior mean (for ref point computation ref pareto frontier)
    with torch.no_grad():
        Y_pmean = model.posterior(X).mean

    outcome_constraints = kwargs.pop("outcome_constraints", None)
    # For HV-based acquisition functions we pass the constraint transform directly
    if outcome_constraints is None:
        cons_tfs = None
    else:
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)
        # Adjust `Y_pmean` to contrain feasible points only.
        feas = torch.stack([c(Y_pmean) <= 0 for c in cons_tfs], dim=-1).all(dim=-1)
        Y_pmean = Y_pmean[feas]

    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    ehvi_kwargs = construct_inputs_EHVI(
        model=model,
        training_data=training_data,
        objective_thresholds=objective_thresholds,
        objective=objective,
        # Pass `Y_pmean` that accounts for constraints to `construct_inputs_EHVI`
        # to ensure that correct non-dominated partitioning is produced.
        Y_pmean=Y_pmean,
        **kwargs,
    )

    sampler = kwargs.get("sampler")
    if sampler is None and isinstance(model, GPyTorchModel):
        sampler = _get_sampler(
            mc_samples=kwargs.get("mc_samples", 128), qmc=kwargs.get("qmc", True)
        )

    add_qehvi_kwargs = {
        "sampler": sampler,
        "X_pending": kwargs.get("X_pending"),
        "constraints": cons_tfs,
        "eta": kwargs.get("eta", 1e-3),
    }
    return {**ehvi_kwargs, **add_qehvi_kwargs}


@acqf_input_constructor(qNoisyExpectedHypervolumeImprovement)
def construct_inputs_qNEHVI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective_thresholds: Tensor,
    objective: Optional[MCMultiOutputObjective] = None,
    X_baseline: Optional[Tensor] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qNoisyExpectedHypervolumeImprovement` constructor."""
    if X_baseline is None:
        X_baseline = _get_dataset_field(
            training_data,
            fieldname="X",
            transform=lambda field: field(),
            first_only=True,
            assert_shared=True,
        )

    # This selects the objectives (a subset of the outcomes) and set each
    # objective threhsold to have the proper optimization direction.
    if objective is None:
        objective = IdentityMCMultiOutputObjective()

    outcome_constraints = kwargs.pop("outcome_constraints", None)
    if outcome_constraints is None:
        cons_tfs = None
    else:
        if isinstance(objective, RiskMeasureMCObjective):
            raise UnsupportedError(
                "Outcome constraints are not supported with risk measures. "
                "Use a feasibility-weighted risk measure instead."
            )
        cons_tfs = get_outcome_constraint_transforms(outcome_constraints)

    sampler = kwargs.get("sampler")
    if sampler is None and isinstance(model, GPyTorchModel):
        sampler = _get_sampler(
            mc_samples=kwargs.get("mc_samples", 128), qmc=kwargs.get("qmc", True)
        )

    if isinstance(objective, RiskMeasureMCObjective):
        ref_point = objective.preprocessing_function(objective_thresholds)
    else:
        ref_point = objective(objective_thresholds)

    return {
        "model": model,
        "ref_point": ref_point,
        "X_baseline": X_baseline,
        "sampler": sampler,
        "objective": objective,
        "constraints": cons_tfs,
        "X_pending": kwargs.get("X_pending"),
        "eta": kwargs.get("eta", 1e-3),
        "prune_baseline": kwargs.get("prune_baseline", True),
        "alpha": kwargs.get("alpha", get_default_partitioning_alpha(model.num_outputs)),
        "cache_pending": kwargs.get("cache_pending", True),
        "max_iep": kwargs.get("max_iep", 0),
        "incremental_nehvi": kwargs.get("incremental_nehvi", True),
        "cache_root": kwargs.get("cache_root", True),
    }


@acqf_input_constructor(qMaxValueEntropy)
def construct_inputs_qMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    candidate_size: int = 1000,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMaxValueEntropy` constructor."""
    inputs_mc = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        **kwargs,
    )

    X = _get_dataset_field(training_data, "X", first_only=True)
    _kw = {"device": X.device, "dtype": X.dtype}
    _rvs = torch.rand(candidate_size, len(bounds), **_kw)
    _bounds = torch.tensor(bounds, **_kw).transpose(0, 1)

    return {
        **inputs_mc,
        "candidate_set": _bounds[0] + (_bounds[1] - _bounds[0]) * _rvs,
        "maximize": kwargs.get("maximize", True),
    }


@acqf_input_constructor(qLowerBoundMaxValueEntropy)
def construct_inputs_qGIBBON(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    candidate_size: int = 1000,
    maximize: bool = True,
    num_optima: int = 128,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMaxValueEntropy` constructor."""
    inputs_mc = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        **kwargs,
    )

    dtype = model.train_targets.dtype
    if kwargs.get('use_gumbel', True):
        from botorch.acquisition.max_value_entropy_search import _sample_max_value_Gumbel
        sobol = SobolEngine(dimension=training_data.X().shape[1], scramble=True)
        candidate_set = sobol.draw(1000)
        optimal_inputs, optimal_outputs = _sample_max_value_Gumbel(
            model,
            candidate_set,
            num_optima
        )
        optimal_outputs = optimal_outputs.squeeze(-1)
    else:
        optimal_inputs, optimal_outputs = get_optimal_samples(
            model=model,
            bounds=torch.as_tensor(bounds, dtype=dtype).T,
            num_optima=num_optima,
            maximize=maximize,
        )
        optimal_inputs, optimal_outputs = optimal_inputs.squeeze(
            -2), optimal_outputs.squeeze(-2)
        optimal_inputs, optimal_outputs = optimal_inputs.to(
            dtype), optimal_outputs.to(dtype)

    X = _get_dataset_field(training_data, "X", first_only=True)
    _kw = {"device": X.device, "dtype": X.dtype}
    _rvs = torch.rand(candidate_size, len(bounds), **_kw)
    _bounds = torch.tensor(bounds, **_kw).transpose(0, 1)

    optimal_outputs = optimal_outputs.squeeze(0)
    return {
        "model": model,
        "candidate_set": _bounds[0] + (_bounds[1] - _bounds[0]) * _rvs,
        "maximize": kwargs.get("maximize", True),
        "input_max_values": optimal_outputs
    }


def construct_inputs_mf_base(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    target_fidelities: Dict[int, Union[int, float]],
    fidelity_weights: Optional[Dict[int, float]] = None,
    cost_intercept: float = 1.0,
    num_trace_observations: int = 0,
    **ignore: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for a multifidelity acquisition function's constructor."""
    if fidelity_weights is None:
        fidelity_weights = {f: 1.0 for f in target_fidelities}

    if set(target_fidelities) != set(fidelity_weights):
        raise RuntimeError(
            "Must provide the same indices for target_fidelities "
            f"({set(target_fidelities)}) and fidelity_weights "
            f" ({set(fidelity_weights)})."
        )

    cost_aware_utility = InverseCostWeightedUtility(
        cost_model=AffineFidelityCostModel(
            fidelity_weights=fidelity_weights, fixed_cost=cost_intercept
        )
    )

    return {
        "target_fidelities": target_fidelities,
        "cost_aware_utility": cost_aware_utility,
        "expand": lambda X: expand_trace_observations(
            X=X,
            fidelity_dims=sorted(target_fidelities),
            num_trace_obs=num_trace_observations,
        ),
        "project": lambda X: project_to_target_fidelity(
            X=X, target_fidelities=target_fidelities
        ),
    }


@acqf_input_constructor(qKnowledgeGradient)
def construct_inputs_qKG(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
    num_fantasies: int = 128,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qKnowledgeGradient` constructor."""

    inputs_mc = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = torch.tensor(bounds, dtype=X.dtype, device=X.device)

    _, current_value = optimize_objective(
        model=model,
        bounds=_bounds.t(),
        q=1,
        target_fidelities=target_fidelities,
        objective=objective,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    return {
        **inputs_mc,
        "num_fantasies": num_fantasies,
        "current_value": current_value.detach().cpu().max(),
    }


@acqf_input_constructor(qMultiFidelityKnowledgeGradient)
def construct_inputs_qMFKG(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    target_fidelities: Dict[int, Union[int, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMultiFidelityKnowledgeGradient` constructor."""

    inputs_mf = construct_inputs_mf_base(
        model=model,
        training_data=training_data,
        target_fidelities=target_fidelities,
        **kwargs,
    )

    inputs_kg = construct_inputs_qKG(
        model=model,
        training_data=training_data,
        bounds=bounds,
        objective=objective,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    return {**inputs_mf, **inputs_kg}


@acqf_input_constructor(qMultiFidelityMaxValueEntropy)
def construct_inputs_qMFMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    target_fidelities: Dict[int, Union[int, float]],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for `qMultiFidelityMaxValueEntropy` constructor."""
    inputs_mf = construct_inputs_mf_base(
        model=model,
        training_data=training_data,
        target_fidelities=target_fidelities,
        **kTradingwargs,
    )

    inputs_qmes = construct_inputs_qMES(
        model=model,
        training_data=training_data,
        bounds=bounds,
        objective=objective,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = torch.tensor(bounds, dtype=X.dtype, device=X.device)
    _, current_value = optimize_objective(
        model=model,
        bounds=_bounds.t(),
        q=1,
        objective=objective,
        posterior_transform=posterior_transform,
        target_fidelities=target_fidelities,
        **kwargs,
    )

    return {
        **inputs_mf,
        **inputs_qmes,
        "current_value": current_value.detach().cpu().max(),
    }


@acqf_input_constructor(AnalyticExpectedUtilityOfBestOption)
def construct_inputs_analytic_eubo(
    model: Model,
    pref_model: Model,
    previous_winner: Optional[Tensor] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for the `AnalyticExpectedUtilityOfBestOption` constructor.

    Args:
        model: The outcome model to be used in the acquisition function.
        pref_model: The preference model to be used in preference exploration

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """
    # construct a deterministic fixed single sample model from `model`
    # i.e., performing EUBO-zeta by default as described
    # in https://arxiv.org/abs/2203.11382
    one_sample_outcome_model = FixedSingleSampleModel(model=model)

    return {
        "pref_model": pref_model,
        "outcome_model": one_sample_outcome_model,
        "previous_winner": previous_winner,
    }


def get_best_f_analytic(
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs,
) -> Tensor:
    if isinstance(training_data, dict) and not _field_is_shared(
        training_data, fieldname="X"
    ):
        raise NotImplementedError("Currently only block designs are supported.")

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        transform=lambda field: field(),
        join_rule=lambda field_tensors: torch.cat(field_tensors, dim=-1),
    )

    posterior_transform = _deprecate_objective_arg(
        posterior_transform=posterior_transform, objective=kwargs.get("objective", None)
    )
    if posterior_transform is not None:
        return posterior_transform.evaluate(Y).max(-1).values
    if Y.shape[-1] > 1:
        raise NotImplementedError(
            "Analytic acquisition functions currently only work with "
            "multi-output models if provided with a `ScalarizedObjective`."
        )
    return Y.max(-2).values.squeeze(-1)


def get_best_f_mc(
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
) -> Tensor:
    if isinstance(training_data, dict) and not _field_is_shared(
        training_data, fieldname="X"
    ):
        raise NotImplementedError("Currently only block designs are supported.")

    Y = _get_dataset_field(
        training_data,
        fieldname="Y",
        transform=lambda field: field(),
        join_rule=lambda field_tensors: torch.cat(field_tensors, dim=-1),
    )

    posterior_transform = _deprecate_objective_arg(
        posterior_transform=posterior_transform,
        objective=objective
        if not isinstance(objective, MCAcquisitionObjective)
        else None,
    )
    if posterior_transform is not None:
        # retain the original tensor dimension since objective expects explicit
        # output dimension.
        Y_dim = Y.dim()
        Y = posterior_transform.evaluate(Y)
        if Y.dim() < Y_dim:
            Y = Y.unsqueeze(-1)
    if objective is None:
        if Y.shape[-1] > 1:
            raise UnsupportedError(
                "Acquisition functions require an objective when "
                "used with multi-output models (execpt for multi-objective"
                "acquisition functions)."
            )
        objective = IdentityMCObjective()
    return objective(Y).max(-1).values


def optimize_objective(
    model: Model,
    bounds: Tensor,
    q: int,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
    qmc: bool = True,
    mc_samples: int = 512,
    seed_inner: Optional[int] = None,
    optimizer_options: Dict[str, Any] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    sequential: bool = False,
    **ignore,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize an objective under the given model.

    Args:
        model: The model to be used in the objective.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The cardinality of input sets on which the objective is to be evaluated.
        objective: The objective to optimize.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        linear_constraints: A tuple of (A, b). Given `k` linear constraints on a
            `d`-dimensional space, `A` is `k x d` and `b` is `k x 1` such that
            `A x <= b`. (Not used by single task models).
        fixed_features: A dictionary of feature assignments `{feature_index: value}` to
            hold fixed during generation.
        target_fidelities: A dictionary mapping input feature indices to fidelity
            values. Defaults to `{-1: 1.0}`.
        qmc: Toggle for enabling (qmc=1) or disabling (qmc=0) use of Quasi Monte Carlo.
        mc_samples: Integer number of samples used to estimate Monte Carlo objectives.
        seed_inner: Integer seed used to initialize the sampler passed to MCObjective.
        optimizer_options: Table used to lookup keyword arguments for the optimizer.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e. according to `round-trip` transformations).
        batch_initial_conditions: A Tensor of initial values for the optimizer.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.

    Returns:
        A tuple containing the best input locations and corresponding objective values.
    """
    if optimizer_options is None:
        optimizer_options = {}

    if objective is not None:
        sampler_cls = SobolQMCNormalSampler if qmc else IIDNormalSampler
        acq_function = qSimpleRegret(
            model=model,
            objective=objective,
            posterior_transform=posterior_transform,
            sampler=sampler_cls(sample_shape=torch.Size([mc_samples]), seed=seed_inner),
        )
    else:
        acq_function = PosteriorMean(
            model=model, posterior_transform=posterior_transform
        )

    if fixed_features:
        acq_function = FixedFeatureAcquisitionFunction(
            acq_function=acq_function,
            d=bounds.shape[-1],
            columns=list(fixed_features.keys()),
            values=list(fixed_features.values()),
        )
        free_feature_dims = list(range(len(bounds)) - fixed_features.keys())
        free_feature_bounds = bounds[:, free_feature_dims]  # (2, d' <= d)
    else:
        free_feature_bounds = bounds

    if linear_constraints is None:
        inequality_constraints = None
    else:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indicies = A[i, :].nonzero(as_tuple=False).squeeze()
            coefficients = -A[i, indicies]
            rhs = -b[i, 0]
            inequality_constraints.append((indicies, coefficients, rhs))

    return optimize_acqf(
        acq_function=acq_function,
        bounds=free_feature_bounds,
        q=q,
        num_restarts=optimizer_options.get("num_restarts", 60),
        raw_samples=optimizer_options.get("raw_samples", 1024),
        options={
            "batch_limit": optimizer_options.get("batch_limit", 8),
            "maxiter": optimizer_options.get("maxiter", 200),
            "nonnegative": optimizer_options.get("nonnegative", False),
            "method": optimizer_options.get("method", "L-BFGS-B"),
        },
        inequality_constraints=inequality_constraints,
        fixed_features=None,  # handled inside the acquisition function
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=True,
        sequential=sequential,
    )


# TODO make single-objective with pairwise and multi-objective with pareto
@acqf_input_constructor(qJointEntropySearch)
def construct_inputs_qJES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: Tensor,
    num_optima: int = 64,
    maximize: bool = True,
    condition_noiseless: bool = True,
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    num_samples: int = 64,
    path_dtype: str = float,
    sample_around_best_std: int = 0.05,
    best_X_frac: float = 0.2,
    **kwargs: Any,
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    dtype = model.train_targets.dtype
    sample_shape = torch.Size([num_optima])
    from copy import deepcopy
    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape)
    
    best_indices = torch.topk(model.train_targets[0], 1 + int(len(model.train_targets[0]) * best_X_frac)).indices
    cands = model.train_inputs[0][0, best_indices].to(path_dtype)
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=torch.as_tensor(bounds, dtype=path_dtype).T,
        sample_around_points=cands,
        sample_around_best_std=sample_around_best_std,
    )

    inputs = {
        "model": model,
        "optimal_inputs": optimal_inputs,
        "optimal_outputs": optimal_outputs,
        "condition_noiseless": condition_noiseless,
        "maximize": maximize,
        "X_pending": X_pending,
        "estimation_type": estimation_type,
        "num_samples": num_samples,
        **kwargs,
    }
    return inputs


@acqf_input_constructor(qPriorExpectedImprovement)
def construct_inputs_qPriorEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    user_prior: UserPrior = None,
    user_prior_value: UserPrior = None,
    decay_beta: float = 3.0,
    prior_floor: float = 1e-9,
    log_acq_floor: float = 1e-30,
    custom_decay_factor: float = 0,
    num_samples: int = 1024,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    path_dtype: torch.dtype = 'float',
    **kwargs: Any
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    sample_shape = torch.Size([num_samples])
    prior_sampler = partial(draw_kernel_feature_paths,
                            num_features=kwargs.get('num_features', 2 ** 13))
    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape, prior_sampler=prior_sampler)

    sampler = PathwiseSampler(sample_shape=sample_shape)
    base_inputs = construct_inputs_mc_base(
        model=model,
        sampler=sampler,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        X_pending=X_pending,
    )
    X_baseline = _get_dataset_field(
        training_data,
        fieldname="X",
        transform=lambda field: field(),
        assert_shared=True,
        first_only=True,
    )
    return {
        **base_inputs,
        "user_prior": user_prior,
        "user_prior_value": user_prior_value,
        "paths": paths,
        "X_baseline": X_baseline,
        "decay_beta": decay_beta,
        **kwargs
    }


@acqf_input_constructor(qPriorLogExpectedImprovement)
def construct_inputs_qPriorLogEI(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    user_prior: UserPrior = None,
    user_prior_value: UserPrior = None,
    decay_beta: float = 3.0,
    prior_floor: float = 1e-9,
    log_acq_floor: float = 1e-30,
    custom_decay_factor: float = 0,
    num_samples: int = 1024,
    resampling_fraction: float = 0.1, 
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    path_dtype: torch.dtype = 'float',
    adaptive_features: bool = False,
    decay_power: float = 1.0,
    **kwargs: Any
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    default_features = kwargs.get('num_features', 2 ** 13)
    relative_beta = max(decay_beta / len(training_data.Y()), 0)

    # after
    # after approx half the run, we want max features
    if adaptive_features:
        # between beta = 4 and beta = 1 / sqrt(2)
        B_START = 2 ** 1.5
        B_STOP  = 1 / 2
        decay_factor = decay_beta / (len(training_data.Y()) ** decay_power)
        adabeta = min(max(decay_factor, B_STOP), B_START)
        feature_shrinkage = (1 - (adabeta - B_STOP) / (B_START - B_STOP)) ** 2
        feature_factor = (MAX_FEATURES / default_features) ** feature_shrinkage
        num_features = int(feature_factor * default_features)
        num_features = num_features - num_features % 2

        num_samples = int(num_samples / feature_factor)
        resampling_fraction = resampling_fraction * feature_factor
    else:
        num_features = default_features

    sample_shape = torch.Size([num_samples])
    prior_sampler = partial(draw_kernel_feature_paths,
                            num_features=num_features)

    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape, prior_sampler=prior_sampler)

    if user_prior_value is not None:
        user_prior_value.setup(training_data.Y(), model.Ymean, model.Ystd)

    sampler = PathwiseSampler(sample_shape=sample_shape)
    base_inputs = construct_inputs_mc_base(
        model=model,
        sampler=sampler,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        X_pending=X_pending,
    )
    X_baseline = _get_dataset_field(
        training_data,
        fieldname="X",
        transform=lambda field: field(),
        assert_shared=True,
        first_only=True,
    )
    if best_f is None:
        best_f = get_best_f_mc(
            training_data=training_data,
            objective=objective,
            posterior_transform=posterior_transform,
        )
    return {
        **base_inputs,
        "user_prior": user_prior,
        "user_prior_value": user_prior_value,
        "paths": paths,
        "X_baseline": X_baseline,
        "X_pending": X_pending,
        'resampling_fraction': resampling_fraction,
        "decay_beta": decay_beta,
        'best_f': best_f,
        'decay_power': decay_power,
        **kwargs,
    }


@acqf_input_constructor(qPriorUpperConfidenceBound)
def construct_inputs_qPriorUCB(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    user_prior: UserPrior = None,
    user_prior_value: UserPrior = None,
    decay_beta: float = 3.0,
    prior_floor: float = 1e-9,
    log_acq_floor: float = 1e-30,
    custom_decay_factor: float = 0,
    num_samples: int = 1024,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    path_dtype: torch.dtype = 'float',
    **kwargs: Any
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    sample_shape = torch.Size([num_samples])
    prior_sampler = partial(draw_kernel_feature_paths,
                            num_features=kwargs.get('num_features', 2 ** 13))

    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape, prior_sampler=prior_sampler)

    if user_prior_value is not None:
        user_prior_value.setup(training_data.Y(), model.Ymean, model.Ystd)

    sampler = PathwiseSampler(sample_shape=sample_shape)

    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )
    beta = 0.2 * math.log(len(training_data.Y()) ** 2)

    return {
        **base_inputs,
        "user_prior": user_prior,
        "user_prior_value": user_prior_value,
        "paths": paths,
        "beta": beta,
        "decay_beta": decay_beta
    }


# TODO make single-objective with pairwise and multi-objective with pareto
@acqf_input_constructor(qPriorJointEntropySearch)
def construct_inputs_qPriorJES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: Tensor,
    user_prior: UserPrior = None,
    user_prior_value: UserPrior = None,
    decay_beta: float = 5.0,
    prior_floor: float = 1e-9,
    num_samples: int = 64,
    maximize: bool = True,
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    path_dtype: torch.dtype = 'float',
    gp_fraction: float = 1.0,
    **kwargs: Any
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    dtype = model.train_targets.dtype
    sample_shape = torch.Size([num_samples])
    from copy import deepcopy
    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape)
    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=torch.as_tensor(bounds, dtype=path_dtype).T,
    )
    if user_prior_value is not None:
        user_prior_value.setup(training_data.Y(), model.Ymean, model.Ystd)

    sampler = PathwiseSampler(sample_shape=sample_shape)
    dtype = model.train_targets.dtype
    optimal_inputs, optimal_outputs = optimal_inputs.to(
        dtype), optimal_outputs.to(dtype)

    inputs = {
        'model': model,
        'optimal_inputs': optimal_inputs,
        'optimal_outputs': optimal_outputs,
        'sampler': sampler,
        'paths': paths,
        'user_prior': user_prior,
        "user_prior_value": user_prior_value,
        'decay_beta': decay_beta,
        'prior_floor': prior_floor,
        'X_pending': X_pending,
        'gp_fraction': gp_fraction
    }
    return inputs


# TODO make single-objective with pairwise and multi-objective with pareto
@acqf_input_constructor(qPriorMaxValueEntropySearch)
def construct_inputs_qPriorMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: Tensor,
    user_prior: UserPrior = None,
    user_prior_value: UserPrior = None,
    decay_beta: float = 5.0,
    prior_floor: float = 1e-9,
    num_samples: int = 1024,
    maximize: bool = True,
    X_pending: Optional[Tensor] = None,
    estimation_type: str = "LB",
    path_dtype: torch.dtype = 'float',
    num_optima: float = 64,
    resampling_fraction: float = 0.1,
    plot: bool = False,
    adaptive_features: bool = False,
    decay_power: float = 1.0,
    custom_decay: float = None,
    **kwargs: Any
):
    if path_dtype == 'float':
        path_dtype = torch.float32
    if path_dtype == 'double':
        path_dtype = torch.float64

    default_features = kwargs.get('num_features', 2 ** 13)
    relative_beta = max(decay_beta / len(training_data.Y()), 0)

    # after
    # after approx half the run, we want max features
    if adaptive_features:
        # between beta = 4 and beta = 1 / sqrt(2)
        B_START = 2 ** 1.5
        B_STOP  = 1 / 2
        decay_factor = decay_beta / (len(training_data.Y()) ** decay_power)
        adabeta = min(max(decay_factor, B_STOP), B_START)
        feature_shrinkage = (1 - (adabeta - B_STOP) / (B_START - B_STOP)) ** 2
        feature_factor = (MAX_FEATURES / default_features) ** feature_shrinkage
        num_features = int(feature_factor * default_features)
        num_features = num_features - num_features % 2

        num_samples = int(num_samples / feature_factor)
        resampling_fraction = resampling_fraction * feature_factor

    else:
        num_features = default_features

    sample_shape = torch.Size([num_samples])
    prior_sampler = partial(draw_kernel_feature_paths,
                            num_features=num_features)

    paths = draw_matheron_paths(
        deepcopy(model).to(path_dtype), sample_shape=sample_shape, prior_sampler=prior_sampler)

    optimal_inputs, optimal_outputs = optimize_posterior_samples(
        paths=paths,
        bounds=torch.as_tensor(bounds, dtype=path_dtype).T,
        # subset_indices=torch.arange(num_optima),
    )
    # Quick fix to unrecognized keyword arg above: Take subset indices from the optimal inputs and outputs
    optimal_inputs = optimal_inputs[:num_optima]
    optimal_outputs = optimal_outputs[:num_optima]

    if user_prior_value is not None:
        user_prior_value.setup(training_data.Y(), model.Ymean, model.Ystd)

    sampler = PathwiseSampler(sample_shape=sample_shape)

    dtype = model.train_targets.dtype

    inputs = {
        'model': model,
        'sampler': sampler,
        'paths': paths,
        'optimal_outputs': optimal_outputs,
        'user_prior': user_prior,
        "user_prior_value": user_prior_value,
        'decay_beta': decay_beta,
        'prior_floor': prior_floor,
        'X_pending': X_pending,
        'num_optima': num_optima,
        'resampling_fraction': resampling_fraction,
        'plot': plot,
        'decay_power': decay_power,
        'custom_decay': custom_decay,
    }
    return inputs


@acqf_input_constructor(PiBO)
def construct_inputs_PiBO(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    user_prior: UserPrior = None,
    decay_beta: float = 5.0,
    prior_floor: float = 1e-9,
    log_acq_floor: float = 1e-30,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    X_baseline: Optional[Tensor] = None,
    prune_baseline: Optional[bool] = False,
    cache_root: Optional[bool] = True,
    **kwargs: Any,
) -> Dict[str, Any]:

    raw_acqf_kwargs = construct_inputs_qNEI(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        X_pending=X_pending,
        sampler=sampler,
        X_baseline=X_baseline,
        prune_baseline=True,
        **kwargs
    )
    return {
        'user_prior': user_prior,
        'raw_acqf_kwargs': raw_acqf_kwargs,
        'decay_beta': decay_beta,
        'prior_floor': prior_floor,
        'log_acq_floor': log_acq_floor,
    }


@acqf_input_constructor(qPriorKnowledgeGradient)
def construct_inputs_qPriorKG(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: List[Tuple[float, float]],
    user_prior: UserPrior = None,
    decay_beta: float = 3.0,
    prior_floor: float = 1e-9,
    log_acq_floor: float = 1e-30,
    custom_decay_factor: float = 0,
    num_samples: int = 128,
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    best_f: Optional[Union[float, Tensor]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    sample_shape = torch.Size([num_samples])
    paths = draw_matheron_paths(model, sample_shape=sample_shape)
    sampler = PathwiseSampler(sample_shape=sample_shape)

    base_inputs = construct_inputs_mc_base(
        model=model,
        training_data=training_data,
        objective=objective,
        posterior_transform=posterior_transform,
        sampler=sampler,
        X_pending=X_pending,
    )

    X = _get_dataset_field(training_data, "X", first_only=True)
    _bounds = torch.tensor(bounds, dtype=X.dtype, device=X.device)

    _, current_value = optimize_objective(
        model=model,
        bounds=_bounds.t(),
        q=1,
        objective=objective,
        posterior_transform=posterior_transform,
        **kwargs,
    )

    return {
        **base_inputs,
        "current_value": current_value,
        "num_fantasies": num_samples,
        "user_prior": user_prior,
        "paths": paths,
        "decay_beta": decay_beta,
        **kwargs
    }


@acqf_input_constructor(AnalyticThompsonSampling)
def construct_inputs_ts_ana(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    objective: Optional[MCAcquisitionObjective] = None,
    posterior_transform: Optional[PosteriorTransform] = None,
    X_pending: Optional[Tensor] = None,
    sampler: Optional[MCSampler] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic MC acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """

    return {
        "model": model,
        "posterior_transform": posterior_transform,
    }


@acqf_input_constructor(DiscreteThompsonSampling)
def construct_inputs_ts_disc(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    posterior_transform: Optional[PosteriorTransform] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    r"""Construct kwargs for basic MC acquisition functions.

    Args:
        model: The model to be used in the acquisition function.
        training_data: Dataset(s) used to train the model.
        objective: The objective to be used in the acquisition function.
        posterior_transform: The posterior transform to be used in the
            acquisition function.
        X_pending: A `batch_shape, m x d`-dim Tensor of `m` design points
            that have points that have been submitted for function evaluation
            but have not yet been evaluated.
        sampler: The sampler used to draw base samples. If omitted, uses
            the acquisition functions's default sampler.

    Returns:
        A dict mapping kwarg names of the constructor to values.
    """

    return {
        "model": model,
        "posterior_transform": posterior_transform,
    }
