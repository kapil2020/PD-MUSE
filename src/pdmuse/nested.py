"""Nested-logit probability tools and two-stage PD-MUSE recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .data import ChoiceDataset, validate_choice_arrays
from .exceptions import DataValidationError, NotFittedError, OptimizationError
from .metrics import log_loss
from .mnl import PDMuseMNL
from .utils import logsumexp_masked, safe_log, sigmoid, softmax_masked


def normalize_nests(
    nests: Sequence[Any] | dict[Any, Any],
    n_alternatives: int,
    alt_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return a nest label for each alternative."""

    if isinstance(nests, dict):
        labels: list[Any] = []
        for idx in range(n_alternatives):
            keys = [idx]
            if alt_names is not None:
                keys.append(str(alt_names[idx]))
            found = [nests[key] for key in keys if key in nests]
            if not found:
                raise DataValidationError(f"missing nest assignment for alternative {idx}.")
            labels.append(found[0])
        return np.asarray(labels, dtype=object)

    labels_arr = np.asarray(list(nests), dtype=object)
    if labels_arr.shape != (n_alternatives,):
        raise DataValidationError("nests must have one label per alternative.")
    return labels_arr


def nested_probabilities(
    X: np.ndarray,
    beta: np.ndarray,
    nests: Sequence[Any] | dict[Any, Any],
    lambdas: dict[Any, float] | Sequence[float] | float,
    *,
    availability: Optional[np.ndarray] = None,
    alt_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute standard nested-logit probabilities.

    Parameters
    ----------
    X:
        Choice array with shape ``(n_choices, n_alternatives, n_features)``.
    beta:
        Taste vector.
    nests:
        Nest label per alternative, or a mapping from alternative name/index to nest label.
    lambdas:
        Dissimilarity per nest. Singleton nests can be set to one.
    """

    X_arr = np.asarray(X, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)
    if X_arr.ndim != 3 or beta_arr.shape != (X_arr.shape[2],):
        raise DataValidationError("X and beta have incompatible shapes.")
    n_choices, n_alternatives, _ = X_arr.shape
    availability_arr = (
        np.ones((n_choices, n_alternatives), dtype=bool)
        if availability is None
        else np.asarray(availability, dtype=bool)
    )
    if availability_arr.shape != (n_choices, n_alternatives):
        raise DataValidationError("availability must match X shape.")

    nest_labels = normalize_nests(nests, n_alternatives, alt_names)
    unique_nests = list(dict.fromkeys(nest_labels.tolist()))
    lambda_map = _lambda_map(lambdas, unique_nests)
    utilities = np.einsum("njk,k->nj", X_arr, beta_arr)

    inclusive = np.full((n_choices, len(unique_nests)), -np.inf, dtype=float)
    conditional = np.zeros((n_choices, n_alternatives), dtype=float)
    nest_available = np.zeros_like(inclusive, dtype=bool)

    for g_idx, nest in enumerate(unique_nests):
        lam = lambda_map[nest]
        if lam <= 0.0 or lam > 1.0:
            raise DataValidationError("nested-logit lambdas must be in (0, 1].")
        mask = nest_labels == nest
        available_g = availability_arr[:, mask]
        nest_available[:, g_idx] = available_g.any(axis=1)
        scaled = utilities[:, mask] / lam
        rows = nest_available[:, g_idx]
        if np.any(rows):
            inclusive[rows, g_idx] = lam * logsumexp_masked(scaled[rows], available_g[rows])
            conditional[np.ix_(rows, mask)] = softmax_masked(scaled[rows], available_g[rows])

    upper = softmax_masked(inclusive, nest_available)
    probabilities = np.zeros_like(conditional)
    for g_idx, nest in enumerate(unique_nests):
        mask = nest_labels == nest
        probabilities[:, mask] = upper[:, [g_idx]] * conditional[:, mask]
    return probabilities


def nested_log_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    nests: Sequence[Any] | dict[Any, Any],
    lambdas: dict[Any, float] | Sequence[float] | float,
    *,
    availability: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    alt_names: Optional[Sequence[str]] = None,
) -> float:
    """Evaluate the standard nested-logit log-likelihood."""

    X_arr, y_arr, availability_arr, weights = validate_choice_arrays(
        X, y, availability, sample_weight
    )
    probs = nested_probabilities(
        X_arr, beta, nests, lambdas, availability=availability_arr, alt_names=alt_names
    )
    return float(np.sum(weights * safe_log(probs[np.arange(X_arr.shape[0]), y_arr])))


@dataclass
class ScalarLogitResult:
    coefficient: float
    converged: bool
    n_iter: int
    objective: float
    gradient: float
    message: str


class NestedLogitTwoStage:
    """Two-stage PD-MUSE estimator for one non-singleton target nest.

    Stage one fits the full-sample PD-MUSE multinomial logit and uses those
    taste coefficients as the reported behavioral coefficients. An auxiliary
    conditional PD-MUSE fit on the target nest supplies the upper-stage outside
    normalization used to recover the nest dissimilarity. This reproduces the
    Apollo two-stage PD-MUSE table in the manuscript: the taste coefficients are
    the PD-MUSE/MNL coefficients, and the second stage adds the nest
    dissimilarity.
    """

    def __init__(
        self,
        *,
        target_nest: Optional[Any] = None,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        if not (0.0 < min_lambda <= max_lambda):
            raise DataValidationError("lambda bounds must satisfy 0 < min_lambda <= max_lambda.")
        self.target_nest = target_nest
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def fit(
        self,
        X: np.ndarray | ChoiceDataset,
        y: Optional[np.ndarray] = None,
        nests: Optional[Sequence[Any] | dict[Any, Any]] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        alt_names: Optional[Sequence[str]] = None,
    ) -> "NestedLogitTwoStage":
        if isinstance(X, ChoiceDataset):
            dataset = X
            X_arr, y_arr, availability_arr, weights = (
                dataset.X,
                dataset.y,
                dataset.availability,
                dataset.sample_weight,
            )
            feature_names = dataset.feature_names
            alt_names = dataset.alt_names
        else:
            if y is None:
                raise DataValidationError("y is required when X is not a ChoiceDataset.")
            X_arr, y_arr, availability_arr, weights = validate_choice_arrays(
                X, y, availability, sample_weight
            )
        if nests is None:
            raise DataValidationError("nests are required.")

        n_choices, n_alternatives, n_features = X_arr.shape
        nest_labels = normalize_nests(nests, n_alternatives, alt_names)
        target = self._choose_target_nest(nest_labels)
        target_mask = nest_labels == target
        outside_mask = ~target_mask
        if target_mask.sum() < 2:
            raise DataValidationError("target nest must contain at least two alternatives.")
        if outside_mask.sum() < 1:
            raise DataValidationError("two-stage estimator needs at least one outside alternative.")

        chosen_in_target = target_mask[y_arr]

        stage1_model = PDMuseMNL(tol=self.tol, max_iter=self.max_iter)
        stage1_model.fit(
            X_arr,
            y_arr,
            availability=availability_arr,
            sample_weight=weights,
            feature_names=feature_names,
            alt_names=alt_names,
        )
        beta = stage1_model.coef_.copy()

        if chosen_in_target.sum() < max(5, n_features + 1):
            raise DataValidationError("not enough choices inside the target nest.")
        X_within = X_arr[chosen_in_target][:, target_mask, :]
        availability_within = availability_arr[chosen_in_target][:, target_mask]
        target_positions = np.flatnonzero(target_mask)
        y_within = np.searchsorted(target_positions, y_arr[chosen_in_target])
        weights_within = weights[chosen_in_target]
        auxiliary_model = PDMuseMNL(tol=self.tol, max_iter=self.max_iter)
        auxiliary_model.fit(
            X_within,
            y_within,
            availability=availability_within,
            sample_weight=weights_within,
            feature_names=feature_names,
        )
        auxiliary_beta = auxiliary_model.coef_.copy()

        target_lse = self._target_inclusive_value(X_arr, beta, target_mask, availability_arr)
        outside_offset = self._outside_inclusive_value(
            X_arr, auxiliary_beta, outside_mask, availability_arr
        )
        target_available = availability_arr[:, target_mask].any(axis=1)
        outside_available = availability_arr[:, outside_mask].any(axis=1)
        upper_train = target_available & outside_available & np.isfinite(target_lse)
        if int(np.sum(upper_train)) < 5:
            raise DataValidationError(
                "not enough rows with both target and outside alternatives available "
                "for the upper-stage logit."
            )
        upper_y = chosen_in_target.astype(float)
        upper_result = _fit_bounded_scalar_logit_with_offset(
            target_lse[upper_train],
            outside_offset[upper_train],
            upper_y[upper_train],
            weights[upper_train],
            lower=self.min_lambda,
            upper=self.max_lambda,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        lam = upper_result.coefficient

        self.coef_ = beta
        self.gamma_ = beta
        self.lambda_ = float(lam)
        self.lambdas_ = {
            label: (float(lam) if label == target else 1.0)
            for label in np.unique(nest_labels)
        }
        self.target_nest_ = target
        self.nest_labels_ = nest_labels
        self.target_mask_ = target_mask
        self.outside_mask_ = outside_mask
        self.stage1_model_ = stage1_model
        self.auxiliary_within_model_ = auxiliary_model
        self.auxiliary_coef_ = auxiliary_beta
        self.baseline_mnl_ = stage1_model
        self.upper_result_ = upper_result
        self.upper_feature_ = target_lse
        self.upper_offset_ = outside_offset
        self.upper_train_mask_ = upper_train
        self.X_ = X_arr
        self.y_ = y_arr
        self.availability_ = availability_arr
        self.sample_weight_ = weights
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"beta_{idx}" for idx in range(n_features)
        ]
        self.alt_names_ = list(alt_names) if alt_names is not None else [
            f"alt_{idx}" for idx in range(n_alternatives)
        ]
        self.probabilities_ = self.predict_proba()
        self.log_likelihood_ = float(
            np.sum(weights * safe_log(self.probabilities_[np.arange(n_choices), y_arr]))
        )
        self.log_loss_ = log_loss(y_arr, self.probabilities_, weights)
        self.n_parameters_ = n_features + 1
        self.baseline_log_likelihood_ = stage1_model.log_likelihood_
        self.baseline_log_loss_ = log_loss(y_arr, stage1_model.probabilities_, weights)
        self.log_likelihood_gain_ = self.log_likelihood_ - stage1_model.log_likelihood_
        return self

    def predict_proba(
        self, X: Optional[np.ndarray] = None, *, availability: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict probabilities using the standard nested-logit form.

        The two stages recover the full-sample PD-MUSE taste vector and a target
        nest dissimilarity. For prediction we report the standard nested-logit
        probability system with that target nest set to the recovered
        dissimilarity and all other nests set to one.
        """

        self._check_is_fitted()
        X_arr = self.X_ if X is None else np.asarray(X, dtype=float)
        if X_arr.ndim != 3 or X_arr.shape[2] != self.gamma_.shape[0]:
            raise DataValidationError("X must have shape (n_choices, n_alternatives, n_features).")
        availability_arr = (
            self.availability_
            if X is None and availability is None
            else np.ones(X_arr.shape[:2], dtype=bool)
            if availability is None
            else np.asarray(availability, dtype=bool)
        )
        return nested_probabilities(
            X_arr,
            self.coef_,
            self.nest_labels_,
            self.lambdas_,
            availability=availability_arr,
        )

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(X, availability=availability), axis=1)

    def log_likelihood(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        self._check_is_fitted()
        if X is None:
            X_arr, y_arr, weights = self.X_, self.y_, self.sample_weight_
            probs = self.probabilities_
        else:
            if y is None:
                raise DataValidationError("y is required when X is provided.")
            X_arr, y_arr, _, weights = validate_choice_arrays(X, y, availability, sample_weight)
            probs = self.predict_proba(X_arr, availability=availability)
        return float(np.sum(weights * safe_log(probs[np.arange(X_arr.shape[0]), y_arr])))

    def _choose_target_nest(self, nest_labels: np.ndarray) -> Any:
        if self.target_nest is not None:
            if self.target_nest not in set(nest_labels.tolist()):
                raise DataValidationError(f"unknown target nest {self.target_nest!r}.")
            return self.target_nest
        counts = {label: int(np.sum(nest_labels == label)) for label in np.unique(nest_labels)}
        candidates = [label for label, count in counts.items() if count >= 2]
        if len(candidates) != 1:
            raise DataValidationError(
                "target_nest is required when there is not exactly one non-singleton nest."
            )
        return candidates[0]

    @staticmethod
    def _target_inclusive_value(
        X: np.ndarray,
        beta: np.ndarray,
        target_mask: np.ndarray,
        availability: np.ndarray,
    ) -> np.ndarray:
        utilities = np.einsum("njk,k->nj", X, beta)
        target_available = availability[:, target_mask].any(axis=1)
        target_lse = np.full(X.shape[0], np.nan, dtype=float)
        if np.any(target_available):
            target_lse[target_available] = logsumexp_masked(
                utilities[target_available][:, target_mask],
                availability[target_available][:, target_mask],
            )
        return target_lse

    @staticmethod
    def _outside_inclusive_value(
        X: np.ndarray,
        beta: np.ndarray,
        outside_mask: np.ndarray,
        availability: np.ndarray,
    ) -> np.ndarray:
        utilities = np.einsum("njk,k->nj", X, beta)
        outside_available = availability[:, outside_mask].any(axis=1)
        offset = np.full(X.shape[0], np.nan, dtype=float)
        if np.any(outside_available):
            offset[outside_available] = logsumexp_masked(
                utilities[outside_available][:, outside_mask],
                availability[outside_available][:, outside_mask],
            )
        return offset

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("fit must be called before this method.")


class NestedLogitMLE:
    """Maximum-likelihood nested logit for one estimated non-singleton nest.

    This class is included as a benchmark and refinement tool. A common workflow
    is to fit ``PDMuseMNL`` first, then refine a supported nesting structure by
    likelihood. The implementation estimates one target nest dissimilarity in
    ``(min_lambda, max_lambda]`` and keeps all other nests at one.
    """

    def __init__(
        self,
        *,
        target_nest: Optional[Any] = None,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0,
        l2_penalty: float = 0.0,
        tol: float = 1e-4,
        max_iter: int = 300,
        verbose: bool = False,
    ) -> None:
        if not (0.0 < min_lambda < max_lambda <= 1.0):
            raise DataValidationError(
                "lambda bounds must satisfy 0 < min_lambda < max_lambda <= 1."
            )
        if l2_penalty < 0.0:
            raise DataValidationError("l2_penalty must be nonnegative.")
        self.target_nest = target_nest
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)
        self.l2_penalty = float(l2_penalty)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)

    def fit(
        self,
        X: np.ndarray | ChoiceDataset,
        y: Optional[np.ndarray] = None,
        nests: Optional[Sequence[Any] | dict[Any, Any]] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        alt_names: Optional[Sequence[str]] = None,
        initial_beta: Optional[np.ndarray] = None,
        initial_lambda: float = 0.9,
    ) -> "NestedLogitMLE":
        if isinstance(X, ChoiceDataset):
            dataset = X
            X_arr, y_arr, availability_arr, weights = (
                dataset.X,
                dataset.y,
                dataset.availability,
                dataset.sample_weight,
            )
            feature_names = dataset.feature_names
            alt_names = dataset.alt_names
        else:
            if y is None:
                raise DataValidationError("y is required when X is not a ChoiceDataset.")
            X_arr, y_arr, availability_arr, weights = validate_choice_arrays(
                X, y, availability, sample_weight
            )
        if nests is None:
            raise DataValidationError("nests are required.")

        n_choices, n_alternatives, n_features = X_arr.shape
        nest_labels = normalize_nests(nests, n_alternatives, alt_names)
        target = self._choose_target_nest(nest_labels)
        target_mask = nest_labels == target
        if target_mask.sum() < 2:
            raise DataValidationError("target nest must contain at least two alternatives.")

        if initial_beta is None:
            initial_model = PDMuseMNL(tol=min(self.tol, 1e-8), max_iter=150)
            initial_model.fit(
                X_arr,
                y_arr,
                availability=availability_arr,
                sample_weight=weights,
                feature_names=feature_names,
                alt_names=alt_names,
            )
            beta0 = initial_model.coef_
            self.initial_mnl_ = initial_model
        else:
            beta0 = np.asarray(initial_beta, dtype=float)
            if beta0.shape != (n_features,):
                raise DataValidationError("initial_beta must have shape (n_features,).")
        lam0 = float(np.clip(initial_lambda, self.min_lambda + 1e-6, self.max_lambda - 1e-8))
        params = np.concatenate([beta0, [self._lambda_to_eta(lam0)]])

        objective, gradient = self._objective_gradient(
            params, X_arr, y_arr, nest_labels, target, availability_arr, weights
        )
        inv_hessian = np.eye(params.shape[0])
        weight_scale = max(float(np.sum(weights)), 1.0)
        gradient_tol = max(self.tol, self.tol * np.sqrt(weight_scale))
        converged = False
        message = "maximum iterations reached"

        for iteration in range(1, self.max_iter + 1):
            grad_norm = float(np.max(np.abs(gradient)))
            if grad_norm <= gradient_tol:
                converged = True
                message = "converged"
                break

            direction = -inv_hessian @ gradient
            if not np.all(np.isfinite(direction)) or float(np.dot(gradient, direction)) >= 0.0:
                inv_hessian = np.eye(params.shape[0])
                direction = -gradient
            directional = float(np.dot(gradient, direction))

            accepted = False
            step_size = 1.0
            for _ in range(40):
                candidate = params + step_size * direction
                candidate_objective, candidate_gradient = self._objective_gradient(
                    candidate, X_arr, y_arr, nest_labels, target, availability_arr, weights
                )
                if candidate_objective <= objective + 1e-4 * step_size * directional:
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                message = "line search failed"
                break

            step = candidate - params
            grad_delta = candidate_gradient - gradient
            curvature = float(np.dot(step, grad_delta))
            if curvature > 1e-10:
                rho = 1.0 / curvature
                identity = np.eye(params.shape[0])
                left = identity - rho * np.outer(step, grad_delta)
                right = identity - rho * np.outer(grad_delta, step)
                inv_hessian = left @ inv_hessian @ right + rho * np.outer(step, step)
            else:
                inv_hessian = np.eye(params.shape[0])

            params = candidate
            objective = candidate_objective
            gradient = candidate_gradient
            if self.verbose:
                lam = self._eta_to_lambda(params[-1])
                print(
                    f"iter={iteration} nll={objective:.8f} "
                    f"grad_inf={np.max(np.abs(gradient)):.3e} lambda={lam:.4f}"
                )
        else:
            iteration = self.max_iter

        if not converged and float(np.max(np.abs(gradient))) <= gradient_tol:
            converged = True
            message = "converged"

        beta = params[:-1]
        lam = self._eta_to_lambda(params[-1])
        self.coef_ = beta
        self.lambda_ = float(lam)
        self.lambdas_ = {
            label: (float(lam) if label == target else 1.0)
            for label in np.unique(nest_labels)
        }
        self.target_nest_ = target
        self.nest_labels_ = nest_labels
        self.X_ = X_arr
        self.y_ = y_arr
        self.availability_ = availability_arr
        self.sample_weight_ = weights
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"beta_{idx}" for idx in range(n_features)
        ]
        self.alt_names_ = list(alt_names) if alt_names is not None else [
            f"alt_{idx}" for idx in range(n_alternatives)
        ]
        self.probabilities_ = self.predict_proba()
        self.log_likelihood_ = float(
            np.sum(weights * safe_log(self.probabilities_[np.arange(n_choices), y_arr]))
        )
        self.n_parameters_ = n_features + 1
        self.converged_ = converged
        self.n_iter_ = iteration
        self.result_ = ScalarLogitResult(
            coefficient=float(lam),
            converged=converged,
            n_iter=iteration,
            objective=float(objective),
            gradient=float(np.max(np.abs(gradient))),
            message=message,
        )
        return self

    def predict_proba(
        self,
        X: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        X_arr = self.X_ if X is None else np.asarray(X, dtype=float)
        availability_arr = (
            self.availability_
            if X is None and availability is None
            else np.ones(X_arr.shape[:2], dtype=bool)
            if availability is None
            else np.asarray(availability, dtype=bool)
        )
        return nested_probabilities(
            X_arr,
            self.coef_,
            self.nest_labels_,
            self.lambdas_,
            availability=availability_arr,
        )

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.argmax(self.predict_proba(X, availability=availability), axis=1)

    def _choose_target_nest(self, nest_labels: np.ndarray) -> Any:
        if self.target_nest is not None:
            if self.target_nest not in set(nest_labels.tolist()):
                raise DataValidationError(f"unknown target nest {self.target_nest!r}.")
            return self.target_nest
        counts = {label: int(np.sum(nest_labels == label)) for label in np.unique(nest_labels)}
        candidates = [label for label, count in counts.items() if count >= 2]
        if len(candidates) != 1:
            raise DataValidationError(
                "target_nest is required when there is not exactly one non-singleton nest."
            )
        return candidates[0]

    def _eta_to_lambda(self, eta: float) -> float:
        return float(
            self.min_lambda
            + (self.max_lambda - self.min_lambda) * sigmoid(np.array([eta]))[0]
        )

    def _lambda_to_eta(self, lam: float) -> float:
        scaled = (lam - self.min_lambda) / (self.max_lambda - self.min_lambda)
        scaled = float(np.clip(scaled, 1e-12, 1.0 - 1e-12))
        return float(np.log(scaled / (1.0 - scaled)))

    def _objective_gradient(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        nest_labels: np.ndarray,
        target: Any,
        availability: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        beta = params[:-1]
        lam = self._eta_to_lambda(params[-1])
        components = _nested_components(X, beta, nest_labels, target, lam, availability)
        chosen_group = components["chosen_group_lookup"][y]
        chosen_is_target = nest_labels[y] == target
        upper = components["upper"]
        conditional = components["conditional"]
        chosen_prob = conditional[np.arange(X.shape[0]), y] * upper[
            np.arange(X.shape[0]), chosen_group
        ]
        nll = float(-np.sum(weights * safe_log(chosen_prob)))
        if self.l2_penalty:
            nll += 0.5 * self.l2_penalty * float(np.dot(beta, beta))

        expected_upper = np.einsum("ng,ngk->nk", upper, components["expected_x"])
        chosen_x = X[np.arange(X.shape[0]), y]
        chosen_expected = components["expected_x"][np.arange(X.shape[0]), chosen_group, :]
        chosen_lambda = np.where(chosen_is_target, lam, 1.0)
        grad_log_beta = (
            chosen_x / chosen_lambda[:, None]
            + (1.0 - 1.0 / chosen_lambda)[:, None] * chosen_expected
            - expected_upper
        )
        grad_beta = -np.einsum("n,nk->k", weights, grad_log_beta)
        if self.l2_penalty:
            grad_beta = grad_beta + self.l2_penalty * beta

        target_idx = components["target_index"]
        target_available = components["nest_available"][:, target_idx]
        d_iv = np.zeros(X.shape[0], dtype=float)
        rows = target_available
        d_iv[rows] = (
            components["log_den"][rows, target_idx]
            - components["expected_v"][rows, target_idx] / lam
        )
        chosen_v = np.einsum("nk,k->n", chosen_x, beta)
        conditional_part = np.zeros(X.shape[0], dtype=float)
        chosen_target = chosen_is_target
        conditional_part[chosen_target] = (
            components["expected_v"][chosen_target, target_idx] - chosen_v[chosen_target]
        ) / (lam * lam)
        grad_log_lambda = (chosen_target.astype(float) - upper[:, target_idx]) * d_iv
        grad_log_lambda = grad_log_lambda + conditional_part
        dlam_deta = (lam - self.min_lambda) * (self.max_lambda - lam) / (
            self.max_lambda - self.min_lambda
        )
        grad_eta = -float(np.sum(weights * grad_log_lambda) * dlam_deta)
        return nll, np.concatenate([grad_beta, [grad_eta]])

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("fit must be called before this method.")


def _lambda_map(
    lambdas: dict[Any, float] | Sequence[float] | float, unique_nests: Sequence[Any]
) -> dict[Any, float]:
    if isinstance(lambdas, dict):
        missing = [nest for nest in unique_nests if nest not in lambdas]
        if missing:
            raise DataValidationError(f"missing lambda values for nests: {missing}")
        return {nest: float(lambdas[nest]) for nest in unique_nests}
    if np.isscalar(lambdas):
        return {nest: float(lambdas) for nest in unique_nests}
    values = list(lambdas)
    if len(values) != len(unique_nests):
        raise DataValidationError("lambda sequence must match the number of nests.")
    return {nest: float(value) for nest, value in zip(unique_nests, values, strict=True)}


def _nested_components(
    X: np.ndarray,
    beta: np.ndarray,
    nest_labels: np.ndarray,
    target: Any,
    target_lambda: float,
    availability: np.ndarray,
) -> dict[str, np.ndarray | int]:
    n_choices, n_alternatives, n_features = X.shape
    unique_nests = list(dict.fromkeys(nest_labels.tolist()))
    target_index = unique_nests.index(target)
    utilities = np.einsum("njk,k->nj", X, beta)
    inclusive = np.full((n_choices, len(unique_nests)), -np.inf, dtype=float)
    log_den = np.full_like(inclusive, np.nan, dtype=float)
    conditional = np.zeros((n_choices, n_alternatives), dtype=float)
    expected_x = np.zeros((n_choices, len(unique_nests), n_features), dtype=float)
    expected_v = np.zeros((n_choices, len(unique_nests)), dtype=float)
    nest_available = np.zeros_like(inclusive, dtype=bool)
    chosen_group_lookup = np.empty(n_alternatives, dtype=int)

    for g_idx, nest in enumerate(unique_nests):
        lam = target_lambda if nest == target else 1.0
        mask = nest_labels == nest
        chosen_group_lookup[mask] = g_idx
        available_g = availability[:, mask]
        rows = available_g.any(axis=1)
        nest_available[:, g_idx] = rows
        if not np.any(rows):
            continue
        scaled = utilities[rows][:, mask] / lam
        log_den_rows = logsumexp_masked(scaled, available_g[rows])
        probs = softmax_masked(scaled, available_g[rows])
        inclusive[rows, g_idx] = lam * log_den_rows
        log_den[rows, g_idx] = log_den_rows
        conditional[np.ix_(rows, mask)] = probs
        X_group = X[rows][:, mask, :]
        expected_x[rows, g_idx, :] = np.einsum("nj,njk->nk", probs, X_group)
        expected_v[rows, g_idx] = np.einsum("nj,nj->n", probs, utilities[rows][:, mask])

    upper = softmax_masked(inclusive, nest_available)
    return {
        "upper": upper,
        "conditional": conditional,
        "expected_x": expected_x,
        "expected_v": expected_v,
        "log_den": log_den,
        "nest_available": nest_available,
        "target_index": target_index,
        "chosen_group_lookup": chosen_group_lookup,
    }


def _fit_bounded_scalar_logit(
    feature: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
    tol: float,
    max_iter: int,
) -> ScalarLogitResult:
    feature = np.asarray(feature, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if feature.ndim != 1:
        raise DataValidationError("upper-stage feature must be one-dimensional.")

    def objective(coef: float) -> float:
        eta = coef * feature
        return float(np.sum(weights * (np.logaddexp(0.0, eta) - y * eta)))

    def grad_hess(coef: float) -> tuple[float, float]:
        p = sigmoid(coef * feature)
        grad = float(np.sum(weights * (p - y) * feature))
        hess = float(np.sum(weights * p * (1.0 - p) * feature * feature))
        return grad, hess

    grid = np.linspace(lower, upper, 101)
    grid_values = np.array([objective(float(value)) for value in grid])
    coef = float(grid[int(np.argmin(grid_values))])
    converged = False
    message = "maximum iterations reached"
    obj = objective(coef)
    grad, hess = grad_hess(coef)

    iteration = 0
    for _iteration in range(1, max_iter + 1):
        iteration = _iteration
        if abs(grad) <= tol:
            converged = True
            message = "converged"
            break
        if hess <= 1e-14:
            raise OptimizationError("upper-stage scalar logit Hessian is numerically zero.")
        step = grad / hess
        direction = -step
        directional = grad * direction
        accepted = False
        step_size = 1.0
        for _ in range(40):
            candidate = min(max(coef + step_size * direction, lower), upper)
            candidate_obj = objective(candidate)
            if candidate_obj <= obj + 1e-4 * step_size * directional:
                coef = candidate
                obj = candidate_obj
                grad, hess = grad_hess(coef)
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            message = "line search failed"
            break
    return ScalarLogitResult(float(coef), converged, iteration, float(obj), float(grad), message)


def _fit_bounded_scalar_logit_with_offset(
    feature: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
    tol: float,
    max_iter: int,
) -> ScalarLogitResult:
    feature = np.asarray(feature, dtype=float)
    offset = np.asarray(offset, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if feature.ndim != 1 or offset.shape != feature.shape:
        raise DataValidationError("upper-stage feature and offset must be one-dimensional.")
    finite = np.isfinite(feature) & np.isfinite(offset)
    if not np.all(finite):
        feature = feature[finite]
        offset = offset[finite]
        y = y[finite]
        weights = weights[finite]

    def objective(coef: float) -> float:
        eta_target = coef * feature
        return float(
            np.sum(
                weights
                * (np.logaddexp(eta_target, offset) - y * eta_target - (1.0 - y) * offset)
            )
        )

    def grad_hess(coef: float) -> tuple[float, float]:
        p = sigmoid(coef * feature - offset)
        grad = float(np.sum(weights * (p - y) * feature))
        hess = float(np.sum(weights * p * (1.0 - p) * feature * feature))
        return grad, hess

    grid = np.linspace(lower, upper, 101)
    grid_values = np.array([objective(float(value)) for value in grid])
    coef = float(grid[int(np.argmin(grid_values))])
    converged = False
    message = "maximum iterations reached"
    obj = objective(coef)
    grad, hess = grad_hess(coef)

    iteration = 0
    for _iteration in range(1, max_iter + 1):
        iteration = _iteration
        if abs(grad) <= tol:
            converged = True
            message = "converged"
            break
        if hess <= 1e-14:
            raise OptimizationError("upper-stage scalar logit Hessian is numerically zero.")
        direction = -grad / hess
        directional = grad * direction
        accepted = False
        step_size = 1.0
        for _ in range(40):
            candidate = min(max(coef + step_size * direction, lower), upper)
            candidate_obj = objective(candidate)
            if candidate_obj <= obj + 1e-4 * step_size * directional:
                coef = candidate
                obj = candidate_obj
                grad, hess = grad_hess(coef)
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            message = "line search failed"
            break
    return ScalarLogitResult(float(coef), converged, iteration, float(obj), float(grad), message)
