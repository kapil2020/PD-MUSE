"""PD-MUSE multinomial logit estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .data import ChoiceDataset, validate_choice_arrays
from .exceptions import DataValidationError, NotFittedError
from .utils import (
    entropy,
    format_feature_names,
    logsumexp_masked,
    pinv_symmetric,
    safe_log,
    softmax_masked,
)


@dataclass
class OptimizationResult:
    """Small optimizer result object exposed on fitted models."""

    converged: bool
    n_iter: int
    objective: float
    gradient_norm: float
    message: str


class PDMuseMNL:
    """Multinomial logit through the PD-MUSE dual.

    The fitted coefficients minimize

    ``sum_n log(sum_j exp(beta' x_nj)) - sum_n beta' x_n,y[n]``,

    which is exactly the dual objective derived in the manuscript and the
    negative multinomial-logit log-likelihood.
    """

    def __init__(
        self,
        *,
        l2_penalty: float = 0.0,
        tol: float = 1e-8,
        max_iter: int = 200,
        armijo: float = 1e-4,
        max_line_search: int = 40,
        verbose: bool = False,
    ) -> None:
        if l2_penalty < 0.0:
            raise DataValidationError("l2_penalty must be nonnegative.")
        self.l2_penalty = float(l2_penalty)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.armijo = float(armijo)
        self.max_line_search = int(max_line_search)
        self.verbose = bool(verbose)

    def fit(
        self,
        X: np.ndarray | ChoiceDataset,
        y: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        alt_names: Optional[Sequence[str]] = None,
        initial_beta: Optional[np.ndarray] = None,
    ) -> "PDMuseMNL":
        """Fit the PD-MUSE/MNL estimator."""

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

        n_choices, n_alternatives, n_features = X_arr.shape
        beta = (
            np.zeros(n_features, dtype=float)
            if initial_beta is None
            else np.asarray(initial_beta, dtype=float)
        )
        if beta.shape != (n_features,):
            raise DataValidationError("initial_beta must have shape (n_features,).")
        if not np.all(np.isfinite(beta)):
            raise DataValidationError("initial_beta contains NaN or infinite values.")

        objective, gradient, hessian, probabilities = self._objective_grad_hess(
            beta, X_arr, y_arr, availability_arr, weights
        )
        converged = False
        message = "maximum iterations reached"
        gradient_norm = float(np.max(np.abs(gradient)))

        for iteration in range(1, self.max_iter + 1):
            if gradient_norm <= self.tol:
                converged = True
                message = "converged"
                break

            step = self._newton_step(hessian, gradient)
            descent = -step
            directional = float(np.dot(gradient, descent))
            if not np.isfinite(directional) or directional >= 0.0:
                descent = -gradient
                directional = -float(np.dot(gradient, gradient))

            accepted = False
            step_size = 1.0
            for _ in range(self.max_line_search):
                candidate = beta + step_size * descent
                candidate_objective = self._objective(
                    candidate, X_arr, y_arr, availability_arr, weights
                )
                if candidate_objective <= objective + self.armijo * step_size * directional:
                    beta = candidate
                    accepted = True
                    break
                step_size *= 0.5

            if not accepted:
                message = "line search failed to improve the dual objective"
                break

            objective, gradient, hessian, probabilities = self._objective_grad_hess(
                beta, X_arr, y_arr, availability_arr, weights
            )
            gradient_norm = float(np.max(np.abs(gradient)))
            if self.verbose:
                print(
                    f"iter={iteration} objective={objective:.8f} "
                    f"grad_inf={gradient_norm:.3e} step={step_size:.3g}"
                )

        self.coef_ = beta
        self.feature_names_ = format_feature_names(
            list(feature_names) if feature_names else None, n_features
        )
        self.alt_names_ = (
            [str(name) for name in alt_names]
            if alt_names is not None
            else [f"alt_{idx}" for idx in range(n_alternatives)]
        )
        self.X_ = X_arr
        self.y_ = y_arr
        self.availability_ = availability_arr
        self.sample_weight_ = weights
        self.probabilities_ = probabilities
        self.log_likelihood_ = self.log_likelihood()
        self.dual_objective_ = objective
        self.entropy_ = float(np.sum(weights * entropy(probabilities)) / np.sum(weights))
        self.observed_information_ = self._hessian(beta, X_arr, availability_arr, weights)
        self.n_iter_ = iteration if "iteration" in locals() else 0
        self.converged_ = converged
        self.n_parameters_ = n_features
        self.result_ = OptimizationResult(
            converged=converged,
            n_iter=self.n_iter_,
            objective=float(objective),
            gradient_norm=float(gradient_norm),
            message=message,
        )
        if not converged and self.verbose:
            print(f"PD-MUSE MNL did not fully converge: {message}")
        return self

    def decision_function(
        self, X: Optional[np.ndarray] = None, *, availability: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self._check_is_fitted()
        X_arr = self.X_ if X is None else np.asarray(X, dtype=float)
        if X_arr.ndim != 3 or X_arr.shape[2] != self.coef_.shape[0]:
            raise DataValidationError("X must have shape (n_choices, n_alternatives, n_features).")
        return np.einsum("njk,k->nj", X_arr, self.coef_)

    def predict_proba(
        self, X: Optional[np.ndarray] = None, *, availability: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return fitted choice probabilities."""

        self._check_is_fitted()
        X_arr = self.X_ if X is None else np.asarray(X, dtype=float)
        if X_arr.ndim != 3 or X_arr.shape[2] != self.coef_.shape[0]:
            raise DataValidationError("X must have shape (n_choices, n_alternatives, n_features).")
        if availability is None and X is None:
            availability_arr = self.availability_
        elif availability is None:
            availability_arr = np.ones(X_arr.shape[:2], dtype=bool)
        else:
            availability_arr = np.asarray(availability, dtype=bool)
        utilities = np.einsum("njk,k->nj", X_arr, self.coef_)
        return softmax_masked(utilities, availability_arr)

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the highest-probability alternative index."""

        return np.argmax(self.predict_proba(X, availability=availability), axis=1)

    def log_likelihood(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate the multinomial-logit log-likelihood."""

        self._check_is_fitted(allow_during_fit=True)
        if X is None:
            X_arr, y_arr, availability_arr, weights = (
                self.X_,
                self.y_,
                self.availability_,
                self.sample_weight_,
            )
        else:
            if y is None:
                raise DataValidationError("y is required when X is provided.")
            X_arr, y_arr, availability_arr, weights = validate_choice_arrays(
                X, y, availability, sample_weight
            )
        probs = self.predict_proba(X_arr, availability=availability_arr)
        return float(np.sum(weights * safe_log(probs[np.arange(X_arr.shape[0]), y_arr])))

    def dual_objective(
        self,
        beta: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        *,
        availability: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate the PD-MUSE dual objective."""

        self._check_is_fitted(allow_during_fit=True)
        beta_arr = self.coef_ if beta is None else np.asarray(beta, dtype=float)
        if X is None:
            X_arr, y_arr, availability_arr, weights = (
                self.X_,
                self.y_,
                self.availability_,
                self.sample_weight_,
            )
        else:
            if y is None:
                raise DataValidationError("y is required when X is provided.")
            X_arr, y_arr, availability_arr, weights = validate_choice_arrays(
                X, y, availability, sample_weight
            )
        return self._objective(beta_arr, X_arr, y_arr, availability_arr, weights)

    def covariance(self, kind: str = "observed") -> np.ndarray:
        """Return observed-information or sandwich covariance."""

        self._check_is_fitted()
        H_inv = pinv_symmetric(self.observed_information_)
        if kind == "observed":
            return H_inv
        if kind != "sandwich":
            raise ValueError("kind must be 'observed' or 'sandwich'.")
        scores = self.per_choice_scores()
        meat = scores.T @ scores
        return H_inv @ meat @ H_inv

    def standard_errors(self, kind: str = "observed") -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.covariance(kind=kind)), 0.0))

    def per_choice_scores(self) -> np.ndarray:
        """Return per-choice log-likelihood score contributions."""

        self._check_is_fitted()
        expected = np.einsum("nj,njk->nk", self.probabilities_, self.X_)
        chosen = self.X_[np.arange(self.X_.shape[0]), self.y_]
        return self.sample_weight_[:, None] * (chosen - expected)

    def kkt_residuals(self) -> dict[str, float]:
        """Return primal-dual residuals for the manuscript's MNL KKT system."""

        self._check_is_fitted()
        utilities = self.decision_function()
        log_den = logsumexp_masked(utilities, self.availability_)
        alpha = log_den - 1.0
        mask = self.availability_
        alpha_available = np.repeat(alpha, mask.sum(axis=1))
        stationarity = (
            -np.log(np.clip(self.probabilities_[mask], 1e-300, 1.0))
            - 1.0
            - alpha_available
            + utilities[mask]
        )

        observed_totals = np.einsum(
            "n,nk->k", self.sample_weight_, self.X_[np.arange(self.X_.shape[0]), self.y_]
        )
        model_totals = np.einsum("n,nj,njk->k", self.sample_weight_, self.probabilities_, self.X_)
        moment_gap = model_totals - observed_totals
        denom = np.maximum(1.0, np.abs(observed_totals))
        return {
            "stationarity_max_abs": float(np.max(np.abs(stationarity))),
            "normalization_max_abs": float(np.max(np.abs(self.probabilities_.sum(axis=1) - 1.0))),
            "moment_max_abs": float(np.max(np.abs(moment_gap))),
            "moment_max_relative": float(np.max(np.abs(moment_gap) / denom)),
            "score_inf_norm": float(np.max(np.abs(-moment_gap))),
            "penalty_gradient_inf_norm": float(np.max(np.abs(self.l2_penalty * self.coef_))),
        }

    def summary(self, covariance: str = "observed") -> pd.DataFrame:
        """Return a coefficient table."""

        self._check_is_fitted()
        se = self.standard_errors(kind=covariance)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0.0, self.coef_ / se, np.nan)
        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "estimate": self.coef_,
                "std_error": se,
                "z": z,
            }
        )

    def _objective(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        availability: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        utilities = np.einsum("njk,k->nj", X, beta)
        log_den = logsumexp_masked(utilities, availability)
        chosen = utilities[np.arange(X.shape[0]), y]
        penalty = 0.5 * self.l2_penalty * float(np.dot(beta, beta))
        return float(np.sum(weights * (log_den - chosen)) + penalty)

    def _objective_grad_hess(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        availability: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        utilities = np.einsum("njk,k->nj", X, beta)
        probabilities = softmax_masked(utilities, availability)
        objective = self._objective(beta, X, y, availability, weights)
        expected = np.einsum("nj,njk->nk", probabilities, X)
        chosen = X[np.arange(X.shape[0]), y]
        gradient = np.einsum("n,nk->k", weights, expected - chosen)
        if self.l2_penalty:
            gradient = gradient + self.l2_penalty * beta
        hessian = self._hessian_from_probabilities(probabilities, X, weights)
        if self.l2_penalty:
            hessian = hessian + self.l2_penalty * np.eye(X.shape[2])
        return objective, gradient, hessian, probabilities

    def _hessian(
        self,
        beta: np.ndarray,
        X: np.ndarray,
        availability: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        utilities = np.einsum("njk,k->nj", X, beta)
        probabilities = softmax_masked(utilities, availability)
        hessian = self._hessian_from_probabilities(probabilities, X, weights)
        if self.l2_penalty:
            hessian = hessian + self.l2_penalty * np.eye(X.shape[2])
        return hessian

    @staticmethod
    def _hessian_from_probabilities(
        probabilities: np.ndarray, X: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        expected = np.einsum("nj,njk->nk", probabilities, X)
        second = np.einsum("n,nj,njk,njl->kl", weights, probabilities, X, X)
        outer = np.einsum("n,nk,nl->kl", weights, expected, expected)
        hessian = second - outer
        return 0.5 * (hessian + hessian.T)

    @staticmethod
    def _newton_step(hessian: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        jitter = 0.0
        identity = np.eye(hessian.shape[0])
        for _ in range(8):
            try:
                return np.linalg.solve(hessian + jitter * identity, gradient)
            except np.linalg.LinAlgError:
                jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
        return np.linalg.lstsq(hessian + jitter * identity, gradient, rcond=None)[0]

    def _check_is_fitted(self, allow_during_fit: bool = False) -> None:
        if allow_during_fit and hasattr(self, "coef_"):
            return
        if not hasattr(self, "coef_"):
            raise NotFittedError("fit must be called before this method.")


MultinomialLogit = PDMuseMNL
