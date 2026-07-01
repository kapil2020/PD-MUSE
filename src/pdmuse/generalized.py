"""Generalized-entropy and inverse-product differentiation logit tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .exceptions import DataValidationError, NotFittedError
from .utils import entropy, pinv_symmetric, softmax_masked


def generalized_entropy(
    shares: np.ndarray,
    groups: Sequence[Sequence[Any]],
    mu: Sequence[float],
) -> np.ndarray:
    """Compute inverse-product generalized entropy for one or many markets."""

    share_arr = np.asarray(shares, dtype=float)
    if share_arr.ndim == 1:
        share_arr = share_arr[None, :]
    if np.any(share_arr < 0.0) or not np.allclose(share_arr.sum(axis=1), 1.0):
        raise DataValidationError("shares must be nonnegative and sum to one in each market.")

    group_arrays = _validate_groups(groups, share_arr.shape[1])
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.shape != (len(group_arrays),):
        raise DataValidationError("mu must have one value per grouping characteristic.")
    if np.any(mu_arr < 0.0) or np.sum(mu_arr) > 1.0 + 1e-12:
        raise DataValidationError("mu values must be nonnegative and sum to at most one.")

    total = (1.0 - float(np.sum(mu_arr))) * entropy(share_arr, axis=1)
    for weight, labels in zip(mu_arr, group_arrays, strict=True):
        group_shares = _group_shares(share_arr, labels)
        total = total + weight * entropy(group_shares, axis=1)
    return total


@dataclass
class InverseShareResult:
    coefficients: np.ndarray
    mu: np.ndarray
    residual_sum_squares: float
    rank: int
    covariance: np.ndarray


class InverseProductDifferentiationLogit:
    """Recover IPDL taste and grouping parameters from the linear inverse-share system."""

    def __init__(self, *, reference: int = 0, rcond: Optional[float] = None) -> None:
        self.reference = int(reference)
        self.rcond = rcond

    def fit_inverse_shares(
        self,
        shares: np.ndarray,
        X: np.ndarray,
        groups: Sequence[Sequence[Any]],
        *,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "InverseProductDifferentiationLogit":
        """Fit the linear differenced inverse-share regression.

        The stacked regression is

        ``log(q_j/q_r) = beta' (x_j - x_r) - sum_d mu_d * b_d(j, r)``,

        where ``b_d(j, r)`` is the differenced group-share contrast from the
        manuscript's inverse-share equation.
        """

        shares_arr, X_arr, weights = _validate_market_arrays(shares, X, sample_weight)
        design, target = inverse_share_regression_matrix(
            shares_arr, X_arr, groups, reference=self.reference
        )
        if weights is not None:
            row_weights = np.repeat(weights, shares_arr.shape[1] - 1)
            scale = np.sqrt(row_weights)
            design_fit = design * scale[:, None]
            target_fit = target * scale
        else:
            design_fit = design
            target_fit = target

        solution, residuals, rank, _ = np.linalg.lstsq(design_fit, target_fit, rcond=self.rcond)
        n_features = X_arr.shape[2]
        beta = solution[:n_features]
        mu = solution[n_features:]
        fitted = design @ solution
        residual = target - fitted
        rss = float(np.dot(residual, residual))
        dof = max(1, design.shape[0] - design.shape[1])
        sigma2 = rss / dof
        covariance = sigma2 * pinv_symmetric(design.T @ design)

        self.coef_ = beta
        self.mu_ = mu
        self.groups_ = [np.asarray(g, dtype=object) for g in groups]
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"beta_{idx}" for idx in range(n_features)
        ]
        self.result_ = InverseShareResult(beta, mu, rss, int(rank), covariance)
        self.n_parameters_ = int(solution.shape[0])
        return self

    def inverse_utilities(
        self,
        shares: np.ndarray,
        groups: Optional[Sequence[Sequence[Any]]] = None,
    ) -> np.ndarray:
        """Return the inverse-share right-hand side implied by fitted ``mu``."""

        self._check_is_fitted()
        share_arr = np.asarray(shares, dtype=float)
        if share_arr.ndim == 1:
            share_arr = share_arr[None, :]
        group_arrays = _validate_groups(groups or self.groups_, share_arr.shape[1])
        log_q = np.log(np.clip(share_arr, 1e-300, 1.0))
        out = (1.0 - float(np.sum(self.mu_))) * log_q
        for weight, labels in zip(self.mu_, group_arrays, strict=True):
            group_shares = _group_shares(share_arr, labels)
            expanded = np.zeros_like(share_arr)
            unique = list(dict.fromkeys(labels.tolist()))
            for g_idx, label in enumerate(unique):
                expanded[:, labels == label] = group_shares[:, [g_idx]]
            out = out + weight * np.log(np.clip(expanded, 1e-300, 1.0))
        return out

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise NotFittedError("fit_inverse_shares must be called before this method.")


def inverse_share_regression_matrix(
    shares: np.ndarray,
    X: np.ndarray,
    groups: Sequence[Sequence[Any]],
    *,
    reference: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the linear inverse-share regression matrix."""

    shares_arr, X_arr, _ = _validate_market_arrays(shares, X, None)
    n_markets, n_alternatives, n_features = X_arr.shape
    if reference < 0 or reference >= n_alternatives:
        raise DataValidationError("reference alternative is outside the valid range.")
    group_arrays = _validate_groups(groups, n_alternatives)
    log_q = np.log(np.clip(shares_arr, 1e-300, 1.0))

    rows = []
    target = []
    for m in range(n_markets):
        for j in range(n_alternatives):
            if j == reference:
                continue
            dx = X_arr[m, j, :] - X_arr[m, reference, :]
            log_diff = log_q[m, j] - log_q[m, reference]
            group_terms = []
            for labels in group_arrays:
                group_share = _group_share_lookup(shares_arr[m], labels)
                gj = labels[j]
                gr = labels[reference]
                bracket = (
                    np.log(np.clip(group_share[gj], 1e-300, 1.0))
                    - np.log(np.clip(group_share[gr], 1e-300, 1.0))
                    - log_diff
                )
                group_terms.append(-bracket)
            rows.append(np.concatenate([dx, np.asarray(group_terms, dtype=float)]))
            target.append(log_diff)
    return np.vstack(rows), np.asarray(target, dtype=float)


def mnl_market_shares(utilities: np.ndarray) -> np.ndarray:
    """Convenience function for the grouping-free IPDL anchor."""

    utilities = np.asarray(utilities, dtype=float)
    return softmax_masked(utilities, np.ones_like(utilities, dtype=bool))


def _validate_market_arrays(
    shares: np.ndarray,
    X: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    shares_arr = np.asarray(shares, dtype=float)
    if shares_arr.ndim == 1:
        shares_arr = shares_arr[None, :]
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 3 or X_arr.shape[:2] != shares_arr.shape:
        raise DataValidationError("X must have shape (n_markets, n_alternatives, n_features).")
    if np.any(shares_arr <= 0.0) or not np.allclose(shares_arr.sum(axis=1), 1.0):
        raise DataValidationError("shares must be positive and sum to one in each market.")
    if sample_weight is None:
        return shares_arr, X_arr, None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (shares_arr.shape[0],):
        raise DataValidationError("sample_weight must have one value per market.")
    return shares_arr, X_arr, weights


def _validate_groups(groups: Sequence[Sequence[Any]], n_alternatives: int) -> list[np.ndarray]:
    if not groups:
        raise DataValidationError("at least one grouping characteristic is required.")
    arrays = []
    for labels in groups:
        arr = np.asarray(list(labels), dtype=object)
        if arr.shape != (n_alternatives,):
            raise DataValidationError(
                "each grouping characteristic needs one label per alternative."
            )
        arrays.append(arr)
    return arrays


def _group_shares(shares: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique = list(dict.fromkeys(labels.tolist()))
    out = np.zeros((shares.shape[0], len(unique)), dtype=float)
    for idx, label in enumerate(unique):
        out[:, idx] = shares[:, labels == label].sum(axis=1)
    return out


def _group_share_lookup(shares: np.ndarray, labels: np.ndarray) -> dict[Any, float]:
    return {label: float(shares[labels == label].sum()) for label in dict.fromkeys(labels.tolist())}
