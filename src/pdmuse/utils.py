"""Numerical utilities for stable choice-model calculations."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .exceptions import DataValidationError


def as_2d_float(array: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(array, dtype=float)
    if out.ndim != 2:
        raise DataValidationError(f"{name} must be a two-dimensional array.")
    if not np.all(np.isfinite(out)):
        raise DataValidationError(f"{name} contains NaN or infinite values.")
    return out


def logsumexp_masked(values: np.ndarray, availability: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute row-wise log-sum-exp with optional availability masking."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise DataValidationError("values must be a two-dimensional array.")

    if availability is None:
        availability = np.ones(values.shape, dtype=bool)
    else:
        availability = np.asarray(availability, dtype=bool)
        if availability.shape != values.shape:
            raise DataValidationError("availability must match values shape.")

    if np.any(~availability.any(axis=1)):
        raise DataValidationError("each choice situation needs at least one available alternative.")

    masked = np.where(availability, values, -np.inf)
    max_values = np.max(masked, axis=1, keepdims=True)
    shifted = np.where(availability, np.exp(masked - max_values), 0.0)
    sums = np.sum(shifted, axis=1)
    return np.ravel(max_values) + np.log(sums)


def softmax_masked(values: np.ndarray, availability: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute row-wise softmax probabilities with unavailable alternatives set to zero."""

    values = np.asarray(values, dtype=float)
    if availability is None:
        availability = np.ones(values.shape, dtype=bool)
    else:
        availability = np.asarray(availability, dtype=bool)
    denom = logsumexp_masked(values, availability)[:, None]
    probs = np.zeros_like(values, dtype=float)
    shifted = values - denom
    probs[availability] = np.exp(shifted[availability])
    row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable logistic transform."""

    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def safe_log(values: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Log with clipping for metrics."""

    return np.log(np.clip(values, eps, 1.0))


def entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy with the convention 0 log 0 = 0."""

    probabilities = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, -probabilities * np.log(probabilities), 0.0)
    return np.sum(terms, axis=axis)


def pinv_symmetric(matrix: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Pseudo-inverse for a symmetric positive semi-definite matrix."""

    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    cutoff = rcond * max(1.0, float(np.max(np.abs(values))))
    inverse_values = np.where(values > cutoff, 1.0 / values, 0.0)
    return (vectors * inverse_values) @ vectors.T


def format_feature_names(
    names: Optional[list[str] | tuple[str, ...]],
    n_features: int,
) -> list[str]:
    """Return feature names with a deterministic fallback."""

    if names is None:
        return [f"beta_{idx}" for idx in range(n_features)]
    if len(names) != n_features:
        raise DataValidationError("feature_names length must equal the number of features.")
    return [str(name) for name in names]
