"""Metrics and model-comparison helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .data import validate_choice_arrays
from .utils import safe_log


def log_loss(
    y: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-15,
) -> float:
    """Mean negative log probability assigned to the chosen alternative."""

    probabilities = np.asarray(probabilities, dtype=float)
    y = np.asarray(y, dtype=int)
    if probabilities.ndim != 2 or probabilities.shape[0] != y.shape[0]:
        raise ValueError("probabilities must have shape (n_choices, n_alternatives).")
    weights = (
        np.ones(y.shape[0], dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight)
    )
    chosen = probabilities[np.arange(y.shape[0]), y]
    return float(-np.sum(weights * safe_log(chosen, eps=eps)) / np.sum(weights))


def accuracy(
    y: np.ndarray,
    probabilities: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Weighted top-1 choice prediction accuracy."""

    y = np.asarray(y, dtype=int)
    pred = np.argmax(probabilities, axis=1)
    weights = (
        np.ones(y.shape[0], dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight)
    )
    return float(np.sum(weights * (pred == y)) / np.sum(weights))


def aic(log_likelihood: float, n_parameters: int) -> float:
    return float(2 * n_parameters - 2 * log_likelihood)


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    return float(np.log(n_observations) * n_parameters - 2 * log_likelihood)


def compare_models(
    models: Iterable[object],
    X: np.ndarray,
    y: np.ndarray,
    availability: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare fitted models on log-likelihood, log loss, accuracy, AIC, and BIC."""

    X, y, availability, weights = validate_choice_arrays(X, y, availability, sample_weight)
    rows = []
    for model in models:
        probs = model.predict_proba(X, availability=availability)
        ll = float(np.sum(weights * safe_log(probs[np.arange(X.shape[0]), y])))
        n_params = int(getattr(model, "n_parameters_", np.size(getattr(model, "coef_", []))))
        rows.append(
            {
                "model": model.__class__.__name__,
                "log_likelihood": ll,
                "log_loss": log_loss(y, probs, weights),
                "accuracy": accuracy(y, probs, weights),
                "aic": aic(ll, n_params),
                "bic": bic(ll, n_params, X.shape[0]),
                "n_parameters": n_params,
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
