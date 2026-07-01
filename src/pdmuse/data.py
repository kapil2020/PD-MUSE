"""Data containers and validation helpers for long-format choice data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .exceptions import DataValidationError


def validate_choice_arrays(
    X: np.ndarray,
    y: np.ndarray,
    availability: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize choice arrays.

    Parameters
    ----------
    X:
        Array of shape ``(n_choices, n_alternatives, n_features)``.
    y:
        Chosen alternative as integer labels of shape ``(n_choices,)`` or a one-hot
        matrix of shape ``(n_choices, n_alternatives)``.
    availability:
        Boolean mask of shape ``(n_choices, n_alternatives)``. Unavailable alternatives
        receive probability zero.
    sample_weight:
        Nonnegative choice-situation weights.
    """

    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 3:
        raise DataValidationError("X must have shape (n_choices, n_alternatives, n_features).")
    if not np.all(np.isfinite(X_arr)):
        raise DataValidationError("X contains NaN or infinite values.")

    n_choices, n_alternatives, _ = X_arr.shape
    if n_choices == 0 or n_alternatives < 2:
        raise DataValidationError("X must contain at least one choice and two alternatives.")

    y_arr = np.asarray(y)
    if y_arr.ndim == 2:
        if y_arr.shape != (n_choices, n_alternatives):
            raise DataValidationError("one-hot y must match the first two dimensions of X.")
        if not np.allclose(y_arr.sum(axis=1), 1.0):
            raise DataValidationError("each row of one-hot y must contain exactly one choice.")
        y_idx = np.argmax(y_arr, axis=1).astype(int)
    elif y_arr.ndim == 1:
        if y_arr.shape[0] != n_choices:
            raise DataValidationError("y length must equal the number of choice situations.")
        if not np.allclose(y_arr, np.round(y_arr)):
            raise DataValidationError("integer y labels are required.")
        y_idx = y_arr.astype(int)
    else:
        raise DataValidationError(
            "y must be one-dimensional labels or a two-dimensional one-hot array."
        )

    if np.any(y_idx < 0) or np.any(y_idx >= n_alternatives):
        raise DataValidationError("y contains an alternative index outside the valid range.")

    if availability is None:
        availability_arr = np.ones((n_choices, n_alternatives), dtype=bool)
    else:
        availability_arr = np.asarray(availability, dtype=bool)
        if availability_arr.shape != (n_choices, n_alternatives):
            raise DataValidationError("availability must have shape (n_choices, n_alternatives).")
    if np.any(~availability_arr.any(axis=1)):
        raise DataValidationError("each choice situation needs at least one available alternative.")
    if not np.all(availability_arr[np.arange(n_choices), y_idx]):
        raise DataValidationError("chosen alternatives must be available.")

    if sample_weight is None:
        weight_arr = np.ones(n_choices, dtype=float)
    else:
        weight_arr = np.asarray(sample_weight, dtype=float)
        if weight_arr.shape != (n_choices,):
            raise DataValidationError("sample_weight must have shape (n_choices,).")
        if np.any(weight_arr < 0.0) or not np.all(np.isfinite(weight_arr)):
            raise DataValidationError("sample_weight must be finite and nonnegative.")
    if float(np.sum(weight_arr)) <= 0.0:
        raise DataValidationError("sample_weight must have positive total weight.")

    return X_arr, y_idx, availability_arr, weight_arr


@dataclass
class ChoiceDataset:
    """Validated long-format choice data stored as dense choice arrays."""

    X: np.ndarray
    y: np.ndarray
    availability: Optional[np.ndarray] = None
    sample_weight: Optional[np.ndarray] = None
    alt_names: Optional[Sequence[str]] = None
    feature_names: Optional[Sequence[str]] = None
    choice_ids: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        X, y, availability, sample_weight = validate_choice_arrays(
            self.X, self.y, self.availability, self.sample_weight
        )
        self.X = X
        self.y = y
        self.availability = availability
        self.sample_weight = sample_weight

        _, n_alternatives, n_features = X.shape
        if self.alt_names is None:
            self.alt_names = tuple(f"alt_{idx}" for idx in range(n_alternatives))
        else:
            if len(self.alt_names) != n_alternatives:
                raise DataValidationError("alt_names length must equal the number of alternatives.")
            self.alt_names = tuple(str(name) for name in self.alt_names)

        if self.feature_names is None:
            self.feature_names = tuple(f"x_{idx}" for idx in range(n_features))
        else:
            if len(self.feature_names) != n_features:
                raise DataValidationError("feature_names length must equal the number of features.")
            self.feature_names = tuple(str(name) for name in self.feature_names)

        if self.choice_ids is not None and len(self.choice_ids) != X.shape[0]:
            raise DataValidationError(
                "choice_ids length must equal the number of choice situations."
            )

    @property
    def n_choices(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_alternatives(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[2])

    @property
    def y_one_hot(self) -> np.ndarray:
        out = np.zeros((self.n_choices, self.n_alternatives), dtype=float)
        out[np.arange(self.n_choices), self.y] = 1.0
        return out

    @classmethod
    def from_long_dataframe(
        cls,
        frame: pd.DataFrame,
        choice_id_col: str,
        alternative_col: str,
        choice_col: str,
        feature_cols: Sequence[str],
        availability_col: Optional[str] = None,
        weight_col: Optional[str] = None,
        alternative_order: Optional[Sequence[Any]] = None,
    ) -> "ChoiceDataset":
        """Create a dataset from one row per choice situation and alternative."""

        required = [choice_id_col, alternative_col, choice_col, *feature_cols]
        if availability_col is not None:
            required.append(availability_col)
        if weight_col is not None:
            required.append(weight_col)
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise DataValidationError(f"missing columns: {missing}")

        choice_ids = list(pd.unique(frame[choice_id_col]))
        alternatives = (
            list(alternative_order)
            if alternative_order is not None
            else list(pd.unique(frame[alternative_col]))
        )
        choice_index = {value: idx for idx, value in enumerate(choice_ids)}
        alt_index = {value: idx for idx, value in enumerate(alternatives)}

        n_choices = len(choice_ids)
        n_alternatives = len(alternatives)
        n_features = len(feature_cols)
        X = np.zeros((n_choices, n_alternatives, n_features), dtype=float)
        availability = np.zeros((n_choices, n_alternatives), dtype=bool)
        y = np.full(n_choices, -1, dtype=int)
        weights = np.ones(n_choices, dtype=float)

        column_index = {name: idx for idx, name in enumerate(frame.columns)}
        for row in frame.itertuples(index=False, name=None):
            choice_id = row[column_index[choice_id_col]]
            alternative = row[column_index[alternative_col]]
            if alternative not in alt_index:
                raise DataValidationError(f"unknown alternative {alternative!r}.")
            n = choice_index[choice_id]
            j = alt_index[alternative]
            X[n, j, :] = [float(row[column_index[col]]) for col in feature_cols]
            availability[n, j] = (
                bool(row[column_index[availability_col]]) if availability_col is not None else True
            )
            if bool(row[column_index[choice_col]]):
                if y[n] != -1:
                    raise DataValidationError(
                        f"choice situation {choice_id!r} has multiple choices."
                    )
                y[n] = j
            if weight_col is not None:
                weights[n] = float(row[column_index[weight_col]])

        bad = [choice_ids[idx] for idx in np.flatnonzero(y < 0)]
        if bad:
            raise DataValidationError(f"choice situations with no chosen alternative: {bad[:5]}")

        return cls(
            X=X,
            y=y,
            availability=availability,
            sample_weight=weights,
            alt_names=tuple(str(value) for value in alternatives),
            feature_names=tuple(feature_cols),
            choice_ids=tuple(choice_ids),
        )

    def to_long_dataframe(self) -> pd.DataFrame:
        """Return a long-format pandas frame."""

        rows: list[dict[str, Any]] = []
        choice_ids = self.choice_ids or tuple(range(self.n_choices))
        for n in range(self.n_choices):
            for j, alt in enumerate(self.alt_names or ()):
                row = {
                    "choice_id": choice_ids[n],
                    "alternative": alt,
                    "chosen": int(self.y[n] == j),
                    "available": bool(self.availability[n, j]),
                    "sample_weight": float(self.sample_weight[n]),
                }
                for k, name in enumerate(self.feature_names or ()):
                    row[name] = float(self.X[n, j, k])
                rows.append(row)
        return pd.DataFrame(rows)
