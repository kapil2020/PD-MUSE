"""Apollo mode-choice data helpers used by examples and tests."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ALTERNATIVES = ["car", "bus", "air", "rail"]
FEATURES = [
    "ASC bus",
    "ASC air",
    "ASC rail",
    "Time car",
    "Time bus",
    "Time air",
    "Time rail",
    "Access time",
    "Cost",
    "Wi-fi",
    "Food",
]
BUS_RAIL_NESTS = np.array(["car", "bus_rail", "air", "bus_rail"], dtype=object)
GROUND_NESTS = np.array(["ground", "ground", "air", "ground"], dtype=object)


def apollo_csv_path() -> Path:
    """Return the packaged Apollo CSV path."""

    return Path(resources.files("pdmuse.datasets").joinpath("apollo_modeChoiceData.csv"))


def apollo_dictionary_path() -> Path:
    """Return the packaged Apollo data-dictionary PDF path."""

    return Path(resources.files("pdmuse.datasets").joinpath("apollo_modeChoiceData_dictionary.pdf"))


def load_apollo_mode_choice(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the Apollo mode-choice CSV.

    Parameters
    ----------
    path:
        Optional external CSV path. If omitted, the packaged Apollo sample is used.
    """

    csv_path = Path(path) if path is not None else apollo_csv_path()
    return pd.read_csv(csv_path, na_values=["NA"])


def apollo_sp_choice_arrays(
    path: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return the manuscript's stated-preference feature arrays.

    The returned tuple is ``(X, y, availability, sp_frame)``. ``X`` has shape
    ``(7000, 4, 11)`` for the packaged data, ``y`` is zero-based, and the four
    alternatives are ordered as car, bus, air, rail.
    """

    frame = load_apollo_mode_choice(path)
    sp = frame.loc[frame["SP"] == 1].reset_index(drop=True)
    n_rows = len(sp)
    X = np.zeros((n_rows, 4, len(FEATURES)), dtype=float)
    availability = np.zeros((n_rows, 4), dtype=bool)

    for j, alternative in enumerate(ALTERNATIVES):
        availability[:, j] = sp[f"av_{alternative}"].astype(bool)

    X[:, 1, 0] = 1.0
    X[:, 2, 1] = 1.0
    X[:, 3, 2] = 1.0

    X[:, 0, 3] = sp["time_car"]
    X[:, 1, 4] = sp["time_bus"]
    X[:, 2, 5] = sp["time_air"]
    X[:, 3, 6] = sp["time_rail"]

    X[:, 1, 7] = sp["access_bus"]
    X[:, 2, 7] = sp["access_air"]
    X[:, 3, 7] = sp["access_rail"]

    X[:, 0, 8] = sp["cost_car"]
    X[:, 1, 8] = sp["cost_bus"]
    X[:, 2, 8] = sp["cost_air"]
    X[:, 3, 8] = sp["cost_rail"]

    X[:, 2, 9] = sp["service_air"] == 2
    X[:, 3, 9] = sp["service_rail"] == 2
    X[:, 2, 10] = sp["service_air"] == 3
    X[:, 3, 10] = sp["service_rail"] == 3

    X[~availability] = 0.0
    y = sp["choice"].astype(int).to_numpy() - 1
    return X, y, availability, sp
