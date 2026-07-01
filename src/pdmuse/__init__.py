"""PD-MUSE: primal-dual maximum-entropy choice modelling."""

from ._version import __version__
from .apollo import (
    ALTERNATIVES,
    BUS_RAIL_NESTS,
    FEATURES,
    GROUND_NESTS,
    apollo_csv_path,
    apollo_dictionary_path,
    apollo_sp_choice_arrays,
    load_apollo_mode_choice,
)
from .data import ChoiceDataset
from .generalized import (
    InverseProductDifferentiationLogit,
    generalized_entropy,
    inverse_share_regression_matrix,
    mnl_market_shares,
)
from .metrics import accuracy, aic, bic, compare_models, log_loss
from .mnl import MultinomialLogit, PDMuseMNL
from .nested import NestedLogitMLE, NestedLogitTwoStage, nested_log_likelihood, nested_probabilities

__all__ = [
    "__version__",
    "ALTERNATIVES",
    "FEATURES",
    "BUS_RAIL_NESTS",
    "GROUND_NESTS",
    "ChoiceDataset",
    "PDMuseMNL",
    "MultinomialLogit",
    "NestedLogitTwoStage",
    "NestedLogitMLE",
    "InverseProductDifferentiationLogit",
    "accuracy",
    "aic",
    "bic",
    "compare_models",
    "apollo_csv_path",
    "apollo_dictionary_path",
    "apollo_sp_choice_arrays",
    "generalized_entropy",
    "inverse_share_regression_matrix",
    "load_apollo_mode_choice",
    "log_loss",
    "mnl_market_shares",
    "nested_log_likelihood",
    "nested_probabilities",
]
