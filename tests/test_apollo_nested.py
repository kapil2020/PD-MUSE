import numpy as np

from pdmuse import (
    BUS_RAIL_NESTS,
    GROUND_NESTS,
    NestedLogitMLE,
    NestedLogitTwoStage,
    PDMuseMNL,
    apollo_sp_choice_arrays,
)
from pdmuse.nested import nested_log_likelihood, nested_probabilities


def test_nested_probabilities_on_apollo_sum_to_one():
    X, y, availability, _ = apollo_sp_choice_arrays()
    model = PDMuseMNL().fit(X, y, availability=availability)
    lambdas = {"car": 1.0, "bus_rail": 0.7620432397830639, "air": 1.0}
    probabilities = nested_probabilities(
        X,
        model.coef_,
        BUS_RAIL_NESTS,
        lambdas,
        availability=availability,
    )

    assert probabilities.shape == X.shape[:2]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert nested_log_likelihood(
        X,
        y,
        model.coef_,
        BUS_RAIL_NESTS,
        lambdas,
        availability=availability,
    ) < 0.0


def test_apollo_nested_mle_matches_manuscript_tables():
    X, y, availability, _ = apollo_sp_choice_arrays()
    mnl = PDMuseMNL().fit(X, y, availability=availability)

    ground = NestedLogitMLE(target_nest="ground", max_iter=250).fit(
        X,
        y,
        nests=GROUND_NESTS,
        availability=availability,
        initial_beta=mnl.coef_,
        initial_lambda=0.9,
    )
    bus_rail = NestedLogitMLE(target_nest="bus_rail", max_iter=250).fit(
        X,
        y,
        nests=BUS_RAIL_NESTS,
        availability=availability,
        initial_beta=mnl.coef_,
        initial_lambda=0.9,
    )

    assert ground.converged_
    assert bus_rail.converged_
    assert np.isclose(ground.lambda_, 0.9262923390058238, atol=1e-6)
    assert np.isclose(ground.log_likelihood_, -5597.345512594084, atol=1e-6)
    assert np.isclose(bus_rail.lambda_, 0.7620432397830639, atol=1e-6)
    assert np.isclose(bus_rail.log_likelihood_, -5586.144763510514, atol=1e-6)


def test_apollo_two_stage_pdmuse_matches_table_two_lambda_and_coefficients():
    X, y, availability, _ = apollo_sp_choice_arrays()
    mnl = PDMuseMNL().fit(X, y, availability=availability)
    two_stage = NestedLogitTwoStage(target_nest="bus_rail", max_iter=200).fit(
        X,
        y,
        nests=BUS_RAIL_NESTS,
        availability=availability,
    )

    assert np.allclose(two_stage.coef_, mnl.coef_, atol=1e-8)
    assert np.isclose(two_stage.lambda_, 0.7107202582556533, atol=1e-8)
    assert np.isclose(round(two_stage.lambda_, 3), 0.711)
    assert np.isclose(two_stage.log_likelihood_, -5625.622992840676, atol=1e-6)
