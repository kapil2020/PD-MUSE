import numpy as np

from pdmuse import PDMuseMNL, apollo_sp_choice_arrays

EXPECTED_MNL_COEF = np.array(
    [
        0.06241069,
        0.23827658,
        -1.48136974,
        -0.01160219,
        -0.01736846,
        -0.01948337,
        -0.00636493,
        -0.02319268,
        -0.05875594,
        0.93755547,
        0.40955973,
    ]
)


def test_apollo_mnl_matches_manuscript_table():
    X, y, availability, _ = apollo_sp_choice_arrays()
    model = PDMuseMNL(tol=1e-8, max_iter=100).fit(X, y, availability=availability)

    assert model.converged_
    assert np.isclose(model.log_likelihood_, -5598.900604665886, atol=1e-6)
    assert np.allclose(model.coef_, EXPECTED_MNL_COEF, atol=5e-7)
    assert np.isclose(model.dual_objective_, -model.log_likelihood_, atol=1e-8)


def test_apollo_mnl_kkt_residuals_are_small():
    X, y, availability, _ = apollo_sp_choice_arrays()
    model = PDMuseMNL(tol=1e-8, max_iter=100).fit(X, y, availability=availability)
    residuals = model.kkt_residuals()

    assert residuals["stationarity_max_abs"] < 1e-12
    assert residuals["normalization_max_abs"] < 1e-12
    assert residuals["moment_max_abs"] < 1e-6
