import numpy as np

from pdmuse import (
    InverseProductDifferentiationLogit,
    PDMuseMNL,
    apollo_sp_choice_arrays,
    generalized_entropy,
    mnl_market_shares,
)


def test_generalized_entropy_at_zero_mu_matches_apollo_mnl_entropy():
    X, y, availability, _ = apollo_sp_choice_arrays()
    model = PDMuseMNL().fit(X, y, availability=availability)
    shares = model.probabilities_[:25]
    groups = [["ground", "ground", "air", "ground"], ["car", "scheduled", "scheduled", "scheduled"]]

    got = generalized_entropy(shares, groups, [0.0, 0.0])
    expected_terms = np.zeros_like(shares)
    positive = shares > 0.0
    expected_terms[positive] = shares[positive] * np.log(shares[positive])
    expected = -np.sum(expected_terms, axis=1)

    assert np.allclose(got, expected)


def test_inverse_share_regression_recovers_apollo_mnl_anchor():
    X, y, availability, _ = apollo_sp_choice_arrays()
    model = PDMuseMNL().fit(X, y, availability=availability)
    all_available = availability.all(axis=1)
    X_market = X[all_available][:500]
    utilities = np.einsum("njk,k->nj", X_market, model.coef_)
    shares = mnl_market_shares(utilities)
    groups = [["ground", "ground", "air", "ground"], ["car", "scheduled", "scheduled", "scheduled"]]

    recovered = InverseProductDifferentiationLogit(reference=0).fit_inverse_shares(
        shares,
        X_market,
        groups,
    )

    assert np.allclose(recovered.coef_, model.coef_, atol=1e-6)
    assert np.allclose(recovered.mu_, [0.0, 0.0], atol=1e-6)
