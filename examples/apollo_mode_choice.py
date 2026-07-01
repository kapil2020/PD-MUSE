"""Reproduce the Apollo stated-preference mode-choice benchmark.

Usage:
    python examples/apollo_mode_choice.py
    python examples/apollo_mode_choice.py /path/to/apollo_modeChoiceData.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdmuse import (
    ALTERNATIVES,
    BUS_RAIL_NESTS,
    FEATURES,
    GROUND_NESTS,
    NestedLogitMLE,
    NestedLogitTwoStage,
    PDMuseMNL,
    apollo_sp_choice_arrays,
    compare_models,
)


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    X, y, availability, sp = apollo_sp_choice_arrays(csv_path)

    mnl = PDMuseMNL(tol=1e-8, max_iter=100).fit(
        X,
        y,
        availability=availability,
        feature_names=FEATURES,
        alt_names=ALTERNATIVES,
    )
    two_stage = NestedLogitTwoStage(target_nest="bus_rail", max_iter=200).fit(
        X,
        y,
        nests=BUS_RAIL_NESTS,
        availability=availability,
        feature_names=FEATURES,
        alt_names=ALTERNATIVES,
    )
    ground_mle = NestedLogitMLE(target_nest="ground", max_iter=250).fit(
        X,
        y,
        nests=GROUND_NESTS,
        availability=availability,
        feature_names=FEATURES,
        alt_names=ALTERNATIVES,
        initial_beta=mnl.coef_,
        initial_lambda=0.9,
    )
    bus_rail_mle = NestedLogitMLE(target_nest="bus_rail", max_iter=250).fit(
        X,
        y,
        nests=BUS_RAIL_NESTS,
        availability=availability,
        feature_names=FEATURES,
        alt_names=ALTERNATIVES,
        initial_beta=mnl.coef_,
        initial_lambda=0.9,
    )

    print(f"SP observations: {len(sp)}")
    print("\nPD-MUSE MNL")
    print(mnl.summary().to_string(index=False))
    print(f"Log-likelihood: {mnl.log_likelihood_:.6f}")
    print(f"Dual + log-likelihood: {mnl.dual_objective_ + mnl.log_likelihood_:.3e}")
    print(f"KKT residuals: {mnl.kkt_residuals()}")

    print("\nPD-MUSE two-stage: bus-rail nest")
    print(f"Lambda: {two_stage.lambda_:.6f}")
    print(f"Log-likelihood: {two_stage.log_likelihood_:.6f}")
    print(f"Gain over MNL: {two_stage.log_likelihood_ - mnl.log_likelihood_:.6f}")

    print("\nNested MLE benchmarks")
    print(f"Ground lambda: {ground_mle.lambda_:.6f}")
    print(f"Ground log-likelihood: {ground_mle.log_likelihood_:.6f}")
    print(f"Bus-rail lambda: {bus_rail_mle.lambda_:.6f}")
    print(f"Bus-rail log-likelihood: {bus_rail_mle.log_likelihood_:.6f}")
    print(
        compare_models([mnl, two_stage, ground_mle, bus_rail_mle], X, y, availability=availability)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
