# PD-MUSE

PD-MUSE is a Python package for primal-dual maximum-entropy choice modelling.
It is built around the mode-choice formulation in the PD-MUSE manuscript and is
validated on the Apollo mode-choice data included with this repository.

The central result implemented in the package is the multinomial-logit
equivalence: the PD-MUSE dual objective is exactly the negative MNL
log-likelihood. The fitted dual variables are the utility coefficients, and the
primal variables are the fitted choice probabilities.

## Why This Package Is Useful

- **One estimation and prediction object**: the package returns both behavioral
  coefficients and fitted probabilities from the same primal-dual model.
- **Manuscript reproducibility**: the Apollo stated-preference benchmark is
  packaged and tested, including MNL, nested MLE benchmarks, and the bus-rail
  two-stage PD-MUSE result.
- **Numerical diagnostics**: `PDMuseMNL` reports KKT residuals, dual objective
  equality, observed-information covariance, sandwich covariance, log loss, AIC,
  and BIC.
- **Nested-logit comparison**: the package includes the PD-MUSE two-stage
  bus-rail recovery and nested-logit likelihood benchmarks to compare against
  baseline MNL.
- **Generalized entropy tools**: inverse-product differentiation logit helpers
  are included for generalized-entropy and inverse-share workflows.
- **GitHub and PyPI ready**: packaging metadata, license, citation file,
  manifest, CI workflow, tests, examples, and build checks are included.

## Installation

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

From a built wheel:

```bash
python -m pip install dist/pd_muse-0.1.0-py3-none-any.whl
```

After publication to PyPI:

```bash
python -m pip install pd-muse
```

## Quick Start With Apollo Data

The package includes the Apollo mode-choice CSV and data dictionary under
`src/pdmuse/datasets/`.

```python
from pdmuse import PDMuseMNL, apollo_sp_choice_arrays

X, y, availability, sp = apollo_sp_choice_arrays()

model = PDMuseMNL(tol=1e-8, max_iter=100).fit(
    X,
    y,
    availability=availability,
)

print(model.summary())
print(model.log_likelihood_)
print(model.kkt_residuals())
```

## Reproduce The Apollo Tables

Run:

```bash
python examples/apollo_mode_choice.py
```

Expected headline results on the 7,000 stated-preference observations:

| Model | Key result |
| --- | --- |
| PD-MUSE MNL | log-likelihood `-5598.900605` |
| PD-MUSE dual check | dual + log-likelihood `-9.095e-13` |
| PD-MUSE two-stage bus-rail | lambda `0.710720`, which rounds to `0.711` |
| Nested MLE ground nest | lambda `0.926292`, log-likelihood `-5597.345513` |
| Nested MLE bus-rail nest | lambda `0.762043`, log-likelihood `-5586.144764` |

The MNL coefficients match the manuscript table to rounding:

```text
ASC bus       0.062
ASC air       0.238
ASC rail     -1.481
Time car     -0.012
Time bus     -0.017
Time air     -0.019
Time rail    -0.006
Access time  -0.023
Cost         -0.059
Wi-fi         0.938
Food          0.410
```

## Package Features

### `PDMuseMNL`

Fits the PD-MUSE multinomial logit dual:

```text
min_beta sum_n log sum_j exp(beta' x_nj) - beta' t
```

The class provides:

- `coef_`
- `probabilities_`
- `log_likelihood_`
- `dual_objective_`
- `summary()`
- `kkt_residuals()`
- `covariance(kind="observed")`
- `covariance(kind="sandwich")`

### `NestedLogitTwoStage`

Reproduces the manuscript's two-stage PD-MUSE bus-rail result. The first stage
fits the full-sample PD-MUSE MNL coefficients. The second stage recovers the
bus-rail dissimilarity parameter.

```python
from pdmuse import BUS_RAIL_NESTS, NestedLogitTwoStage, apollo_sp_choice_arrays

X, y, availability, _ = apollo_sp_choice_arrays()

two_stage = NestedLogitTwoStage(target_nest="bus_rail").fit(
    X,
    y,
    nests=BUS_RAIL_NESTS,
    availability=availability,
)

print(two_stage.lambda_)
print(two_stage.log_likelihood_)
```

### `NestedLogitMLE`

Fits a nested-logit likelihood benchmark for one non-singleton nest. It is used
to reproduce the Apollo ground and bus-rail nested-logit benchmark columns.

### Generalized Entropy And IPDL Tools

The package includes:

- `generalized_entropy`
- `InverseProductDifferentiationLogit`
- `inverse_share_regression_matrix`
- `mnl_market_shares`

These support generalized-entropy diagnostics and inverse-share recovery.

## Data Format For Your Own Dataset

For custom data, provide dense choice arrays:

- `X`: shape `(n_choices, n_alternatives, n_features)`
- `y`: zero-based chosen alternative index
- `availability`: boolean mask with shape `(n_choices, n_alternatives)`

You can also create a `ChoiceDataset` from a long pandas table:

```python
from pdmuse import ChoiceDataset

dataset = ChoiceDataset.from_long_dataframe(
    frame,
    choice_id_col="choice_id",
    alternative_col="mode",
    choice_col="chosen",
    feature_cols=["time", "cost", "access_time"],
    availability_col="available",
)
```

## Development And Deployment

Run checks before pushing:

```bash
python -m ruff check .
python -m pytest
python examples/apollo_mode_choice.py
python -m build
python -m twine check dist/*
```

GitHub Actions is configured in `.github/workflows/ci.yml`.

## Repository Contents

- `src/pdmuse/`: package source
- `src/pdmuse/datasets/`: Apollo sample data and dictionary
- `examples/apollo_mode_choice.py`: manuscript-table reproduction script
- `tests/`: Apollo-based regression tests
- `docs/theory.md`: model notes
- `docs/manuscript_package_section.md`: manuscript-ready package section

## Citation

If you use PD-MUSE, please cite the manuscript and the package. 
