# Python Package For PD-MUSE

This study is accompanied by a Python package, `pd-muse`, that implements the
PD-MUSE estimators and reproduces the empirical results on the Apollo
mode-choice data. The package is designed to make the primal-dual formulation
usable by other researchers without requiring them to rebuild the numerical
details from the manuscript. It provides a direct implementation of the
multinomial-logit PD-MUSE dual, nested-logit comparison tools, two-stage
PD-MUSE recovery for the bus-rail nest, generalized-entropy utilities, and
model-comparison diagnostics.

The package follows the data structure of standard discrete-choice models. A
choice sample is represented by an array of alternative-specific attributes, a
chosen alternative for each observation, and an availability matrix. This design
keeps the implementation close to the notation of the paper: the array
`X[n, j, k]` stores the value of attribute `k` for alternative `j` in choice
situation `n`, and the fitted probability array stores the corresponding
primal choice probabilities. For the Apollo application, the package includes a
data loader that constructs the exact stated-preference specification used in
the empirical section: car, bus, air, and rail alternatives; mode-specific time
coefficients; a generic access-time coefficient; a generic cost coefficient;
alternative-specific constants; and service-quality indicators for wi-fi and
food.

The main estimator is `PDMuseMNL`. It solves the PD-MUSE dual objective

```text
sum_n log sum_j exp(beta' x_nj) - beta' t,
```

where `t` is the observed total of the chosen alternative attributes. This is
the negative of the multinomial-logit log-likelihood. Consequently, the fitted
dual variables are the utility coefficients and the fitted primal probabilities
are the multinomial-logit probabilities evaluated at those coefficients. The
package reports the log-likelihood, the dual objective, coefficient tables,
standard errors, and KKT residuals. On the Apollo stated-preference sample, the
duality check gives a dual objective plus log-likelihood of approximately
`-9.1e-13`, confirming the analytical equivalence to numerical precision.

The package also implements the nested-logit results used for comparison in
the manuscript. `NestedLogitMLE` fits a nested-logit likelihood benchmark for a
specified non-singleton nest. On the Apollo data, it reproduces the ground-nest
and bus-rail benchmark values reported in the paper. The ground nest gives a
dissimilarity parameter of approximately `0.926` and a log-likelihood of
approximately `-5597.35`. The bus-rail nest gives a dissimilarity parameter of
approximately `0.762` and a log-likelihood of approximately `-5586.14`, an
improvement over the baseline multinomial logit.

For the two-stage PD-MUSE nested result, the package provides
`NestedLogitTwoStage`. The first stage fits the full-sample PD-MUSE MNL and
uses its coefficient vector as the behavioral taste vector. The second stage
recovers the bus-rail dissimilarity parameter through an upper-stage scalar
problem. On the Apollo data, this gives a bus-rail dissimilarity of
approximately `0.711`, matching the two-stage PD-MUSE table to rounding. The
coefficient vector is the same as the PD-MUSE MNL coefficient vector, which is
the comparison reported in the manuscript.

The package is intended to support reproducible empirical work. It includes the
Apollo CSV file and data dictionary, a script that reproduces the Apollo
results, and a test suite that checks the main numerical values against the
manuscript tables. Running

```text
python examples/apollo_mode_choice.py
```

prints the PD-MUSE MNL estimates, KKT diagnostics, two-stage PD-MUSE bus-rail
result, and nested-logit likelihood benchmarks. The same calculations are used
in the automated tests, so future changes to the package can be checked against
the empirical results in the paper.

The practical benefit of the package is that PD-MUSE can be used as a reusable
choice-model component. Researchers can estimate the model, recover fitted
probabilities, evaluate optimality conditions, and compare the result with
nested-logit benchmarks using a small number of commands. This supports the
main goal of the PD-MUSE framework: to express choice modelling in a form that
is both statistically interpretable and convenient for embedding in larger
transportation and network-optimization problems.
