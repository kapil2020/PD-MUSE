# Theory Notes

## Multinomial Logit

PD-MUSE starts from the entropy primal:

```text
maximize    - sum_n sum_j p_nj log p_nj
subject to  sum_j p_nj = 1
            sum_n sum_j p_nj x_nj = sum_n x_n,y[n]
            p_nj >= 0
```

The KKT conditions give the multinomial-logit probability:

```text
p_nj = exp(beta' x_nj) / sum_k exp(beta' x_nk)
```

The dual objective is:

```text
g(beta) = sum_n log sum_j exp(beta' x_nj) - beta' t.
```

The MNL log-likelihood is `-g(beta)`. Therefore, the PD-MUSE dual solution and
the MNL maximum-likelihood solution are identical for the same feature
specification. The package verifies this identity on the Apollo
stated-preference sample.

## Nested Logit

The nested-logit entropy decomposes into an across-nest entropy and a
within-nest conditional entropy. In the package, the Apollo bus-rail
two-stage PD-MUSE workflow is implemented as:

1. Fit the full-sample PD-MUSE MNL to recover the taste coefficients.
2. Fit an upper-stage scalar model to recover the bus-rail dissimilarity.
3. Evaluate the resulting nested-logit probabilities using the recovered
   dissimilarity and the PD-MUSE taste coefficients.

The package also provides `NestedLogitMLE` to reproduce the maximum-likelihood
ground-nest and bus-rail benchmark columns from the manuscript.

## Generalized Entropy And IPDL

For the inverse-product differentiation logit, generalized entropy is a convex
combination of alternative-level entropy and group-level entropies. The
inverse-share relationship is linear after differencing against a reference
alternative:

```text
log(q_j / q_r)
  = beta' (x_j - x_r)
    - sum_d mu_d * [log(q_Gd(j) / q_Gd(r)) - log(q_j / q_r)].
```

`InverseProductDifferentiationLogit.fit_inverse_shares` solves this linear
system for market-share data. The package tests the grouping-free anchor using
Apollo feature arrays and the fitted Apollo MNL probabilities.
