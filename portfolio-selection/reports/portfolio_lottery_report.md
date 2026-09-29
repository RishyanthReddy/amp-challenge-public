# Exact finite-population draw probabilities

Hypergeometric values are exact for a uniform draw without replacement. The seeded 10,000-draw Monte Carlo estimates are compared with the exact values; each error must be within four binomial standard errors (or one simulation step). These are probabilities over model-threshold labels, not assay-hit probabilities. The threshold is predicted AMP probability ≥ 0.70 plus the synthesis-filter flag.

| Portfolio | N | Threshold-positive | Draw k | Expected positives | Exact P(≥1) | MC P(≥1) | Error | Exact P(≥5) | MC P(≥5) | Exact P(≥10) | MC P(≥10) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top 50 | 50 | 50 | 25 | 25.000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Top 100 | 100 | 100 | 25 | 25.000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Convergence check by event threshold

| Portfolio | Event | Exact probability | Monte Carlo probability | Absolute error | Four standard errors | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Top 50 | ≥1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
| Top 50 | ≥5 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
| Top 50 | ≥10 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
| Top 100 | ≥1 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
| Top 100 | ≥5 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
| Top 100 | ≥10 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | PASS |
