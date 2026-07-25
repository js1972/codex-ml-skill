# Evaluation, Explainability, and Production Readiness

## Contents

- [Honest evaluation](#honest-evaluation)
- [Uncertainty and practical value](#uncertainty-and-practical-value)
- [Error and subgroup analysis](#error-and-subgroup-analysis)
- [Explainability](#explainability)
- [Inference testing](#inference-testing)
- [Security and dependencies](#security-and-dependencies)
- [Monitoring and retraining](#monitoring-and-retraining)

## Honest evaluation

- Select model family, hyperparameters, features, threshold, calibration and
  post-processing without holdout targets.
- Refit the frozen pipeline on train+validation when methodologically valid.
- Evaluate once on holdout using the same metric implementation and data
  contract.
- Report validation and holdout separately.
- If holdout influences a later choice, stop calling it an unbiased final test.

Never compare a holdout result with a benchmark from different rows, horizon,
label definition or metric implementation.

## Uncertainty and practical value

Report more than a point estimate:

- fold variation for cross-validation;
- bootstrap confidence intervals when rows are independent;
- group/block/bootstrap or backtest variation when dependence exists;
- event counts and effective sample size;
- practical improvement versus baseline and operational tolerance.

Do not use “within margin of error” without estimating uncertainty. A
statistically detectable gain can still be operationally irrelevant.

## Error and subgroup analysis

Inspect:

- highest-impact false positives/negatives or residuals;
- errors over time and by source/entity/product/region where permitted;
- performance on rare classes and boundary cases;
- missingness/unseen-category failures;
- calibration and threshold behavior by important segment;
- support counts and uncertainty for every slice.

Evaluate supplied sensitive attributes when they matter to the use case and
show support counts. Label statistically unstable small-group results clearly.
State fairness limitations when attributes are unavailable or labels are
biased.

## Explainability

Choose the method from the question:

- coefficients for standardized sparse linear models;
- permutation importance on validation for model-agnostic global importance;
- SHAP for supported models when local/global attribution is useful;
- partial dependence/ALE for response shape, preferring ALE under correlated
  features;
- representative error examples with enough context to diagnose them.

Explain that:

- importance is not causality;
- correlated features share or swap attribution;
- preprocessing changes feature names/meaning;
- background/reference data affects SHAP;
- explanations can be unstable across folds.

Save feature-name mappings and compute explanations on held-out validation or
an approved sample, not on sealed holdout before final evaluation.

## Inference testing

Test:

- a representative batch round-trip;
- one row and empty input;
- missing optional and required columns;
- extra columns;
- wrong dtypes/units/timezones;
- unseen categories;
- all-missing permissible fields;
- output row alignment and probability/threshold semantics;
- target and post-event columns are not required.

Compare generated predictions with the saved expected/golden sample within a
documented tolerance.

## Security and dependencies

- Warn that `joblib` and pickle-based artifacts must come from a trusted source.
- Record exact Python and library versions and platform information.
- Produce a dependency lock or fully pinned requirements for inference.
- Avoid embedding credentials or authenticated source URLs in artifacts.
- Hash data/config/model files without claiming the hash anonymizes them.
- Document native-library and CPU/GPU requirements.

## Monitoring and retraining

Provide a practical monitoring section even when deployment is out of scope:

- input schema failures;
- missingness/category/range drift;
- prediction/score distribution drift;
- calibration and outcome performance after labels mature;
- subgroup or entity mix;
- latency, memory and error rates;
- retraining trigger, cadence and owner.

Do not recommend automatic retraining solely on feature drift. Require outcome
evidence or a domain-approved policy when possible.
