# Evaluation, Explainability, and Production Readiness

## Contents

- [Honest evaluation](#honest-evaluation)
- [Evaluation design choices](#evaluation-design-choices)
- [Improvement evidence lineage](#improvement-evidence-lineage)
- [Incumbent comparison](#incumbent-comparison)
- [Uncertainty and practical value](#uncertainty-and-practical-value)
- [Error and subgroup analysis](#error-and-subgroup-analysis)
- [Explainability](#explainability)
- [Inference testing](#inference-testing)
- [Security and dependencies](#security-and-dependencies)
- [Monitoring and retraining](#monitoring-and-retraining)

## Honest evaluation

- Select model family, hyperparameters, features, threshold, calibration and
  post-processing without holdout/external targets or the active outer fold.
- For holdout/external designs, refit the frozen pipeline on permitted
  development data when methodologically valid, then evaluate once on the
  declared final population with the same metric implementation and contract.
- For nested CV, use outer folds to estimate the complete selection procedure.
  After locking it, run that procedure on all permitted data for deployment;
  do not call this specific full-data refit independently tested.
- Report development/inner evidence separately from final/outer evidence.
- If final evidence influences a later choice, stop calling it unbiased.
- Compare every approved execution track on identical evaluation row IDs,
  folds, weights, target semantics, and metric code. Different internal
  preprocessing/build mechanics are expected; different evidence is not.

Never compare a final result with a benchmark from different rows, horizon,
label definition or metric implementation.

## Evaluation design choices

Declare one design before target-aware modeling:

| Design | Use when | Required interpretation |
|---|---|---|
| `holdout` | Enough independent groups/events remain after development splitting | One untouched internal test, opened once |
| `nested_cv` | Small/grouped/rare data cannot support a useful fixed holdout | Outer-fold estimate includes model selection inside each outer training fold; no independent final test |
| `external_test` | A genuinely separate site, cohort, period, or source exists | Report domain differences and fingerprint the external set |
| `prospective_validation` | Deployment conditions or labels require future silent testing | Do not claim validated performance until outcomes mature |

Never tune once globally and call ordinary cross-validation “nested.” Every
feature, preprocessing, family, hyperparameter, calibration, and threshold
choice must occur inside each outer training partition. Preserve groups/time in
both inner and outer resampling.

If data are too small for any design to estimate the critical harm/rare-class
metric, report **evaluation insufficient** rather than manufacturing a score.

For selectively observed outcomes, state whether each result estimates
performance on historically labeled support, the full eligible population or
the enacted review/treatment queue. A prospective cohort is not sufficient by
itself when the incumbent or candidate policy still determines which outcomes
become observable; predeclare the representative audit, randomized allocation
or other identified label-acquisition design.

## Improvement evidence lineage

Preserve parent and ancestor final evidence through append-only references in
each improvement run. Keep these references separate from the current run's
own final-evaluation exposure. For every prior result that was inspected or
used, record its run ID, immutable artifact hashes, evaluation-population
fingerprint, first-opened time and purpose, values viewed, and the descendant
decisions it influenced.

Do not mutate an immutable parent when its result later influences a
descendant. Append that influence event to the descendant's lineage instead.
Do not re-seal the same evaluation population under a new run or filename. If
parent evidence influenced the hypothesis, feature set, roster, calibration,
threshold or deployment choice, treat it as benchmark-selection evidence for
the descendant and require untouched future/external evidence for a new
unbiased claim.

## Incumbent comparison

Ask whether an incumbent model, rule or verified historical benchmark exists
and record the answer. Compare an available incumbent only when its version,
prediction contract, cohort and metric implementation can be aligned with the
candidate. If no valid incumbent exists, say so and use the declared naive and
fixed baselines; do not block completion or manufacture one.

When candidate and incumbent policies select different rows and selection
controls label observation, comparing outcomes only among each policy's
observed selections is not a fair same-population comparison. Use independent
representative labels, a randomized/interleaved experiment, or a justified
off-policy design with support, exchangeability and policy-grain propensities.
Otherwise report the comparison as unidentified.

## Uncertainty and practical value

Report more than a point estimate:

- fold variation for cross-validation;
- bootstrap confidence intervals when rows are independent;
- group/block/bootstrap or backtest variation when dependence exists;
- event counts and effective sample size;
- practical improvement versus baseline and operational tolerance.

Do not use “within margin of error” without estimating uncertainty. A
statistically detectable gain can still be operationally irrelevant.

For fixed-capacity decisions, retain complete eligible scheduling batches in
each resample or evaluation fold. Recompute scores and rerun the entire queue
policy—including eligibility, entity caps/deduplication, tie-breaking and
top-k selection—before recalculating the metric. Never bootstrap only the
already selected rows or their scores.

Preserve both repeated-entity and serial time dependence using an appropriate
declared design, such as paired scheduling periods, entity clusters, temporal
blocks or a justified multiway procedure. Candidate-incumbent intervals should
use paired differences on the same resampled batches or the experiment's
assignment units. Report the number of independent scheduling periods and
events; weighting uncertainty and effective sample size do not remove
unidentified selective-label bias.

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

Run the real unified inference entry point for every retained backend on a
nonempty temporary fixture. Inspect the actual output, then remove fixtures and
prediction files after recording the checks in `validation.json`. Do not
substitute a schema-only check, mocked predictor, or copied expected file.
Verify finite values, row/identifier alignment, output column semantics, and
task constraints such as probability bounds, forecast horizons, or
anomaly-score direction.

For every retained backend, declare and execute the four required
`validation.json` case kinds from `artifacts.md`:

- `representative`: at least two rows, executed at least twice;
- `single_row`: one row, executed at least twice;
- `empty_input`: success with zero rows and the exact output schema;
- `missing_required_column`: controlled non-zero exit with an actionable
  message.

Every case must dispatch its actual declared backend. Require representative
and one-row outputs to be byte-identical across repeats. Verify finite
prediction/probability values, probability bounds, output row count, and input
identifier order. Do not substitute a schema-only check, mocked predictor, or
copied expected file.

Run each inference attempt in a fresh subprocess so validation exercises a
cold import and model load rather than the warm training process. For
AutoGluon and other native-library backends, set bounded native thread
environment variables before importing numerical/model libraries. A warm
prediction is useful for latency measurement but does not satisfy cold-start
handoff validation.

Also test optional-column omission, the declared extra-column policy, wrong
dtypes/units/timezones, unseen categories, all-missing permissible fields, and
that target/post-event columns are not required when applicable. For capacity
workflows, test `score_rows` independently from whole-batch `select_queue`,
including empty, sub-capacity, tied, and ineligible batches.

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
