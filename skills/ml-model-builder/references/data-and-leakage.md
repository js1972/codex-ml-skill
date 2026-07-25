# Data Contracts, Splitting, and Leakage

## Contents

- [Data contract](#data-contract)
- [Target and prediction moment](#target-and-prediction-moment)
- [Split decision](#split-decision)
- [Leakage audit](#leakage-audit)
- [Preprocessing decisions](#preprocessing-decisions)
- [Distribution checks](#distribution-checks)
- [Inference contract](#inference-contract)

## Data contract

Record:

- source locations and immutable fingerprints;
- row grain and primary/entity keys;
- decision/action grain when it differs from a source row;
- schema, units, timezone and semantic type;
- target definition and label-generation timestamp;
- prediction/scoring moment;
- feature availability and update cadence;
- join/concat rules and expected cardinality;
- sensitive attributes and prohibited uses;
- partition rule and random seed.

Do not concatenate matching schemas automatically. Source membership can encode
site, period, customer or collection-process differences. Confirm semantics and
assess `dataset_source` as both a drift indicator and possible leakage proxy.

## Target and prediction moment

Derive a target only after confirming its business meaning. Exclude every
source column used to calculate the target unless it is genuinely available at
prediction time and does not reveal the answer.

Ask:

- When exactly is the prediction made?
- When does the label become known?
- Which features exist, with what delay, at that moment?
- Is the target actionable and stable over time?
- Are censored or not-yet-matured labels present?

For outcome data, account for label delay and right censoring. A recent negative
may simply not have had time to become positive.

For delayed binary outcomes, place a maturity gap/purge between the latest
feature timestamp and each validation cutoff that is at least the outcome
window plus known ingestion delay. Apply this at every temporal fold, not only
the final holdout.

## Split decision

Choose splits in this order:

1. If predicting future periods, use chronological or rolling-origin splits.
2. If rows share an entity, household, account, device, patient, document, or
   source event, keep related rows in one partition unless the deployment goal
   explicitly predicts future rows for known entities.
3. If both time and groups matter, use grouped-temporal evaluation that
   reproduces the deployment scenario.
4. Otherwise use stratified random splits for classification and random splits
   for regression.

Do not select a split solely from row count. Check that every evaluation fold
has enough target events, horizon coverage and representative entities. Use
repeated or nested cross-validation when a single split would be unstable.

Persist split assignments or the deterministic rule. Never allow a target-aware
decision to inspect held-out targets.

It is acceptable to verify aggregate event support once while constructing the
split. Record the check, freeze the rule, and then seal holdout labels; do not
revisit those counts to choose features, thresholds, or models.

## Leakage audit

Check:

- direct copies, encodings or arithmetic derivatives of the target;
- post-event statuses, resolutions, actual dates and final outcomes;
- labels aggregated into historical or group statistics without fold isolation;
- entity overlap, duplicate events and near-duplicate text across partitions;
- target encoding, imputation, scaling, feature selection or SMOTE fit outside
  training folds;
- time-window features that include the current/future row;
- source-system or data-quality fields created after the prediction moment;
- proxy features that encode protected attributes or operational decisions;
- labels generated using future observation windows;
- target leakage through filenames, paths, row order or manually curated data.

Association strength is only a screening clue. Leakage can be nonlinear,
categorical or semantically obvious with weak marginal correlation.

## Preprocessing decisions

Make transformations conditional on task and model:

- **Missingness:** preserve informative missing indicators when appropriate.
  Do not drop a column only because missingness exceeds a universal threshold.
- **Duplicates:** determine whether they are accidental copies, legitimate
  repeated events, weighting, or leakage across partitions.
- **Low variance:** retain rare but important indicators when supported by the
  domain or target.
- **Correlated features:** multicollinearity mainly affects coefficient
  stability; tree models do not require automatic removal. Prefer regularized
  models or grouped interpretation.
- **Skew:** transform only when it benefits the candidate model or
  interpretation. Trees rarely require it.
- **Outliers:** distinguish errors, rare valid events and the target signal.
  Prefer robust models/losses before deletion or winsorization.
- **Categoricals:** handle unseen values. Use one-hot for manageable
  cardinality; consider native categorical models, hashing or frequency
  encoding for high cardinality. Fit encoders inside folds.
- **Resampling:** prefer appropriate metrics, class weights and threshold
  selection before SMOTE. If used, choose SMOTE/SMOTENC appropriately and run
  it only inside training folds. Never use it for temporal data without a
  defensible temporal design.
- **Text:** keep TF-IDF and vocabulary fitting inside folds; deduplicate or
  group near-identical documents before splitting.

## Distribution checks

After splitting, compare feature-only distributions across train, validation
and holdout without inspecting holdout labels. Report:

- unseen categories and range violations;
- population/entity/source mix;
- time coverage and regime changes;
- missingness changes;
- numeric drift using effect sizes and robust plots.

Use these checks to explain generalization risk, not to retune against holdout.

## Inference contract

Save a machine-readable contract containing:

- required/optional columns and dtypes;
- allowed missingness and categories;
- units and timezone;
- feature order where required;
- prediction moment and excluded post-event fields;
- target column exclusion;
- identifier passthrough for row alignment/output, kept separate from model
  features;
- current as-of scoring-population and eligibility rules when producing an
  operational queue or outreach list;
- behavior for extra/missing columns and unseen categories;
- model/dependency version and trusted-artifact warning.

Make `infer.py` fail with actionable messages for incompatible input. Do not
silently coerce ambiguous dates, units or identifiers.
