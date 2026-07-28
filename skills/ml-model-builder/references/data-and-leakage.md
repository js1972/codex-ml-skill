# Data Contracts, Splitting, and Leakage

## Contents

- [Data contract](#data-contract)
- [Target and prediction moment](#target-and-prediction-moment)
- [Cohort and label observation](#cohort-and-label-observation)
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
- source population, eligibility, cohort dates and sampling mechanism;
- label-observation mechanism and inclusion probabilities or weights;
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

## Cohort and label observation

Define the modeling cohort before splitting. Record the source population,
index/as-of date, eligibility and exclusion rules, observation window, outcome
ascertainment, and every case-control, negative-sampling or review-selection
stage.

- Do not treat an unreviewed, unobserved or not-yet-mature label as negative.
- Describe what makes a label observable when review, treatment or prior policy
  selects labeled rows. Use propensity weighting only with justified
  exchangeability and positivity; otherwise report the unidentified bias.
- When constructing a sample, split the eligible population first and apply
  negative sampling only within training folds. If the source is already
  sampled, preserve its design and obtain an unsampled, representative
  evaluation and calibration population whenever population probabilities,
  precision or workload estimates matter.
- Record sampling probabilities and use design-valid training/evaluation
  weights when needed. State the assumptions; weighting does not repair labels
  missing not at random.
- Correct case-control prevalence before interpreting calibrated
  probabilities, predictive values or thresholds. Report both weighted
  population estimates and unweighted sample support when weights are used.
- Preserve group, time and source structure during sampling, weighting and
  evaluation. Check weight concentration and effective sample size.

When review, treatment or a prior policy controls which labels become
observable, define a label-acquisition and policy-evaluation contract before
training or making performance claims. Record:

- the action and label-observation grain and whether the action can change the
  outcome rather than merely reveal it;
- the acquisition policy and version for each period, including randomized
  audit/exploration allocation and logged selection probabilities at the row,
  entity or whole-batch policy grain as applicable;
- the estimand: performance on historically labeled cases, the full eligible
  population, or the queue produced by a specified policy;
- support/positivity and exchangeability assumptions, weight concentration and
  effective sample size;
- how candidate and incumbent outcomes will be obtained on comparable eligible
  populations for evaluation and calibration.

A deterministic historical policy has no support where its selection
probability is zero. Row-level propensity weighting cannot repair that, and it
may also be insufficient for a batch-relative top-k policy whose actions depend
on the other eligible rows. Require an independently labeled representative
sample, randomized exploration, or a prospective randomized/interleaved policy
comparison when those regions or policies matter. Silent prospective scoring
does not create validation labels when only reviewed or treated cases reveal
the outcome. If valid label acquisition is unavailable, scope metrics and
calibration claims to the observed support and report population or alternative
policy performance as unidentified.

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

For future-event classification or regression that will score later rows from
known entities, use a declared `known_entity_temporal` overlap policy rather
than forcing a cold-start group split. The same entity may appear in an earlier
training period and a later evaluation period only when:

- known-entity recurrence is part of the deployment contract;
- partitions remain strictly chronological and duplicate/source-event rows do
  not cross them;
- every feature and historical aggregate is reconstructed as of the fold's
  scoring origin, and every training label was observable by that origin;
- target encoders, preprocessing, permutation and resampling preserve the same
  causal time boundary;
- the manifest records the intentional entity overlap, time direction and
  support, and results report known-entity and cold-start/new-entity slices
  separately when both occur in production.

This policy does not permit random entity overlap, future-informed histories,
or splitting rows from the same contemporaneous action/source event. Use a
group-disjoint policy when deployment is for unseen entities or when safe
as-of reconstruction is unavailable.

Do not select a split solely from row count. Check that every evaluation fold
has enough target events, horizon coverage and representative entities. For
small grouped or rare-outcome datasets, prefer repeated/nested outer
cross-validation when a separate holdout would be too small to support the
claimed metric. State that outer CV is an internal generalization estimate and
still requires future/external validation for high-stakes deployment.

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
  encoding for high cardinality. Fit encoders inside folds. For target
  encoding, generate training values by cross-fitting, ordered encoding or a
  smoothed leave-one-out scheme, then fit the validation/inference mapping on
  the corresponding training fold only. Preserve group and time boundaries;
  leave-one-row-out encoding is not safe when related rows or future outcomes
  remain in the encoding pool.
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
- sampling-weight and label-observation semantics when training data are not a
  representative cohort;
- behavior for extra/missing columns and unseen categories;
- model/dependency version and trusted-artifact warning.

Make `infer.py` fail with actionable messages for incompatible input. Do not
silently coerce ambiguous dates, units or identifiers.
