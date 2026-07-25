# AutoGluon Comparison

## Contents

- [When to run](#when-to-run)
- [Fair comparison](#fair-comparison)
- [Final evaluation](#final-evaluation)
- [Deployment comparison](#deployment-comparison)
- [Required records](#required-records)

## When to run

Run only after explicit opt-in. Explain that AutoGluon adds a large optional
dependency set and extra runtime.

Skip or reframe when:

- the task is true forecasting (use an appropriate time-series benchmark
  rather than the tabular API);
- anomaly labels do not exist;
- data is too small for a meaningful comparison;
- the installed version cannot honor the required split/metric semantics.

## Fair comparison

- Use identical development and final/outer evaluation boundaries.
- Give both approaches the same target, exclusions, metric implementation and
  available-feature contract.
- Pass training data to `TabularPredictor.fit` and explicit validation through
  `tuning_data` or the version-equivalent mechanism.
- Keep holdout/external/active outer-fold targets out of tuning, ensembling and
  early stopping.
- Record AutoGluon version, preset, time limit, included/excluded families and
  hardware.

Use `medium_quality` as a bounded benchmark only when it fits the user's
budget. Do not imply it is AutoGluon's maximum achievable quality.

## Final evaluation

Compare permitted development evidence first. Finalize/refit both candidates
where valid, then score them under the same declared evaluation design. For
nested CV this means rerunning each complete AutoML selection inside every
outer training fold; skip the comparison when that cost is unjustified.

Do not silently replace the transparent pipeline. If the user chooses based on
holdout, label that set a benchmark-selection set and require future/external
data for an unbiased selected-model estimate.

Describe observed gaps as small/moderate/large only as plain-language bands;
do not call them confidence intervals.

## Deployment comparison

Measure both candidates in the same environment:

- serialized artifact/directory size;
- warm median and p95 latency at batch size 1 and a representative batch;
- cold start and peak memory when measurable;
- runtime/dependency/native-library requirements.

Record hardware, versions, warm-up, repetitions, sample and batch size. Label
unmeasured dimensions. Do not publish universal latency/memory multipliers or
claim serverless/edge compatibility without checking the actual target.

## Required records

Populate `metrics.json.autogluon` with:

- `attempted`, `reason`;
- `version`, `preset`, `time_limit_seconds`;
- development and declared final/outer metrics;
- direction-aware gap versus the main pipeline;
- execution mode/task ID;
- measured deployment properties and context.

Save the predictor directory under `artefacts/autogluon_predictor/` and document
trusted loading/inference requirements without replacing `model.joblib`.
