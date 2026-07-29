# AutoGluon Track

## Contents

- [Role](#role)
- [Approval and inputs](#approval-and-inputs)
- [Fair evaluation](#fair-evaluation)
- [Execution](#execution)
- [Selection and serving](#selection-and-serving)
- [Artifacts](#artifacts)

## Role

Treat AutoGluon as an autonomous model-building backend, not as another
classical estimator or an Optuna candidate. The skill supplies an eligible raw
table and experiment boundaries; AutoGluon owns preprocessing, model
construction, tuning, bagging, stacking, and ensembling.

Do not:

- transform the data through the classical preprocessing pipeline;
- externally tune AutoGluon's models with Optuna;
- engineer features solely for AutoGluon unless the user explicitly approves a
  domain-required feature contract shared by all tracks;
- serialize AutoGluon as `model.joblib`;
- describe its internal constituent search as the skill's classical search.

“Raw” means the source-valued eligible feature columns after mandatory removal
of target sources, post-outcome fields, identifiers not intended as features,
and prohibited data. It does not permit leakage or invalid dtypes/semantics.

## Approval and inputs

Include AutoGluon in the mandatory experiment approval. Present:

- include or decline;
- task and target;
- raw eligible feature contract;
- evaluation folds and primary metric;
- preset;
- wall-time, CPU count, parallel jobs, GPU flag, memory, and disk budget;
- any family/resource restrictions;
- expected dependency size and installation impact.

Do not probe or install AutoGluon before approval. If installation is required,
obtain any additional approval required for the large dependency.

Use this track for supported tabular classification or regression. Reframe or
decline when the task requires a different AutoGluon API, the data cannot
support the evaluation design, or the installed release cannot honor the
approved split and metric semantics.

## Fair evaluation

Use the same:

- source population and target definition;
- prediction-time eligible features;
- weights and label-observation semantics;
- development/evaluation row IDs and folds;
- primary metric implementation;
- final evidence boundary.

Pass raw fold-training rows to `TabularPredictor.fit`. Supply explicit
validation/tuning rows when the API and approved design support them. Keep
holdout, external, prospective, and active outer-fold targets out of
AutoGluon model building, dynamic stacking, calibration, and early stopping.

Under nested CV, rerun the complete AutoGluon builder inside every outer
training partition. Decline the track when that cost exceeds the approved
budget rather than running an incomparable shortcut.

## Execution

Let AutoGluon manage its own pipeline. Record:

- AutoGluon and Python versions;
- preset, time limit, resource limits, and dynamic-stacking settings;
- included/excluded internal model families when constrained;
- fold IDs and row counts supplied;
- native leaderboard summary;
- failures, elapsed time, peak memory, and disk usage when measurable;
- stop reason.

Do not claim that one preset or time limit is AutoGluon's maximum achievable
quality. Report it as the approved bounded build.

## Selection and serving

Compare AutoGluon with other approved tracks first on permitted development
evidence and then under the declared final evaluation design. Use the same rows
and metric code. Measure deployment properties in the same environment when
they affect selection:

- predictor-directory size;
- warm median and p95 latency at batch size 1 and a representative batch;
- cold start and peak memory when measurable;
- runtime, native-library, CPU/GPU, and disk requirements.

Permit AutoGluon to be the predictive or operational winner. Do not silently
replace the approved default backend; record the selection decision and exact
inference command.

## Artifacts

Keep:

```text
backends/autogluon/
└── predictor/
```

Record build settings, metrics, directory hash, resource results, and inference
requirements in root `run.json`. Use root `train.py --backend autogluon` to
rebuild and root `infer.py --backend autogluon` for new rows. Test that
inference loads the native predictor and preserves row alignment.

Do not create separate AutoGluon config, metric, model-card, or report files.
Do not copy the classical model, transformed data, Optuna study, split
manifests, fixtures, or root reports into its directory.
