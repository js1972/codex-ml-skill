# AutoGluon Track

## Contents

- [Role](#role)
- [Approval and inputs](#approval-and-inputs)
- [Fair evaluation](#fair-evaluation)
- [Execution](#execution)
- [Selection and serving](#selection-and-serving)
- [Deployment packaging](#deployment-packaging)
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

Under `build.training_diagnostics`, record
`fit_summary_captured`, `elapsed_seconds`, and `stop_reason` before cloning.
Keep the full native leaderboard and internal component failures in their
dedicated backend fields rather than flattening them into free text.

When the approved `parallel_jobs` is `1`, explicitly pass
`fold_fitting_strategy="sequential_local"` through the supported AutoGluon fit
configuration. Record the effective strategy and why it was selected. Do not
allow AutoGluon's default parallel runner to introduce Ray workers or
restricted system inspection into a deliberately single-job run.

Keep a canonical `internal_failures` list under the AutoGluon backend. Use an
empty list when none occurred. For each failed, skipped, or unavailable
internal component, record:

- component/model family;
- build stage;
- status;
- concrete reason;
- impact on the overall track.

An internal model failure does not make the whole AutoGluon track failed when
other constituent models complete and a valid predictor is produced. Include
the ledger in `run.json`, `report.html`, and `results.md`.

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

Test cold-start inference in a new process, not only with the already-loaded
training process. In `infer.py`, before importing NumPy, pandas, Torch,
AutoGluon, or any adapter that imports them, set:

```python
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
```

Then load the packaged predictor and run representative and single-row
inference in fresh subprocesses. Record these runtime safeguards under the
AutoGluon backend.

Permit AutoGluon to be the predictive or operational winner. Do not silently
replace the approved default backend; record the selection decision and exact
inference command.

## Deployment packaging

Treat `TabularPredictor.clone_for_deployment(path=..., model="best")` as the
standard final packaging step for a retained AutoGluon backend. The clone keeps
the winning model and dependencies required for prediction, removes unrelated
models and fit-only material, and is prediction-only.

Use this order:

1. Capture the bounded-build metrics, native leaderboard, internal failures,
   training diagnostics, selected model, and original predictor size.
2. Measure current disk use and ensure the approved disk budget can tolerate
   the original predictor and clone existing at the same time.
3. Create the clone at a temporary deployment path:

   ```python
   deployment = predictor.clone_for_deployment(
       path=str(deployment_path),
       model="best",
       return_clone=True,
   )
   ```

4. Compare `predict` and, when applicable, `predict_proba` from the original
   and clone on the same temporary representative fixture. Require identical
   labels and numerically equivalent probabilities within a recorded
   tolerance.
5. Run root `infer.py --backend autogluon` against the clone in a fresh
   subprocess with the native thread limits above.
6. Move the validated clone to `backends/autogluon/predictor` and point
   `run.json.backends.autogluon.build.predictor_path` there.
7. Remove the full training predictor only after the equivalence and
   cold-start checks pass. Retain it only when continued AutoGluon analysis was
   explicitly requested; store it at
   `backends/autogluon/training_predictor` and record the reason.
8. Record final deployment-predictor bytes and temporary peak packaging bytes.

Do not call `save_space()` on the only copy before capturing diagnostics or
before creating a validated deployment clone.

## Artifacts

Keep:

```text
backends/autogluon/
└── predictor/
```

The retained `predictor/` is the validated deployment clone. Record build
settings, native leaderboard, internal failures, clone-equivalence evidence,
directory hash, final and peak disk bytes, runtime safeguards, resource
results, and inference requirements in root `run.json`. Use root
`train.py --backend autogluon` to rebuild and root
`infer.py --backend autogluon` for new rows. Test that cold-start inference
loads the clone and preserves row alignment.

Do not create separate AutoGluon config, metric, model-card, or report files.
Do not copy the classical model, transformed data, Optuna study, split
manifests, fixtures, or root reports into its directory.
