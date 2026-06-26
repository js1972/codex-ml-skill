# Artifact Layout and Formats

Use the `artefacts/` directory for model outputs. Create it if missing.
Also write `results.md` in the project root with a concise final summary.

## Expected Files

- `artefacts/train.py`
- `artefacts/infer.py`
- `artefacts/model.joblib` (full preprocessor+model pipeline as a single joblib
  object — use joblib, not pickle; joblib handles NumPy arrays more reliably)
- `artefacts/metrics.json`
- `artefacts/config.json`
- Optional explainability output: `artefacts/shap_summary.html` — beeswarm
  summary plot of top 20 features by mean absolute SHAP value
- Optional AutoML comparison output (only when user opted in):
  `artefacts/autogluon_predictor/` — full AutoGluon predictor directory
  (not a single file). Load with
  `TabularPredictor.load('artefacts/autogluon_predictor')`.
- `results.md` (summary in project root)

## Scripts Expectations

- `train.py` should:
  - load data and apply all profiling decisions (drop columns, record
    transforms)
  - split into train/validation/holdout before any fitting
  - fit the best pipeline on train+validation combined after Optuna completes
  - evaluate on the holdout test set for final metrics
  - save the full preprocessor+model pipeline together as `model.joblib`
  - write `metrics.json` and `config.json`
- `infer.py` should:
  - load the saved `model.joblib` pipeline
  - CLI usage: `python infer.py --input data.csv --output predictions.csv`
  - apply identical preprocessing to `train.py` (handled automatically by the
    loaded pipeline)
  - output a CSV with a `prediction` column alongside the original columns

## metrics.json (suggested structure)

```json
{
  "task_type": "classification",
  "metric": "f1_macro",
  "baseline": {
    "score": 0.72,
    "eval_set": "validation",
    "details": {
      "accuracy": 0.75
    }
  },
  "signal_check": {
    "ran": true,
    "permutations": 5,
    "real_baseline_score": 0.72,
    "shuffled_mean": 0.49,
    "shuffled_std": 0.02,
    "signal_detected": true,
    "user_overrode_no_signal": false
  },
  "stacking": {
    "attempted": true,
    "base_learners": ["LightGBM", "XGBoost", "RandomForest"],
    "meta_learner": "LogisticRegression",
    "best_single_score": 0.79,
    "ensemble_score": 0.81,
    "adopted": true,
    "reason": "ensemble beat best single by 2.5%",
    "time_limit_seconds_used": null,
    "time_limit_source": "spec_unlimited"
  },
  "ceiling_check": {
    "near_ceiling": false,
    "family_score_spread_pct": 0.018,
    "baseline_to_best_gain_pct": 0.072,
    "note": "Top-3 families spread 1.8%; baseline-to-best gain 7.2% — headroom remains."
  },
  "autogluon": {
    "attempted": true,
    "reason": null,
    "time_limit_seconds": 300,
    "preset": "medium_quality",
    "holdout_score": 0.881,
    "holdout_score_vs_main_pct": 0.007
  },
  "final": {
    "score": 0.81,
    "eval_set": "holdout_test",
    "model": "stacking_ensemble",
    "details": {
      "accuracy": 0.83
    }
  }
}
```

## config.json (suggested structure)

```json
{
  "dataset": {
    "locations": ["data.csv"],
    "format": "csv",
    "merge_strategy": "concat",
    "source_column": "dataset_source"
  },
  "task_type": "classification",
  "target": "label",
  "target_derivation": null,
  "metric": "f1_macro",
  "splits": {
    "strategy": "random_80_10_10",
    "train": 0.8,
    "validation": 0.1,
    "holdout_test": 0.1
  },
  "bounds": {
    "baseline_trials": 1,
    "main_trials": 500,
    "main_hours": 8,
    "main_seconds_used": 28800,
    "main_seconds_source": "spec_default",
    "stacking_seconds_used": null,
    "stacking_seconds_source": "spec_unlimited",
    "early_stop_no_improve_trials": 25,
    "early_stop_min_relative_gain": 0.001
  },
  "optuna": {
    "sampler": "TPESampler",
    "seed": 42,
    "cv_folds": null
  },
  "environment": {
    "venv_path": ".venv",
    "packages": {
      "scikit-learn": "1.4.2",
      "optuna": "3.6.1",
      "xgboost": "2.0.3"
    }
  },
  "reproducibility": {
    "random_seed": 42
  },
  "feature_engineering": {
    "date_parts": ["month", "quarter", "day_of_week", "is_weekend"],
    "lags": {
      "target": [1, 7, 14],
      "rolling_windows": [7, 30]
    },
    "transforms": ["log1p:amount"],
    "text": {
      "method": "tfidf",
      "max_features": 5000
    }
  },
  "feature_handling": {
    "excluded": [],
    "missing_values": {
      "numeric": "median",
      "categorical": "mode"
    },
    "class_imbalance": {
      "strategy": "class_weight_balanced",
      "imbalance_ratio": null
    }
  },
  "explainability": {
    "requested": false
  },
  "signal_check": {
    "enabled": true,
    "permutations": 5,
    "threshold_stds": 2
  },
  "stacking": {
    "enabled": true,
    "min_families": 3,
    "max_base_learners": 5,
    "family_score_window": 0.1,
    "acceptance_relative_gain": 0.005
  },
  "comparison": {
    "autogluon": false
  }
}
```

## results.md (suggested structure)

```markdown
# Model Results

## Data profile
- Rows: 1,599
- Columns: 12
- Missing values: 0.0%
- Target distribution: 25% positive, 75% negative

## Profiling decisions
- Dropped columns: [list any dropped and why]
- Imputed columns: [list strategy per column]
- Transforms applied in pipeline: [e.g., log1p on amount]
- Outlier handling: [e.g., capped at 1st/99th percentile for column X]

## Signal check
- Real baseline: 0.72 (validation)
- Shuffled-label baselines: 0.49 ± 0.02 (5 permutations)
- Verdict: signal detected — proceeded to iteration
- What this means: the model is clearly learning something real from the
  features, not just memorising noise.

## Best model
- Model: stacking ensemble of LightGBM + XGBoost + RandomForest
- Meta-learner: LogisticRegression
- Metric: AUC = 0.91 (holdout test set)
- Best single model (LightGBM): AUC = 0.89 (validation)
- Baseline AUC: 0.83 (validation)
- What AUC means: probability the model ranks a positive example above a
  negative one; higher is better.

## AutoML comparison (only if user opted in)

Include this section only when `autogluon.attempted == true`. If the user
did not opt in, omit the section entirely. If they opted in but the run
was skipped (small data, install failure, time-series task), include the
section briefly stating why.

- AutoGluon score: AUC = 0.881 (same holdout, medium_quality preset, 5 min budget)
- Transparent pipeline score: AUC = 0.875 (same holdout)
- Gap: +0.7% in AutoGluon's favour (within margin of error)
- Verdict: "Within margin of error — the transparent pipeline is competitive
  with off-the-shelf AutoML on this dataset. Use the transparent pipeline if
  you value inspectability; use AutoGluon if you only care about the score."

(Adjust the verdict per the bands in SKILL.md §4.7: within 1% → "competitive";
1–3% → "small but real gap"; >3% → "AutoGluon meaningfully better, consider
its diverse model zoo".)

Inference with the AutoGluon model:
```bash
python -c "from autogluon.tabular import TabularPredictor; \
  p = TabularPredictor.load('artefacts/autogluon_predictor'); \
  print(p.predict(...))"
```

## Ceiling check
- Top-3 family validation spread: 1.8% relative (LightGBM 0.89, XGBoost 0.88,
  RandomForest 0.875)
- Baseline → best gain: 7.2% relative
- Verdict: headroom remains — different models still produce meaningfully
  different scores, and the iteration step beat the baseline by a non-trivial
  margin.

(If `near_ceiling = true`, replace the verdict with the plain-language
explanation from SKILL.md §4.6 — e.g. "All families clustered within ~1%
and stacking did not help; further trials are unlikely to improve this
score.")

## Training process
- Split: random 80/10/10 (train/validation/holdout), stratified
- Preprocessing: median imputation + one-hot encoding (fit on training fold only)
- Feature engineering: date parts (month, day_of_week), lag(1, 7), rolling mean
- Optimizer: Optuna TPESampler, seed 42
- Trials run: 87 (converged — 25 non-improving trials with < 0.1% gain)
- Stacking: 3 base learners selected, ensemble beat best single by 2.2% → adopted
- Seed: 42

## What to try next

Generate this section from the run's state — do not boilerplate it. Use the
rules below:

- **If `near_ceiling = true`**: list (in this order) collecting new features,
  reframing the target, joining external context, and — if any text fields
  exist — trying an LLM-based approach. Do not suggest "more trials" or
  "different hyperparameters"; those will not help.
- **If stacking was skipped because too few families converged**: suggest
  adding the missing families (e.g. "CatBoost was not installed; installing
  it could enable stacking").
- **If baseline → best gain was modest (<5%) but signal was clearly present**:
  suggest feature engineering targeted at the top SHAP features.
- **If SHAP was not run**: suggest running it to identify which features
  are driving the score — a one-line recommendation, not a paragraph.
- **Always**: name the single most important data lever the user could pull,
  in one sentence.

## Inference
- Run: `python artefacts/infer.py --input new_data.csv --output predictions.csv`

## Artifacts
- artefacts/model.joblib
- artefacts/train.py
- artefacts/infer.py
- artefacts/metrics.json
- artefacts/config.json

---
## Reproducibility

This footer makes the artefact on disk the source of truth — if a chat
summary disagrees with these numbers, trust this footer.

- Random seed: 42
- Trials run: 87 (Optuna TPESampler, convergence: 25 non-improving with <0.1% gain)
- Final eval set: holdout_test (532 rows, never touched during search)
- Baseline algorithm: LogisticRegression (fixed per task type)
- Source repo commit: <git rev-parse HEAD if available>
- Python: 3.x.y · scikit-learn x.y · optuna x.y · lightgbm x.y · xgboost x.y
- Generated: <ISO timestamp>
```

## No-signal results.md variant

If the signal check fails and the user opts to halt, the `results.md` should
clearly state that no model was produced. Example:

```markdown
# Model Results — No Signal Detected

## Verdict
This dataset does not contain enough signal to predict the target reliably.

## Signal check
- Real baseline: 0.51 (validation)
- Shuffled-label baselines: 0.49 ± 0.02 (5 permutations)
- The real baseline is within 2 standard deviations of random — the features
  cannot reliably distinguish real labels from shuffled ones.

## What to try instead
- Collect more or different features for each row.
- Reconsider how the target is defined or derived.
- If most of the signal might live in free text, consider an LLM-based approach.
- Confirm there are no upstream data issues (wrong join key, stale labels).

## Artifacts
- artefacts/metrics.json (includes the signal-check result)
- artefacts/config.json
- No model was saved.
```
