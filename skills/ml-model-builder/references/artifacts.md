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
  "final": {
    "score": 0.79,
    "eval_set": "holdout_test",
    "details": {
      "accuracy": 0.81
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

## Best model
- Model: RandomForestClassifier
- Metric: AUC = 0.91 (holdout test set)
- Baseline AUC: 0.83 (validation set)
- What AUC means: probability the model ranks a positive example above a
  negative one; higher is better.

## Training process
- Split: random 80/10/10 (train/validation/holdout), stratified
- Preprocessing: median imputation + one-hot encoding (fit on training fold only)
- Feature engineering: date parts (month, day_of_week), lag(1, 7), rolling mean
- Optimizer: Optuna TPESampler, seed 42
- Trials run: 87 (converged — 25 non-improving trials with < 0.1% gain)
- Seed: 42

## Inference
- Run: `python artefacts/infer.py --input new_data.csv --output predictions.csv`

## Artifacts
- artefacts/model.joblib
- artefacts/train.py
- artefacts/infer.py
- artefacts/metrics.json
- artefacts/config.json
```
