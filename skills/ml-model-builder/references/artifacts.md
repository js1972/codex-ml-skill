# Artifact Layout and Formats

Use the `artefacts/` directory for model outputs. Create it if missing.
Also write `results.md` in the project root with a concise final summary.

## Expected Files

- `artefacts/train.py`
- `artefacts/infer.py`
- `artefacts/model.<ext>` (choose a consistent format, e.g., joblib or pickle)
- `artefacts/metrics.json`
- `artefacts/config.json`
- Optional explainability output (e.g., `artefacts/shap_summary.html`)
- `results.md` (summary of best model, metrics, training process)

## metrics.json (suggested structure)

```json
{
  "task_type": "classification",
  "metric": "f1_macro",
  "baseline": {
    "score": 0.72,
    "details": {
      "accuracy": 0.75
    }
  },
  "final": {
    "score": 0.79,
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
    "locations": [
      "data.csv"
    ],
    "format": "csv",
    "merge_strategy": "concat",
    "source_column": "dataset_source"
  },
  "task_type": "classification",
  "target": "label",
  "target_derivation": null,
  "split_strategy": "random_80_20",
  "metric": "f1_macro",
  "bounds": {
    "baseline_minutes": 5,
    "baseline_trials": 2,
    "main_minutes": 30,
    "main_trials": 25,
    "early_stop_trials": 6
  },
  "environment": {
    "venv_path": ".venv",
    "packages": {
      "scikit-learn": "1.4.2",
      "xgboost": "2.0.3"
    }
  },
  "reproducibility": {
    "random_seed": 42
  },
  "feature_engineering": {
    "date_parts": [
      "month",
      "quarter",
      "day_of_week",
      "is_weekend"
    ],
    "lags": {
      "target": [
        1,
        7,
        14
      ],
      "rolling_windows": [
        7,
        30
      ]
    },
    "transforms": [
      "log1p:amount"
    ],
    "text": {
      "method": "tfidf",
      "max_features": 5000
    }
  },
  "feature_handling": {
    "excluded": [],
    "missing_values": "impute"
  },
  "explainability": {
    "requested": false
  }
}
```

## Scripts Expectations

- `train.py` should:
  - load data, preprocess, split, train, evaluate
  - save the model and preprocessing pipeline together
  - write metrics.json and config.json
- `infer.py` should:
  - load the same pipeline
  - accept an input file and output predictions

## results.md (suggested structure)

```markdown
# Model Results

## Data profile
- Rows: 1,599
- Columns: 12
- Missing values: 0.0%
- Target distribution: 25% positive, 75% negative

## Best model
- Model: RandomForestClassifier
- Metric: AUC = 0.91
- Baseline AUC: 0.83
- What AUC means: probability the model ranks a positive example above a
  negative one; higher is better.

## Training process
- Split: random 80/20, stratified
- Preprocessing: imputation + one-hot encoding
- Feature engineering: date parts (month, day_of_week), lag(1, 7), rolling mean
- Bounds: 30 minutes or 25 trials, early stop after 6 no-improve
- Seed: 42

## Artifacts
- artefacts/model.joblib
- artefacts/train.py
- artefacts/infer.py
- artefacts/metrics.json
- artefacts/config.json
```
