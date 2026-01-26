# Codex ML Skill - Full Project Spec

## 0) Objective
Build an OpenAI Codex skill that helps users create classical machine learning
models from their dataset. The skill gathers requirements, builds a quick
baseline, then iterates to a stronger model and saves code and artifacts.

---

## 1) Functional Requirements
- Accept a user description of the ML problem.
- Ask follow-up questions until minimum inputs are collected.
- Support task types: classification, regression, time series forecasting,
  anomaly detection.
- Allow dataset location(s) from the current working folder or an HTTP(S) URL.
- Support multiple files and ask how to combine them if needed.
- Create a `.venv` in the current working directory before any Python runs or
  dependency installation.
- Install dependencies only inside the `.venv`.
- Build a quick, cheap baseline model first and report metrics.
- Iterate to improve the model within user-provided bounds (with defaults).
- If the task is better suited to an LLM (e.g., text-heavy with very small
  labeled data), recommend that path and ask whether to proceed with classical
  ML anyway.
- Apply default feature engineering (date parts, lags, transforms, encoding)
  with user override and record it in `artefacts/config.json`.
- Include non-sklearn models (e.g., XGBoost, LightGBM, CatBoost) in the
  iteration stage when appropriate, installing them into the venv if needed.
- Save outputs into `artefacts/`:
  - `train.py` (training pipeline)
  - `infer.py` (inference pipeline)
  - model artifacts (e.g., `model.pkl` or `model.joblib`)
  - `metrics.json` (baseline + final)
  - `config.json` (captured inputs, splits, metric, bounds)
  - optional explainability outputs (e.g., `shap_summary.html`) if requested
- Provide a final user-facing summary of the best model, metrics, training
  process, and artifact locations.
- Save the same summary to `results.md` in the project root.
- Require a random seed for reproducibility and store it in `artefacts/config.json`.
- Run a quick target leakage guard and confirm any exclusions with the user.
- Include a data profile summary in `results.md`.
- Explain metrics in simple, non-expert terms in the summary and `results.md`.

---

## 2) Clarification Flow and Minimum Inputs
### Always required before training
- Dataset location(s) (local path(s) or URL(s))
- Task type (classification, regression, time series, anomaly)
- Evaluation metric (or accept defaults)
- Split strategy (random split or time-based split)
- Choose split/CV based on time ordering and entity grouping; use time-based
  splits for time-dependent or financial data.

### Additional requirements by task
- Classification/Regression: target column
- Time series forecasting: time column, forecast horizon, data frequency
- Anomaly detection:
  - Supervised: target/label column
  - Unsupervised: entity/time columns (if applicable)

### Optional questions (ask if not supplied)
- Feature exclusions
- Handling missing values
- Class imbalance strategy (classification)
- Target transformation rules (threshold, date difference)
- Explainability (SHAP) preference
- Training bounds (time and trial caps)

---

## 3) Dataset Handling
- Local file: CSV or Parquet in the current working folder.
- URL: HTTP(S) link to a CSV or Parquet file.
- Multiple files: default to row-wise concat when schemas align.
- If target is derived, record the rule in `artefacts/config.json`.
- Authenticated cloud storage is out of scope.

---

## 4) Training Workflow
1. Load dataset and validate schema.
2. Create baseline model:
   - Simple preprocessing
   - Task-appropriate baseline algorithm
   - Evaluate and store baseline metrics
3. Iterate to stronger model:
   - Improved preprocessing, feature handling, and model selection
   - Use the chosen metric to select the best model
   - Respect training bounds
4. Export artifacts and scripts to `artefacts/`.

---

## 5) Training Bounds (Ask the user, use defaults if not provided)
Defaults:
- Baseline budget: 5 minutes or 2 trials, whichever comes first
- Main training: 30 minutes or 25 trials, whichever comes first
- Early stop: stop after 6 non-improving trials

---

## 6) Out of Scope
- SAP-specific architecture or products
- UI frontend or history view
- API layers, auth, caching, or persistence beyond local artifacts
- Cloud bucket ingestion requiring auth
