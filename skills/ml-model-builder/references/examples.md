# Example Prompts and Expected Clarifications

Use these examples to guide the clarification flow and defaults.

## Example 1: Wine quality classification

Prompt:
Wine quality classification. Using the classic wine dataset located at:
- data/winequality-red.csv
- data/winequality-white.csv
Train a binary classifier to determine high quality wines. Treat quality >= 7
as high quality (column name is `quality`). Use AUC as the model evaluation
metric.

Clarifications:
- Confirm merge strategy for the two files (default: concat with
  `dataset_source` column).
- Confirm derived target rule: `is_high_quality = quality >= 7`.
- Confirm split strategy (default: random 80/20 with stratification).

## Example 2: Journal entry anomaly detection

Prompt:
Anomaly detection over finance journal entries. Dataset is located at
data/journal entries with 5 percent injected anomalies.csv
and is a dummy dataset with 5 percent injected anomalies.

Clarifications:
- Ask if a label column exists for anomalies. If yes, treat as supervised.
- If no label, confirm unsupervised and use contamination=0.05 by default.
- Ask for entity/time columns if present.

## Example 3: Payment delay forecasting

Prompt:
Payment delay forecasting with dataset located at
data/accounts receivable entries.csv. Use a time-based
split. Column `NetDueDate` is when payment was due and `ClearingDate` is when it
was actually paid.

Clarifications:
- Confirm target derivation: `delay_days = ClearingDate - NetDueDate`.
- Confirm time column for split: `NetDueDate`.
- Ask if the goal is per-invoice regression or aggregated time-series
  forecasting (default: per-invoice regression with time-based split).
