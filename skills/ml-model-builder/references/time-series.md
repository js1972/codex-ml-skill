# Time-Series Forecasting and Time-Dependent Prediction

## Contents

- [Choose the correct problem](#choose-the-correct-problem)
- [Forecast contract](#forecast-contract)
- [Panel profiling](#panel-profiling)
- [Validation](#validation)
- [Baselines and models](#baselines-and-models)
- [Multi-horizon strategy](#multi-horizon-strategy)
- [Feature safety](#feature-safety)
- [Metrics and uncertainty](#metrics-and-uncertainty)
- [Replenishment decisions](#replenishment-decisions)
- [Special cases](#special-cases)

## Choose the correct problem

Distinguish:

- forecasting future values of one or many series;
- per-event regression/classification evaluated forward in time;
- survival/time-to-event prediction with censoring;
- anomaly detection on temporal observations.

Do not call per-row payment-delay regression “time-series forecasting” merely
because it has dates. Route survival/censoring problems explicitly; ordinary
regression can bias labels when outcomes have not matured.

Future-event classification or regression for recurring known entities remains
a supervised tabular task. Use the declared `known_entity_temporal` policy from
`data-and-leakage.md` when later rows for known entities match deployment; do
not relabel the task as forecasting merely to permit entity overlap.

## Forecast contract

Record:

- forecast origin and horizon;
- data frequency and timezone;
- one series or panel/entity key;
- target aggregation and reconciliation needs;
- whether the target is observed sales or latent unconstrained demand;
- known-future versus observed-only covariates;
- historical vintages of planned covariates such as promotions and prices;
- update cadence and retraining frequency;
- cold-start behavior for new entities;
- operational cost of late/early or under/over forecasts.

## Panel profiling

Report target-blind full-population series counts and distributions of history
length, coverage and gaps. Compute target levels, zeros and other
value-dependent diagnostics only on the permitted development or
backtest-training population. Bound detailed traces to at most the configured
`--max-panel-series` sample (default 12), record its deterministic selection,
and do not infer panel-wide quality from those examples. Aggregate large remote
panels near the source as described in `large-data.md`.

## Validation

Use rolling-origin or expanding-window backtests with multiple origins. Cover
enough independent origins to represent important seasonal, promotional and
regime conditions; two origins are only a minimal smoke test. Match:

- production horizon;
- retraining cadence;
- gap/embargo needed for delayed features or labels;
- entity availability;
- seasonal coverage.

Keep feature engineering inside each backtest fold. Never use random CV. Hold
out the final horizon(s) for the one-time final evaluation.

For panel data, report both aggregate and per-series/segment results. Weighting
must match business importance, not merely row count.

## Baselines and models

Always compare:

- last value;
- seasonal naive using the relevant seasonal period;
- drift or simple moving reference when appropriate.

Candidate models may include:

- ETS/SARIMAX for suitable low-dimensional series;
- regularized autoregression;
- gradient boosting with safe lag/rolling/calendar features;
- global panel models when series share structure;
- intermittent-demand methods for sparse counts.

Use Prophet only when its assumptions and dependency cost fit the problem; it
is not a universal default.

## Multi-horizon strategy

Choose and record one of:

- **direct** horizon-specific models, which avoid recursive error propagation
  but cost more and may be inconsistent across horizons;
- **recursive** one-step models, which are compact but feed predictions back as
  inputs and can accumulate error;
- **multi-output/global** models, which learn horizons jointly but require a
  clear output tensor/table and loss weighting.

Match training targets and features to the production issuance schedule. Test
every horizon and cumulative lead-time demand; do not assume a model that is
good at day 1 is good at day 28.

## Feature safety

- Create target lags with lag >= 1.
- Apply `.shift(1)` before rolling target statistics.
- For horizon `h`, ensure every feature is available at forecast origin for all
  steps being predicted.
- Treat weather, price, staffing or plan fields as known-future only when their
  future values truly exist at serving time.
- Reconstruct covariates from the plan/schedule version that existed at each
  historical forecast origin. Final revised schedules and realized promotions
  leak later information into backtests.
- Fit imputation, scaling and encoders within each temporal fold.
- Use causal gap filling; interpolation that reads future values leaks.
- Encode holidays/calendar features from deterministic calendars, not future
  outcomes.

## Metrics and uncertainty

Use MAE/RMSE with units. Use:

- MASE or RMSSE for comparison across series;
- sMAPE cautiously and with its limitations stated;
- MAPE only away from zero;
- pinball loss and empirical coverage for quantile forecasts.

Report error by horizon, time period, series/segment and demand scale. Include
prediction intervals when decisions depend on uncertainty, and validate both
coverage and width.

Distinguish confidence intervals for an estimated mean from predictive
intervals for future observations. Record requested quantiles and whether
coverage is marginal per horizon, simultaneous across the path, or for
cumulative lead-time demand. Use temporally valid calibration data, report
quantile crossing, and do not celebrate nominal coverage produced by uselessly
wide intervals.

## Replenishment decisions

Choose a primary measure aligned with inventory action: cumulative lead-time
demand error, asymmetric under/overage cost, service-level loss, or weighted
pinball loss. Day-level MAE alone can select the wrong replenishment model.
When inputs permit, simulate the downstream inventory policy with lead times,
order constraints, on-hand stock and backorders.

## Special cases

- **Intermittent demand:** report zero frequency and inter-arrival behavior;
  avoid models dominated by zeros.
- **Stockouts and lost sales:** observed sales can be censored below latent
  demand. Make target treatment a required decision: use availability/on-hand
  data and a justified lost-sales/censored-demand method, or exclude/downweight
  affected periods with sensitivity analysis. Never teach the model that zero
  inventory proves zero demand.
- **Promotions and price:** distinguish planned values known at forecast origin
  from realized values. Treat historical promotion assignment as potentially
  endogenous; forecasting association does not estimate promotion lift.
- **Regime changes:** show backtest performance before/after the change and
  avoid averaging away failure.
- **Short series:** prefer pooled/global or simple baselines; do not fit complex
  seasonal models without enough cycles.
- **New entities:** simulate age-at-origin cohorts, define metadata/global
  fallbacks, and do not leak post-opening history into cold-start evaluation.
- **Dense calendars:** distinguish a true zero from closure, not ranged,
  pre-opening, stockout and missing ETL; do not fill missing dates blindly.
- **Hierarchy:** ask whether forecasts must reconcile across product/site/region;
  choose a reconciliation method and verify point coherence. State that
  independently reconciled quantiles need not form coherent distributions.
- **Metric weighting:** report macro and business-weighted views. Guard
  MASE/RMSSE against zero or unstable denominators in short/intermittent series.
- **Censoring:** route to survival analysis or mature-label cohorts rather than
  treating incomplete outcomes as negatives or zero delays.
