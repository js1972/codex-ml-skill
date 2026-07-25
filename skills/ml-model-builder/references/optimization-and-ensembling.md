# Signal Diagnostics, Optimization, and Ensembling

## Contents

- [Learnability diagnostics](#learnability-diagnostics)
- [Cross-validation objective](#cross-validation-objective)
- [Search design](#search-design)
- [Convergence and budget](#convergence-and-budget)
- [Failure and resource handling](#failure-and-resource-handling)
- [Stacking](#stacking)
- [Search diagnostics](#search-diagnostics)

## Learnability diagnostics

Use diagnostics to avoid wasting compute, but do not confuse them with a proof
that no useful model exists.

### Supervised tabular

Compare:

- naive target-only prediction;
- fixed regularized linear/logistic probe;
- one shallow nonlinear tree/boosting probe when nonlinear signal is plausible.

Use cross-validated permutation tests when practical. Preserve exchangeability:

- shuffle labels within valid groups/blocks when dependence exists;
- do not permute time-series labels across time;
- rerun the full fold-local preprocessing pipeline per permutation.

Run permutation tests only when their decision value justifies the compute.
When used, 20 permutations provide only a coarse screen; if the result is near
the decision boundary and matters, extend to 99+ within budget. Report the
one-sided empirical p-value with +1 correction and a practical effect size.

Warn or stop when all appropriate probes fail materially, but let the user
override after explaining the evidence. Do not halt nonlinear search because
only a linear probe failed.

### Forecasting

Compare fixed candidate probes with last-value and seasonal-naive baselines
over multiple rolling origins. Require improvement that matters at the stated
horizon; a statistical p-value is secondary to stable backtest gain.

### Unsupervised anomaly detection

Do not run a “signal test.” Use rank stability, top-k overlap, perturbation
sanity checks and domain review as described in `anomaly-detection.md`.

## Cross-validation objective

Use the same folds for all candidates. Match deployment:

- stratified folds for ordinary classification;
- grouped folds for repeated entities;
- temporal/rolling folds for future prediction;
- grouped-temporal folds when both apply.

Use repeated or nested CV for small/noisy data when compute permits. A fixed
5,000-row cutoff is not a valid scientific rule.

Keep holdout/external targets outside Optuna and every model-selection
decision. Under nested CV, rerun the complete selection process inside each
outer training fold and never expose the active outer-fold targets.

## Search design

1. Select eligible families from the task/data reference.
2. Give each family a small, comparable initial design.
3. Run separate conditional search spaces per family or a correctly
   conditional joint study.
4. Optimize the agreed primary validation metric.
5. Track secondary metrics, fit time, prediction time, model size and failures.
6. Keep feature engineering/preprocessing inside folds.

Use seeded Optuna TPE after an initial random exploration. Avoid huge,
uninformed ranges. Use pruning only for estimators with meaningful intermediate
results. Do not let a cheap family receive most trials merely because it
finishes faster.

## Convergence and budget

Ask for an elapsed-time/compute budget. If unspecified:

- cover every eligible family with a minimal initial design;
- continue adaptive search while meaningful improvements occur;
- enforce a finite failsafe appropriate to the environment.

Define convergence before the run using:

- minimum completed trials per family;
- patience on the cross-validated incumbent;
- practical minimum improvement;
- uncertainty/noise of the validation estimate;
- remaining time.

Record the exact stop reason: budget, convergence, user stop, failures, or
insufficient candidates. Avoid a universal “25 trials and 0.1%” rule.

## Failure and resource handling

- Catch expected invalid hyperparameter combinations and mark failed trials.
- Stop/reduce parallelism on memory pressure rather than corrupting the run.
- Set estimator and BLAS thread counts to avoid nested oversubscription.
- Record CPU/GPU, threads, elapsed time and peak memory when available.
- Resume from persistent Optuna storage for long studies.
- Validate that trial failures are not systematically removing one family.

## Stacking

Stack only when:

- at least two or three genuinely suitable families are competitive;
- enough training data exists for out-of-fold meta-features;
- candidates make complementary errors;
- the compute/deployment budget permits added complexity.

Use out-of-fold predictions created with the same task-aware folds.
Hyperparameters must be selected without leaking the meta-validation target;
use nested folds or freeze candidate configurations before generating OOF
predictions.

Use a simple regularized meta-learner. Do not add class weights automatically
when calibrated probabilities matter. Evaluate calibration after stacking.

Adopt stacking only when gain is repeatable across folds and practically
meaningful—not merely above a fixed 0.5% on one split. Otherwise keep the
simpler single model.

## Search diagnostics

Report:

- best score by family with fold variation;
- improvement trajectory and elapsed compute;
- failed/pruned trial counts;
- family diversity and error correlation;
- baseline-to-best improvement;
- whether gains have plateaued within the explored search.

Call this a **search plateau**, not a predictive ceiling. Recommend new data,
features, task reframing or alternative model classes when evidence supports
those actions.
