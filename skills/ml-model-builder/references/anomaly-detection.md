# Anomaly Detection

## Contents

- [Frame the review process](#frame-the-review-process)
- [Supervised detection](#supervised-detection)
- [Unsupervised detection](#unsupervised-detection)
- [Validation and stability](#validation-and-stability)
- [Daily capacity queues](#daily-capacity-queues)
- [Candidate methods](#candidate-methods)
- [Outputs and limitations](#outputs-and-limitations)

## Frame the review process

Ask:

- What counts as anomalous and who decides?
- Are labels complete, delayed, biased toward previously detected cases, or
  available only for reviewed rows?
- Is the goal novelty detection on future observations or outlier description
  within the current dataset?
- How many cases can humans review per day/week?
- What is the cost of a missed event and an unnecessary review?
- Must anomalies be diverse rather than near-duplicates?

The operational output is usually a ranked review queue, not a binary truth.

## Supervised detection

Treat reliable anomaly labels as rare-event classification. Follow
`supervised-tabular.md`, emphasizing:

- PR-AUC, recall at review capacity and precision at k;
- time/group-aware validation;
- delayed-label and selection-bias analysis;
- class weights and thresholding before synthetic resampling;
- calibration only when label sampling supports probability interpretation.

Do not treat injected synthetic anomalies as proof of real-world performance.
Report them as a controlled test only.

## Unsupervised detection

Without labels, do not report accuracy, F1, AUC, “signal detected,” or a
validated contamination rate. Produce:

- anomaly scores and stable ranks;
- top-k cases for domain review;
- reason codes or contributing features where defensible;
- cluster/entity/time composition of the review queue;
- sensitivity to contamination, seed, scaling and feature set;
- a plan to capture review outcomes as future labels.

Use user-provided contamination as an operational review proportion, not an
estimated truth unless independently justified.

## Validation and stability

Assess:

- rank correlation and top-k overlap across seeds/folds/time windows;
- whether anomalies are duplicates or one dominant subgroup;
- score drift over time;
- synthetic/semi-synthetic perturbation recovery only as a limited sanity
  check;
- expert review agreement and precision at reviewed k when feedback exists.

For temporal data, evaluate future windows. For grouped data, ensure anomalies
do not arise merely from unseen entities unless that is the intended goal.

Compare top-k overlap across seeds/configurations on the **same scoring
population**. Different days contain different transactions, so across time
compare score/rank distributions, queue composition, concentration, persistent
entity rates and reviewed yield—not raw transaction overlap.

## Daily capacity queues

For a top-k operational queue, record:

- scoring timezone, cutoff and eligible population;
- historical reference window and whether it rolls or stays frozen;
- exclusion rules and late-arriving-data behavior;
- capacity by scheduling unit and behavior when fewer than k rows are eligible;
- deterministic tie-breaker, usually a stable transaction identifier;
- identifier passthrough for output while excluding it from model features;
- duplicate/near-duplicate handling and optional entity/merchant caps;
- score direction, within-batch rank, review flag and reason codes;
- cold-start behavior for unseen entities and insufficient history.

Exactly k selections require the complete scoring batch; a row-wise threshold
cannot guarantee daily capacity under score drift. Separate `score_rows` from a
batch `select_queue` step, and test empty, sub-capacity and tied batches. Apply
the same interface split to supervised rare-event queues.

For an unlabeled anomaly run, make `selection.capacity.limit` match
`metrics.anomaly_evaluation.review_capacity`, use `anomaly_score` rather than
probability semantics in `inference_test.json`, and run the same fixed-capacity
queue cases required by `artifacts.md`.

Fit reference transformations on historical data only. Do not let the current
day contaminate its own reference distribution. Freeze competing configurations
before a prospective analyst pilot.

Without labels, do not optimize model families against a fabricated scalar
accuracy objective. Compare a small defensible set using stability,
concentration/diversity, perturbation sanity checks, runtime, explanation
quality and blinded domain review. Stop when added complexity has no operational
evidence.

## Candidate methods

- IsolationForest for scalable mixed-shape outlier scoring after appropriate
  preprocessing.
- Robust covariance only for roughly elliptical, low-dimensional numeric data.
- LocalOutlierFactor for local structure; use `novelty=True` and correct
  train/inference semantics for new-data scoring.
- OneClassSVM only for suitably scaled, modest datasets because it can be
  expensive and sensitive.
- PCA/reconstruction methods for correlated numeric structure.
- Domain rules and robust univariate/multivariate scores as interpretable
  baselines.

Do not compare heterogeneous anomaly scores without orienting and normalizing
them. Do not ensemble merely because methods are diverse.

## Outputs and limitations

Record:

- review budget/top-k;
- score direction and transformation;
- feature set and scaling;
- contamination parameter and its meaning;
- stability measures;
- reviewed examples and outcomes with enough detail for domain validation;
- known blind spots and populations overrepresented in the queue.

Capture analyst feedback with transaction ID, scorer/data version, review
timestamp, reviewer decision, reason code and label-maturity status. Unreviewed
transactions are unknown—not negatives. Report selection bias because reviewed
labels come from the model-created queue.

Inference must return a continuous score, rank or thresholded review flag with
the threshold rationale. State clearly when no labeled estimate of real-world
precision or recall exists.

For adversarial settings, document poisoning/evasion assumptions, protect
reference windows from untrusted updates where possible, and monitor abrupt
feature/queue changes. Validate that reason codes are stable enough to support
review and do not claim local fidelity that the method cannot provide.
