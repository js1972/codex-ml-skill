# Governance and Execution

## Contents

- [Decision policy](#decision-policy)
- [Mandatory execution approval](#mandatory-execution-approval)
- [Question design](#question-design)
- [Progress and managed execution](#progress-and-managed-execution)
- [Track-specific budgets](#track-specific-budgets)
- [Classical candidate ledger](#classical-candidate-ledger)
- [Backend additions and improvement evidence](#backend-additions-and-improvement-evidence)
- [Audit trail](#audit-trail)

## Decision policy

Classify decisions by consequence:

- **Explicit approval** — require confirmation for the target, prediction
  moment, features, split/evaluation semantics, primary metric, each execution
  track and its budget, remote data transfer, high-stakes deployment, and any
  irreversible action.
- **Recommendation** — group material modeling-preflight findings in a concise
  table with a recommended action and allow the user to approve or override
  them.
- **Default** — apply only low-risk reversible mechanics such as seed 42, safe
  output names, and bounded concurrency; record them.
- **Required safeguard** — do not allow convenience to introduce leakage,
  invalid evaluation, unsafe deserialization, or false performance claims.
  Explain the constraint and offer a valid alternative.

Treat thresholds as context-dependent screening triggers, not scientific
constants.

## Mandatory execution approval

Complete source inspection and modeling preflight before proposing execution,
but do not fit, build, query, or install a large optional backend until the
user approves one experiment plan.

Present:

| Decision | Required detail |
|---|---|
| Problem | Target/label meaning, prediction moment, row/decision grain, intended and prohibited uses |
| Features | Included feature contract and material exclusions |
| Evaluation | Population, split design, primary metric, uncertainty method |
| Classical | Include/decline, families, Optuna/search and compute budget |
| AutoGluon | Include/decline, preset, runtime estimate, run-to-completion or time-limited mode, optional time limit and resource budget |
| SAP RPT | Include/decline, accessible model IDs/access route, full-context or truncation plan, context sizes, retrieval strategies/readiness, input format/seed, request budget, named destination and transferred data scope |
| Ablation | When requested or material to a feature/source decision: include/decline, feature groups and hypotheses, full-pipeline retraining budget, and development-only evidence boundary |
| Selection | Operational constraints used alongside predictive quality |

Recommend a concrete choice for each track. Require an explicit response that
confirms or changes it. A request such as “train the best model” is not enough
to infer that AutoGluon or SAP RPT should be excluded. Conversely, an explicit
request to include SAP RPT already answers that track's include/decline
question; show it as selected rather than asking the user to choose it again.

Use the host's native structured question tool for the consolidated approval
whenever it is available, such as Codex `request_user_input` or Claude Code
`AskUserQuestion`. Provide a recommended “approve plan” option plus concise
alternatives to change the plan. Never make approval depend on typing an exact
sentence or magic phrase. If the host exposes no structured question tool,
accept an ordinary semantic yes/no or requested changes.

Before invoking that question, finish all safe read-only preflight checks that
could reveal a later blocker: dependency availability, expected installs,
AutoGluon runtime range, disk/resources, RPT CLI/access readiness, and transfer
scope. For RPT, discover accessible model IDs, deployed capacities, whether
full fold-valid context fits, retrieval-extra availability, input size/format,
and the estimated model/context/retrieval request matrix. If a capacity probe
would transfer data, include that probe inside the proposed transfer and
request budget rather than running it before approval. Put every foreseeable
decision in one structured question invocation so the user can approve the
complete unattended run and walk away.

Record the confirmed target/features in `problem`, the evaluation plan in
`evaluation`, and set `approval.scope.target`, `feature_contract`,
`split_design`, and `primary_metric` to true only after the user confirms those
current values. Record each track's selection/status/budget, the approved RPT
configuration plan when selected, and the approval time in `approval`.
Initialize `approval.amendments` and
`approval.remote_transfers` as lists, even when empty.

When an ablation is selected, record its approved groups, hypotheses, maximum
variants, and incremental wall-time budget in `approval.analyses`. Do not add an
ablation after the approval gate merely because a feature-importance result is
interesting; obtain an amendment first if it was not foreseeable.

For each later approved plan change, add one amendment with a unique ID,
approval timestamp, reason, and a non-empty list of structured
`path`/`before`/`after` changes. Use `lineage.notes` only for scientific
interpretation or evidence exposure, not as the sole approval record.

Do not start one track while the overall plan awaits approval. After approval,
an unavailable dependency or access route may pause only that approved track
while other approved tracks continue. Do not substitute an unapproved backend.
Do not ask routine follow-up questions about choices already disclosed and
approved. Ask again only when new authority or a material unforeseeable scope
change is required.

When SAP RPT is selected, include the named destination and structured
feature/label/query/identifier transfer scope in the consolidated experiment
approval. That one approval authorizes RPT requests within the disclosed scope;
do not obtain a second confirmation before the first request. Record a unique
transfer approval ID, timestamp, backend, destination, purpose, and data scope
in `approval.remote_transfers`; reference that ID from the completed RPT
backend. Ask again only when the destination or transferred fields,
sensitivity, row volume, or purpose materially expands, or when an independent
external policy requires another approval. Never request or store credentials.

## Question design

Ask the smallest number of questions required to avoid the wrong experiment.

1. Resolve a problem-framing question before inspection only when plausible
   answers materially change scientific validity, leakage, safety, or the
   decision being supported.
2. Otherwise infer the most reasonable target meaning, row grain, prediction
   moment, and label semantics as explicitly provisional values.
3. Inspect source data only for modeling preflight.
4. Present the provisional semantics, findings, and proposed resolutions
   together in the single approval gate:

   | Finding | Why it matters | Recommendation | Decision |
   |---|---|---|---|

5. Present all remaining decisions together through one invocation of the
   host's structured question tool when available; do not request a typed
   approval sentence or deliberately defer a known question until execution.
6. Separate unrelated high-impact decisions such as redefining the target or
   changing the prediction moment.

Explain jargon in one sentence and state the consequence of each option. Avoid
one-column-at-a-time prompting. Do not write `approval.scope` confirmations
until the user approves the provisional or corrected values.

## Progress and managed execution

Use the host task UI when available. Maintain one item per major stage, not per
model trial. Send concise user-visible updates at stage boundaries and during
long work with useful new information such as completed folds, elapsed budget,
best permitted development score, or current backend.

Write long-running entry points to disk and persist resumable progress. Use the
host's managed-process/session mechanism for classical search, AutoGluon,
large explainability runs, and long RPT request batches. Record the execution
mechanism and session/task identifier in `run.json` when available.

Do not shorten an approved budget merely to fit a foreground tool timeout.
When AutoGluon `run_to_completion` is approved, continue even when the runtime
estimate is several hours. A long duration is not itself a reason to interrupt
or seek another confirmation.

## Track-specific budgets

Do not express every backend as Optuna trials:

- **Every approved track:** positive `cpu_count`, `parallel_jobs`, and
  `memory_gb`, plus boolean `gpu_enabled`.
- **Classical:** `candidate_families`, `minimum_family_coverage`,
  `time_limit_seconds`, and `optuna_trials`.
- **AutoGluon:** `preset`, `run_mode`, `runtime_estimate`,
  `time_limit_seconds`, and `disk_gb`. Use `run_mode: "run_to_completion"` with
  `time_limit_seconds: null`, or `run_mode: "time_limited"` with a positive
  approved limit. Default best-model requests without a deadline to
  run-to-completion.
- **SAP RPT:** `max_context_rows`, `max_request_rows` for context plus query,
  `max_query_batch_rows` per call, `max_columns`, `max_requests`,
  `max_retries`, and `timeout_seconds`, plus an upfront plan containing model
  IDs, full-context behavior, context-size candidates, retrieval strategies
  and extra status, input format, reproducibility seed, estimated
  configurations, remote-transfer approval, and any latency/cost envelope.

Ask before materially exceeding any approved budget.

## Classical candidate ledger

Freeze the classical roster from task, data, and deployment criteria before
dependency inspection. Do not treat package absence as scientific
unsuitability, but do not universally force the installation or execution of
every possible library. Propose a defensible family set and coverage budget in
the single approval gate.

Give each candidate exactly one ledger row under
`run.json.backends.classical.candidates` with:

- a unique non-empty `name`;
- a non-empty `family`;
- a non-empty `consideration_basis` explaining task/data/deployment fit;
- `status`: `completed`, `failed`, or `excluded`;
- a finite `score` for a completed candidate;
- a concrete `reason` for a failed or excluded candidate.

Record every family in the approved roster exactly once, including eligible
families that fail and proposed families excluded for scientific, deployment,
dependency, or approved-budget reasons. Do not populate the ledger with every
library that could theoretically solve the task. Keep detailed dependency and
coverage information in `backends.classical.evidence`.

Do not put AutoGluon or SAP RPT in this ledger. They are independent approved
tracks with different execution semantics.

## Backend additions and improvement evidence

Keep approved backends that share the same source fingerprint, target, eligible
features, splits, evaluation rows, weights, and metric implementation in one
experiment directory.

When the user adds an optional backend later:

1. Verify that the experiment contract is still identical.
2. Obtain explicit approval for the added track and budget.
3. Add only its required backend artifacts.
4. Update `approval.tracks`, add a structured `approval.amendments` record and
   any `approval.remote_transfers` record, then update `backends`,
   `inference.backends`, selection, `lineage.notes`, and `validation.json`.
5. Refresh the inclusive `report.html` and `results.md`.

Do not create a full child run or copy the existing model, fold assignments,
predictions, report assets, fixtures, or search trials merely to add a
comparison.

Create a new run when the source data, target, feature contract, split,
primary metric, modeling hypothesis, or released winner changes materially.
Record the prior run ID and only the evidence required to explain the change;
do not copy the prior directory.

Treat any opened holdout/external/outer-fold result as historical evidence.
Do not use it to select a descendant and still claim that descendant has an
unbiased evaluation on the same population. Require untouched future/external
evidence for that claim or label the result development-only.

## Audit trail

Use the consolidated `run.json` as the machine-readable authority. Record:

- source fingerprints and bounded queries;
- problem, cohort, label, feature, and evaluation contracts;
- modeling-preflight findings and approved resolutions;
- the explicit experiment approval and approved later changes;
- structured remote-transfer approvals and their backend references;
- track selections, budgets, dependencies, execution status, failures, and
  results;
- split fingerprint and overlap/order/support audits;
- final-evidence access and selection influence;
- selected predictive and operational winners;
- pinned environment in `requirements.lock`, backend runtime evidence, and the
  `inference` contract;
- prior-run references and same-experiment backend additions in
  `lineage.notes`.

Keep only trial or out-of-fold evidence required for a material result, and
record its concise summary under the relevant `backends` entry. The files on
disk are authoritative. Reconcile any conflicting chat or report statement
before handoff.
