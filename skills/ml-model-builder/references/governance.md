# Governance and Execution

## Contents

- [Decision policy](#decision-policy)
- [Question design](#question-design)
- [Progress reporting](#progress-reporting)
- [Managed processes](#managed-processes)
- [Analysis-to-model transitions](#analysis-to-model-transitions)
- [Budget and dependency control](#budget-and-dependency-control)
- [Improvement runs](#improvement-runs)
- [Audit trail](#audit-trail)

## Decision policy

Classify decisions by consequence:

- **Ask** — obtain explicit approval for choices that change the business
  question, target, prediction moment, split semantics, irreversible data
  handling, evaluation interpretation, large dependency installation, or
  deployment decision.
- **Recommend** — present a grouped table of material data-quality/modeling
  findings with a recommended action and let the user approve or override the
  group or individual rows.
- **Default** — apply low-risk reversible mechanics such as seed 42, artifact
  names, plot sampling seed, and safe output directories; record them.
- **Required safeguard** — do not allow user convenience to silently introduce
  target leakage, holdout reuse, unsafe deserialization, or false performance
  claims. Explain the constraint and offer a valid alternative.

Do not encode universal thresholds as scientific truths. A percentage is a
screening trigger that must be interpreted in context.

## Question design

Ask the smallest number of questions needed to avoid a wrong workflow.

1. Ask blocking problem-framing questions before loading data.
2. Profile data.
3. Present related findings in a concise decision table:

   | Finding | Why it matters | Recommendation | Decision |
   |---|---|---|---|

4. Separate unrelated high-impact questions, such as redefining the target or
   changing the prediction moment.
5. Explain jargon in one sentence and state the consequence of each option.

Avoid dozens of one-column-at-a-time prompts. Transparency means visible
reasoning and recorded decisions, not unnecessary interruption.

## Progress reporting

Use the host task/plan UI when available (`update_plan` in Codex or
`TodoWrite` in Claude Code). Maintain one item per major workflow stage, not
one item per chart or model trial.

Send user-visible updates:

```text
▶ Stage N — <name>
✓ Stage N complete — <decision or result>
```

During long work, update at least once per minute with useful information such
as completed trials, best validation score, elapsed budget, or current model
family. Do not repeat unchanged status.

## Managed processes

Write long-running Python entry points to disk and make them persist progress
and results. Launch them through the host's managed-process/session mechanism:

- Codex: retain the command session ID and poll it.
- Claude Code: use background Bash and poll the task ID.

Use managed execution for model search, stacking, AutoGluon, large SHAP runs,
or any operation likely to exceed a foreground tool timeout. Record
`execution_mode: managed_process` and the session/task ID when available.

Do not shorten the user-approved ML budget merely to fit a tool timeout.

## Analysis-to-model transitions

Before modeling a population used in analysis-only mode, inspect the prior
report and audit trail for full-data target statistics, plots or decisions. A
later split does not reseal overlapping target-exposed rows. Treat them as
discovery/development data and use a fresh external or prospective population
for an unbiased final estimate. If none is available, report development or
previously exposed benchmark evidence, not an untouched estimate.

Record `config.analysis.pre_partition_target_exposure` with:

- `status`: `none`, `development_only` or `full_population`;
- `source`, `final_population_overlap`, `values_viewed` and
  `decisions_influenced`.

Use `none` only with a null source, no overlap and empty audit arrays. A
`development_only` exposure cannot overlap final evaluation. For
`full_population`, use a disjoint external/prospective final population or, if
it overlaps, set `evaluation.independent_test: false`, record influenced
decisions and set `run_manifest.json.evaluation_exposure.status` to
`benchmark_selection`; never call it sealed.

## Budget and dependency control

- Ask for a compute or elapsed-time budget before expensive search.
- Use a conservative default when the user has no preference: enough to cover
  each eligible family at least once, then adaptive search within a bounded
  budget.
- Freeze roster membership and suitability from task, data and deployment
  criteria even when installed-package state is already known. Never use an
  import, package list or prior environment observation as an eligibility
  criterion.
- Control CPU threads, process parallelism, memory, and GPU use explicitly.
- Treat XGBoost, LightGBM and CatBoost as normal modeling dependencies. Install
  them inside the project environment when selected unless a concrete
  incompatibility or resource constraint applies. Ask before large optional
  systems such as AutoGluon.
- Record installation failures and continue with valid alternatives when
  possible.

Give every candidate exactly one unique ledger row with `family`,
`consideration_basis`, `suitability_status`, `dependency_status`,
`execution_status` and `reason`. Make `consideration_basis` explain
task/data/deployment fit without referring to installed packages. Every
supervised-tabular ledger must contain `xgboost`, `lightgbm` and `catboost`,
including exclusions. Use only:

- `suitability_status`: `eligible` or `excluded`;
- `dependency_status`: `installed`, `installed_for_run`, `not_required`,
  `installation_failed` or `user_declined`;
- `execution_status`: `attempted`, `excluded`, `installation_failed`,
  `user_declined` or `deferred_by_budget`.

Use `eligible`, an available dependency state (`installed`,
`installed_for_run` or `not_required`), and `attempted` for a run. Use
`excluded/not_required/excluded` only for an environment-independent
task/data/deployment reason. Use matching `installation_failed` or
`user_declined` dependency/execution states for an eligible family. Use
`deferred_by_budget` only with an available dependency and record the approved
wall-time/compute cap, estimated minimum coverage, remaining budget and
quantified shortfall. Require a concrete `reason` for every non-attempted row;
use null only for an attempted row with no failure. Package absence alone is
never an exclusion.

If an available family starts but cannot complete, keep
`execution_status: attempted` and record its `metrics.json` family-result
status as `failed`, with the reason and completed-trial count. Do not invent a
`failed` execution status.

## Improvement runs

Create every improvement attempt as a new immutable run. Do not overwrite,
relabel or delete the parent run. Record the run ID, parent run ID, parent
artifact hashes, code revision, changed hypothesis, and data/split
fingerprints.

Maintain a final-evidence exposure ledger with the evaluation-population
fingerprint, first-opened time and purpose, values viewed, and decisions they
influenced. Treat exposed holdout, external or outer-fold results as historical
benchmarks only; do not use them to select a descendant and still claim an
unbiased result. Use untouched future/external evidence for that claim, or
save the development-only child as explicitly incomplete with evaluation
pending. It must not pass the completed-run validator or deployment handoff
until a valid final evaluation contract and evidence exist.

## Audit trail

Record user choices/defaults, assumptions, rejected alternatives, task route,
compute budget, candidate roster, dependency outcomes and warning overrides in
the selected run's `config.json`.

Record split assignments/fingerprint and overlap/order audits in
`split_manifest.json`. Record run/parent identity, code revision, current
data/split fingerprints, changed hypothesis and final-evidence exposure in
`run_manifest.json`. Record metrics, family results, failures and actual usage
in `metrics.json`.

The artifacts on disk are authoritative. If a chat summary conflicts with
them, fix the summary or artifacts before handoff.
