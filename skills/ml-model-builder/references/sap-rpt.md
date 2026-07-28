# SAP RPT Comparison

## Contents

- [Position in the workflow](#position-in-the-workflow)
- [Offer and setup boundary](#offer-and-setup-boundary)
- [When to run](#when-to-run)
- [Fair evaluation](#fair-evaluation)
- [Context and query construction](#context-and-query-construction)
- [Response validation](#response-validation)
- [Cost, scale, and reproducibility](#cost-scale-and-reproducibility)
- [Artifacts and interpretation](#artifacts-and-interpretation)

## Position in the workflow

Treat SAP RPT as an explicit opt-in remote foundation-model benchmark beside
AutoGluon. Do not add it to the mandatory classical candidate roster and do not
let dependency availability affect XGBoost, LightGBM, or CatBoost coverage.

Use RPT for supervised tabular classification or regression only. Do not use
the tabular interface as a native forecasting, anomaly-detection, causal,
uplift, or survival model. Reframe only when ordinary row-level supervision is
scientifically valid.

The internal playground is useful because it avoids requiring the user to
provision an SAP BTP subaccount, AI Core, and RPT deployments. Label the CLI
and its results as an internal, non-production benchmark. If RPT is best,
report that result, but keep the deployable local pipeline separate until a
supported production serving route is designed and tested.

## Offer and setup boundary

Offer RPT once during model building or improvement for a compatible task,
whether or not this skill performed the earlier EDA:

> Would you like to benchmark SAP RPT? This optional experiment uses SAP's
> internal playground, requires you to install and configure the private RPT
> CLI, and sends selected training context and prediction rows to that service.

After opt-in:

1. Check whether `sap-rpt` is available without changing the environment.
2. If it is absent or unconfigured, direct the user to
   `https://github.tools.sap/DL-COE/rpt-cli`.
3. Ask the user to clone the repository, follow its current installation
   instructions, and complete the interactive playground configuration.
4. Pause only the RPT comparison until the user confirms completion. Continue
   local modeling when useful.
5. Never run the private clone or interactive configuration on the user's
   behalf, and never request, display, log, or store credentials or tokens.
6. After confirmation, inspect the installed CLI help/version and perform only
   non-secret readiness checks before the first request.

Do not hard-code mutable installation instructions from the private repository
into this skill. Record the installed package version and source revision
reported by the user's checkout. Pin the version/revision used by the run.

Before the first remote request, obtain explicit confirmation that the selected
training-context features and labels plus query features may be sent to the
named SAP endpoint. This is a remote-execution decision, not an automatic
redaction rule; do not hide or transform values merely because RPT is used.

## When to run

Prefer an initial, development-only pilot when:

- the task is classification or regression;
- the table contains meaningful business attributes rather than mostly opaque
  identifiers;
- labels and context rows are representative of the scoring population;
- the deployed RPT limits support the required columns, classes, context, and
  query batch;
- remote transfer, latency, and request volume are acceptable.

Skip or report the comparison as unavailable when:

- the task is incompatible;
- the user declines the remote request or private CLI setup;
- no fold-valid representative context can be constructed;
- the required probability semantics cannot be validated;
- offline, edge, strict low-latency, or high-volume serving is mandatory;
- the evaluation design would require an unaffordable number of remote calls.

Use a two-stage budget. First run one fixed RPT model/deployment and one
defensible context policy on development evidence. Try another RPT deployment,
context size, or selector only if the pilot is competitive within the observed
fold variation and the extra requests are justified. Never search RPT against
the sealed final evaluation set.

## Fair evaluation

- Reuse the persisted split assignments, target, feature-availability
  contract, eligible rows, weights, and primary metric implementation.
- Build every request context from the corresponding training fold only.
  Never include validation, holdout, active outer-fold, or future labels.
- Under nested CV, repeat the complete RPT context/model selection inside each
  outer-training partition or label the RPT result development-only.
- For temporal validation, use only rows and matured labels available before
  the fold's scoring origin. For grouped validation, preserve the declared
  group boundary in both context and query rows.
- Freeze the RPT model/deployment, schema, context policy, query batching,
  probability handling, and optional calibration before final evaluation.
- Opt in before opening final evidence. If an RPT final result influences a
  later model choice, mark the evaluation exposure `benchmark_selection` and
  require new future/external evidence for an unbiased selected-model claim.

Native preprocessing may differ from a local estimator, but information may
not. Use an explicit RPT schema. Represent numeric-looking identifiers and
category codes as strings when they are not quantities. Keep row identifiers
outside model features and use them to verify response alignment.

## Context and query construction

Discover and record the actual deployed limits; do not assume a model-card or
older CLI limit matches the active playground deployment. Use a conservative
query batch until readiness testing confirms the limit.

Use the full fold-training population when it fits. Otherwise create one
deterministic context per fold:

- classification: preserve representative class proportions and guarantee
  support for every evaluated class; record intentional rebalancing because it
  changes the apparent prior;
- regression: preserve the target range, tails, important groups, and time
  regimes using training rows only;
- grouped data: select at the group unit before selecting rows;
- temporal data: use an as-of or predeclared recency policy.

Record context row identifiers, order, selector name/version, seed, size, and a
SHA-256 fingerprint. Do not default to batch-dependent or target-informed
retrieval. When a stochastic selector is justified, test a small number of
seeds on development data, then freeze one policy.

Keep the same context for every query batch in a fold. Test a small development
sample for repeated-call stability and singleton-versus-batch invariance. Save
the production-equivalent batch size.

## Response validation

Use a project-local benchmark script that invokes the configured CLI in a
fail-closed way. Do not depend blindly on its sklearn wrapper. Require:

- process success and a successful status in the parsed response payload;
- the expected response count and unique row-identifier alignment;
- finite predictions with task-valid values;
- an explicitly recorded model ID and actual deployment ID;
- retained request IDs, timings, retries, and failure reasons.

Do not assume a display `model_id` proves which remote deployment handled the
request. Record the actual routing/deployment metadata used by the CLI.

For classification, request every class when feasible. Verify that returned
classes are known and unique, confidences are finite and within `[0, 1]`, and
each row forms a complete distribution that sums approximately to one. If the
service returns only truncated top-k confidence, report label/top-k metrics
only; do not calculate log loss, Brier score, calibration, average precision,
or AUC as if missing classes had zero probability. Fit any calibrator only from
task-valid out-of-fold RPT scores.

For regression, validate finite point predictions. Treat returned intervals as
unspecified until their semantics are documented, then measure empirical
coverage and width before relying on them.

## Cost, scale, and reproducibility

Calculate planned calls before running:

`sum(ceil(query_rows_per_fold / verified_batch_limit))`

RPT retransmits context with requests and is not a streaming or full-data
trainer. Record context/query rows, columns, approximate request bytes, call
count, failures/retries, wall time, median/p95 latency, and throughput.
For data larger than local resources, construct fold-valid bounded contexts and
queries where the data already lives; do not download the full source merely
to use RPT.

Remote results have limited point-in-time reproducibility. Record client
version/revision, auth mode, endpoint, model and deployment IDs, schema,
context/query fingerprints, selector seed, batch size, request IDs, timestamp,
and raw response hashes. Never store service credentials.

## Artifacts and interpretation

Record the conditional contracts in `config.json.comparison.sap_rpt` and
`metrics.json.sap_rpt` as defined in `artifacts.md`. Keep supporting evidence
under `<run-dir>/sap_rpt/`, including the project-local benchmark script,
protocol, context manifest, request/response manifests, prediction rows, and
hashes. Exclude tokens and credential files.

Keep RPT out of `config.json.search.candidates`,
`metrics.json.search.best_family`, and the deployable
`metrics.json.selection.model`. Report it separately with:

- development and any permitted final metric;
- comparison against the local winner on the same population;
- context coverage and probability limitations;
- latency/request volume and failures;
- `role: benchmark_only`;
- `reproducibility_status: limited_remote_service`;
- the internal, non-production serving limitation.

If RPT wins, recommend a separate productionization decision covering an
approved SAP AI Core/API route, exact deployment, production context
selection/storage, authentication, quotas, resilience, observability,
residency, monitoring, and validation on that exact serving path.
